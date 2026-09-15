"""Foreground, W&B-monitored paired test of uncertain-background masking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import (resolve, split_tiles, source_manifest, digest, write_json, online_run)
from .data import TrainingDataset, collate
from .validate import validate
from detection.centernet_detector import CenterNetDetector
from detection.centernet_loss import CenterNetLoss
from detection.train_centernet import _cached_forward


def forward_loss(model, batch, criterion, device, use_ignore):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        pred = _cached_forward(model, batch['features'].to(device))
    pred = {k: v.float() for k, v in pred.items()}
    losses = criterion(pred, [c.to(device) for c in batch['centroids']],
                       ignore_masks=batch['ignore'].to(device) if use_ignore else None)
    if not all(torch.isfinite(v).all() for v in losses.values()):
        raise FloatingPointError('Nonfinite loss')
    return losses


def train(prepared, out, arm, device, check_only=False, epochs=None):
    payload = torch.load(prepared / 'labels.pt', map_location='cpu', weights_only=False)
    cfg = dict(payload['config'])
    if epochs is not None:
        if epochs < 1:
            raise ValueError('epochs must be positive')
        cfg['epochs'] = epochs
    train_ids, val_ids = split_tiles(cfg)
    if not check_only and (not payload['complete'] or set(payload['tiles']) != set(train_ids)):
        raise ValueError('Full training requires proposals prepared for every training tile')
    if set(payload['tiles']) & set(val_ids):
        raise ValueError('Training/validation overlap')
    if not set(payload['validation_tiles']) <= set(val_ids):
        raise ValueError('Unexpected validation tile IDs')
    manifest = json.loads((prepared / 'source_sha256.json').read_text())
    for key in ('initial_checkpoint', 'vis_labels', 'nir_labels'):
        if digest(resolve(cfg[key])) != manifest[cfg[key]]:
            raise ValueError(f'{key} changed after proposals were prepared')
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('An accessible GPU is required')
    torch.set_num_threads(4)
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator().manual_seed(cfg['seed'])
    dataset = TrainingDataset(cfg, payload['tiles'], arm)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True,
                        generator=generator, num_workers=0 if check_only else cfg['workers'],
                        pin_memory=True, collate_fn=collate)
    model = CenterNetDetector.load(str(resolve(cfg['initial_checkpoint'])), encoder=None, device=device)
    criterion = CenterNetLoss(sigma=2.0)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'config.json', {**cfg, 'arm': arm, 'check_only': check_only,
                                   'prepared': str(prepared.resolve()),
                                   'prepared_epochs': payload['config']['epochs'],
                                   'epoch_override': epochs,
                                   'cosine_T_max': cfg['epochs'],
                                   'labels_sha256': digest(prepared / 'labels.pt'),
                                   'train_tiles': sorted(payload['tiles']),
                                   'validation_tiles': payload['validation_tiles'],
                                   'precision': 'bf16 convolutions, float32 loss',
                                   'selection': 'final fixed epoch; validation only for monitoring'})
    write_json(out / 'source_sha256.json', source_manifest(cfg))
    if check_only:
        # Real-data execution check: backward only. No optimizer, no weight
        # update, no checkpoint, and no W&B upload during this diagnostic.
        original_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.train()
        batch = next(iter(loader))
        losses = forward_loss(model, batch, criterion, device, arm == 'unknown')
        losses['loss_total'].backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError('Nonfinite gradients')
        # Training-mode BatchNorm changes running buffers even without an
        # optimizer. Restore them before checking the initial model's metrics.
        model.load_state_dict(original_state)
        model.zero_grad(set_to_none=True)
        result = {'losses': {k: float(v.detach()) for k, v in losses.items()},
                  'ignored_fraction': float(batch['ignore'].float().mean()),
                  'weight_updates': 0, 'batchnorm_buffers_restored': True,
                  'validation': validate(model, cfg, payload['validation_tiles'][:1], device)}
        write_json(out / 'check.json', result)
        print(json.dumps(result, indent=2), flush=True)
        return
    run = online_run(cfg, out, 'train', f'detection2-{arm}-{cfg["epochs"]}ep', {'arm': arm})
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    start = time.monotonic()
    global_step = 0
    try:
        run.define_metric('train/*', step_metric='train/step')
        run.define_metric('val/*', step_metric='epoch')
        initial = validate(model, cfg, payload['validation_tiles'], device)
        write_json(out / 'validation_epoch_00.json', initial)
        run.log({'epoch': 0, **initial['metrics']})
        with (out / 'history.jsonl').open('w', buffering=1) as log:
            for epoch in range(1, cfg['epochs'] + 1):
                model.train()
                epoch_losses = []
                for step, batch in enumerate(loader):
                    optimizer.zero_grad(set_to_none=True)
                    losses = forward_loss(model, batch, criterion, device, arm == 'unknown')
                    losses['loss_total'].backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError('Nonfinite gradient norm')
                    optimizer.step()
                    global_step += 1
                    scalar = {k: float(v.detach()) for k, v in losses.items()}
                    epoch_losses.append(scalar['loss_total'])
                    if step % 20 == 0 or step + 1 == len(loader):
                        row = {'epoch': epoch, 'train/step': global_step,
                               'train/loss': scalar['loss_total'], 'train/heatmap_loss': scalar['loss_hm'],
                               'train/offset_loss': scalar['loss_off'],
                               'train/n_sources': scalar['n_sources'], 'train/gradient_norm': float(grad_norm),
                               'train/ignored_fraction': float(batch['ignore'].float().mean()),
                               'train/lr': optimizer.param_groups[0]['lr'],
                               'elapsed_seconds': time.monotonic() - start}
                        run.log(row)
                        log.write(json.dumps(row) + '\n')
                        print(json.dumps(row), flush=True)
                scheduler.step()
                model.save(str(out / f'epoch_{epoch:02d}.pt'))
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(), 'epoch': epoch,
                            'global_step': global_step, 'loader_rng': generator.get_state(),
                            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                            'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
                            'config': cfg, 'arm': arm}, out / 'training_state.pt')
                result = validate(model, cfg, payload['validation_tiles'], device)
                write_json(out / f'validation_epoch_{epoch:02d}.json', result)
                row = {'epoch': epoch, 'train/step': global_step,
                       'train/epoch_mean_loss': float(np.mean(epoch_losses)), **result['metrics']}
                run.log(row)
                log.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
            model.save(str(out / 'final.pt'))
        run.summary['completed_epochs'] = cfg['epochs']
        run.summary['elapsed_minutes'] = (time.monotonic() - start) / 60
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arm', choices=('control', 'unknown'), required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int, help='Override training budget without regenerating fixed proposals')
    p.add_argument('--check-only', action='store_true', help='One backward pass, no optimizer or weight updates')
    a = p.parse_args()
    train(a.prepared, a.out, a.arm, torch.device(a.device), a.check_only, a.epochs)


if __name__ == '__main__':
    main()
