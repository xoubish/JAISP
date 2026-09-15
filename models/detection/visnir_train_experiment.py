"""Paired, fixed-budget fine-tuning of the production head on frozen features.

Run once with --extra-labels and once without it. Both runs share initialization,
data order, four cached augmentations, optimizer and schedule. Patch 25 is never
read by the training loop or used to choose a checkpoint. The final epoch is the
predeclared comparison; intermediate checkpoints are for diagnosis only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'models'))
from detection.cached_dataset import CachedFeatureDataset, collate_cached
from detection.centernet_detector import CenterNetDetector
from detection.centernet_loss import CenterNetLoss
from detection.train_centernet import _cached_forward


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--extra-labels', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'config.json').exists():
        raise FileExistsError(f'{args.out} already contains a run; use a new directory')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This experiment requires an accessible GPU')
    torch.backends.cudnn.benchmark = True
    base = ROOT/'checkpoints/q1_detection_v11/centernet_vis_sep.pt'
    dataset = CachedFeatureDataset(
        str(ROOT/'data/cached_features_v11_q1'), str(ROOT/'data/rubin_tiles_all'),
        str(ROOT/'data/euclid_tiles_all_q1'), labels_mode='vis_sep',
        extra_labels=str(args.extra_labels) if args.extra_labels else None)
    indices = [i for i, (tid, _) in enumerate(dataset._samples)
               if not tid.endswith('_patch_25')]
    train_ids = sorted({dataset._samples[i][0] for i in indices})
    assert len(train_ids) == 682 and len(indices) == 2728
    if args.extra_labels:
        extra = torch.load(args.extra_labels, weights_only=False, map_location='cpu')
        assert set(extra['promoted']) == set(train_ids)
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(42),
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0, collate_fn=collate_cached)
    model = CenterNetDetector.load(str(base), encoder=None, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = CenterNetLoss(sigma=2.0)
    config = dict(vars(args), seed=42, baseline=str(base),
                  baseline_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                  n_parameters=sum(p.numel() for p in model.parameters()),
                  n_train_samples=len(indices), train_tiles=train_ids,
                  precision='bfloat16 convolutions; float32 loss',
                  checkpoint_selection='final fixed epoch; no validation selection',
                  encoder='frozen v11 cached features', excluded_patches=[25])
    (args.out/'config.json').write_text(json.dumps(config, default=str, indent=2)+'\n')
    print(f'Start: {len(indices)} samples; {args.epochs} epochs; {device}', flush=True)
    start = time.monotonic()
    with (args.out/'history.jsonl').open('w', buffering=1) as log:
        for epoch in range(args.epochs):
            model.train()
            losses = []
            ep_start = time.monotonic()
            for step, batch in enumerate(loader):
                optimizer.zero_grad(set_to_none=True)
                features = batch['features'].to(device, non_blocking=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    out = _cached_forward(model, features)
                out = {k: v.float() for k, v in out.items()}
                loss = criterion(out, [c.to(device) for c in batch['centroids']])['loss_total']
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at epoch={epoch+1}, step={step}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
                if (step+1) % 100 == 0 or step == 0:
                    row = dict(epoch=epoch+1, step=step+1, steps_per_epoch=len(loader),
                               loss=float(np.mean(losses[-100:])),
                               elapsed_s=time.monotonic()-start)
                    log.write(json.dumps(row)+'\n')
                    print(json.dumps(row), flush=True)
            scheduler.step()
            model.save(str(args.out/f'epoch_{epoch+1:02d}.pt'))
            row = dict(epoch=epoch+1, mean_loss=float(np.mean(losses)),
                       epoch_s=time.monotonic()-ep_start, next_lr=scheduler.get_last_lr()[0])
            log.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
        model.save(str(args.out/'final.pt'))
    print(f'Finished in {(time.monotonic()-start)/60:.1f} minutes', flush=True)


if __name__ == '__main__':
    main()
