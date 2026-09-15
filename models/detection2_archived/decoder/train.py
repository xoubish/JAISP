"""Two-GPU training of a learned renderer with fixed object positions.

Launch through torch.distributed.run. --check-only performs real forward and
backward passes, including distributed gradient synchronization, with zero
optimizer updates and no W&B run.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from ..common import ROOT, digest, write_json
from .data import CropTiles, BANDS, to_device
from .model import ObjectRenderer, objective
from .monitor import validate, curves


def atomic_save(value, path):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def parameter_hash(model):
    h = hashlib.sha256()
    for p in model.parameters(): h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train(prepared, out, resume=False, check_only=False):
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if not torch.cuda.is_available(): raise RuntimeError('An accessible GPU is required')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(4)
    if world > 1: dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    barrier = (lambda: dist.barrier(device_ids=[local_rank])) if world > 1 else (lambda: None)
    meta = json.loads((prepared/'metadata.json').read_text())
    cfg = meta['config']
    if not meta['complete'] and not check_only:
        raise ValueError('Partial preparation is accepted for --check-only, not full training')
    if world != 2 and not check_only:
        raise ValueError('This experiment is configured for two GPUs; launch with torch.distributed.run --nproc-per-node=2')
    if set(meta['train_tiles']) & set(meta['validation_tiles']): raise ValueError('Training/validation overlap')
    if any(t.endswith('_patch_25') for t in meta['train_tiles']): raise ValueError('Patch 25 appears in training')
    if not all(t.endswith('_patch_25') for t in meta['validation_tiles']): raise ValueError('Unexpected validation field')
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']+rank)
    lock = None
    if rank == 0:
        out.mkdir(parents=True, exist_ok=resume)
        lock = (out/'training.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    barrier()
    code_paths = list(Path(__file__).parent.glob('*.py'))+[ROOT/'models/detection2/object_decoder.py']
    manifest = {'prepared_metadata_sha256': digest(prepared/'metadata.json'),
                'selected_checkpoint_sha256': meta['selected_baseline']['checkpoint_sha256'],
                'source_sha256': {str(p): digest(p) for p in code_paths}, 'world_size': world}
    if rank == 0:
        if resume:
            if json.loads((out/'source_sha256.json').read_text()) != manifest:
                raise ValueError('Prepared data, code or GPU count changed; cannot resume this run')
        else:
            write_json(out/'config.json', {**cfg, 'prepared': str(prepared), 'world_size': world,
                       'scenes_per_step': world*cfg['train_crops_per_tile'], 'precision': 'float32',
                       'checkpoint_selection': 'final fixed epoch', 'check_only': check_only})
            write_json(out/'source_sha256.json', manifest)
    raw_model = ObjectRenderer(meta['feature_dim'], cfg, meta['band_shapes'], meta['stamp_sizes'], meta['pixel_scales_arcsec']).to(device)
    model = DDP(raw_model, device_ids=[local_rank]) if world > 1 else raw_model
    dataset = CropTiles(prepared, meta['train_tiles'])
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=cfg['seed']) if world > 1 else None
    loader = DataLoader(dataset, batch_size=None, sampler=sampler, shuffle=sampler is None,
                        num_workers=0 if check_only else cfg['workers_per_gpu'], pin_memory=True,
                        persistent_workers=not check_only and cfg['workers_per_gpu'] > 0)
    val_loader = DataLoader(CropTiles(prepared, meta['validation_tiles']), batch_size=None,
                            num_workers=0 if check_only else cfg['workers_per_gpu'], pin_memory=True) if rank == 0 else None
    if check_only:
        batch = to_device(next(iter(loader)), device)
        before = parameter_hash(raw_model)
        timings = []
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); start = time.monotonic()
            output = model(batch['features'], batch['present'], batch['positions'])
            loss, reconstruction = objective(output, batch, cfg)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite execution-check loss')
            loss.backward()
            torch.cuda.synchronize(); timings.append(time.monotonic()-start)
        grads = [p.grad for p in raw_model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads): raise FloatingPointError('Invalid gradients')
        grad_norm = float(torch.sqrt(sum(g.square().sum() for g in grads)))
        assert grad_norm > 0 and parameter_hash(raw_model) == before
        row = {'rank': rank, 'loss': float(loss), 'reconstruction_huber': float(reconstruction),
               'gradient_norm': grad_norm, 'optimizer_updates': 0, 'parameters_unchanged': True,
               'scenes': len(batch['features']), 'max_objects': batch['features'].shape[1],
               'real_objects': int(batch['present'].sum()), 'seconds_per_step': timings,
               'peak_memory_gb': torch.cuda.max_memory_allocated()/1e9}
        results = [None]*world
        if world > 1: dist.all_gather_object(results, row)
        else: results = [row]
        if rank == 0:
            validate(raw_model, val_loader, device, cfg, out, 0)
            curves(out)
            write_json(out/'check.json', {'ranks': results, 'optimizer_updates': 0,
                'gradient_norm_spread': max(r['gradient_norm'] for r in results)-min(r['gradient_norm'] for r in results),
                'parameter_count': sum(p.numel() for p in raw_model.parameters())})
            print(json.dumps({'execution_check': results}), flush=True)
        barrier()
        if world > 1: dist.destroy_process_group()
        if lock: lock.close()
        return

    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=cfg['minimum_learning_rate'])
    start_epoch, step = 0, 0
    if resume:
        state = torch.load(out/'training_state.pt', map_location='cpu', weights_only=False)
        raw_model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); start_epoch, step = state['epoch'], state['step']
        torch.set_rng_state(state['rng'][rank]['cpu']); torch.cuda.set_rng_state(state['rng'][rank]['cuda'], device)

    def save_training_state(epoch):
        rng = {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device)}
        rngs = [None]*world
        if world > 1:
            dist.all_gather_object(rngs, rng)
        else:
            rngs = [rng]
        if rank == 0:
            atomic_save({'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(), 'epoch': epoch, 'step': step, 'rng': rngs},
                        out/'training_state.pt')
        barrier()

    if not resume:
        save_training_state(0)
    run = None
    if rank == 0:
        import wandb
        wc = cfg['wandb']
        if wc['mode'] != 'online': raise ValueError('Online W&B is required for training')
        record_path = out/'wandb_run.json'
        record = json.loads(record_path.read_text()) if record_path.exists() else {'id': wandb.util.generate_id()}
        write_json(record_path, record)
        run = wandb.init(entity=wc['entity'], project=wc['project'], group=wc['group'],
                         id=record['id'], resume='allow' if resume else 'never',
                         name='Learned object decoder — fixed catalogue', job_type='decoder-warmup',
                         mode='online', dir=str(out), config={**cfg, 'world_size': world,
                         'training_tiles': len(meta['train_tiles']), 'validation_tiles': len(meta['validation_tiles']),
                         'kind': 'ordinary reconstruction from full-image cached features; fixed positions'},
                         settings=wandb.Settings(init_timeout=45, disable_code=True, disable_git=True))
        record['url'] = run.url; write_json(record_path, record)
        print('W&B:', run.url, flush=True)
        run.define_metric('val/*', step_metric='epoch')
        run.define_metric('train/*', step_metric='train/step')
        from .report import publish_report
        try:
            publish_report(out)
        except Exception as exc:
            write_json(out/'report_error.json', {'exception': type(exc).__name__, 'message': str(exc)})
            print('Report creation failed; numeric W&B logging continues. Retry with decoder.report.', flush=True)
    barrier()
    started = time.monotonic()
    try:
        if rank == 0 and start_epoch == 0:
            metrics = validate(raw_model, val_loader, device, cfg, out, 0)
            run.log(metrics); curves(out)
        barrier()
        with (out/f'history_rank{rank}.jsonl').open('a') as history:
            for epoch in range(start_epoch+1, cfg['epochs']+1):
                if sampler: sampler.set_epoch(epoch)
                model.train()
                accum = torch.zeros(3, device=device, dtype=torch.float64)
                for j, raw in enumerate(loader):
                    batch = to_device(raw, device)
                    optimizer.zero_grad(set_to_none=True)
                    output = model(batch['features'], batch['present'], batch['positions'])
                    loss, reconstruction = objective(output, batch, cfg)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite renderer loss')
                    loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
                    optimizer.step(); step += 1
                    accum += torch.stack([loss.detach().double(), reconstruction.detach().double(), loss.new_tensor(1.).double()])
                    if j % cfg['log_every_steps'] == 0 or j+1 == len(loader):
                        values = torch.stack([loss.detach(), reconstruction.detach(), grad.detach(), output['centering_arcsec2'].detach()])
                        if world > 1: dist.all_reduce(values); values /= world
                        if rank == 0:
                            row = {'epoch': epoch, 'train/step': step, 'train/loss': float(values[0]),
                                   'train/reconstruction_huber': float(values[1]), 'train/gradient_norm': float(values[2]),
                                   'train/template_centroid_rms_arcsec': float(values[3].sqrt()),
                                   'train/learning_rate': optimizer.param_groups[0]['lr'],
                                   'elapsed_minutes': (time.monotonic()-started)/60}
                            run.log(row); history.write(json.dumps(row)+'\n'); history.flush()
                            write_json(out/'status.json', {'stage': 'training', 'epoch': epoch,
                                       'step_in_epoch': j+1, 'steps_per_epoch': len(loader),
                                       'completed_epoch': epoch-1, 'total_epochs': cfg['epochs']})
                            print(json.dumps(row), flush=True)
                scheduler.step()
                if world > 1: dist.all_reduce(accum)
                barrier()
                if rank == 0:
                    metrics = validate(raw_model, val_loader, device, cfg, out, epoch)
                    metrics.update({'train/step': step, 'train/epoch_mean_loss': float(accum[0]/accum[2]),
                                    'train/epoch_mean_reconstruction_huber': float(accum[1]/accum[2])})
                    run.log(metrics); history.write(json.dumps(metrics)+'\n'); history.flush(); curves(out)
                    atomic_save({'model': raw_model.state_dict(), 'metadata': meta, 'epoch': epoch}, out/f'epoch_{epoch:02d}.pt')
                barrier()
                save_training_state(epoch)
                if rank == 0:
                    write_json(out/'status.json', {'stage': 'training', 'completed_epoch': epoch, 'total_epochs': cfg['epochs']})
                barrier()
        if rank == 0:
            atomic_save({'model': raw_model.state_dict(), 'metadata': meta, 'epoch': cfg['epochs']}, out/'final.pt')
            run.summary['completed_epochs'] = cfg['epochs']; run.finish()
            write_json(out/'status.json', {'stage': 'complete', 'completed_epoch': cfg['epochs']})
        barrier()
    except BaseException as exc:
        if rank == 0:
            write_json(out/'status.json', {'stage': 'failed', 'exception': type(exc).__name__, 'message': str(exc)})
            if run: run.finish(exit_code=1)
        raise
    finally:
        if world > 1 and dist.is_initialized(): dist.destroy_process_group()
        if lock: lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    train(a.prepared.resolve(), a.out.resolve(), a.resume, a.check_only)


if __name__ == '__main__':
    main()
