"""Per-band numeric diagnostics and local image/reconstruction/residual panels."""
from __future__ import annotations

import json
import numpy as np
import torch

from ..common import write_json
from .data import BANDS, to_device
from .model import objective


def gallery(out, batch, output, epoch, scene=0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(BANDS), 3, figsize=(8, 19), constrained_layout=True)
    for i, band in enumerate(BANDS):
        target = batch['targets'][i][scene].detach().cpu().numpy()
        prediction = output['images'][i][scene].detach().cpu().numpy()
        valid = batch['valid'][i][scene].detach().cpu().numpy()
        for j, image in enumerate([target, prediction]):
            shown = np.where(valid, np.arcsinh(image/3), np.nan)
            axes[i, j].imshow(shown, origin='lower', cmap='gray', vmin=-.5, vmax=3)
        axes[i, 2].imshow(np.where(valid, target-prediction, np.nan), origin='lower',
                         cmap='RdBu_r', vmin=-3, vmax=3)
        positions = batch['positions'][scene, batch['present'][scene], i].detach().cpu().numpy()
        for ax in axes[i, :2]:
            ax.scatter(positions[:, 0], positions[:, 1], s=16, marker='+', c='#f472b6', linewidths=.5)
            ax.set_xlim(-.5, target.shape[1]-.5); ax.set_ylim(-.5, target.shape[0]-.5)
        axes[i, 0].set_ylabel(band, fontsize=9)
        for ax in axes[i]: ax.set_xticks([]); ax.set_yticks([])
    for ax, title in zip(axes[0], ['Observed', 'Reconstructed', 'Observed − reconstructed']): ax.set_title(title)
    fig.suptitle(f'Fixed-catalogue renderer | epoch {epoch} | {batch["tile_id"]}\n'
                 'Observed/model: identical fixed asinh stretch. Residual: ±3 training MAD units.\n'
                 'Crosses: fixed detector positions. Ordinary reconstruction, not hidden-pixel prediction.', fontsize=10)
    fig.savefig(out/f'reconstruction_epoch_{epoch:02d}.png', dpi=120)
    plt.close(fig)


@torch.no_grad()
def validate(model, loader, device, cfg, out, epoch):
    model.eval()
    # Columns: residual SSE, zero-signal SSE, count, source SSE/count,
    # background SSE/count, learned-background-only SSE.
    totals = np.zeros((len(BANDS), 8), np.float64)
    loss_sum, scenes = 0., 0
    for index, raw in enumerate(loader):
        batch = to_device(raw, device)
        output = model(batch['features'], batch['present'], batch['positions'])
        _, recon = objective(output, batch, cfg)
        n = len(batch['features']); loss_sum += float(recon)*n; scenes += n
        for i, (prediction, target, valid, source) in enumerate(zip(output['images'], batch['targets'], batch['valid'], batch['source'])):
            residual = (prediction-target).double().square()
            source = source & valid; background = valid & ~source
            bg_error = (output['background'][:, i, None, None]-target).double().square()
            values = [residual[valid].sum(), target.double().square()[valid].sum(), valid.sum(),
                      residual[source].sum(), source.sum(), residual[background].sum(), background.sum(), bg_error[valid].sum()]
            totals[i] += np.array([float(v) for v in values])
        if index == 0:
            gallery(out, batch, output, epoch)
    metrics = {'epoch': epoch, 'val/reconstruction_huber': loss_sum/max(scenes, 1), 'val/scenes': scenes}
    gains, object_gains = [], []
    for band, row in zip(BANDS, totals):
        sse, zero, n, source_sse, source_n, bg_sse, bg_n, learned_bg_sse = row
        if not n or not zero:
            raise ValueError('Empty or degenerate validation band')
        prefix = 'val/'+band+'/'
        gain = 100*(1-sse/zero)
        object_gain = 100*(1-sse/max(learned_bg_sse, 1e-12))
        metrics.update({prefix+'rmse': float(np.sqrt(sse/n)), prefix+'background_baseline_rmse': float(np.sqrt(zero/n)),
                        prefix+'reconstruction_gain_percent': float(gain), prefix+'gain_over_learned_background_percent': float(object_gain),
                        prefix+'source_rmse': float(np.sqrt(source_sse/max(source_n, 1))),
                        prefix+'background_rmse': float(np.sqrt(bg_sse/max(bg_n, 1)))})
        gains.append(gain); object_gains.append(object_gain)
    metrics['val/mean_reconstruction_gain_percent'] = float(np.mean(gains))
    metrics['val/mean_gain_over_learned_background_percent'] = float(np.mean(object_gains))
    write_json(out/f'validation_epoch_{epoch:02d}.json', {'metrics': metrics, 'band_sums': totals.tolist(),
        'band_sum_columns': ['residual_sse', 'zero_signal_sse', 'pixels', 'source_sse', 'source_pixels',
                             'background_sse', 'background_pixels', 'learned_background_only_sse'],
        'scope': 'Fixed overlapping image crops in held-out patch 25; cached features see full images. '
                 'Renderer reconstruction metrics, not detector completeness/purity or independent-pixel statistics.'})
    return metrics


def curves(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = [json.loads(p.read_text())['metrics'] for p in sorted(out.glob('validation_epoch_*.json'))]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for band in BANDS:
        axes[0].plot([r['epoch'] for r in records], [r[f'val/{band}/reconstruction_gain_percent'] for r in records], 'o-', markersize=3, label=band)
        axes[1].plot([r['epoch'] for r in records], [r[f'val/{band}/source_rmse'] for r in records], 'o-', markersize=3, label=band)
    axes[0].axhline(0, color='gray', linewidth=1)
    axes[0].set(title='Improvement over zero-signal background after training-data normalization', ylabel='MSE reduction (%)')
    axes[1].set(title='Residual within 1.5 arcsec of fixed proposals', ylabel='RMSE (training MAD units)', xlabel='Epoch')
    for ax in axes: ax.grid(alpha=.2)
    axes[0].legend(ncol=5, fontsize=8)
    fig.savefig(out/'reconstruction_curves.png', dpi=150)
    plt.close(fig)
