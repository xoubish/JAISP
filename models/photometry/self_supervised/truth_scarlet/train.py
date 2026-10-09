"""Train or evaluate a truth-supervised amortised-scarlet variant, logging to Weights & Biases.

    python -m models.photometry.self_supervised.truth_scarlet.train --variant A1 --device cuda:0
    python -m models.photometry.self_supervised.truth_scarlet.train --variant A0 --eval-only \
        --init models/photometry/self_supervised/runs/amortised_scarlet/fixedbg/epoch1.pt

Loss per batch (architecture_brainstorm fig. 3):
    lambda_f * Huber((f - f*) / sigma_oracle)               flux, in noise units
  + lambda_t * per-source model-image error / its norm       light ownership
  + lambda_chi * mean_b log(chi2_b / dof_b)                  explain every pixel
  + lambda_bias * sum over S/N bins of (batch mean signed normalised error)^2
sigma_oracle is the error of the true-template fit, so the head cannot lower the
flux loss by inflating its own conditional errors.

Learning rate: linear warm-up, then cosine decay to --lr-final of the peak.
Validation every --eval-every training scenes: injected validation scenes (vs true
flux) and real validation-tile scenes (vs MER), with the calibrated mixture
photometer as a fixed reference. Checkpoints are selected on the injection
metric `val/median_abs_chi` (mean over bands of median |f - f*| / sigma_oracle),
never on chi-square.
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..core import BANDS
from ..scene_features import SceneEncoder
from .model import TruthScarletHead, TruthScarlet, VARIANTS
from .scenes import OUT, load_scene

ROOT = Path(__file__).resolve().parents[4]
EUCLID = [b for b in BANDS if b.startswith('euclid_')]
SNR_BINS = (0, 3, 10, 30, np.inf)
SHORT = {b: b.split('_')[1] for b in BANDS}


# ----------------------------------------------------------------------------- data
class Scenes(torch.utils.data.Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, k):
        scene, truth = load_scene(self.paths[k]); return scene, truth, self.paths[k].stem


def identity(item): return item   # keep numpy PSF stamps as numpy (default conversion would make them tensors)


def loader(paths, shuffle, workers):
    return torch.utils.data.DataLoader(Scenes(paths), batch_size=None, shuffle=shuffle, num_workers=workers, collate_fn=identity,
                                       persistent_workers=workers > 0, prefetch_factor=4 if workers else None)


# ----------------------------------------------------------------------------- loss
def scene_loss(out, scene, truth, device, a):
    """Per-scene loss terms and the normalised signed errors used by the batch bias penalty."""
    flux_terms, template_terms, chis, snrs = [], [], [], []
    for band, r in out['results'].items():
        idx = r['source_indices']; t = truth[band]
        f_true = torch.as_tensor(t['flux'], device=device, dtype=torch.float64)[idx]
        sigma = torch.as_tensor(t['error'], device=device, dtype=torch.float64)[idx]
        chi = (r['flux'] - f_true) / sigma
        flux_terms.append(F.huber_loss(chi, torch.zeros_like(chi), delta=a.huber, reduction='none'))
        chis.append(chi); snrs.append((f_true / sigma).detach())
        d = scene['bands'][band]
        w = torch.where(d['mask'].to(device), torch.rsqrt(d['variance'].to(device).clamp_min(1e-30)), torch.zeros_like(d['variance'].to(device))).flatten().double()
        target = torch.as_tensor(t['profiles'], device=device, dtype=torch.float64).reshape(len(t['flux']), -1)[idx] * f_true[:, None]
        model = (out['templates'][band][:, idx].double() * r['flux'][None]).T
        norm = ((target * w) ** 2).sum(1).clamp_min(a.template_floor)
        template_terms.append((((model - target) * w) ** 2).sum(1) / norm)
    return dict(flux=torch.cat(flux_terms).mean(), template=torch.cat(template_terms).mean(), chi2=out['total'].double(),
                chi=torch.cat(chis), snr=torch.cat(snrs))


def bias_penalty(chis, snrs, clip):
    """Squared batch-mean signed error per S/N bin; errors clipped at the Huber scale so bright outliers cannot dominate."""
    chi, snr = torch.cat(chis).clamp(-clip, clip), torch.cat(snrs); total = chi.new_zeros(())
    for lo, hi in zip(SNR_BINS[:-1], SNR_BINS[1:]):
        sel = (snr >= lo) & (snr < hi)
        if int(sel.sum()) >= 4: total = total + chi[sel].mean() ** 2
    return total


# ----------------------------------------------------------------------------- evaluation
def run_injections(model, paths, device):
    rows, failures = [], 0
    for path in paths:
        scene, truth = load_scene(path)
        try:
            with torch.enable_grad(): out = model.run(scene)
        except (ValueError, RuntimeError): failures += 1; continue
        for band, r in out['results'].items():
            for j, f, e in zip(r['source_indices'].tolist(), r['flux'].tolist(), r['error'].tolist()):
                rows.append(dict(name=path.stem, source=j, band=band, flux=f, error=e, reduced_chi2=float(r['chi2'] / max(r['dof'], 1))))
    return pd.DataFrame(rows), failures


def injection_frame(rows, truth, manifest):
    d = rows.merge(truth, on=['name', 'source', 'band'], how='inner').merge(manifest, on='name', how='left')
    d['chi'] = (d.flux - d.truth_flux) / d.oracle_error; d['frac'] = (d.flux - d.truth_flux) / d.truth_flux
    d['context'] = np.where(d.n_sources == 1, 'isolated', np.where(d.min_sep_arcsec < .7, 'blend <0.7"', 'blend >=0.7"'))
    return d


def injection_metrics(d, prefix):
    m = {}
    per_band = d.groupby('band').chi.apply(lambda x: np.median(np.abs(x)))
    m[f'{prefix}/median_abs_chi'] = float(per_band.mean()); m[f'{prefix}/mean_abs_chi'] = float(np.abs(d.chi).mean())
    for b, v in per_band.items(): m[f'{prefix}_band/{SHORT[b]}_median_abs_chi'] = float(v)
    for lo, hi in zip(SNR_BINS[:-1], SNR_BINS[1:]):
        sel = (d.true_snr >= lo) & (d.true_snr < hi)
        if sel.sum(): m[f'{prefix}_snr/median_frac_bias_{lo}-{hi}'] = float(np.median(d.frac[sel])); m[f'{prefix}_snr/median_abs_chi_{lo}-{hi}'] = float(np.median(np.abs(d.chi[sel])))
    for c, g in d.groupby('context'): m[f'{prefix}_context/median_abs_chi_{c}'] = float(np.median(np.abs(g.chi)))
    m[f'{prefix}/median_frac_bias'] = float(np.median(d.frac))
    return m


def run_real(model, paths, device):
    rows, failures = [], 0
    for path in paths:
        scene, _ = load_scene(path)
        try:
            with torch.enable_grad(): out = model.run(scene)
        except (ValueError, RuntimeError): failures += 1; continue
        for band in EUCLID:
            r = out['results'][band]
            if 0 not in r['source_indices'].tolist(): continue
            k = r['source_indices'].tolist().index(0)
            rows.append(dict(name=path.stem, band=band, flux=float(r['flux'][k]), error=float(r['error'][k])))
    return pd.DataFrame(rows), failures


def real_frame(rows, manifest):
    parts = []
    for band in EUCLID:
        g = rows[rows.band == band].merge(manifest[['name', 'mer_is_star', f'{band}__ref_ujy', f'{band}__ujy_per_native']], on='name')
        g = g.rename(columns={f'{band}__ref_ujy': 'ref_ujy', f'{band}__ujy_per_native': 'conv'}); g['flux_ujy'] = g.flux * g.conv
        ok = (g.flux_ujy > 0) & (g.ref_ujy > 0)
        g['mag'] = np.where(ok, 23.9 - 2.5 * np.log10(g.flux_ujy.where(ok)), np.nan); g['ref_mag'] = np.where(ok, 23.9 - 2.5 * np.log10(g.ref_ujy.where(ok)), np.nan)
        parts.append(g)
    d = pd.concat(parts, ignore_index=True); d['dmag'] = d.mag - d.ref_mag
    return d


def real_metrics(d, prefix):
    m = {}
    for b, g in d.groupby('band'):
        x = g.dmag.dropna()
        if len(x): m[f'{prefix}/{SHORT[b]}_nmad_mag'] = float(1.4826 * np.median(np.abs(x - np.median(x)))); m[f'{prefix}/{SHORT[b]}_median_dmag'] = float(np.median(x))
    nm = [v for k, v in m.items() if k.endswith('_nmad_mag')]
    if nm: m[f'{prefix}/mean_nmad_mag'] = float(np.mean(nm))
    return m


# ----------------------------------------------------------------------------- plots
COLORS = {'isolated': '#3182bd', 'blend >=0.7"': '#31a354', 'blend <0.7"': '#e6550d'}


def fig_flux_scatter(d, ref, title):
    fig, axs = plt.subplots(2, 5, figsize=(20, 8.4), sharex=True, sharey=True)
    for ax, band in zip(axs.flat, BANDS):
        if ref is not None:
            g = ref[ref.band == band]; ax.scatter(g.true_snr, g.flux / g.oracle_error, s=3, c='#bdbdbd', alpha=.5, label='mixture (reference)', rasterized=True)
        g = d[d.band == band]
        for c, col in COLORS.items():
            h = g[g.context == c]; ax.scatter(h.true_snr, h.flux / h.oracle_error, s=4, c=col, alpha=.6, label=c, rasterized=True)
        ax.plot([.5, 500], [.5, 500], 'k', lw=.8)
        ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=1); ax.set_xlim(.5, 500); ax.set_ylim(-10, 500)
        mm = np.median(np.abs(g.chi)); rm = np.median(np.abs(ref[ref.band == band].chi)) if ref is not None else np.nan
        ax.set_title(f'{SHORT[band]}   med|Δ|/σ {mm:.2f} (mixture {rm:.2f})', fontsize=9)
    for ax in axs[1]: ax.set_xlabel('true flux / σ_oracle')
    for ax in axs[:, 0]: ax.set_ylabel('measured flux / σ_oracle')
    axs[0, 0].legend(fontsize=7, markerscale=3, loc='upper left')
    fig.suptitle(title); fig.tight_layout(); return fig


def running(x, y, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x < hi)
        out.append((np.sqrt(lo * hi), *np.percentile(y[sel], [16, 50, 84])) if sel.sum() >= 10 else (np.sqrt(lo * hi), np.nan, np.nan, np.nan))
    return np.array(out)


def fig_frac_vs_snr(d, ref, title):
    edges = np.geomspace(.7, 300, 14)
    fig, axs = plt.subplots(2, 5, figsize=(20, 7.6), sharex=True, sharey=True)
    for ax, band in zip(axs.flat, BANDS):
        for frame, col, lab, ls in ((ref, '#636363', 'mixture', '--'), (d, '#d94801', 'this model', '-')):
            if frame is None: continue
            g = frame[frame.band == band]; r = running(g.true_snr.to_numpy(), g.frac.to_numpy(), edges)
            ax.plot(r[:, 0], r[:, 2], ls, color=col, label=lab); ax.fill_between(r[:, 0], r[:, 1], r[:, 3], color=col, alpha=.12)
        ax.axhline(0, color='k', lw=.6); ax.set_xscale('log'); ax.set_xlim(.7, 300); ax.set_ylim(-1, 1); ax.set_title(SHORT[band], fontsize=9)
    for ax in axs[1]: ax.set_xlabel('true S/N')
    for ax in axs[:, 0]: ax.set_ylabel('(f − f*) / f*  (16/50/84%)')
    axs[0, 0].legend(fontsize=8); fig.suptitle(title); fig.tight_layout(); return fig


def fig_real(d, ref, title):
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6), sharex=True, sharey=True)
    for ax, band in zip(axs, EUCLID):
        if ref is not None:
            g = ref[ref.band == band]; ax.scatter(g.ref_mag, g.dmag, s=4, c='#bdbdbd', alpha=.6, label='mixture (reference)', rasterized=True)
        g = d[d.band == band]
        ax.scatter(g.ref_mag[~g.mer_is_star], g.dmag[~g.mer_is_star], s=5, c='#d94801', alpha=.7, label='this model, galaxies', rasterized=True)
        ax.scatter(g.ref_mag[g.mer_is_star], g.dmag[g.mer_is_star], s=12, marker='*', c='#08519c', label='this model, MER stars', rasterized=True)
        ax.axhline(0, color='k', lw=.6); ax.set_ylim(-1, 1); ax.set_xlabel('MER AB mag')
        x = g.dmag.dropna(); nm = 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan
        ax.set_title(f'{SHORT[band]}  NMAD {nm:.3f}, median {np.median(x) if len(x) else np.nan:+.3f}', fontsize=9)
    axs[0].set_ylabel('this − MER [mag]'); axs[0].legend(fontsize=7, markerscale=2); fig.suptitle(title); fig.tight_layout(); return fig


def fig_residuals(model, paths, device, title):
    show = ('euclid_VIS', 'rubin_r', 'euclid_H')
    fig, axs = plt.subplots(len(paths), 2 + len(show), figsize=(3.1 * (2 + len(show)), 3.1 * len(paths)), squeeze=False)
    for row, path in zip(axs, paths):
        scene, truth = load_scene(path)
        try:
            with torch.enable_grad(): out = model.run(scene)
        except (ValueError, RuntimeError) as exc:
            row[0].set_title(f'failed: {exc}'[:60], fontsize=7); [a.axis('off') for a in row]; continue
        d = scene['bands']['euclid_VIS']; r = out['results']['euclid_VIS']
        image = d['image'].numpy(); model_img = r['model'].detach().cpu().numpy(); s = np.sqrt(np.median(d['variance'].numpy()))
        vmax = np.percentile(image / s, 99.5)
        for a, im, t in ((row[0], image / s, 'VIS data / σ'), (row[1], model_img / s, 'VIS model / σ')):
            a.imshow(np.arcsinh(im), origin='lower', cmap='magma', vmin=-1, vmax=np.arcsinh(vmax)); a.set_title(t, fontsize=8)
        for a, band in zip(row[2:], show):
            db = scene['bands'][band]; rb = out['results'][band]
            chi = np.where(db['mask'].numpy(), (db['image'].numpy() - rb['model'].detach().cpu().numpy()) / np.sqrt(db['variance'].numpy()), np.nan)
            a.imshow(chi, origin='lower', cmap='RdBu_r', vmin=-4, vmax=4)
            idx = rb['source_indices'].tolist()
            tag = ', '.join(f'{f / truth[band]["flux"][j]:.2f}' for j, f in zip(idx, rb['flux'].tolist()))
            a.set_title(f'{SHORT[band]} χ resid, χ²/dof {float(rb["chi2"] / max(rb["dof"], 1)):.2f}\nf/f* = {tag}', fontsize=8)
        for a in row: a.set_xticks([]); a.set_yticks([])
        for a in row[:2]: a.plot(*d['positions'].numpy().T, '+', c='c', ms=6)
        row[0].set_ylabel(path.stem, fontsize=8)
    fig.suptitle(title); fig.tight_layout(); return fig


def gallery_paths(val_dir, manifest):
    picks = []
    for sel in (manifest[(manifest.n_sources == 1) & (manifest.vis_snr < 4)], manifest[(manifest.n_sources == 1) & (manifest.vis_snr > 30)],
                manifest[(manifest.n_sources == 2) & (manifest.min_sep_arcsec < .6) & (manifest.vis_snr > 8)], manifest[manifest.n_sources == 3]):
        if len(sel): picks.append(val_dir / 'scenes' / f'{sel.iloc[0]["name"]}.npz')
    return picks


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--variant', choices=sorted(VARIANTS), required=True)
    p.add_argument('--data', default=str(OUT)); p.add_argument('--output', default='')
    p.add_argument('--foundation', default='models/checkpoints/jaisp_v11_q1_soft/checkpoint_best.pt')
    p.add_argument('--init', default='', help='head checkpoint to start from (required for A0)')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--resume', default='', help="checkpoint to continue from ('auto' = <output>/last.pt); continues the same W&B run")
    p.add_argument('--epochs', type=int, default=20); p.add_argument('--lr', type=float, default=3e-4); p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--batch', type=int, default=8); p.add_argument('--width', type=int, default=64); p.add_argument('--steps', type=int, default=2)
    p.add_argument('--lambda-flux', type=float, default=1.); p.add_argument('--lambda-template', type=float, default=1.)
    p.add_argument('--lambda-chi2', type=float, default=.1); p.add_argument('--lambda-bias', type=float, default=.5)
    p.add_argument('--huber', type=float, default=3.); p.add_argument('--template-floor', type=float, default=25.)
    p.add_argument('--eval-every', type=int, default=8000, help='training scenes between validations')
    p.add_argument('--lr-final', type=float, default=.05, help='cosine decay to this fraction of --lr by the last step')
    p.add_argument('--val-limit', type=int, default=800); p.add_argument('--real-limit', type=int, default=500)
    p.add_argument('--train-limit', type=int, help='training scenes per epoch, for smoke tests')
    p.add_argument('--workers', type=int, default=6); p.add_argument('--device', default='cuda:0'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--wandb-project', default='jaisp-photometry'); p.add_argument('--wandb-group', default='truth_scarlet')
    p.add_argument('--wandb-mode', default='online', choices=('online', 'offline', 'disabled')); p.add_argument('--name', default='')
    a = p.parse_args()
    import wandb
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    flags = VARIANTS[a.variant]; data = Path(a.data); device = torch.device(a.device)
    name = a.name or a.variant; out_dir = Path(a.output or data / 'runs' / name); out_dir.mkdir(parents=True, exist_ok=True)
    if a.variant == 'A0' and not (a.eval_only and a.init): raise SystemExit('A0 is the existing chi-square checkpoint: use --eval-only --init <ckpt>')

    head = TruthScarletHead(width=a.width, steps=a.steps, per_band=flags['per_band'], implicit_psf=flags['implicit_psf']).to(device)
    if a.init:
        state = torch.load(ROOT / a.init if not Path(a.init).is_absolute() else a.init, map_location=device, weights_only=False)['head']
        missing, unexpected = head.load_state_dict(state, strict=False)
        if unexpected: raise ValueError(f'unexpected keys in {a.init}: {unexpected[:5]}')
        print(f'initialised from {a.init}; new parameters: {sorted({k.split(".")[0] for k in missing})}', flush=True)
    resume = None; run_id = None
    if a.resume:
        resume_path = out_dir / 'last.pt' if a.resume == 'auto' else Path(a.resume)
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        head.load_state_dict(resume['head'])
        run_id = resume.get('wandb_id')
        if run_id is None and (out_dir / 'wandb/latest-run').exists():   # checkpoints written before resume support
            run_id = Path(os.path.realpath(out_dir / 'wandb/latest-run')).name.split('-')[-1]
        print(f"resuming from {resume_path} at {resume['seen']} scenes, W&B run {run_id}", flush=True)
    encoder = None
    if flags['foundation']: encoder = SceneEncoder(ROOT / a.foundation); encoder.encoder.to(device)
    model = TruthScarlet(head, encoder, device, foundation=flags['foundation'])

    train_paths = sorted((data / 'train/scenes').glob('scene_*.npz'))
    val_paths = sorted((data / 'val/scenes').glob('scene_*.npz'))[:a.val_limit]
    real_paths = sorted((data / 'val_real/scenes').glob('*.npz'))[:a.real_limit]
    truth = pd.read_csv(data / 'val/truth.csv'); truth['name'] = truth.scene.map(lambda s: f'scene_{s:05d}')
    manifest = pd.read_csv(data / 'val/scenes.csv'); manifest['name'] = manifest.scene.map(lambda s: f'scene_{s:05d}')
    real_manifest = pd.read_csv(data / 'val_real/manifest.csv') if (data / 'val_real/manifest.csv').exists() else None
    ref_inj = ref_real = None
    if (data / 'val/reference_mixture.csv').exists():
        ref_inj = injection_frame(pd.read_csv(data / 'val/reference_mixture.csv'), truth, manifest)
        ref_inj = ref_inj[ref_inj.name.isin({p.stem for p in val_paths})]
    if real_manifest is not None and (data / 'val_real/reference_mixture.csv').exists():
        rr = pd.read_csv(data / 'val_real/reference_mixture.csv'); rr = rr[(rr.source == 0) & rr.band.isin(EUCLID) & rr.name.isin({p.stem for p in real_paths})]
        ref_real = real_frame(rr, real_manifest)
    gallery = gallery_paths(data / 'val', manifest)

    config = dict(vars(a), **flags, train_scenes=len(train_paths), val_scenes=len(val_paths), real_scenes=len(real_paths),
                  head_version=TruthScarletHead.version, n_parameters=sum(p.numel() for p in head.parameters()))
    run = wandb.init(project=a.wandb_project, group=a.wandb_group, name=name, config=config, mode=a.wandb_mode, dir=str(out_dir),
                     **(dict(id=run_id, resume='allow') if run_id else {}))
    ref_metrics = {}
    if ref_inj is not None: ref_metrics.update(injection_metrics(ref_inj, 'mixture_val'))
    if ref_real is not None: ref_metrics.update(real_metrics(ref_real, 'mixture_real'))
    for k, v in ref_metrics.items(): run.summary[k] = v
    print(json.dumps(dict(variant=a.variant, train=len(train_paths), val=len(val_paths), real=len(real_paths), **{k: v for k, v in ref_metrics.items() if '/' in k and k.count('_') < 4})), flush=True)

    best = math.inf; seen = 0; optimizer = None
    if resume is not None:
        seen = int(resume['seen'])
        best = resume.get('best', math.inf)
        if best == math.inf and (out_dir / 'best.pt').exists():
            best = torch.load(out_dir / 'best.pt', map_location='cpu', weights_only=False)['metrics']['val/median_abs_chi']

    def evaluate(epoch):
        nonlocal best
        head.eval(); t0 = time.time()
        rows, fail = run_injections(model, val_paths, device); d = injection_frame(rows, truth, manifest)
        log = injection_metrics(d, 'val'); log['val/failures'] = fail
        log['val/mean_reduced_chi2_log'] = float(np.log(rows.drop_duplicates(['name', 'band']).reduced_chi2).mean())
        figs = dict(flux_scatter=fig_flux_scatter(d, ref_inj, f'{name}: injected validation, {seen} training scenes'),
                    frac_vs_snr=fig_frac_vs_snr(d, ref_inj, f'{name}: fractional flux error vs S/N, {seen} training scenes'),
                    residuals=fig_residuals(model, gallery, device, f'{name}: residual gallery, {seen} training scenes'))
        if real_paths:
            rrows, rfail = run_real(model, real_paths, device); rd = real_frame(rrows, real_manifest)
            log.update(real_metrics(rd, 'real')); log['real/failures'] = rfail
            figs['real_vs_mer'] = fig_real(rd, ref_real, f'{name}: real validation tiles vs MER, {seen} training scenes')
        for k, v in ref_metrics.items():
            if k in ('mixture_val/median_abs_chi', 'mixture_real/mean_nmad_mag'): log[k] = v   # flat reference lines on the same charts
        plots = out_dir / 'plots'; plots.mkdir(exist_ok=True)
        for k, f in figs.items(): f.savefig(plots / f'{k}_{seen:07d}.png', dpi=90); log[f'plots/{k}'] = wandb.Image(f); plt.close(f)
        log.update(epoch=epoch, seen_scenes=seen, eval_seconds=time.time() - t0,
                   **({'gpu/max_memory_gb': torch.cuda.max_memory_allocated(device) / 1e9} if device.type == 'cuda' else {}))
        run.log(log, step=seen)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in log.items() if not k.startswith('plots/') and '_band/' not in k}), flush=True)
        if log['val/median_abs_chi'] < best: best = log['val/median_abs_chi']; improved = True
        else: improved = False
        state = dict(head=head.state_dict(), variant=a.variant, flags=flags, seen=seen, epoch=epoch, metrics={k: v for k, v in log.items() if not k.startswith('plots/')},
                     optimizer=optimizer.state_dict() if optimizer is not None else None, best=best, wandb_id=run.id,
                     metadata=dict(foundation=a.foundation, width=a.width, steps=a.steps, head=TruthScarletHead.version, fixed_background=True,
                                   per_band=flags['per_band'], implicit_psf=flags['implicit_psf'], use_foundation=flags['foundation']))
        torch.save(state, out_dir / 'last.pt')
        if improved:
            torch.save(state, out_dir / 'best.pt'); run.summary['best/val_median_abs_chi'] = best; run.summary['best/seen_scenes'] = seen
        head.train()

    if a.eval_only:
        evaluate(0); run.finish(); return

    optimizer = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    if resume is not None and resume.get('optimizer'): optimizer.load_state_dict(resume['optimizer'])
    paths_all = train_paths[:a.train_limit] if a.train_limit else train_paths
    total_steps = max(1, a.epochs * len(paths_all) // a.batch); step0 = seen // a.batch
    def lr_factor(s):
        s = s + step0
        if s < a.warmup: return (s + 1) / a.warmup
        progress = min(1., (s - a.warmup) / max(1, total_steps - a.warmup))
        return a.lr_final + (1 - a.lr_final) * .5 * (1 + math.cos(math.pi * progress))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    if resume is None: evaluate(0)
    next_eval = (seen // a.eval_every + 1) * a.eval_every; step = step0
    for epoch in range(seen // len(paths_all), a.epochs):
        paths = paths_all
        if seen > epoch * len(paths_all):   # resumed mid-epoch: the remaining number of scenes, freshly shuffled
            paths = [paths_all[i] for i in np.random.default_rng([a.seed, epoch]).permutation(len(paths_all))[:(epoch + 1) * len(paths_all) - seen]]
        head.train(); batch = []; skipped = 0; t0 = time.time(); seen0 = seen
        for scene, truth_s, stem in loader(paths, True, a.workers):
            try:
                out = model.run(scene)
                batch.append(scene_loss(out, scene, truth_s, device, a))
            except (ValueError, RuntimeError) as exc:
                if 'out of memory' in str(exc): raise
                skipped += 1
            seen += 1
            if len(batch) == a.batch:
                L = {k: torch.stack([b[k] for b in batch]).mean() for k in ('flux', 'template', 'chi2')}
                L['bias'] = bias_penalty([b['chi'] for b in batch], [b['snr'] for b in batch], a.huber)
                total = a.lambda_flux * L['flux'] + a.lambda_template * L['template'] + a.lambda_chi2 * L['chi2'] + a.lambda_bias * L['bias']
                optimizer.zero_grad(); total.backward()
                gn = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.); optimizer.step(); schedule.step(); step += 1
                if step % 10 == 0:
                    run.log({'train/loss': float(total), **{f'train/{k}': float(v) for k, v in L.items()}, 'train/grad_norm': float(gn),
                             'train/lr': schedule.get_last_lr()[0], 'train/skipped': skipped, 'epoch': epoch,
                             'train/scenes_per_s': (seen - seen0) / max(time.time() - t0, 1e-9)}, step=seen)
                if step % 50 == 0: print(f'epoch {epoch} step {step} seen {seen} loss {float(total):.3f} ' + ' '.join(f'{k} {float(v):.3f}' for k, v in L.items()) + f' ({time.time() - t0:.0f}s, {(seen - seen0) / max(time.time() - t0, 1e-9):.2f} scenes/s)', flush=True)
                batch = []
            if seen >= next_eval: evaluate(epoch); next_eval += a.eval_every
    evaluate(a.epochs - 1); run.finish()


if __name__ == '__main__': main()
