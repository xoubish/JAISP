"""Tables, precomputations and figures for the A6_v4 report notebook (io/30_a6_v4_photometry_head.ipynb).

    python -m models.photometry.self_supervised.truth_scarlet.report_a6 precompute   # aperture photometry + example cutouts (GPU if available)
"""
import json, itertools, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from ..core import BANDS
from .scenes import OUT as D, TILES, regions_of, load_scene

EU = ['euclid_VIS', 'euclid_Y', 'euclid_J', 'euclid_H']
SHORT = {b: b.split('_')[1] for b in BANDS}
RUN = 'A6_v4'
COL = dict(A6='#d94801', A8='#7a0177', A7='#c51b8a', mixture='#3182bd', tractor='#31a354', MER='#636363', aperture='#984ea3', oracle='#999999')
nmad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) > 5 else np.nan
mag = lambda f: np.where(f > 0, 23.9 - 2.5 * np.log10(np.where(f > 0, f, 1)), np.nan)


# ----------------------------------------------------------------------------- precompute
def aperture_photometry(radius_arcsec=1.0, annulus=(2.5, 3.5)):
    """Fixed circular aperture on the central source of every real validation scene, annulus-median sky."""
    rows = []
    for path in sorted((D / 'val_real/scenes').glob('*.npz')):
        z = np.load(path)
        for b in BANDS:
            im, var, m = z[b + '__image'], z[b + '__variance'], z[b + '__mask']; s2p = z[b + '__sky_to_pixel']
            scale = np.sqrt(abs(np.linalg.det(np.linalg.inv(s2p)))); x, y = z[b + '__positions'][0]
            yy, xx = np.indices(im.shape); r = np.hypot(xx - x, yy - y) * scale
            ok = m & np.isfinite(im) & np.isfinite(var); ap = ok & (r <= radius_arcsec); an = ok & (r >= annulus[0]) & (r <= annulus[1])
            if ap.sum() < 3 or an.sum() < 20: continue
            bg = np.median(im[an]); flux = float((im[ap] - bg).sum()); err = float(np.sqrt(var[ap].sum() + ap.sum() ** 2 * np.median(var[an]) / an.sum()))
            rows.append(dict(name=path.stem, band=b, flux=flux, error=err, n_pix=int(ap.sum())))
    out = pd.DataFrame(rows); out.to_csv(D / 'diag_real_aperture.csv', index=False); return out


def example_cutouts(n=8, seed=20261015, device=None, tto_steps=0, snr_min=10., prior=1.):
    """Run A6 (+ per-scene optimisation of the shapes when tto_steps > 0) on random held-out sources spanning
    magnitude; save data/model/residual images and every method's flux."""
    import torch
    from . import tto
    from ..scene_features import SceneEncoder
    from .model import TruthScarletHead, TruthScarlet, VARIANTS
    ROOT = D.parents[4]
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    st = torch.load(D / 'runs' / RUN / 'best.pt', map_location=dev, weights_only=False); meta = st['metadata']
    head = TruthScarletHead(per_band=True, perturbation=meta['perturbation'], eps_bound=meta['eps_bound'], steps=meta['steps'], centroid=True).to(dev)
    head.load_state_dict(st['head']); head.eval()
    enc = SceneEncoder(ROOT / 'models/checkpoints/jaisp_v11_q1_soft/checkpoint_best.pt'); enc.encoder.to(dev); model = TruthScarlet(head, enc, dev)
    man = pd.read_csv(D / 'val_real/manifest.csv'); man['vismag'] = mag(man['euclid_VIS__ref_ujy'].to_numpy())
    rng = np.random.default_rng(seed); picks = []
    for lo, hi in [(19, 21), (21, 22.5), (22.5, 23.5), (23.5, 24.5), (24.5, 25.5), (25.5, 27)]:
        c = man[man.vismag.between(lo, hi)]
        if len(c): picks += list(rng.choice(c.name.to_numpy(), min(2, len(c)), replace=False))
    picks = picks[:n]; arrays = {}; info = []
    for name in picks:
        scene, _ = load_scene(D / 'val_real/scenes' / f'{name}.npz')
        with torch.enable_grad(): out = model.run(scene)
        results = out['results']; objective = float(out['total'])
        if tto_steps: results, objective, _ = tto.optimise(model, scene, steps=tto_steps, lr=.02, snr_min=snr_min, prior=prior)
        for b in ('euclid_VIS', 'rubin_r', 'euclid_Y', 'euclid_J', 'euclid_H'):
            d = scene['bands'][b]; r = results[b]
            arrays[f'{name}__{b}__data'] = d['image'].numpy(); arrays[f'{name}__{b}__model'] = r['model'].detach().cpu().numpy()
            arrays[f'{name}__{b}__sigma'] = np.sqrt(d['variance'].numpy()); arrays[f'{name}__{b}__mask'] = d['mask'].numpy(); arrays[f'{name}__{b}__positions'] = d['positions'].numpy()
            idx = r['source_indices'].tolist(); k = idx.index(0) if 0 in idx else None
            info.append(dict(name=name, band=b, flux=float(r['flux'][k]) if k is not None else np.nan, error=float(r['error'][k]) if k is not None else np.nan,
                             n_fit=len(idx), reduced_chi2=float(r['chi2'] / max(r['dof'], 1)), tto_steps=tto_steps,
                             objective_amortised=float(out['total']), objective=objective))
    np.savez_compressed(D / 'report_examples.npz', **arrays); pd.DataFrame(info).to_csv(D / 'report_examples.csv', index=False)
    return picks


# ----------------------------------------------------------------------------- tables
def real_table(run=RUN, who='A6'):
    """Per (source, band) magnitudes for the head, mixture, Tractor, aperture and MER on the real validation tiles, with source metadata."""
    man = pd.read_csv(D / 'val_real/manifest.csv')
    mer = pd.concat([pd.read_csv(TILES / f'region_{r:03d}/mer_match.csv')[['source', 'mer_det_quality_flag', 'mer_semimajor_axis', 'n_mer_within_1arcsec', 'mer_point_like_prob']].assign(region=r) for r in man.region.unique()])
    man = man.merge(mer, on=['region', 'source'], how='left')
    def native(rows):
        out = []
        for b in EU:
            g = rows[rows.band == b].merge(man[['name', f'{b}__ujy_per_native']], on='name'); g['mag'] = mag((g.flux * g[f'{b}__ujy_per_native']).to_numpy()); out.append(g[['name', 'band', 'mag']])
        return pd.concat(out).set_index(['name', 'band']).mag
    cols = {}
    mix = pd.read_csv(D / 'val_real/reference_mixture.csv'); cols['mixture'] = native(mix[mix.source == 0])
    cols[who] = native(pd.read_csv(D / f'diag_real_{run}.csv'))
    if (D / 'diag_real_aperture.csv').exists(): cols['aperture'] = native(pd.read_csv(D / 'diag_real_aperture.csv'))
    tr = []
    for r in man.region.unique():
        p = TILES / f'region_{r:03d}/tractor_fluxes.csv'
        if p.exists(): t = pd.read_csv(p); t['name'] = [f'region{r:03d}_source{int(s)}' for s in t.source]; tr.append(t[['name', 'band', 'flux_ujy']])
    if tr: t = pd.concat(tr); t = t[t.band.isin(EU)]; t['mag'] = mag(t.flux_ujy.to_numpy()); cols['tractor'] = t.set_index(['name', 'band']).mag
    ref = []
    for b in EU:
        g = man[['name', f'{b}__ref_ujy']].copy(); g['band'] = b; g['mag'] = mag(g[f'{b}__ref_ujy'].to_numpy()); ref.append(g[['name', 'band', 'mag']])
    cols['MER'] = pd.concat(ref).set_index(['name', 'band']).mag
    df = pd.DataFrame(cols).dropna()
    vis = df.xs('euclid_VIS', level='band').MER; df['vismag'] = df.index.get_level_values(0).map(vis)
    meta = man.set_index('name')
    df['clean'] = df.index.get_level_values(0).map((meta.mer_det_quality_flag == 0) & (meta.n_mer_within_1arcsec <= 1) & (meta.mer_point_like_prob < .2))
    df['size'] = df.index.get_level_values(0).map(meta.mer_semimajor_axis); df['near'] = df.index.get_level_values(0).map(meta.nearest_arcsec)
    return df


def hat(df, methods):
    """N-cornered hat: solve pairwise NMAD^2 = s_a^2 + s_b^2 for each method's own scatter."""
    pair = {(a, b): nmad(df[a] - df[b]) for a, b in itertools.combinations(methods, 2)}
    A = np.zeros((len(pair), len(methods))); y = np.zeros(len(pair))
    for i, ((a, b), v) in enumerate(pair.items()): A[i, methods.index(a)] = 1; A[i, methods.index(b)] = 1; y[i] = v ** 2
    own = np.linalg.lstsq(A, y, rcond=None)[0]
    return pair, {m: float(np.sqrt(max(s, 0))) for m, s in zip(methods, own)}


def injection_table(folder, run=RUN, who='A6'):
    t = pd.read_csv(D / folder / 'truth.csv'); t['name'] = t.scene.map(lambda s: f'scene_{s:05d}')
    m = pd.read_csv(D / folder / 'scenes.csv'); m['name'] = m.scene.map(lambda s: f'scene_{s:05d}')
    base = t.merge(m[['name', 'mode', 'min_sep_arcsec', 'n_sources']], on='name')
    def frame(rows):
        d = rows.merge(base, on=['name', 'source', 'band']); d['chi'] = (d.flux - d.truth_flux) / d.oracle_error; d['frac'] = (d.flux - d.truth_flux) / d.truth_flux; return d
    runs = {'oracle': base.assign(flux=base.oracle_flux, chi=(base.oracle_flux - base.truth_flux) / base.oracle_error, frac=(base.oracle_flux - base.truth_flux) / base.truth_flux),
            'mixture': frame(pd.read_csv(D / folder / 'reference_mixture.csv')), who: frame(pd.read_csv(D / f'diag_{folder}_{run}.csv'))}
    common = set.intersection(*[set(zip(f.name, f.source, f.band)) for f in runs.values()])
    for k in runs: runs[k] = runs[k][[x in common for x in zip(runs[k].name, runs[k].source, runs[k].band)]].reset_index(drop=True)
    for f in runs.values(): f['context'] = np.where(f.min_sep_arcsec > 2, 'isolated', np.where(f.min_sep_arcsec < .7, 'blend <0.7"', 'blend 0.7-2"'))
    return runs


def training_curves(run=RUN):
    rows = [json.loads(l) for l in open(D / 'runs' / run / 'log.txt') if l.startswith('{"val/median_abs_chi')]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def fig_architecture():
    """Five-stage left-to-right view; details in the notebook text."""
    fig, ax = plt.subplots(figsize=(16, 6.2)); ax.set_xlim(0, 16); ax.set_ylim(0, 6.2); ax.axis('off')
    def box(x, y, w, h, title, body, color, title_fs=10.5, fs=8.3):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03,rounding_size=0.15', fc=color, ec='#444', lw=1.2))
        ax.text(x + w / 2, y + h - .32, title, ha='center', va='center', fontsize=title_fs, weight='bold')
        ax.text(x + w / 2, y + (h - .6) / 2, body, ha='center', va='center', fontsize=fs, linespacing=1.45)
    def arrow(x0, x1, y, text=''):
        ax.annotate('', xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle='-|>', lw=1.6, color='#222', mutation_scale=16))
        if text: ax.text((x0 + x1) / 2, y + .18, text, ha='center', fontsize=7.5, style='italic')
    y0, h = 2.0, 3.3
    box(.2, y0, 2.6, h, '1  Scene', '12" cutout, 10 bands\n(Rubin ugrizy, Euclid VIS Y J H)\nimages, variances, masks\nall detections inside\n(up to 40), their PSFs', '#f0f0f0')
    box(3.4, y0, 2.9, h, '2  Features (frozen)', 'JAISP foundation encoder\nfused 10-band bottleneck\n+ VIS stem window\n\n+ raw 10-band pixel windows\naround each source', '#c9c9c9')
    box(6.9, y0, 3.2, h, '3  Shape network (learned)', 'per source, a CNN predicts\n• elliptical radial profile\n• free-form modulation (×7)\n• per-band size & colour mix\n• centroid correction ≤ 0.2"\n\n3 refinement passes on the residual', '#9ecae1')
    box(10.7, y0, 2.4, h, '4  Render', 'shape × PSF of each band\nat corrected position\n→ unit-flux template\nper source, per band', '#a1d99b')
    box(13.7, y0, 2.1, h, '5  Flux solve', 'linear least squares:\nall sources jointly,\nper band, signed,\nfixed robust sky\n\n→ fluxes + covariance', '#a1d99b')
    arrow(2.8, 3.4, y0 + h / 2); arrow(6.3, 6.9, y0 + h / 2); arrow(10.1, 10.7, y0 + h / 2, 'shapes'); arrow(13.1, 13.7, y0 + h / 2, 'templates')
    top = y0 + h; ax.plot([14.7, 14.7, 8.5, 8.5], [top, top + .45, top + .45, top + .08], color='#d94801', ls='--', lw=1.2)
    ax.annotate('', xy=(8.5, top + .02), xytext=(8.5, top + .3), arrowprops=dict(arrowstyle='-|>', lw=1.2, color='#d94801', mutation_scale=14))
    ax.text(11.6, top + .55, 'residual fed back to the shape network (3 passes)', ha='center', fontsize=8, color='#d94801', style='italic')
    ax.add_patch(FancyBboxPatch((.2, .35), 15.6, 1.15, boxstyle='round,pad=0.03', fc='#fff5f0', ec='#a50f15', lw=1))
    ax.text(8, 1.17, 'Training (only stage 3 learns; the foundation is frozen, stages 4–5 have no weights)', ha='center', fontsize=9.5, weight='bold', color='#a50f15')
    ax.text(8, .68, 'Real galaxies injected into real scenes at known flux → loss = flux error (noise units) + bright-end fractional error + per-source image error '
                   '+ pixel χ² + running bias correction.\nThe network never outputs a flux: fluxes are always solved from the pixels, so they carry covariances and stay linear in the data.',
            ha='center', fontsize=8.3, linespacing=1.5)
    ax.set_title('A6_v4 photometry head: how a scene becomes fluxes', fontsize=13, weight='bold', pad=6); return fig


LINEAGE = [('A0', 'A0: χ² only'), ('A1', 'A1: truth loss'), ('A2', 'A2: +per-band'), ('A2_v2', 'A2 v2: real scenes'), ('A2_v3', 'A2 v3: size–mag'),
           ('A5_v3b', 'A5 v3b: free shapes'), ('A2_v4', 'A2 v4: loss+data'), ('A5_v4', 'A5 v4'), ('A6_v4', 'A6 v4: centroid'), ('A7_v4', 'A7 v4: point template'), ('A8_v4', 'A8 v4: bright-shape term')]


def lineage_table():
    rows = []
    for run, label in LINEAGE:
        p = D / 'runs' / run / 'log.txt'
        if not p.exists(): continue
        r = [json.loads(l) for l in open(p) if l.startswith('{"val/median_abs_chi')]; best = min(r, key=lambda d: d['val/median_abs_chi'])
        on_v3 = run.endswith(('_v3', '_v3b', '_v4'))   # these runs validated on val_v3; earlier ones on v1/v2 sets (not comparable)
        rows.append(dict(run=run, label=label, real_nmad=best['real/mean_nmad_mag'], real_vis=best['real/VIS_nmad_mag'], val_v3=best['val/median_abs_chi'] if on_v3 else np.nan,
                         close_v3=best['val_context/median_abs_chi_blend <0.7"'] if on_v3 else np.nan))
    return pd.DataFrame(rows)


def fig_lineage(mixture_real=.133, mixture_val=2.164, mixture_close=2.183):
    t = lineage_table(); x = np.arange(len(t))
    fig, axs = plt.subplots(1, 2, figsize=(16, 4.6))
    axs[0].plot(x, t.real_nmad, 'o-', c=COL['A6'], label='all 4 bands'); axs[0].plot(x, t.real_vis, 's--', c='#fdae6b', label='VIS'); axs[0].axhline(mixture_real, c=COL['mixture'], ls='--', label='mixture (4 bands)')
    axs[0].set_ylabel('NMAD vs MER [mag], real held-out tiles'); axs[0].set_title('real tiles: every head, best checkpoint')
    ok = t.val_v3.notna(); axs[1].plot(x[ok], t.val_v3[ok], 'o-', c=COL['A6'], label='overall'); axs[1].plot(x[ok], t.close_v3[ok], 's--', c='#e6550d', label='close blends < 0.7"')
    axs[1].axhline(mixture_val, c=COL['mixture'], ls='--', label='mixture overall'); axs[1].axhline(mixture_close, c=COL['mixture'], ls=':', label='mixture close blends')
    axs[1].set_ylabel('median |f − f*| / σ_oracle on val_v3'); axs[1].set_title('injections with known truth (heads validated on val_v3 only)')
    for a in axs: a.set_xticks(x); a.set_xticklabels(t.label, rotation=30, ha='right', fontsize=8); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle('Progress across the head lineage (each is a fine-tune of the previous; data versions v1→v4)'); fig.tight_layout(); return fig


def fig_training(curves, mixture_val=2.164, mixture_real=.133, a5_real=.146, oracle_bias=(-.168, -.042, -.012)):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2)); x = curves.seen_scenes / 1000
    axs[0].plot(x, curves['val/median_abs_chi'], 'o-', c=COL['A6'], label='A6_v4'); axs[0].axhline(mixture_val, c=COL['mixture'], ls='--', label='mixture'); axs[0].set_ylabel('median |f − f*| / σ_oracle (val_v3)'); axs[0].set_title('injections: overall')
    axs[1].plot(x, curves['real/mean_nmad_mag'], 'o-', c=COL['A6'], label='A6_v4'); axs[1].axhline(mixture_real, c=COL['mixture'], ls='--', label='mixture'); axs[1].axhline(a5_real, c='#fdae6b', ls=':', label='A5_v4 (init)'); axs[1].set_ylabel('NMAD vs MER [mag], 4 Euclid bands'); axs[1].set_title('real held-out tiles')
    for k, lab, ob, col in (('val_snr/median_frac_bias_0-3', 'S/N < 3', oracle_bias[0], '#1f77b4'), ('val_snr/median_frac_bias_3-10', 'S/N 3–10', oracle_bias[1], '#ff7f0e'), ('val_snr/median_frac_bias_10-30', 'S/N 10–30', oracle_bias[2], '#2ca02c')):
        axs[2].plot(x, 100 * curves[k], 'o-', c=col, label=lab); axs[2].axhline(100 * ob, c=col, ls=':', lw=1)
    axs[2].axhline(0, c='k', lw=.6); axs[2].set_ylabel('median fractional bias [%]'); axs[2].set_title('injections: faint-end bias (dotted: oracle with true shapes)')
    for a in axs: a.set_xlabel('training scenes seen [thousand]'); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.tight_layout(); return fig


def fig_flux_scatter(runs, who='A6', title='', snr_min=3.):
    """Measured / true flux against true S/N (S/N >= snr_min), per band: faint points for the head, prominent running
    median and 16-84% bands for the head and the mixture, oracle (true shapes) median as the noise-floor reference."""
    edges = np.geomspace(snr_min, 500, 12)
    def running(x, y):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s_ = (x >= lo) & (x < hi); out.append((np.sqrt(lo * hi), *np.percentile(y[s_], [16, 50, 84])) if s_.sum() >= 12 else (np.sqrt(lo * hi), np.nan, np.nan, np.nan))
        return np.array(out)
    fig, axs = plt.subplots(2, 5, figsize=(20, 7.6), sharex=True, sharey=True)
    for ax, band in zip(axs.flat, BANDS):
        d = runs[who][(runs[who].band == band) & (runs[who].true_snr >= snr_min)]
        ax.scatter(d.true_snr, d.flux / d.truth_flux, s=3, c=COL['A6'], alpha=.15, rasterized=True)
        for k, col, lab in (('mixture', COL['mixture'], 'mixture'), (who, COL['A6'], 'A6_v4')):
            g = runs[k][(runs[k].band == band) & (runs[k].true_snr >= snr_min)]; r = running(g.true_snr.to_numpy(), (g.flux / g.truth_flux).to_numpy())
            ax.fill_between(r[:, 0], r[:, 1], r[:, 3], color=col, alpha=.3, lw=0); ax.plot(r[:, 0], r[:, 2], '-', c=col, lw=2, label=f'{lab}: median, 16–84%')
        g = runs['oracle'][(runs['oracle'].band == band) & (runs['oracle'].true_snr >= snr_min)]; r = running(g.true_snr.to_numpy(), (g.flux / g.truth_flux).to_numpy())
        ax.plot(r[:, 0], r[:, 1], ':', c='k', lw=1); ax.plot(r[:, 0], r[:, 3], ':', c='k', lw=1, label='oracle (true shapes) 16–84%: noise floor')
        ax.axhline(1, c='k', lw=.7); ax.set_xscale('log'); ax.set_xlim(snr_min, 500); ax.set_ylim(.4, 1.6); ax.grid(alpha=.3)
        a = runs[who][(runs[who].band == band) & (runs[who].true_snr >= snr_min)]; m = runs['mixture'][(runs['mixture'].band == band) & (runs['mixture'].true_snr >= snr_min)]
        ax.set_title(f'{SHORT[band]}   med|Δ|/σ: A6 {np.median(np.abs(a.chi)):.2f}, mixture {np.median(np.abs(m.chi)):.2f}', fontsize=9)
    for ax in axs[1]: ax.set_xlabel('true S/N')
    for ax in axs[:, 0]: ax.set_ylabel('measured / true flux')
    axs[0, 0].legend(fontsize=7.5, loc='upper right'); fig.suptitle(title); fig.tight_layout(); return fig


def fig_frac_vs_snr(runs, who='A6', title=''):
    edges = np.geomspace(.7, 300, 14)
    def running(x, y):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (x >= lo) & (x < hi); out.append((np.sqrt(lo * hi), *np.percentile(y[s], [16, 50, 84])) if s.sum() >= 10 else (np.sqrt(lo * hi), np.nan, np.nan, np.nan))
        return np.array(out)
    fig, axs = plt.subplots(2, 5, figsize=(20, 7.5), sharex=True, sharey=True)
    for ax, band in zip(axs.flat, BANDS):
        for k, col, ls in (('oracle', COL['oracle'], ':'), ('mixture', COL['mixture'], '--'), (who, COL['A6'], '-')):
            g = runs[k][runs[k].band == band]; r = running(g.true_snr.to_numpy(), g.frac.to_numpy())
            ax.plot(r[:, 0], r[:, 2], ls, c=col, label=k); ax.fill_between(r[:, 0], r[:, 1], r[:, 3], color=col, alpha=.1)
        ax.axhline(0, c='k', lw=.6); ax.set_xscale('log'); ax.set_xlim(.7, 300); ax.set_ylim(-1, 1); ax.set_title(SHORT[band], fontsize=9); ax.grid(alpha=.3)
    for ax in axs[1]: ax.set_xlabel('true S/N')
    for ax in axs[:, 0]: ax.set_ylabel('(f − f*) / f*  (16/50/84%)')
    axs[0, 0].legend(fontsize=8); fig.suptitle(title); fig.tight_layout(); return fig


def fig_real_vs_mer(df, methods=('A6', 'mixture', 'tractor', 'aperture'), ylim=.6):
    fig, axs = plt.subplots(len(methods), 4, figsize=(18, 3.4 * len(methods)), sharex=True, sharey=True)
    edges = [19, 21, 22, 23, 24, 25, 26, 27]
    for row, m in zip(axs, methods):
        for ax, band in zip(row, EU):
            g = df[df.index.get_level_values(1) == band]; d = g[m] - g.MER
            ax.scatter(g.MER, d, s=4, c=COL.get(m, 'k'), alpha=.45, rasterized=True); ax.axhline(0, c='k', lw=.6)
            xs, lo_, med, hi_ = [], [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                s_ = g.MER.between(lo, hi)
                if s_.sum() >= 8: xs.append((lo + hi) / 2); q = np.percentile(d[s_], [16, 50, 84]); lo_.append(q[0]); med.append(q[1]); hi_.append(q[2])
            ax.plot(xs, med, 'k-', lw=1.3); ax.fill_between(xs, lo_, hi_, color='k', alpha=.12)
            ax.set_ylim(-ylim, ylim); ax.set_xlim(19, 27.5); ax.grid(alpha=.3)
            ax.set_title(f'{m}  {SHORT[band]}:  NMAD {nmad(d):.3f}, median {np.median(d):+.3f}', fontsize=9)
    for ax in axs[-1]: ax.set_xlabel('MER AB mag')
    for ax in axs[:, 0]: ax.set_ylabel('method − MER [mag]')
    fig.suptitle('Real held-out tiles: each method against MER (black: running median and 16–84% band)'); fig.tight_layout(); return fig


def fig_bright_isolated(df, vis_max=22.5, near_min=3.):
    """Direct magnitude comparison for bright, isolated MER galaxies: where any method should agree with MER."""
    sel = (df.vismag < vis_max) & (df.near > near_min) & df.clean.fillna(False)
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6))
    for ax, band in zip(axs, EU):
        g = df[sel & (df.index.get_level_values(1) == band)]
        for m, mk in (('tractor', 's'), ('mixture', '^'), ('A6', 'o')):
            ax.scatter(g.MER, g[m], s=28, marker=mk, c=COL[m], alpha=.8, label=f'{m}: NMAD {nmad(g[m] - g.MER):.3f}, {np.mean(np.abs(g[m] - g.MER) < .1):.0%} within 0.1')
        lo, hi = g.MER.min() - .3, g.MER.max() + .3; ax.plot([lo, hi], [lo, hi], 'k-', lw=.8); ax.plot([lo, hi], [lo + .1, hi + .1], 'k:', lw=.6); ax.plot([lo, hi], [lo - .1, hi - .1], 'k:', lw=.6)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xlabel('MER AB mag'); ax.set_title(f'{SHORT[band]}  (N = {len(g)})', fontsize=10); ax.legend(fontsize=7.5, loc='upper left'); ax.grid(alpha=.3)
    axs[0].set_ylabel('measured AB mag')
    fig.suptitle(f'Bright isolated galaxies (MER VIS < {vis_max}, no detection within {near_min}", MER galaxy class): measured vs MER, dotted = ±0.1 mag', fontsize=11); fig.tight_layout(); return fig


def fig_vs_floor(runs, who='A6'):
    """Fractional scatter by true S/N for the head and the mixture, against the oracle (true-shape) noise floor."""
    bins = [(3, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 1e9)]; labels = ['3–5', '5–10', '10–20', '20–50', '50–100', '>100']
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4))
    for ax, ctx in zip(axs, (None, 'isolated')):
        for k, col, mk in (('oracle', COL['oracle'], 'x'), ('mixture', COL['mixture'], '^'), (who, COL['A6'], 'o')):
            f = runs[k]; f = f if ctx is None else f[f.context == ctx]
            ax.plot(range(len(bins)), [100 * nmad(f.frac[(f.true_snr >= lo) & (f.true_snr < hi)]) for lo, hi in bins], marker=mk, c=col, label=k if k != 'oracle' else 'oracle (true shapes): noise floor')
        ax.set_xticks(range(len(bins))); ax.set_xticklabels(labels); ax.set_xlabel('true S/N'); ax.set_yscale('log'); ax.set_ylabel('NMAD of fractional flux error [%]'); ax.grid(alpha=.3, which='both')
        ax.set_title('all injected sources' if ctx is None else 'isolated injected sources (no source within 2")', fontsize=10); ax.legend(fontsize=8)
    fig.suptitle('Injections with known truth: scatter vs the noise floor'); fig.tight_layout(); return fig


def fig_breakdown(df, methods=('A6', 'mixture', 'tractor', 'aperture')):
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.4))
    specs = (('vismag', [18, 22, 23, 24, 25, 27], 'MER VIS magnitude'), ('size', [0, 1.5, 2.05, 2.85, 4.3, 50], 'MER semimajor axis [VIS px]'), ('near', [0, 1.5, 3, 6, 100], 'nearest other detection [arcsec]'))
    for ax, (col, edges, lab) in zip(axs, specs):
        labels = [f'{lo}–{hi}' if hi < 50 else f'>{lo}' for lo, hi in zip(edges[:-1], edges[1:])]; x = np.arange(len(labels)); w = .8 / len(methods)
        for i, m in enumerate(methods):
            vals = [nmad(df[m][df[col].between(lo, hi)] - df.MER[df[col].between(lo, hi)]) for lo, hi in zip(edges[:-1], edges[1:])]
            ax.bar(x + (i - len(methods) / 2 + .5) * w, vals, w, color=COL.get(m, 'k'), label=m)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_xlabel(lab); ax.set_ylabel('NMAD vs MER [mag], 4 bands'); ax.grid(axis='y', alpha=.3)
    axs[0].legend(fontsize=8); fig.suptitle('Where each method differs from MER'); fig.tight_layout(); return fig


def example_table():
    """AB magnitudes per method for the example sources (Euclid bands)."""
    info = pd.read_csv(D / 'report_examples.csv'); df = real_table(); man = pd.read_csv(D / 'val_real/manifest.csv').set_index('name'); names = list(dict.fromkeys(info.name)); rows = []
    for name in names:
        for b in EU:
            if (name, b) in df.index:
                r = df.loc[(name, b)]; e = info[(info.name == name) & (info.band == b)]
                row = dict(source=name, band=SHORT[b], A6=round(r.A6, 2))
                if len(e) and e.tto_steps.iloc[0]: row['A6+opt'] = round(float(mag(np.array([e.flux.iloc[0] * man.loc[name, f'{b}__ujy_per_native']]))[0]), 2)
                row.update(mixture=round(r.mixture, 2), tractor=round(r.tractor, 2), aperture=round(r.aperture, 2), MER=round(r.MER, 2)); rows.append(row)
    return pd.DataFrame(rows).set_index(['source', 'band'])


def fig_examples(names=None, bands=('euclid_VIS', 'rubin_r', 'euclid_Y', 'euclid_H')):
    z = np.load(D / 'report_examples.npz'); info = pd.read_csv(D / 'report_examples.csv')
    names = names or list(dict.fromkeys(info.name))
    fig, axs = plt.subplots(len(names), 3 * len(bands), figsize=(2.2 * 3 * len(bands), 2.3 * len(names)), squeeze=False)
    for row, name in zip(axs, names):
        for j, b in enumerate(bands):
            data, model, sig, m = (z[f'{name}__{b}__{k}'] for k in ('data', 'model', 'sigma', 'mask')); s = np.median(sig[m]) if m.any() else 1.
            st = lambda im: np.arcsinh(im / (2 * s)); vmax = np.percentile(st(data[m]), 99.7); h, w = data.shape
            for a, im, t in ((row[3 * j], data, 'data'), (row[3 * j + 1], model, 'A6 model')):
                a.imshow(np.where(m, st(im), np.nan), origin='lower', cmap='magma', vmin=-1, vmax=vmax)
            chi = np.where(m, (data - model) / sig, np.nan); row[3 * j + 2].imshow(chi, origin='lower', cmap='RdBu_r', vmin=-4, vmax=4)
            p = z[f'{name}__{b}__positions']; inside = (p[:, 0] > -1) & (p[:, 0] < w) & (p[:, 1] > -1) & (p[:, 1] < h)
            row[3 * j].plot(p[1:][inside[1:], 0], p[1:][inside[1:], 1], '+', c='c', ms=5); row[3 * j].plot(p[0, 0], p[0, 1], 'o', mfc='none', mec='lime', ms=10)
            r = info[(info.name == name) & (info.band == b)].iloc[0]
            row[3 * j].set_title(f'{SHORT[b]} data ({r.n_fit} fitted)', fontsize=7.5); row[3 * j + 1].set_title('A6 model' + (' + per-scene opt.' if r.tto_steps else ''), fontsize=7.5); row[3 * j + 2].set_title(f'χ residual, χ²/dof {r.reduced_chi2:.2f}', fontsize=7.5)
            for a in row[3 * j:3 * j + 3]: a.set_xlim(-.5, w - .5); a.set_ylim(-.5, h - .5)
        row[0].set_ylabel(name.replace('region', 'r').replace('_source', ' s'), fontsize=8)
        for a in row: a.set_xticks([]); a.set_yticks([])
    fig.suptitle('Random held-out sources: data, A6 model' + (' after per-scene shape optimisation' if info.tto_steps.iloc[0] else '') + ' and χ residual (lime: measured source; cyan: neighbours fitted jointly)', fontsize=10, y=1.0)
    fig.tight_layout(); return fig


def fig_shapes(rows=5, cols=5, stretch=300.):
    """Gallery for 100 held-out sources (report_shapes.npz from scratch shape_gallery.py): VIS data, the PSF-convolved
    model of the measured source in the same units and stretch, and the intrinsic (pre-PSF) shape."""
    path = D / 'report_shapes.npz'; opt = False   # the validated inference path is the amortised head; optimised galleries are kept in report_shapes_tto*.npz
    z = np.load(path, allow_pickle=True); pick = np.linspace(0, len(z['vismag']) - 1, rows * cols).astype(int)
    shapes, data, model, vm = (z[k][pick] for k in ('shapes', 'data', 'model', 'vismag'))
    fig, axs = plt.subplots(rows, 3 * cols, figsize=(1.25 * 3 * cols, 1.3 * rows + .6))
    for i in range(rows * cols):
        r, c = divmod(i, cols); a_d, a_m, a_s = axs[r, 3 * c], axs[r, 3 * c + 1], axs[r, 3 * c + 2]
        a_d.imshow(data[i], origin='lower', cmap='magma', vmin=-.5, vmax=3); a_m.imshow(model[i], origin='lower', cmap='magma', vmin=-.5, vmax=3)
        a_s.imshow(np.arcsinh(shapes[i] / shapes[i].max() * stretch), origin='lower', cmap='magma'); a_d.text(2, 52, f'{vm[i]:.1f}', color='w', fontsize=7)
        for a in (a_d, a_m, a_s): a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f'{rows * cols} held-out sources spanning VIS 19–26, sorted by magnitude. Per source: VIS data (S/N) | PSF-convolved model of that source' + (' after per-scene optimisation' if opt else '') + f', same units and stretch | intrinsic shape before the PSF (asinh ×300, shows wings far below the noise); 6.1" boxes', fontsize=10, y=.998)
    fig.tight_layout(); return fig


def fig_tto_examples(names=('region019_source110', 'region041_source138', 'region024_source170'), bands=('euclid_VIS', 'rubin_r', 'euclid_H')):
    """Bright sources: amortised model vs 60 ungated steps of per-scene optimisation (report_examples.npz vs tto_examples_60.npz)."""
    before = np.load(D / 'report_examples.npz'); after = np.load(D / 'tto_examples_60.npz')
    fig, axs = plt.subplots(len(names), 4 * len(bands), figsize=(2.3 * 4 * len(bands), 2.4 * len(names)))
    for row, name in zip(axs, names):
        for j, b in enumerate(bands):
            data, sig, m = after[f'{name}__{b}__data'], after[f'{name}__{b}__sigma'], after[f'{name}__{b}__mask']; s_ = np.median(sig[m]); st = lambda im: np.arcsinh(im / (2 * s_)); vmax = np.percentile(st(data[m]), 99.7)
            mb, ma = before[f'{name}__{b}__model'], after[f'{name}__{b}__model_tto']
            row[4 * j].imshow(np.where(m, st(data), np.nan), origin='lower', cmap='magma', vmin=-1, vmax=vmax); row[4 * j].set_title(f'{SHORT[b]} data', fontsize=8)
            row[4 * j + 1].imshow(np.where(m, st(ma), np.nan), origin='lower', cmap='magma', vmin=-1, vmax=vmax); row[4 * j + 1].set_title('optimised model', fontsize=8)
            for a, mod, t in ((row[4 * j + 2], mb, 'χ amortised'), (row[4 * j + 3], ma, 'χ optimised')):
                chi = np.where(m, (data - mod) / sig, np.nan); a.imshow(chi, origin='lower', cmap='RdBu_r', vmin=-4, vmax=4); a.set_title(f'{t}, χ²/dof {np.nanmean(chi ** 2):.2f}', fontsize=8)
            for a in row[4 * j:4 * j + 4]: a.set_xticks([]); a.set_yticks([])
        row[0].set_ylabel(name.replace('region', 'r').replace('_source', ' s'), fontsize=8)
    fig.suptitle('Per-scene optimisation of the shapes (ungated, 60 steps): the residuals of bright sources vanish, but the fluxes of their neighbours do not survive (table above)', fontsize=10); fig.tight_layout(); return fig


def fig_head_examples(tags=('A6_v4', 'A7_v4_last', 'A8_v4'), labels=('A6', 'A7', 'A8'), names=('region019_source110', 'region041_source138', 'region024_source170'), bands=('euclid_VIS', 'rubin_r', 'euclid_H')):
    """Bright sources: data and the χ residual of each head (core_examples_<tag>.npz from the CPU evaluation script)."""
    ex = {t: np.load(D / f'core_examples_{t}.npz') for t in tags}; core = pd.concat([pd.read_csv(D / f'diag_core_{t}.csv').assign(head=l) for t, l in zip(tags, labels)]).set_index(['head', 'name', 'band'])
    k = 1 + len(tags); fig, axs = plt.subplots(len(names), k * len(bands), figsize=(2.2 * k * len(bands), 2.4 * len(names)))
    for row, name in zip(axs, names):
        for j, b in enumerate(bands):
            e0 = ex[tags[0]]; data, sig, m = e0[f'{name}__{b}__data'], e0[f'{name}__{b}__sigma'], e0[f'{name}__{b}__mask']; s_ = np.median(sig[m]); st = lambda im: np.arcsinh(im / (2 * s_))
            row[k * j].imshow(np.where(m, st(data), np.nan), origin='lower', cmap='magma', vmin=-1, vmax=np.percentile(st(data[m]), 99.7)); row[k * j].set_title(f'{SHORT[b]} data', fontsize=8)
            for i, (t, l) in enumerate(zip(tags, labels)):
                mod = ex[t][f'{name}__{b}__model']; chi = np.where(m, (data - mod) / sig, np.nan); a = row[k * j + 1 + i]
                a.imshow(chi, origin='lower', cmap='RdBu_r', vmin=-4, vmax=4); a.set_title(f'χ {l}: core χ²/dof {core.loc[(l, name, b), "chi2_core"]:.0f}', fontsize=8)
            for a in row[k * j:k * j + k]: a.set_xticks([]); a.set_yticks([])
        row[0].set_ylabel(name.replace('region', 'r').replace('_source', ' s'), fontsize=8)
    fig.suptitle('The same bright sources under each head (amortised, no per-scene optimisation); core = within 0.5″ of the centre', fontsize=10); fig.tight_layout(); return fig


if __name__ == '__main__':
    if sys.argv[1:] == ['precompute']:
        ap = aperture_photometry(); print('aperture rows', len(ap))
        print('examples', example_cutouts())
