"""PNG galleries of truth_scarlet training scenes, one page per scene type, plus a distributions page.

    python -m models.photometry.self_supervised.truth_scarlet.plot_scenes --split train_v2

Each row is one scene: VIS, r and H data (all sources marked), the injected light alone
(truth flux x truth profile, VIS), and VIS data minus the injected light (what the
scene looked like before injection). Green circles: injected sources with truth;
cyan crosses: real detections fitted without truth.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .scenes import OUT

SHOW = (('euclid_VIS', 'VIS'), ('rubin_r', 'r'), ('euclid_H', 'H'))


def stretch(image, sigma): return np.arcsinh(image / (2 * sigma))


def page(folder, manifest, title, path, n=6, seed=0):
    rows = manifest.sample(min(n, len(manifest)), random_state=seed) if len(manifest) else manifest
    if not len(rows): return False
    fig, axs = plt.subplots(len(rows), 5, figsize=(16, 3.3 * len(rows)), squeeze=False)
    for ax_row, (_, r) in zip(axs, rows.iterrows()):
        z = np.load(folder / 'scenes' / f'scene_{int(r.scene):05d}.npz'); nt = int(z['n_truth']) if 'n_truth' in z else len(z['truth__euclid_VIS__flux'])
        for a, (b, short) in zip(ax_row[:3], SHOW):
            im = z[b + '__image']; m = z[b + '__mask']; s = np.sqrt(np.median(z[b + '__variance'][m])) if m.any() else 1
            a.imshow(np.where(m, stretch(im, s), np.nan), origin='lower', cmap='magma', vmin=-1, vmax=4)
            p = z[b + '__positions']; h, w = im.shape; inside = (p[:, 0] > -1) & (p[:, 0] < w) & (p[:, 1] > -1) & (p[:, 1] < h)
            real = inside & (np.arange(len(p)) >= nt)
            a.plot(p[real, 0], p[real, 1], '+', c='c', ms=8, mew=1.2); a.plot(p[:nt, 0], p[:nt, 1], 'o', mfc='none', mec='lime', ms=12, mew=1.2)
            a.set_xlim(-.5, w - .5); a.set_ylim(-.5, h - .5)
            fl = ', '.join(f'{f / e:.0f}' for f, e in zip(z[f'truth__{b}__flux'], z[f'truth__{b}__error']))
            a.set_title(f'{short}  injected S/N: {fl}', fontsize=8)
        b = 'euclid_VIS'; m = z[b + '__mask']; s = np.sqrt(np.median(z[b + '__variance'][m]))
        injected = np.einsum('n,nhw->hw', z[f'truth__{b}__flux'], z[f'truth__{b}__profiles'])
        ax_row[3].imshow(stretch(injected, s), origin='lower', cmap='magma', vmin=-1, vmax=4); ax_row[3].set_title('injected light only (VIS)', fontsize=8)
        ax_row[4].imshow(np.where(m, stretch(z[b + '__image'] - injected, s), np.nan), origin='lower', cmap='magma', vmin=-1, vmax=4)
        ax_row[4].set_title('VIS before injection', fontsize=8)
        label = f"scene {int(r.scene)} · {r['mode']} · {r.kind}\n{int(r.n_sources)} sources ({int(r.n_injected)} injected)\nmin sep {r.min_sep_arcsec:.2f}\""
        if bool(r.get('primary_point_like', False)): label += ' · point-like'
        ax_row[0].set_ylabel(label, fontsize=8)
        for a in ax_row: a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title, fontsize=12, y=1.0); fig.tight_layout(); fig.savefig(path, dpi=80, bbox_inches='tight'); plt.close(fig)
    return True


def distributions(folder, manifest, truth, path):
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    for mode, col in (('real', '#d94801'), ('blank', '#3182bd')):
        m = manifest[manifest['mode'] == mode]
        axs[0, 0].hist(np.log10(m.vis_snr), bins=40, alpha=.6, color=col, label=f'{mode} ({len(m)})')
        axs[0, 1].hist(m.n_sources, bins=np.arange(0.5, 41.5), alpha=.6, color=col, label=mode)
        axs[0, 2].hist(np.clip(m.min_sep_arcsec, 0, 8), bins=40, alpha=.6, color=col, label=mode)
    axs[0, 0].set_xlabel('log10 primary VIS S/N (target)'); axs[0, 1].set_xlabel('sources fitted per scene'); axs[0, 2].set_xlabel('primary: nearest other source [arcsec] (clipped at 8)')
    t = truth.copy(); t['short'] = t.band.str.split('_').str[1]
    order = ['u', 'g', 'r', 'i', 'z', 'y', 'VIS', 'Y', 'J', 'H']
    data = [np.log10(np.clip(t.true_snr[t.short == s], .1, None)) for s in order]
    axs[1, 0].boxplot(data, labels=order, showfliers=False); axs[1, 0].set_ylabel('log10 true S/N per injected source'); axs[1, 0].axhline(np.log10(3), color='k', lw=.6, ls='--')
    q = pd.read_csv(folder / 'donor_quality.csv'); q = q[q.suitable_v2] if 'suitable_v2' in q else q[q.suitable_primary]
    axs[1, 1].hist(np.log10(q.selection_snr), bins=40, color='#636363'); axs[1, 1].set_xlabel(f'log10 donor selection S/N ({len(q)} donors, {int(q.get("point_like", pd.Series(dtype=bool)).sum())} point-like)')
    kinds = manifest.groupby(['mode', 'kind']).size().unstack(fill_value=0)
    kinds.plot.bar(ax=axs[1, 2], rot=0); axs[1, 2].set_ylabel('scenes'); axs[1, 2].set_title('injected sources per scene', fontsize=9)
    for a in axs[0]: a.legend(fontsize=8)
    fig.suptitle(f'{folder.name}: {len(manifest)} scenes', fontsize=12); fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split', default='train_v2'); p.add_argument('--rows', type=int, default=6)
    a = p.parse_args()
    folder = OUT / a.split; dest = folder / 'gallery'; dest.mkdir(exist_ok=True)
    m = pd.read_csv(folder / 'scenes.csv'); t = pd.read_csv(folder / 'truth.csv')
    if 'mode' not in m: m['mode'] = 'blank'; m['n_injected'] = m.n_sources; m['primary_point_like'] = False
    real, blank = m[m['mode'] == 'real'], m[m['mode'] == 'blank']
    pages = [('01_real_isolated', 'Real-scene injection, nothing within 2"', real[real.min_sep_arcsec > 2]),
             ('02_real_close_blend', 'Real-scene injection, another source within 0.7"', real[real.min_sep_arcsec < .7]),
             ('03_real_crowded', 'Real-scene injection, 20 or more sources fitted', real[real.n_sources >= 20]),
             ('04_real_faint', 'Real-scene injection, primary VIS S/N < 3', real[real.vis_snr < 3]),
             ('05_real_bright', 'Real-scene injection, primary VIS S/N > 30', real[real.vis_snr > 30]),
             ('06_point_like', 'Point-like donor (star or compact galaxy) as primary', m[m.primary_point_like.astype(bool)]),
             ('07_blank_pairs_triples', 'Blank-sky injection, pairs and triples', blank[blank.n_injected > 1]),
             ('08_blank_isolated', 'Blank-sky injection, single source', blank[blank.n_injected == 1])]
    for name, title, sel in pages:
        ok = page(folder, sel, f'{a.split}: {title} ({len(sel)} such scenes)', dest / f'{name}.png', n=a.rows)
        print(name, len(sel), 'saved' if ok else 'none')
    distributions(folder, m, t, dest / '00_distributions.png'); print('00_distributions saved')


if __name__ == '__main__': main()
