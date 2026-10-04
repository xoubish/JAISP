"""Collect the detection-catalog measurements and compare them with Tractor and MER.

Everything here reads the per-region products written by the pipeline steps and
writes ``catalog_long.csv`` (one row per detection, band and method),
``catalog_wide.csv`` (one row per detection), summary tables and figures.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.visualization import AsinhStretch, ImageNormalize
from .common import OUT, EUCLID_BANDS, RUBIN_BANDS, ROOT, PSF_CALIBRATION, region_dir, tangent_arcsec, read_json, write_json

METHODS = ('foundation', 'image', 'tractor')  # extended with 'scarlet' by collect() when scarlet_fluxes.csv exists
LABELS = {'foundation': 'JAISP foundation prior', 'image': 'JAISP image prior', 'tractor': 'Tractor VIS profile (fixed positions)', 'scarlet': 'JAISP amortised scarlet'}
COLORS = {'foundation': '#2a78d6', 'image': '#eb6834', 'tractor': '#1baf7a', 'scarlet': '#eda100'}
MAG_EDGES = np.arange(17.5, 27.01, .5)
NISP_REF = 'templfit'  # MER NISP reference: 'templfit' (T-PHOT, as upstream) or 'sersic'
RUBIN_NJY = True  # Rubin deep_coadd pixels are assumed to be in nanojansky (LSST DP1 convention)


def nmad(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def reference(frame, band):
    short = band.split('_')[1].lower()
    if short == 'vis':
        star = frame.mer_is_star.to_numpy(bool)
        flux = np.where(star, frame.mer_flux_vis_psf, frame.mer_flux_vis_sersic); err = np.where(star, frame.mer_fluxerr_vis_psf, frame.mer_fluxerr_vis_sersic)
        return flux, err, np.where(star, 'vis_psf', 'vis_sersic')
    return frame[f'mer_flux_{short}_{NISP_REF}'].to_numpy(float), frame[f'mer_fluxerr_{short}_{NISP_REF}'].to_numpy(float), np.full(len(frame), f'{short}_{NISP_REF}')


def collect():
    parts = []
    for folder in sorted(OUT.glob('region_*')):
        needed = [folder / f for f in ('foundation_fluxes.csv', 'tractor_fluxes.csv', 'mer_match.csv', 'detections.csv')]
        if not all(f.exists() for f in needed): continue
        ours, tractor, mer, det = (pd.read_csv(f) for f in needed)
        tractor = tractor.drop(columns=['flux_native'])
        ours['profile'] = 'mixture'; ours['blobbed'] = True
        frames = [ours, tractor]
        if (folder / 'scarlet_fluxes.csv').exists():
            scarlet = pd.read_csv(folder / 'scarlet_fluxes.csv'); scarlet['profile'] = 'scarlet'; scarlet['blobbed'] = True; frames.append(scarlet)
        rows = pd.concat(frames, ignore_index=True, sort=False)
        rows = rows.merge(det[['source', 'ra', 'dec', 'score', 'shift_px', 'x_vis', 'y_vis']], on='source', how='left')
        rows = rows.merge(mer.drop(columns=['region', 'ra', 'dec']), on='source', how='left')
        parts.append(rows)
    if not parts: raise ValueError('No regions have photometry, Tractor and MER products yet')
    d = pd.concat(parts, ignore_index=True)
    if d.duplicated(['region', 'source', 'band', 'model']).any(): raise ValueError('Duplicate measurement rows')
    global METHODS
    METHODS = ('foundation', 'image', 'tractor') + (('scarlet',) if (d.model == 'scarlet').any() else ())
    rubin = d.band.str.startswith('rubin')
    if RUBIN_NJY:
        d.loc[rubin, 'flux_ujy'] = d.loc[rubin, 'flux_native'] / 1e3; d.loc[rubin, 'error_ujy'] = d.loc[rubin, 'error_native'] / 1e3
    # Sources measured by all three methods in the Euclid bands form the paired comparison sample.
    counts = d[d.band.isin(EUCLID_BANDS)].groupby(['region', 'source']).model.nunique()
    common = counts[counts == len(METHODS)].index
    d['common'] = pd.MultiIndex.from_frame(d[['region', 'source']]).isin(common)
    # Neighbouring tiles overlap by 6 arcsec: a detection may appear twice; keep the first region's copy.
    det = d.drop_duplicates(['region', 'source'])[['region', 'source', 'ra', 'dec']].sort_values(['region', 'source'])
    sky = det[['ra', 'dec']].to_numpy(); duplicate = np.zeros(len(det), bool)
    from scipy.spatial import cKDTree
    center = sky.mean(0); tree = cKDTree(tangent_arcsec(sky, center)); region = det.region.to_numpy()
    for i, j in tree.query_pairs(.3):
        duplicate[max(i, j, key=lambda k: region[k])] = True
    d = d.merge(det.assign(duplicate=duplicate)[['region', 'source', 'duplicate']], on=['region', 'source'])
    d['ref_flux_ujy'] = np.nan; d['ref_error_ujy'] = np.nan; d['ref_column'] = ''
    for band in EUCLID_BANDS:
        sel = (d.band == band) & d.matched.fillna(False).astype(bool)
        flux, err, column = reference(d[sel], band)
        d.loc[sel, 'ref_flux_ujy'] = flux; d.loc[sel, 'ref_error_ujy'] = err; d.loc[sel, 'ref_column'] = column
    d['ref_mag'] = np.where(d.ref_flux_ujy > 0, 23.9 - 2.5 * np.log10(d.ref_flux_ujy.where(d.ref_flux_ujy > 0)), np.nan)
    d['mag'] = np.where(d.flux_ujy > 0, 23.9 - 2.5 * np.log10(d.flux_ujy.where(d.flux_ujy > 0)), np.nan)
    d['snr'] = d.flux_ujy / d.error_ujy
    both = (d.flux_ujy > 0) & (d.ref_flux_ujy > 0)
    d['delta_ab'] = np.where(both, -2.5 * np.log10((d.flux_ujy / d.ref_flux_ujy).where(both)), np.nan)
    d['chi'] = (d.flux_ujy - d.ref_flux_ujy) / np.sqrt(d.error_ujy ** 2 + d.ref_error_ujy ** 2)
    vis_ref = d[d.band == 'euclid_VIS'].drop_duplicates(['region', 'source']).set_index(['region', 'source']).ref_mag
    d['vis_ref_mag'] = pd.MultiIndex.from_frame(d[['region', 'source']]).map(vis_ref)
    d['galaxy'] = d.matched.fillna(False).astype(bool) & ~d.mer_is_star.fillna(False).astype(bool)
    d.to_csv(OUT / 'catalog_long.csv', index=False)
    wide = d.pivot_table(index=['region', 'source', 'ra', 'dec', 'score', 'shift_px', 'matched', 'mer_object_id', 'match_sep_arcsec', 'duplicate', 'common'],
                         columns=['model', 'band'], values=['flux_ujy', 'error_ujy'], aggfunc='first')
    wide.columns = [f'{model}_{band}_{value}' for value, model, band in wide.columns]
    wide.reset_index().to_csv(OUT / 'catalog_wide.csv', index=False)
    return d


def comparison_sample(d):
    """Paired sample: MER-matched, measured by all methods, not a tile-edge duplicate, Euclid bands."""
    return d[d.common & ~d.duplicate & d.matched.fillna(False).astype(bool) & d.band.isin(EUCLID_BANDS) & d.primary_match.fillna(False).astype(bool)]


def subsets(x):
    nn = x.nearest_detection_arcsec if 'nearest_detection_arcsec' in x else x.second_mer_sep_arcsec
    return {'all': np.ones(len(x), bool), 'galaxies': x.galaxy.to_numpy(bool), 'stars': x.mer_is_star.fillna(False).to_numpy(bool),
            'isolated_3arcsec': (x.n_mer_within_1arcsec.fillna(9) <= 1).to_numpy() & (x.second_mer_sep_arcsec.fillna(0) >= 3).to_numpy(),
            'crowded_1.5arcsec': (x.second_mer_sep_arcsec.fillna(99) < 1.5).to_numpy(),
            'clean_mer_flag': (x.mer_det_quality_flag.fillna(1) == 0).to_numpy()}


def summarize(d):
    x = comparison_sample(d); rows = []
    bins = [(-np.inf, np.inf, 'all'), (-np.inf, 21, '<21'), (21, 22, '21-22'), (22, 23, '22-23'), (23, 24, '23-24'), (24, 25, '24-25'), (25, 26, '25-26'), (26, np.inf, '>26')]
    masks = subsets(x)
    for band in EUCLID_BANDS:
        for subset, keep in masks.items():
            for lo, hi, label in bins:
                for method in METHODS:
                    g = x[(x.band == band) & (x.model == method) & keep & (x.ref_mag >= lo) & (x.ref_mag < hi)]
                    e = g.delta_ab.to_numpy(); ok = np.isfinite(e)
                    rows.append(dict(band=band, subset=subset, mag_bin=label, model=method, n=int(len(g)), n_positive=int(ok.sum()),
                                     median_delta_ab=float(np.median(e[ok])) if ok.any() else np.nan, nmad_ab=nmad(e),
                                     outlier_fraction=float((np.abs(e[ok]) > .5).mean()) if ok.any() else np.nan,
                                     chi_nmad=nmad(g.chi), nonpositive_fraction=float((g.flux_ujy <= 0).mean()) if len(g) else np.nan,
                                     median_snr=float(g.snr.median()) if len(g) else np.nan))
    out = pd.DataFrame(rows); out.to_csv(OUT / 'summary.csv', index=False); return out


def paired_bootstrap(d, n_boot=1000, seed=20261002):
    """Tile-level bootstrap of method differences in NMAD and median offset on the common sample."""
    x = comparison_sample(d); rng = np.random.default_rng(seed); rows = []
    regions = np.array(sorted(x.region.unique()))
    for band in EUCLID_BANDS:
        g = x[x.band == band].pivot_table(index=['region', 'source'], columns='model', values='delta_ab')
        g = g.dropna(); region = g.index.get_level_values('region').to_numpy()
        def stats(idx):
            s = g.iloc[idx]
            return {m: (nmad(s[m]), float(np.median(s[m]))) for m in METHODS}
        base = stats(np.arange(len(g))); by_region = {r: np.flatnonzero(region == r) for r in regions}
        samples = []
        for _ in range(n_boot):
            pick = rng.choice(regions, len(regions), replace=True)
            idx = np.concatenate([by_region[r] for r in pick if len(by_region[r])])
            samples.append(stats(idx))
        pairs = [('foundation', 'tractor'), ('foundation', 'image')] + ([('scarlet', 'foundation'), ('scarlet', 'tractor')] if 'scarlet' in METHODS else [])
        for first, other in pairs:
            for k, metric in enumerate(('nmad', 'median')):
                diff = np.array([s[first][k] - s[other][k] for s in samples])
                rows.append(dict(band=band, comparison=f'{first}-{other}', metric=metric, n=len(g),
                                 foundation=base[first][k], other=base[other][k], difference=base[first][k] - base[other][k],
                                 ci_low=float(np.percentile(diff, 2.5)), ci_high=float(np.percentile(diff, 97.5))))
    out = pd.DataFrame(rows); out.to_csv(OUT / 'paired_bootstrap.csv', index=False); return out


def binned(x, y, edges, minimum=8):
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    centers, q, n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() < minimum: continue
        centers.append(np.median(x[m])); q.append(np.percentile(y[m], [16, 50, 84])); n.append(int(m.sum()))
    return np.array(centers), np.array(q).reshape(-1, 3).T, np.array(n)


def _style(ax):
    ax.grid(alpha=.18, lw=.6); ax.spines[['top', 'right']].set_visible(False)


def fig_delta_vs_mag(d, ylimit=1.):
    x = comparison_sample(d); fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for band, ax in zip(EUCLID_BANDS, axes.flat):
        counts = []
        for method in METHODS:
            g = x[(x.band == band) & (x.model == method)]; counts.append(int(np.isfinite(g.delta_ab).sum()))
            inside = np.abs(g.delta_ab) <= ylimit
            ax.scatter(g.ref_mag[inside], g.delta_ab[inside], color=COLORS[method], s=5, alpha=.12, linewidths=0, rasterized=True)
            c, q, n = binned(g.ref_mag, g.delta_ab, MAG_EDGES)
            if len(c):
                ax.plot(c, q[1], color=COLORS[method], lw=2, label=LABELS[method]); ax.fill_between(c, q[0], q[2], color=COLORS[method], alpha=.12, lw=0)
        ax.axhline(0, color='k', lw=.8, ls=':'); ax.set_ylim(-ylimit, ylimit); ax.set_title(band.replace('euclid_', 'Euclid '))
        ax.text(.02, .97, 'positive-flux pairs ' + '/'.join(m[0].upper() for m in METHODS) + ': ' + '/'.join(map(str, counts)), transform=ax.transAxes, va='top', fontsize=8); _style(ax)
        ax.set_xlabel('MER reference AB magnitude')
    for ax in axes[:, 0]: ax.set_ylabel('measured − MER  ΔAB (mag)')
    h, l = axes.flat[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=len(METHODS), fontsize=9, bbox_to_anchor=(.5, .01))
    fig.suptitle('Detection-head sources on real Euclid Q1 tiles: measured minus MER magnitude (median and 16–84% band per 0.5 mag)', y=.995)
    fig.subplots_adjust(left=.07, right=.99, bottom=.11, top=.92, hspace=.3, wspace=.06)
    fig.savefig(OUT / 'delta_ab_vs_mag.png', dpi=160); fig.savefig(OUT / 'delta_ab_vs_mag.pdf'); return fig


def fig_scatter_vs_mag(d):
    x = comparison_sample(d); fig, axes = plt.subplots(3, 4, figsize=(17, 10), sharex=True)
    for j, band in enumerate(EUCLID_BANDS):
        for method in METHODS:
            g = x[(x.band == band) & (x.model == method)]
            cm, qm, n = binned(g.ref_mag, g.delta_ab, MAG_EDGES, minimum=15)
            nm = [nmad(g.delta_ab[(g.ref_mag >= lo) & (g.ref_mag < hi)]) for lo, hi in zip(MAG_EDGES[:-1], MAG_EDGES[1:])]
            cn = [(lo + hi) / 2 for lo, hi in zip(MAG_EDGES[:-1], MAG_EDGES[1:])]
            cs = [nmad(g.chi[(g.ref_mag >= lo) & (g.ref_mag < hi)]) for lo, hi in zip(MAG_EDGES[:-1], MAG_EDGES[1:])]
            keep = np.array([((g.ref_mag >= lo) & (g.ref_mag < hi)).sum() >= 15 for lo, hi in zip(MAG_EDGES[:-1], MAG_EDGES[1:])])
            axes[0, j].plot(cm, qm[1], color=COLORS[method], lw=2, marker='o', ms=4, label=LABELS[method])
            axes[1, j].plot(np.array(cn)[keep], np.array(nm)[keep], color=COLORS[method], lw=2, marker='o', ms=4)
            axes[2, j].plot(np.array(cn)[keep], np.array(cs)[keep], color=COLORS[method], lw=2, marker='o', ms=4)
        axes[0, j].set_title(band.replace('euclid_', 'Euclid ')); axes[0, j].axhline(0, color='k', lw=.8, ls=':'); axes[0, j].set_ylim(-.4, .4)
        axes[1, j].set_ylim(0, .6); axes[2, j].axhline(1, color='k', lw=.8, ls=':'); axes[2, j].set_ylim(0, 5); axes[2, j].set_xlabel('MER reference AB magnitude')
        for ax in axes[:, j]: _style(ax)
    axes[0, 0].set_ylabel('median ΔAB (measured − MER)'); axes[1, 0].set_ylabel('NMAD of ΔAB (mag)'); axes[2, 0].set_ylabel('NMAD of (f − f_MER)/σ_combined')
    h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(.5, .005))
    fig.suptitle('Offset, scatter and error calibration against MER per 0.5 mag bin (≥15 sources per bin)', y=.995)
    fig.subplots_adjust(left=.06, right=.99, bottom=.1, top=.93, hspace=.15, wspace=.2)
    fig.savefig(OUT / 'scatter_vs_mag.png', dpi=160); fig.savefig(OUT / 'scatter_vs_mag.pdf'); return fig


def fig_head_to_head(d, ylimit=.75):
    x = comparison_sample(d); comparisons = [('foundation', 'tractor'), ('foundation', 'image')] + ([('scarlet', 'foundation'), ('scarlet', 'tractor')] if 'scarlet' in METHODS else [])
    fig, axes = plt.subplots(len(comparisons), 4, figsize=(17, 3.75 * len(comparisons)), sharey=True)
    for j, band in enumerate(EUCLID_BANDS):
        g = x[x.band == band].pivot_table(index=['region', 'source'], columns='model', values=['flux_ujy', 'ref_mag', 'mer_is_star'], aggfunc='first')
        mag = g[('ref_mag', 'foundation')]
        for row, (first, other) in enumerate(comparisons):
            color = COLORS[other if first == 'foundation' else first]
            f, o = g[('flux_ujy', first)], g[('flux_ujy', other)]; ok = (f > 0) & (o > 0)
            delta = -2.5 * np.log10(f[ok] / o[ok]); ax = axes[row, j]
            ax.scatter(mag[ok], delta, s=5, alpha=.15, color=color, linewidths=0, rasterized=True)
            c, q, n = binned(mag[ok], delta, MAG_EDGES)
            if len(c): ax.plot(c, q[1], color=color, lw=2); ax.fill_between(c, q[0], q[2], color=color, alpha=.15, lw=0)
            ax.axhline(0, color='k', lw=.8, ls=':'); ax.set_ylim(-ylimit, ylimit); _style(ax)
            ax.text(.02, .97, f'n = {int(ok.sum())}; NMAD = {nmad(delta):.3f}', transform=ax.transAxes, va='top', fontsize=8)
        axes[0, j].set_title(band.replace('euclid_', 'Euclid ')); axes[-1, j].set_xlabel('MER reference AB magnitude')
    for row, (first, other) in enumerate(comparisons): axes[row, 0].set_ylabel(f'{first} − {other}  ΔAB (mag)')
    fig.suptitle('Direct method differences on the same sources, pixels, masks, PSFs and positions (no MER involved in the y-axis)', y=.995)
    fig.subplots_adjust(left=.06, right=.99, bottom=.09, top=.9, hspace=.25, wspace=.06)
    fig.savefig(OUT / 'head_to_head.png', dpi=160); fig.savefig(OUT / 'head_to_head.pdf'); return fig


def fig_trends(d):
    x = comparison_sample(d); fig, axes = plt.subplots(3, 4, figsize=(17, 10.5), sharey=True)
    size_edges = np.array([0, 1.5, 2.5, 3.5, 5, 7, 10, 15, 25, 50]); crowd_edges = np.array([0, .75, 1, 1.5, 2, 3, 4, 6, 10, 20])
    for j, band in enumerate(EUCLID_BANDS):
        for method in METHODS:
            g = x[(x.band == band) & (x.model == method) & (x.ref_mag < 24.5)]
            c, q, n = binned(g.second_mer_sep_arcsec, g.delta_ab, crowd_edges)
            if len(c): axes[0, j].plot(c, q[1], color=COLORS[method], lw=2, marker='o', ms=4, label=LABELS[method]); axes[0, j].fill_between(c, q[0], q[2], color=COLORS[method], alpha=.1, lw=0)
            gg = g[g.galaxy]; c, q, n = binned(gg.mer_semimajor_axis * .1, gg.delta_ab, size_edges * .1)
            if len(c): axes[1, j].plot(c, q[1], color=COLORS[method], lw=2, marker='o', ms=4); axes[1, j].fill_between(c, q[0], q[2], color=COLORS[method], alpha=.1, lw=0)
            for k, (label, keep) in enumerate((('stars', g.mer_is_star.fillna(False).astype(bool)), ('galaxies', g.galaxy))):
                e = g.delta_ab[keep]; e = e[np.isfinite(e)]
                axes[2, j].errorbar(k + (list(METHODS).index(method) - (len(METHODS) - 1) / 2) * .2, np.median(e) if len(e) else np.nan,
                                    yerr=[[np.median(e) - np.percentile(e, 16)], [np.percentile(e, 84) - np.median(e)]] if len(e) else None,
                                    fmt='o', color=COLORS[method], capsize=3)
                axes[2, j].text(k + (list(METHODS).index(method) - (len(METHODS) - 1) / 2) * .2, -.68, str(len(e)), ha='center', fontsize=7, color=COLORS[method])
        axes[0, j].set_title(band.replace('euclid_', 'Euclid ')); axes[0, j].set_xscale('log'); axes[0, j].set_xlabel('distance to nearest other MER source (arcsec)')
        axes[1, j].set_xscale('log'); axes[1, j].set_xlabel('MER semimajor axis (arcsec), galaxies only')
        axes[2, j].set_xticks([0, 1]); axes[2, j].set_xticklabels(['MER stars', 'MER galaxies']); axes[2, j].set_xlim(-.6, 1.6)
        for ax in axes[:, j]: ax.axhline(0, color='k', lw=.8, ls=':'); ax.set_ylim(-.75, .75); _style(ax)
    for ax in axes[:, 0]: ax.set_ylabel('measured − MER  ΔAB (mag)')
    h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(.5, .005))
    fig.suptitle('Offset trends for MER < 24.5 sources: crowding, galaxy size, and star/galaxy class (median with 16–84% band)', y=.995)
    fig.subplots_adjust(left=.06, right=.99, bottom=.1, top=.93, hspace=.35, wspace=.06)
    fig.savefig(OUT / 'trends.png', dpi=160); fig.savefig(OUT / 'trends.pdf'); return fig


def fig_colors(d):
    x = comparison_sample(d); pairs = (('euclid_VIS', 'euclid_H'), ('euclid_Y', 'euclid_J'), ('euclid_J', 'euclid_H'))
    fig, axes = plt.subplots(len(METHODS), len(pairs), figsize=(13, 11), sharex='col', sharey='col')
    for j, (b1, b2) in enumerate(pairs):
        g = x[x.band.isin((b1, b2))].pivot_table(index=['region', 'source', 'model'], columns='band', values=['flux_ujy', 'snr', 'ref_flux_ujy'], aggfunc='first').reset_index()
        ok = (g[('flux_ujy', b1)] > 0) & (g[('flux_ujy', b2)] > 0) & (g[('ref_flux_ujy', b1)] > 0) & (g[('ref_flux_ujy', b2)] > 0) & (g[('snr', b1)] > 10) & (g[('snr', b2)] > 10)
        g = g[ok]; mer = -2.5 * np.log10(g[('ref_flux_ujy', b1)] / g[('ref_flux_ujy', b2)]); ours = -2.5 * np.log10(g[('flux_ujy', b1)] / g[('flux_ujy', b2)])
        for i, method in enumerate(METHODS):
            m = (g['model'] == method).to_numpy(); ax = axes[i, j]
            ax.scatter(mer[m], ours[m], s=6, alpha=.3, color=COLORS[method], linewidths=0, rasterized=True)
            lim = np.nanpercentile(mer[m], [1, 99]) if m.any() else (-1, 3); ax.plot(lim, lim, color='k', lw=.8, ls=':')
            ax.text(.02, .97, f'{LABELS[method]}\nn = {int(m.sum())}, NMAD(ours − MER) = {nmad(ours[m] - mer[m]):.3f}', transform=ax.transAxes, va='top', fontsize=8); _style(ax)
            name = f'{b1.split("_")[1]} − {b2.split("_")[1]}'
            if i == len(METHODS) - 1: ax.set_xlabel(f'MER colour {name} (mag)')
            if j == 0: ax.set_ylabel('measured colour (mag)')
    fig.suptitle('Colours for sources with S/N > 10 in both bands: measured versus MER', y=.995)
    fig.subplots_adjust(left=.07, right=.99, bottom=.06, top=.94, hspace=.12, wspace=.12)
    fig.savefig(OUT / 'colors.png', dpi=160); return fig


def detection_statistics(d):
    completeness = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob('region_*/mer_completeness.csv'))], ignore_index=True)
    completeness = completeness[completeness.in_footprint & (completeness.flux_vis_psf > 0)].copy()
    completeness['vis_mag'] = 23.9 - 2.5 * np.log10(np.where(completeness.is_star, completeness.flux_vis_psf, completeness.flux_vis_sersic.where(completeness.flux_vis_sersic > 0, completeness.flux_vis_psf)))
    detections = d[(d.band == 'euclid_VIS') & (d.model == 'foundation') & ~d.duplicate]
    edges = np.arange(17, 28.01, .5); rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        c = completeness[(completeness.vis_mag >= lo) & (completeness.vis_mag < hi)]; p = detections[(detections.mag >= lo) & (detections.mag < hi)]
        rows.append(dict(mag_low=lo, mag_high=hi, mer_sources=len(c), completeness=float(c.detected.mean()) if len(c) else np.nan,
                         detections=len(p), purity=float(p.matched.mean()) if len(p) else np.nan))
    out = pd.DataFrame(rows); out.to_csv(OUT / 'detection_statistics.csv', index=False); return out


def fig_detection(stats):
    fig, ax = plt.subplots(figsize=(8, 4.5)); c = (stats.mag_low + stats.mag_high) / 2
    ax.plot(c, stats.completeness, color=COLORS['foundation'], lw=2, marker='o', ms=4, label='completeness: MER sources with a detection within 0.5″')
    ax.plot(c, stats.purity, color=COLORS['tractor'], lw=2, marker='s', ms=4, label='purity: detections with a MER source within 0.5″')
    ax.set_xlabel('VIS AB magnitude (MER for completeness; foundation measurement for purity)'); ax.set_ylabel('fraction'); ax.set_ylim(0, 1.02); _style(ax)
    ax.legend(fontsize=8, loc='lower left'); ax.set_title('Detection head versus the MER catalogue inside the photometered footprint')
    fig.tight_layout(); fig.savefig(OUT / 'detection_statistics.png', dpi=160); return fig


def fig_rubin(d):
    g = d[(d.model == 'foundation') & ~d.duplicate]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for band, color in zip(RUBIN_BANDS, ('#4a3aa7', '#2a78d6', '#1baf7a', '#eda100', '#eb6834', '#e34948')):
        r = g[g.band == band]; c, q, n = binned(r.vis_ref_mag, r.snr, MAG_EDGES)
        if len(c): axes[0].plot(c, q[1], color=color, lw=2, marker='o', ms=3, label=band.split('_')[1])
    axes[0].set_yscale('log'); axes[0].set_xlabel('MER VIS AB magnitude'); axes[0].set_ylabel('median Rubin flux S/N (foundation)'); axes[0].legend(fontsize=8, ncol=2); _style(axes[0])
    wide = g[g.band.isin(('rubin_i', 'rubin_z', 'euclid_VIS'))].pivot_table(index=['region', 'source'], columns='band', values=['flux_ujy', 'snr'], aggfunc='first')
    ok = (wide[('snr', 'rubin_i')] > 10) & (wide[('snr', 'euclid_VIS')] > 10) & (wide[('flux_ujy', 'rubin_i')] > 0) & (wide[('flux_ujy', 'euclid_VIS')] > 0)
    axes[1].scatter(wide[('flux_ujy', 'euclid_VIS')][ok], wide[('flux_ujy', 'rubin_i')][ok], s=5, alpha=.3, color=COLORS['foundation'], linewidths=0, rasterized=True)
    lim = (wide[('flux_ujy', 'euclid_VIS')][ok].quantile([.005, .995]).to_numpy()); axes[1].plot(lim, lim, color='k', ls=':', lw=.8)
    axes[1].set_xscale('log'); axes[1].set_yscale('log'); axes[1].set_xlabel('foundation VIS flux (μJy)'); axes[1].set_ylabel('foundation Rubin i flux (μJy, nJy pixels assumed)'); _style(axes[1])
    ratio = -2.5 * np.log10(wide[('flux_ujy', 'rubin_i')][ok] / wide[('flux_ujy', 'euclid_VIS')][ok])
    axes[1].text(.02, .97, f'n = {int(ok.sum())}; median i − VIS = {np.median(ratio):.3f} mag; NMAD {nmad(ratio):.3f}', transform=axes[1].transAxes, va='top', fontsize=8)
    nonpos = [(d[(d.model == 'foundation') & (d.band == b)].flux_ujy <= 0).mean() for b in RUBIN_BANDS + EUCLID_BANDS]
    axes[2].bar(range(10), nonpos, color=[COLORS['foundation']] * 10, width=.6); axes[2].set_xticks(range(10)); axes[2].set_xticklabels([b.split('_')[1] for b in RUBIN_BANDS + EUCLID_BANDS])
    axes[2].set_ylabel('fraction of non-positive signed fluxes'); _style(axes[2])
    fig.suptitle('Rubin bands measured jointly by the foundation photometer (no catalogue reference for ugrizy)', y=.99)
    fig.tight_layout(); fig.savefig(OUT / 'rubin.png', dpi=160); return fig


def fig_example(d, region=None, source=None, half=40):
    """Data, both models and residuals around one bright MER galaxy, all four Euclid bands."""
    import torch
    from .prepare import load_inputs, scene_for_source
    from ..scene_features import SceneEncoder, image_features
    from ..run_mixture import predict_prior
    from ..mixture import fit_multiband
    x = comparison_sample(d)
    if region is None:
        g = x[(x.band == 'euclid_VIS') & (x.model == 'foundation') & x.galaxy & (x.ref_mag.between(19.5, 21)) & (x.valid_fraction > .995) & (x.second_mer_sep_arcsec > 2.5)]
        pick = g.sort_values('ref_mag').iloc[len(g) // 2]; region, source = int(pick.region), int(pick.source)
    inputs = load_inputs(region_dir(region)); sigmas = {b: v['sigma_px'] for b, v in read_json(ROOT / PSF_CALIBRATION).items()}
    scene, info = scene_for_source(inputs, int(np.flatnonzero(inputs['source'] == source)[0]), sigmas)
    cp = torch.load(ROOT / 'models/photometry/self_supervised/runs/q1_mixture_calibrated/priors.pt', map_location='cpu', weights_only=False)
    encoder = SceneEncoder(ROOT / cp['metadata']['original']['foundation_checkpoint'])
    item = dict(scene=scene, foundation=encoder(scene), image=image_features(scene))
    prior = predict_prior(item, cp['heads']['foundation'], cp['population'], 'foundation')
    fits = fit_multiband(scene, prior, prior_precision=cp['heads']['foundation'].get('precision'), strength=cp['metadata']['prior_strength'], band_strength=cp['metadata']['band_strength'])
    with np.load(region_dir(region) / 'tractor_models.npz') as z: tractor = {b: z[b] for b in EUCLID_BANDS}
    fig, axes = plt.subplots(4, 5, figsize=(17, 13.5))
    for i, band in enumerate(EUCLID_BANDS):
        b = scene['bands'][band]; image = b['image'].numpy(); var = b['variance'].numpy(); mask = b['mask'].numpy(); model = fits[band]['model']
        xy = inputs[band + '__positions'][np.flatnonzero(inputs['source'] == source)[0]]; origin = np.rint(xy).astype(int) - 60
        tr = tractor[band][origin[1]:origin[1] + image.shape[0], origin[0]:origin[0] + image.shape[1]]
        c = 60; s = np.s_[c - half:c + half + 1, c - half:c + half + 1]
        vals = image[s][mask[s]]; lo, hi = np.percentile(vals, [2, 99.7]); norm = ImageNormalize(vmin=lo, vmax=hi, stretch=AsinhStretch(a=.1), clip=True)
        sigma = np.sqrt(var[s])
        panels = [('data', image[s], norm), ('foundation model', model[s], norm), ('Tractor model', tr[s], norm),
                  ('(data − foundation)/σ', np.where(mask[s], (image[s] - model[s]) / sigma, np.nan), None), ('(data − Tractor)/σ', np.where(mask[s], (image[s] - tr[s]) / sigma, np.nan), None)]
        for j, (title, arr, nm) in enumerate(panels):
            ax = axes[i, j]
            if nm is not None: ax.imshow(np.ma.masked_where(~mask[s], arr), origin='lower', cmap='gray_r', norm=nm, interpolation='nearest')
            else: ax.imshow(arr, origin='lower', cmap='RdBu_r', vmin=-5, vmax=5, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(title)
            if j == 0: ax.set_ylabel(band.replace('euclid_', 'Euclid '))
        row = x[(x.region == region) & (x.source == source) & (x.band == band)].set_index('model')
        axes[i, 0].text(.02, .97, f'MER {row.ref_mag.iloc[0]:.2f} AB', transform=axes[i, 0].transAxes, va='top', fontsize=8, color='white', bbox=dict(facecolor='black', alpha=.5, pad=1))
        for j, method in ((1, 'foundation'), (2, 'tractor')):
            axes[i, j].text(.02, .97, f'{row.loc[method].mag:.2f} AB (Δ {row.loc[method].delta_ab:+.3f})', transform=axes[i, j].transAxes, va='top', fontsize=8, color='white', bbox=dict(facecolor='black', alpha=.5, pad=1))
    fig.suptitle(f'Region {region:03d}, detection {source}: {2 * half / 10:.0f}″ around a MER galaxy; residuals in pixel-noise units, clipped at ±5', y=.995)
    fig.subplots_adjust(left=.03, right=.99, bottom=.01, top=.95, hspace=.05, wspace=.03)
    fig.savefig(OUT / 'example_scene.png', dpi=140); return fig


def main():
    d = collect(); summary = summarize(d); boot = paired_bootstrap(d); stats = detection_statistics(d)
    x = comparison_sample(d)
    print('rows', len(d), '| detections', d.groupby(['region', 'source']).ngroups, '| paired MER-matched sources', x.groupby(['region', 'source']).ngroups)
    print(summary[(summary.subset == 'all') & (summary.mag_bin == 'all')].pivot(index='band', columns='model', values=['n_positive', 'median_delta_ab', 'nmad_ab']).round(3).to_string())
    print(boot.round(4).to_string())
    for f in (fig_delta_vs_mag, fig_scatter_vs_mag, fig_head_to_head, fig_trends, fig_colors, fig_rubin): plt.close(f(d))
    plt.close(fig_detection(stats)); plt.close(fig_example(d))


if __name__ == '__main__': main()
