"""Brainstorm figures for a truth-supervised amortised-scarlet photometry head.

Schematics only: no checkpoint is loaded and no real pixels are read. The toy
calculation in fig2 is a 1-D illustration of the flux-split degeneracy.

    python -m models.photometry.architecture_brainstorm.make_figures
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent / 'figures'
C = dict(frozen='#c9c9c9', old='#9ecae1', new='#fdae6b', fixed='#a1d99b', data='#f0f0f0', loss='#fbb4b9', test='#dadaeb')
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})


def box(ax, x, y, w, h, text, color, fs=8.5, weight='normal', ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                fc=color, ec='#333333', lw=1.0, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, weight=weight, linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, p, q, text=None, dashed=False, color='#333333', rad=0., fs=7.5, toff=(0, .12)):
    ax.annotate('', xy=q, xytext=p, arrowprops=dict(arrowstyle='-|>', lw=1.2, color=color, mutation_scale=12,
                                                    ls='--' if dashed else '-', connectionstyle=f'arc3,rad={rad}'))
    if text:
        ax.text((p[0] + q[0]) / 2 + toff[0], (p[1] + q[1]) / 2 + toff[1], text, ha='center', va='bottom',
                fontsize=fs, color=color, style='italic')


def canvas(w, h, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.axis('off')
    return fig, ax


# ----------------------------------------------------------------------------- fig 1
def fig_architecture():
    fig, ax = canvas(16, 8.6, 16, 9)
    ax.text(.2, 8.7, 'Proposed head: shared foundation shapes + per-band branch, scene-level linear flux solve',
            fontsize=13, weight='bold', va='top')

    box(ax, .2, 5.3, 2.7, 2.2, 'Scene cutout\n10 bands on native grids\nimage, variance, mask\nall detections in region\n(VIS-anchored positions)', C['data'])
    bands = ['u', 'g', 'r', 'i', 'z', 'y', 'VIS', 'Y', 'J', 'H']
    for j, b in enumerate(bands):
        ax.add_patch(Rectangle((.25 + j * .265, 4.75), .23, .32, fc='#7f7f7f' if j < 6 else '#3f3f3f', ec='none'))
        ax.text(.365 + j * .265, 4.6, b, ha='center', va='top', fontsize=6.5)

    box(ax, 3.4, 6.6, 2.7, 1.5, 'Foundation encoder\n10-band fused\n(frozen)', C['frozen'], weight='bold')
    box(ax, 3.4, 3.9, 2.7, 1.6, 'Raw S/N windows\n10 bands resampled to\n0.1″ grid, 61×61 / source', C['fixed'])

    box(ax, 6.7, 5.1, 2.8, 2.9, 'Shared shape trunk\n(per source, dilated CNN)\n\nellipse q, θ\ncompact + extended\nmonotone radial profiles\nperturbation ε, |log| ≤ 0.5', C['old'])
    ax.text(8.1, 7.75, 'existing', ha='center', fontsize=7, style='italic', color='#08519c')

    box(ax, 6.7, 1.4, 2.8, 3.1, 'Per-band branch  (NEW)\none net, band embedding\n\nin: band-b native pixels,\nPSF_b, residual R_b,\ntrunk features\n\nout: mix w_ib, δε_ib\n(small, bounded)', C['new'])

    box(ax, 10.1, 3.9, 2.4, 2.3, 'Render, per band\nM_ib ⊗ PSF_b(x_i)\nflux-conserving\n→ unit templates T_ib', C['fixed'])
    box(ax, 13.1, 3.9, 2.7, 2.3, 'Linear solve, per band\nf_b = argmin χ²_b\nsigned, all sources jointly\nfixed robust background\n(flux never predicted)', C['fixed'], weight='bold')
    box(ax, 13.1, 6.7, 2.7, 1.5, 'Outputs\nf_ib, Cov(f_b)\ntemplates, model image', C['data'])
    box(ax, 13.1, 1.4, 2.7, 1.8, 'Residual\nR_b = (I_b − model_b) / σ_b\n∂χ²/∂ε', C['data'])

    arrow(ax, (2.9, 6.9), (3.4, 7.3))
    arrow(ax, (2.9, 5.8), (3.4, 4.8))
    arrow(ax, (6.1, 7.35), (6.7, 7.0))
    ax.text(4.75, 6.35, 'bottleneck 15×15×256, VIS stem 61×61×64', ha='center', va='top', fontsize=7.5, style='italic', color='#333333')
    arrow(ax, (6.1, 4.7), (6.7, 5.6))
    arrow(ax, (8.1, 5.1), (8.1, 4.5), 'features', toff=(.45, -.15))
    ax.annotate('', xy=(6.7, 2.0), xytext=(1.5, 4.45), arrowprops=dict(arrowstyle='-|>', lw=1.2, color='#d94801',
                mutation_scale=12, connectionstyle='angle,angleA=-90,angleB=180,rad=8'))
    ax.text(3.2, 1.75, 'band-b pixels at native resolution', fontsize=7.5, color='#d94801', style='italic')
    arrow(ax, (9.5, 6.4), (10.1, 5.6))
    ax.text(9.95, 6.3, 'shared shape M_i', fontsize=7.5, style='italic', color='#333333')
    arrow(ax, (9.5, 3.0), (10.1, 4.4), 'band shape M_ib', toff=(.3, -.35))
    arrow(ax, (12.5, 5.05), (13.1, 5.05))
    arrow(ax, (14.45, 6.2), (14.45, 6.7))
    arrow(ax, (14.45, 3.9), (14.45, 3.2))
    arrow(ax, (13.1, 2.3), (9.5, 2.3), 'refine ×2', dashed=True, color='#d94801')
    arrow(ax, (13.4, 3.2), (9.5, 5.4), dashed=True, color='#08519c', rad=-.25)
    ax.add_patch(FancyBboxPatch((13.35, 8.35), 2.2, .45, boxstyle='round,pad=0.02', fc=C['loss'], ec='#a50f15'))
    ax.text(14.45, 8.57, 'losses: see Fig. 3', ha='center', va='center', fontsize=8, color='#a50f15')

    for k, (lab, col) in enumerate([('frozen', C['frozen']), ('existing, trainable', C['old']),
                                    ('new, trainable', C['new']), ('fixed op, no weights', C['fixed']), ('data', C['data'])]):
        ax.add_patch(Rectangle((.3 + k * 2.6, .35), .35, .3, fc=col, ec='#333333'))
        ax.text(.75 + k * 2.6, .5, lab, va='center', fontsize=8)
    fig.savefig(OUT / 'fig1_architecture.png', dpi=180, bbox_inches='tight'); plt.close(fig)


# ----------------------------------------------------------------------------- fig 2
def profile1d(x, centre, h, psf):
    """Unit-flux 1-D exponential convolved with a Gaussian PSF (pixel = grid step)."""
    p = np.exp(-np.abs(x - centre) / h)
    k = np.exp(-.5 * ((x - x.mean()) / psf) ** 2)
    out = np.convolve(p, k / k.sum(), mode='same')
    return out / out.sum()


def fig_degeneracy():
    x = np.arange(-4, 4.0001, .1); psf = .1
    ca, cb, ha, hb = -.35, .35, .2, .4
    A, B = profile1d(x, ca, ha, psf), profile1d(x, cb, hb, psf)
    S = A + B
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))

    ax = axs[0]
    ax.plot(x, S, 'k', lw=2, label='scene (A+B)')
    ax.plot(x, A, color='#e6550d', label=f'true A, flux {A.sum():.2f}')
    ax.plot(x, B, color='#3182bd', label=f'true B, flux {B.sum():.2f}')
    ax.set_xlim(-2, 2); ax.set_title('a) Truth: 0.7″ pair, equal flux'); ax.set_xlabel('arcsec'); ax.legend(fontsize=7.5, frameon=False)

    ax = axs[1]
    w = 1 / (1 + np.exp((x - .25) / .12))
    A2, B2 = S * w, S * (1 - w)
    ax.plot(x, S, 'k', lw=2, label='model A′+B′ = scene, residual 0')
    ax.plot(x, A2, color='#e6550d', ls='--', label=f'free-form A′, flux {A2.sum():.2f}')
    ax.plot(x, B2, color='#3182bd', ls='--', label=f'free-form B′, flux {B2.sum():.2f}')
    ax.set_xlim(-2, 2); ax.set_title('b) Free-form shapes: same χ², different fluxes'); ax.set_xlabel('arcsec')
    ax.legend(fontsize=7.5, frameon=False)

    ax = axs[2]
    sizes = np.geomspace(.05, 1.2, 60)
    rows = []
    for sa in sizes:
        for sb in sizes:
            for da in (-.06, 0, .06):          # small centroid errors as well
                T = np.stack((profile1d(x, ca + da, sa, psf), profile1d(x, cb, sb, psf)), 1)
                f, *_ = np.linalg.lstsq(T, S, rcond=None)
                rows.append((f[0] / A.sum() - 1, np.sum((S - T @ f) ** 2)))
    err, r2 = np.array(rows).T
    for snr, col in ((100, '#08519c'), (30, '#6baed6'), (10, '#fd8d3c')):
        sigma2 = np.sum(S ** 2) / snr ** 2         # noise level giving this optimal scene S/N
        dchi = r2 / sigma2
        ax.scatter(err * 100, dchi, s=3, color=col, alpha=.5, label=f'scene S/N {snr}')
        ok = dchi < 1
        ax.axvspan(err[ok].min() * 100, err[ok].max() * 100, color=col, alpha=.12)
    ax.axhline(1, color='k', lw=.8, ls=':'); ax.text(-58, 1.25, 'Δχ² = 1', fontsize=7.5)
    ax.set_yscale('log'); ax.set_ylim(1e-2, 1e3); ax.set_xlim(-60, 60)
    ax.set_xlabel('flux error of source A [%]'); ax.set_ylabel('expected Δχ² vs truth')
    ax.set_title('c) Monotone profiles: faint pairs still under-constrained')
    ax.legend(fontsize=7.5, frameon=False, loc='lower right', markerscale=4)
    fig.suptitle('Why a residual-only loss cannot fix flux errors (1-D toy; shaded = flux errors allowed within Δχ² < 1)',
                 fontsize=12, weight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / 'fig2_chi2_degeneracy.png', dpi=180, bbox_inches='tight'); plt.close(fig)


# ----------------------------------------------------------------------------- fig 3
def galaxy(n, xc, yc, re, q, theta, clump=None, seed=0):
    y, x = np.mgrid[:n, :n].astype(float)
    c, s = np.cos(theta), np.sin(theta)
    u, v = (x - xc) * c + (y - yc) * s, -(x - xc) * s + (y - yc) * c
    img = np.exp(-np.sqrt(u ** 2 + (v / q) ** 2) / re) + .6 * np.exp(-(u ** 2 + (v / q) ** 2) / (2 * (re * .35) ** 2))
    if clump:
        img += clump[2] * np.exp(-((x - xc - clump[0]) ** 2 + (y - yc - clump[1]) ** 2) / 2.)
    return img / img.sum()


def show(ax, img, title, lo=None, hi=None, cmap='magma'):
    s = np.arcsinh(img / (np.std(img) * .3 + 1e-12))
    ax.imshow(s, origin='lower', cmap=cmap, vmin=lo if lo is not None else np.percentile(s, 1),
              vmax=hi if hi is not None else np.percentile(s, 99.8))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8.5)


def fig_training():
    rng = np.random.default_rng(3); n = 48
    fig = plt.figure(figsize=(16, 10.4))
    fig.text(.01, .985, 'Training data and loss: truth-scored scenes built from real training-tile galaxies and real sky',
             fontsize=13, weight='bold', va='top')

    def row(y0, label):
        fig.text(.03, y0 + .305, label, fontsize=11, weight='bold', va='top', color='#a50f15')
        return [fig.add_axes([.03 + k * .19, y0, .155, .155 * 16 / 10.4]) for k in range(5)]

    # Row A: dim and transplant
    axs = row(.62, 'A. Dim-and-transplant (faint end, isolated)')
    clean = galaxy(n, 24, 24, 3.2, .6, .5, clump=(4, 3, .004))
    bright = 4000 * clean + rng.normal(0, .25, (n, n))
    show(axs[0], bright, 'real bright donor, S/N > 50\n(training tiles only)')
    show(axs[1], clean, 'noise-free reconstruction M*\nper band (empirical.reconstruct)\nreference flux F_ref')
    alpha = .012
    sky = rng.normal(0, .25, (n, n)) + .05 * rng.normal(0, 1, (n, n)).cumsum(0) / 10
    faint = alpha * 4000 * clean + sky
    st = np.arcsinh(faint / (np.std(faint) * .3)); lo, hi = np.percentile(st, 1), np.percentile(st, 99.8)
    show(axs[2], sky, 'real blank-sky patch\n(training tiles only)', lo, hi)
    show(axs[3], faint, lo=lo, hi=hi, title=f'head input: α·M*⊗PSF_b + sky\nα chosen for S/N 2–30\nsame α in all 10 bands')
    axs[4].axis('off')
    axs[4].text(0, .5, 'truth known exactly:\n\n  f*_b = α · F_ref,b\n  template = M*_b ⊗ PSF_b\n\nF_ref errors are shared by\nbright and faint copies, so this\ntargets faint-end bias, not\nabsolute calibration.',
                fontsize=9, va='center', transform=axs[4].transAxes,
                bbox=dict(boxstyle='round', fc=C['data'], ec='#999999'))

    # Row B: real-pair blends
    axs = row(.31, 'B. Real-pair blends (deblending: who owns which light)')
    gA = galaxy(n, 21, 24, 2.4, .7, .3, seed=1)
    gB = galaxy(n, 28, 26, 4.0, .45, 1.9, clump=(-3, 2, .003), seed=2)
    show(axs[0], gA, 'donor A, M*_A')
    show(axs[1], gB, 'donor B, M*_B')
    fA, fB = 300., 100.
    blend = fA * gA + fB * gB + rng.normal(0, .25, (n, n))
    show(axs[2], blend, 'head input: blend on real sky\nsep 0.3–2″, ratio 1–10,\n+ any real neighbours in patch')
    show(axs[3], blend, 'targets per source\n(contours: f*_A M*_A, f*_B M*_B)')
    for g, col in ((fA * gA, '#fd8d3c'), (fB * gB, '#6baed6')):
        axs[3].contour(g, levels=np.percentile(g, [90, 97, 99.5]), colors=col, linewidths=.9)
    axs[4].axis('off')
    axs[4].text(0, .5, 'truth known exactly per source:\n\n  f*_A,b , f*_B,b\n  M*_A,b , M*_B,b\n\nthe χ² term alone would accept\nany light-trading split (Fig. 2);\nthese labels settle it.',
                fontsize=9, va='center', transform=axs[4].transAxes,
                bbox=dict(boxstyle='round', fc=C['data'], ec='#999999'))
    for y in (.62 + .12, .31 + .12):
        for k in range(3):
            fig.text(.2035 + k * .19, y, '→', fontsize=16, ha='center', va='center')

    # loss box
    ax = fig.add_axes([.03, .015, .94, .26]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((.0, .02), 1, .96, boxstyle='round,pad=0.01', fc='#fff5f0', ec='#a50f15'))
    ax.text(.02, .88, 'Loss per training scene (sum over bands b and truth sources i; real neighbours enter only via χ²)',
            fontsize=10.5, weight='bold', va='top')
    lines = [
        (r'$\lambda_f\;\sum_{ib}\;\mathrm{Huber}\!\left[(\hat f_{ib}-f^*_{ib})\,/\,\sigma_{ib}\right]$',
         'flux in noise units (same metric as the empirical pilot); σ_ib = conditional error from the linear solve'),
        (r'$+\;\lambda_t\;\sum_{ib}\;\|\,(\hat f_{ib}\hat T_{ib}-f^*_{ib}M^*_{ib}\otimes\mathrm{PSF}_b)/\sigma_b\|^2\;/\;\|f^*_{ib}M^*_{ib}\otimes\mathrm{PSF}_b/\sigma_b\|^2$',
         'per-source model image: penalises light trading even when total flux is right'),
        (r'$+\;\lambda_\chi\;\frac{1}{N_b}\sum_b\;\log(\chi^2_b/\mathrm{dof}_b)$',
         'existing objective: keeps explaining every pixel, incl. untruthed neighbours'),
        (r'$+\;\lambda_\beta\;\sum_{\mathrm{S/N\ bin}}\left[\mathrm{mean}_{\mathrm{batch}}\,(\hat f-f^*)/\sigma\right]^2$',
         'optional: explicit batch bias penalty for the −17% faint-end offset'),
    ]
    for k, (eq, note) in enumerate(lines):
        y = .70 - k * .17
        ax.text(.02, y, eq, fontsize=12.5, va='center')
        ax.text(.60, y, note, fontsize=8.5, va='center', color='#555555', style='italic')
    fig.savefig(OUT / 'fig3_training_and_loss.png', dpi=180, bbox_inches='tight'); plt.close(fig)


# ----------------------------------------------------------------------------- fig 4
def fig_protocol():
    fig, ax = canvas(16, 8.2, 16, 8.6)
    ax.text(.2, 8.45, 'Data separation and checkpoint selection', fontsize=13, weight='bold', va='top')

    ax.text(2.9, 7.55, 'TRAIN / VALIDATION', ha='center', fontsize=10, weight='bold')
    box(ax, .2, 5.6, 5.4, 1.6, '43 training tiles\nruns/amortised_scarlet/training_tiles\n(selected to exclude the 28 test tiles)', C['data'])
    box(ax, .2, 3.6, 2.6, 1.5, 'donor library\nS/N > 25, isolated\nreconstructed per band', C['fixed'])
    box(ax, 3.0, 3.6, 2.6, 1.5, 'blank-sky patches\nmulti-band masking\n(as in empirical v2)', C['fixed'])
    box(ax, .2, 1.4, 5.4, 1.7, 'on-the-fly scene generator\nrows A + B of Fig. 3, random α, sep, ratio,\nrotation, mild PSF broadening; whole tiles\nsplit 90 / 10 into train / validation', C['new'])
    arrow(ax, (1.5, 5.6), (1.5, 5.1)); arrow(ax, (4.3, 5.6), (4.3, 5.1))
    arrow(ax, (1.5, 3.6), (1.5, 3.1)); arrow(ax, (4.3, 3.6), (4.3, 3.1))

    ax.text(8.0, 7.55, 'MODEL SELECTION', ha='center', fontsize=10, weight='bold')
    box(ax, 6.3, 3.4, 3.4, 3.8, 'validation injections\n(held-out tiles' + "'" + ' donors + sky)\n\nselect on:\nMAE in noise units\n|median bias| per S/N bin\nseparation < 0.7″ subset\n\nχ² logged, never\nused to select', C['old'])
    arrow(ax, (5.6, 2.25), (7.0, 3.4), 'validation\nscenes', toff=(.25, -.1))

    ax.text(13.0, 7.55, 'FROZEN TESTS (run once, after selection)', ha='center', fontsize=10, weight='bold')
    box(ax, 10.4, 5.9, 5.3, 1.3, '128 known-flux blends\ninjections.py, seed 20261201 (synthetic profiles)', C['test'])
    box(ax, 10.4, 4.3, 5.3, 1.3, 'empirical_injection_pilot_v2 (1000 scenes)\ndonors + sky from the 28 detcat tiles: disjoint', C['test'])
    box(ax, 10.4, 2.7, 5.3, 1.3, '28 real tiles vs MER and Tractor\nNMAD, offset, per band (detcat)', C['test'])
    arrow(ax, (9.7, 5.3), (10.4, 5.3), 'frozen\nckpt', toff=(0, .1))

    ax.text(.2, 1.0, 'Ablations (same generator, same tests)', fontsize=10, weight='bold')
    rows = [('A0', 'existing fixedbg head, χ² only', 'baseline'),
            ('A1', 'same architecture + flux / template losses', 'is the training target the bottleneck?'),
            ('A2', 'A1 + per-band branch', 'does band-specific input help?'),
            ('A3', 'A2 with foundation inputs removed (raw pixels only)', 'any foundation-specific gain?'),
            ('ref', 'mixture q1_mixture_calibrated; Tractor', 'current best / external')]
    for k, (a, b, c) in enumerate(rows):
        y = .62 - k * .2
        ax.text(.3, y, a, fontsize=8.5, weight='bold'); ax.text(1.0, y, b, fontsize=8.5); ax.text(7.0, y, c, fontsize=8.5, style='italic', color='#555555')
    fig.savefig(OUT / 'fig4_protocol.png', dpi=180, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    fig_architecture(); fig_degeneracy(); fig_training(); fig_protocol()
    print('wrote', sorted(p.name for p in OUT.glob('*.png')))
