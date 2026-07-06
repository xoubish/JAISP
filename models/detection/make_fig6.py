"""Fig 6 (detection performance), DR1-style layout, from the Q1 metrics — no GPU.
Production detector = cn_vis_sep (CenterNet + SEP labels).
 (a) wide top : completeness vs VIS mag -- injection all-10 / VIS-only / NISP-only
               + MER-catalogue recovery (dashed) + MER VIS-mag distribution (grey hist).
 (b) bot-left : the same MER recovery binned by MEASURED VIS aperture S/N (zeropoint-free).
 (c) bot-right: MER completeness (VIS<24.5) & purity vs detection threshold, working point 0.30.
(The catalogue-free "extras are real multi-band" check lives in legB_extras_metrics.json.)
"""
import json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
D = REPO / 'checkpoints/q1_detection'
inj = json.load(open(D / 'injection_metrics.json'))['cn_vis_sep']
rec = json.load(open(D / 'mer_recovery.json'))
bak = json.load(open(D / 'bakeoff_metrics.json'))['cn_vis_sep']['rows']
CA, CV, CN, CP, INK = '#1f77b4', '#ff7f0e', '#d62728', '#d62728', '#1c2733'

fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.26, wspace=0.2)
a = fig.add_subplot(gs[0, :]); b = fig.add_subplot(gs[1, 0]); c = fig.add_subplot(gs[1, 1])

# ---- (a) completeness vs VIS magnitude ----
medges = np.array(rec['medges']); hc = (medges[:-1] + medges[1:]) / 2
at = a.twinx()
at.bar(hc, rec['mer_hist'], width=0.38, color='0.86', zorder=0, label='MER VIS-mag distribution')
at.set_yticks([])   # histogram is illustrative context; no right-axis scale needed
a.set_zorder(at.get_zorder() + 1); a.patch.set_visible(False)
mag = np.array(inj['curves']['all']['mag']); cu = {k: np.array(inj['curves'][k]['comp']) for k in ('all', 'vis', 'nisp')}
a.fill_between(mag, cu['vis'], cu['all'], color=CA, alpha=0.15, zorder=2)
a.plot(mag, cu['all'], '-o', color=CA, lw=2.6, ms=6, label='injection: all 10 bands', zorder=5)
a.plot(mag, cu['vis'], '-s', color=CV, lw=2.2, ms=5, label='injection: VIS only', zorder=5)
a.plot(mag, cu['nisp'], '-^', color=CN, lw=2.2, ms=6, label='injection: NISP only (Y/J/H)', zorder=5)
a.plot(rec['rec_mag'], rec['rec_comp'], '--D', color='0.35', lw=1.8, ms=6, mfc='white', zorder=4,
       label='MER catalogue recovery')
a.axhline(50, color='0.6', lw=0.9, ls=':')
a.set_xlim(22, 26.6); a.set_ylim(0, 103)
a.set_xlabel('VIS magnitude', fontsize=12); a.set_ylabel('completeness [%]', fontsize=12)
a.tick_params(labelsize=10)
h1, l1 = a.get_legend_handles_labels(); h2, l2 = at.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, fontsize=9.5, loc='lower left', bbox_to_anchor=(0.0, 1.01),
         ncol=3, frameon=False, columnspacing=1.4, handletextpad=0.4)

# ---- (b) completeness vs measured VIS aperture S/N ----
b.plot(rec['snr_x'], rec['snr_comp'], '-s', color=CA, lw=2.2, ms=6)
b.axhline(50, color='0.6', lw=0.9, ls=':'); b.set_xscale('log'); b.set_ylim(0, 103)
b.set_xlabel('measured VIS aperture S/N', fontsize=12); b.set_ylabel('completeness [%]', fontsize=12)
b.tick_params(labelsize=10)

# ---- (c) completeness & purity vs threshold ----
keys = sorted(bak, key=float); confs = [float(k) for k in keys]
comp = [bak[k]['completeness'] for k in keys]; pur = [bak[k]['purity'] for k in keys]
c.plot(confs, comp, '-o', color=CA, lw=2.2, ms=6, label='completeness (VIS<24.5)')
c.plot(confs, pur, '-^', color=CP, lw=2.2, ms=6, label='purity (vs MER, floor)')
c.axvline(0.30, color='k', lw=1.1, ls='--')
c.text(0.30, 104, 'working point 0.30', fontsize=10, ha='center', color='k')
c.set_ylim(0, 103); c.set_xlabel('detection threshold', fontsize=12); c.set_ylabel('percent [%]', fontsize=12)
c.tick_params(labelsize=10); c.legend(fontsize=10, loc='lower center', frameon=True, framealpha=0.9)

for x in (a, b, c):
    for s in ('top', 'right'): x.spines[s].set_visible(False)
for s in ('top', 'right'): at.spines[s].set_visible(False)
out = REPO / 'paper/figures/fig6_detection_performance.png'
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print('saved', out)
