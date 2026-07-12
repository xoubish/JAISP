# Talk-only diagram: the two detection-head variants and where they read
# the frozen foundation. Style follows paper fig2 (matplotlib boxes).
# Output: fig_detection_heads.png (used on the Detection slide).
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NAVY   = (20/255, 60/255, 110/255)
STEEL  = '#3d5a80'
GOLD   = '#c9930f'
EBLUE  = '#2196c9'
GREEN  = '#2e8b57'
ORANGE = '#be4619'
GRAY   = '#666666'
LIGHT  = '#eef1f5'

fig, ax = plt.subplots(figsize=(8.6, 7.6), dpi=300)
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

def box(x, y, w, h, fc, ec, text, tc='white', fs=11.5, lw=1.6, ls='-', weight='bold'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06,rounding_size=0.12',
                                fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=3))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', color=tc,
            fontsize=fs, fontweight=weight, zorder=4)

def arrow(x0, y0, x1, y1, color, lw=2.4, style='-|>', ls='-'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=16, color=color, lw=lw,
                                 linestyle=ls, zorder=2))

# ---------------- frozen encoder stack (left) ----------------
SX, SW = 0.55, 3.9
# input strip: Rubin + Euclid halves
ax.add_patch(FancyBboxPatch((SX, 8.55), SW/2 - 0.05, 0.95,
             boxstyle='round,pad=0.04,rounding_size=0.1', fc=GOLD, ec='none', zorder=3))
ax.add_patch(FancyBboxPatch((SX + SW/2 + 0.05, 8.55), SW/2 - 0.05, 0.95,
             boxstyle='round,pad=0.04,rounding_size=0.1', fc=EBLUE, ec='none', zorder=3))
ax.text(SX + SW/4, 9.03, 'Rubin\n$ugrizy$ · 0.2$^{\\prime\\prime}$/px', ha='center', va='center',
        color='white', fontsize=10, fontweight='bold', zorder=4)
ax.text(SX + 3*SW/4, 9.03, 'Euclid\nVIS $YJH$ · 0.1$^{\\prime\\prime}$/px', ha='center', va='center',
        color='white', fontsize=10, fontweight='bold', zorder=4)
ax.text(SX + SW/2, 9.78, '10-band input', ha='center', va='center',
        color=NAVY, fontsize=12, fontweight='bold')

arrow(SX + SW/2, 8.50, SX + SW/2, 7.95, NAVY)
box(SX, 6.95, SW, 0.95, LIGHT, STEEL,
    'per-band conv stems\nnative-resolution features', tc=NAVY, fs=10.5)
arrow(SX + SW/2, 6.90, SX + SW/2, 6.35, NAVY)
box(SX, 5.35, SW, 0.95, LIGHT, STEEL,
    'two-stream ConvNeXt\n+ transformer', tc=NAVY, fs=10.5)
arrow(SX + SW/2, 5.30, SX + SW/2, 4.75, NAVY)
box(SX, 3.75, SW, 0.95, NAVY, NAVY,
    'fused 10-band bottleneck\n0.4$^{\\prime\\prime}$/px · 256-d', fs=10.5)

# frozen bracket
ax.add_patch(FancyBboxPatch((SX - 0.32, 3.45), SW + 0.64, 4.85,
             boxstyle='round,pad=0.05,rounding_size=0.18', fc='none', ec=STEEL,
             lw=1.4, linestyle=(0, (4, 3)), zorder=1))
ax.text(SX - 0.32, 3.18, '❄ frozen foundation', color=STEEL, fontsize=11,
        fontweight='bold', ha='left', va='top')

# ---------------- two heads (right) ----------------
HX, HW = 6.15, 3.3
# StemCenterNet reads the stems
box(HX, 6.95, HW, 0.95, 'white', ORANGE, 'StemCenterNet\nreads native per-band stems',
    tc=ORANGE, fs=10.5, ls=(0, (4, 2)))
arrow(SX + SW + 0.34, 7.42, HX - 0.08, 7.42, ORANGE)
# CenterNet reads the fused bottleneck
box(HX, 3.75, HW, 0.95, GREEN, GREEN, 'CenterNet  (production)\nreads fused 10-band features', fs=10.5)
arrow(SX + SW + 0.34, 4.22, HX - 0.08, 4.22, GREEN)

# gains
ax.text(HX + HW/2, 6.60, 'multi-band gain:  +0.03 mag',
        ha='center', va='center', color=GRAY, fontsize=9.5, style='italic')
ax.text(HX + HW/2, 3.40, 'multi-band gain:  +0.45 mag',
        ha='center', va='center', color=GREEN, fontsize=9.5, fontweight='bold', style='italic')

# ---------------- shared output (bottom right) ----------------
# StemCN routes around the right edge; CenterNet drops straight down
ax.add_patch(FancyArrowPatch((HX + HW + 0.12, 7.42), (8.55, 2.30),
             connectionstyle='arc3,rad=-0.32', arrowstyle='-|>', mutation_scale=16,
             color=ORANGE, lw=2.4, linestyle=(0, (4, 2)), zorder=2))
arrow(HX + HW/2, 3.12, HX + HW/2, 2.30, GREEN)

# evidence-map inset
rng = np.random.default_rng(3)
n = 90
img = np.zeros((n, n))
yy, xx = np.mgrid[0:n, 0:n]
peaks = [(22, 25, 2.2, 1.0), (60, 68, 2.0, 0.85), (44, 46, 2.6, 0.65),
         (74, 20, 1.8, 0.5), (18, 70, 2.0, 0.4), (65, 40, 1.7, 0.3)]
for py, px, s, a in peaks:
    img += a * np.exp(-((yy - py)**2 + (xx - px)**2) / (2 * s**2))
img += 0.035 * rng.standard_normal((n, n))
ext = [4.85, 6.35, 0.55, 2.05]
ax.imshow(img, extent=ext, origin='lower', cmap='magma', vmin=0, vmax=1.0, zorder=3)
for py, px, s, a in peaks[:4]:
    ax.plot(ext[0] + (px/n)*(ext[1]-ext[0]), ext[2] + (py/n)*(ext[3]-ext[2]),
            '+', color='white', ms=9, mew=1.6, zorder=5)
ax.add_patch(plt.Rectangle((ext[0], ext[2]), ext[1]-ext[0], ext[3]-ext[2],
             fc='none', ec=NAVY, lw=1.4, zorder=4))

ax.text(6.65, 1.85, 'same head, same output:', color=NAVY, fontsize=11, fontweight='bold',
        ha='left', va='center')
ax.text(6.65, 1.28, 'source-evidence map\n+ sub-pixel offset\n$\\rightarrow$ peak-finding = detections',
        color=NAVY, fontsize=10.5, ha='left', va='center')
ax.text(5.60, 0.30, 'trained on classical 3$\\sigma$ VIS labels', color=GRAY,
        fontsize=9.5, style='italic', ha='center')

plt.tight_layout(pad=0.3)
plt.savefig('fig_detection_heads.png', bbox_inches='tight', facecolor='white')
print('saved fig_detection_heads.png')
