"""Production-head reference numbers on patch 25, border-excluded.

Patch 25 is the holdout of the patchval25 run. For the PRODUCTION head, patch 25
is ~85% training sky, so these numbers are the leakage-inflated reference; the
new head's numbers on the same selection measure the honest baseline.
Border exclusion: drop patch-25 sources that also appear (within 20 mas, same
band) under any non-patch-25 tile in the archive -- those sky positions are
shared with training patches even for the patchval run.
"""
import numpy as np
from scipy.spatial import cKDTree

ROOT = "/home/shemmati/Work/Projects/JAISP/"
BANDS = ["u", "g", "r", "i", "z", "y", "nisp_Y", "nisp_J", "nisp_H"]
SNR_BINS = [(5, 7), (7, 10), (10, 15), (15, 30), (30, 1e9)]

arch = np.load(ROOT + "models/checkpoints/latent_position_v10_no_psf/anchors_centernet_v10.npz",
               allow_pickle=True)

def mags(a): return np.hypot(a[:, 0], a[:, 1]) * 1000.0

pool = {"all": [], "core": []}   # rows: raw_mag, head_mag, snr
per_band = {}
for b in BANDS:
    tiles = arch[f"{b}_tiles"]
    in25 = np.char.endswith(tiles.astype(str), "_patch_25")
    if not in25.any():
        continue
    ra = arch[f"{b}_ra"].astype(np.float64); dec = arch[f"{b}_dec"].astype(np.float64)
    raw, head, snr = arch[f"{b}_raw"], arch[f"{b}_head_resid"], arch[f"{b}_snr"]
    cosd = np.cos(np.deg2rad(np.median(dec)))
    xy = np.column_stack([ra * cosd, dec]) * 3600.0
    tree_out = cKDTree(xy[~in25])
    d, _ = tree_out.query(xy[in25], k=1, distance_upper_bound=0.020)
    core = np.isinf(d)              # patch-25 sources with no counterpart outside patch 25
    i25 = np.where(in25)[0]
    rm, hm, sn = mags(raw[i25]), mags(head[i25]), snr[i25]
    per_band[b] = dict(n=in25.sum(), n_core=int(core.sum()),
                       raw_med=np.median(rm), head_med=np.median(hm),
                       raw_med_c=np.median(rm[core]) if core.any() else np.nan,
                       head_med_c=np.median(hm[core]) if core.any() else np.nan)
    pool["all"].append(np.column_stack([rm, hm, sn]))
    pool["core"].append(np.column_stack([rm[core], hm[core], sn[core]]))

print("PRODUCTION head on patch-25 tiles (leakage-inflated reference)")
print(f"{'band':7s} {'N':>7s} {'Ncore':>6s} | all: raw/head med | border-excl: raw/head med")
for b, r in per_band.items():
    print(f"{b:7s} {r['n']:7d} {r['n_core']:6d} | {r['raw_med']:6.1f} / {r['head_med']:5.1f} "
          f"| {r['raw_med_c']:6.1f} / {r['head_med_c']:5.1f}")

for key in ("all", "core"):
    a = np.vstack(pool[key])
    print(f"\npooled [{key}] N={len(a)}: raw MAE/med {a[:,0].mean():.1f}/{np.median(a[:,0]):.1f}  "
          f"head MAE/med {a[:,1].mean():.1f}/{np.median(a[:,1]):.1f} mas")
    for lo, hi in SNR_BINS:
        m = (a[:, 2] >= lo) & (a[:, 2] < hi)
        if m.sum() > 30:
            print(f"  SNR {lo:>3}-{'inf' if hi>1e8 else int(hi):<4} N={m.sum():6d}  "
                  f"raw med {np.median(a[m,0]):6.1f}  head med {np.median(a[m,1]):6.1f}")

frac = 1 - len(np.vstack(pool['core'])) / len(np.vstack(pool['all']))
print(f"\nborder-strip fraction excluded: {frac:.1%}")
