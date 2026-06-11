"""Side-by-side: production head (leaky split) vs patchval25 head (spatially disjoint)
on the SAME patch-25 sources, border-excluded, per band and per SNR bin.

Production anchors come from the all-790 archive filtered to patch-25 tiles;
patchval anchors from the patch-25-only eval export of the disjoint head.
Border exclusion: drop sources with a counterpart (<20 mas, same band) under any
non-patch-25 tile in the production archive. The production-vs-patchval gap on
identical sources IS the leakage inflation of the published numbers.
"""
import numpy as np
from scipy.spatial import cKDTree

ROOT = "/home/shemmati/Work/Projects/JAISP/"
BANDS = ["u", "g", "r", "i", "z", "y", "nisp_Y", "nisp_J", "nisp_H"]
SNR_BINS = [(5, 7), (7, 10), (10, 15), (15, 30), (30, 1e9)]

prod = np.load(ROOT + "models/checkpoints/latent_position_v10_no_psf/anchors_centernet_v10.npz",
               allow_pickle=True)
pv = np.load(ROOT + "models/checkpoints/latent_position_v10_patchval25/anchors_patch25.npz",
             allow_pickle=True)

def mags(a):
    return np.hypot(a[:, 0], a[:, 1]) * 1000.0

rows = {"prod": [], "pv": []}
print(f"{'band':7s} {'Nmatch':>7s} | prod head med | patchval head med | raw med (check)")
for b in BANDS:
    tiles_p = prod[f"{b}_tiles"]
    in25 = np.char.endswith(tiles_p.astype(str), "_patch_25")
    if not in25.any() or f"{b}_ra" not in pv.files:
        continue
    ra_p = prod[f"{b}_ra"].astype(np.float64); dec_p = prod[f"{b}_dec"].astype(np.float64)
    cosd = np.cos(np.deg2rad(np.median(dec_p)))
    xy_p = np.column_stack([ra_p * cosd, dec_p]) * 3600.0
    # border exclusion from the production archive geometry
    d_out, _ = cKDTree(xy_p[~in25]).query(xy_p[in25], k=1, distance_upper_bound=0.020)
    core = np.isinf(d_out)
    idx25 = np.where(in25)[0][core]

    ra_v = pv[f"{b}_ra"].astype(np.float64); dec_v = pv[f"{b}_dec"].astype(np.float64)
    xy_v = np.column_stack([ra_v * cosd, dec_v]) * 3600.0
    d, j = cKDTree(xy_v).query(xy_p[idx25], k=1, distance_upper_bound=0.010)
    ok = np.isfinite(d)
    m_p = idx25[ok]; m_v = j[ok]

    raw_p, head_p, snr_p = prod[f"{b}_raw"][m_p], prod[f"{b}_head_resid"][m_p], prod[f"{b}_snr"][m_p]
    raw_v, head_v = pv[f"{b}_raw"][m_v], pv[f"{b}_head_resid"][m_v]
    rows["prod"].append(np.column_stack([mags(raw_p), mags(head_p), snr_p]))
    rows["pv"].append(np.column_stack([mags(raw_v), mags(head_v), snr_p]))
    print(f"{b:7s} {ok.sum():7d} | {np.median(mags(head_p)):8.1f}      | {np.median(mags(head_v)):8.1f}          "
          f"| {np.median(mags(raw_p)):6.1f} vs {np.median(mags(raw_v)):6.1f}")

P, V = np.vstack(rows["prod"]), np.vstack(rows["pv"])
print(f"\npooled matched border-excluded N={len(P)}")
print(f"  production head: MAE {P[:,1].mean():5.1f}  median {np.median(P[:,1]):5.1f} mas")
print(f"  patchval  head: MAE {V[:,1].mean():5.1f}  median {np.median(V[:,1]):5.1f} mas")
print(f"\n{'SNR bin':>9s} {'N':>7s} | raw med | prod head med | patchval head med")
for lo, hi in SNR_BINS:
    m = (P[:, 2] >= lo) & (P[:, 2] < hi)
    if m.sum() > 30:
        lab = f"{lo}-{'inf' if hi > 1e8 else int(hi)}"
        print(f"{lab:>9s} {m.sum():7d} | {np.median(P[m,0]):6.1f}  | {np.median(P[m,1]):8.1f}      | {np.median(V[m,1]):8.1f}")
