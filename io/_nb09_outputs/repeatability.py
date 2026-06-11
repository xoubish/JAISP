"""Label-free astrometric repeatability from cross-tile duplicates.

For the same physical source measured in two overlapping tiles, the difference of
offset vectors is independent of the (shared) true offset: it measures per-tile
measurement noise only. Comparing raw vs head-corrected pair differences tests
whether the head denoises individual measurements -- without using labels as truth.
Pairs are split by train/val tile membership: genuine denoising shows up in
val-val pairs; label memorization shows up as train-only improvement.
"""
import sys, os
import numpy as np
sys.path.insert(0, "/home/shemmati/Work/Projects/JAISP")
from scipy.spatial import cKDTree
from models.astrometry2.dataset import discover_tile_pairs, split_tile_pairs

ROOT = "/home/shemmati/Work/Projects/JAISP/"
BANDS = ["u", "g", "r", "i", "z", "y", "nisp_Y", "nisp_J", "nisp_H"]

pairs = discover_tile_pairs(ROOT + "data/rubin_tiles_all", ROOT + "data/euclid_tiles_all")
_, val_pairs = split_tile_pairs(pairs, 0.15, 42)
def stem(p):
    for el in p:
        b = os.path.basename(str(el))
        if b.startswith("tile_") and b.endswith(".npz") and "euclid" not in b:
            return b[:-4]
val_tiles = {stem(p) for p in val_pairs}

arch = np.load(ROOT + "models/checkpoints/latent_position_v10_no_psf/anchors_centernet_v10.npz",
               allow_pickle=True)

SNR_BINS = [(5, 7), (7, 10), (10, 15), (15, 30), (30, 1e9)]
acc = {}  # (split, bin) -> list of (raw_diff_mag, head_diff_mag)

n_pairs_total = 0
for b in BANDS:
    ra = arch[f"{b}_ra"].astype(np.float64)
    dec = arch[f"{b}_dec"].astype(np.float64)
    raw = arch[f"{b}_raw"].astype(np.float64)
    head = arch[f"{b}_head_resid"].astype(np.float64)
    snr = arch[f"{b}_snr"].astype(np.float64)
    tiles = arch[f"{b}_tiles"]
    cosd = np.cos(np.deg2rad(np.median(dec)))
    xy = np.column_stack([ra * cosd, dec]) * 3600.0
    tree = cKDTree(xy)
    prs = tree.query_pairs(r=0.020, output_type="ndarray")  # 20 mas
    if len(prs) == 0:
        continue
    i, j = prs[:, 0], prs[:, 1]
    diff_tile = tiles[i] != tiles[j]
    i, j = i[diff_tile], j[diff_tile]
    n_pairs_total += len(i)
    iv_i = np.isin(tiles[i], list(val_tiles))
    iv_j = np.isin(tiles[j], list(val_tiles))
    split = np.where(iv_i & iv_j, "val-val", np.where(~iv_i & ~iv_j, "train-train", "mixed"))
    draw = np.hypot(*(raw[i] - raw[j]).T) * 1000.0 / np.sqrt(2)   # per-measurement scatter, mas
    dhead = np.hypot(*(head[i] - head[j]).T) * 1000.0 / np.sqrt(2)
    smin = np.minimum(snr[i], snr[j])
    for sp in ("train-train", "val-val", "mixed"):
        msk = split == sp
        for lo, hi in SNR_BINS:
            sel = msk & (smin >= lo) & (smin < hi)
            if sel.sum():
                acc.setdefault((sp, (lo, hi)), []).append(
                    np.column_stack([draw[sel], dhead[sel]]))

print(f"total cross-tile duplicate pairs: {n_pairs_total}")
print(f"\n{'split':12s} {'SNR bin':>9s} {'N pairs':>8s} {'raw rep.':>9s} {'head rep.':>9s} {'ratio':>6s}")
for sp in ("train-train", "val-val", "mixed"):
    for lo, hi in SNR_BINS:
        key = (sp, (lo, hi))
        if key not in acc:
            continue
        a = np.vstack(acc[key])
        rm, hm = np.median(a[:, 0]), np.median(a[:, 1])
        label = f"{lo}-{'inf' if hi > 1e8 else int(hi)}"
        print(f"{sp:12s} {label:>9s} {len(a):8d} {rm:7.1f}   {hm:7.1f}   {hm/rm:6.2f}")
    print()

# pooled all-SNR per split
print("pooled (all SNR):")
for sp in ("train-train", "val-val", "mixed"):
    arrs = [np.vstack(acc[k]) for k in acc if k[0] == sp]
    if not arrs:
        continue
    a = np.vstack(arrs)
    print(f"  {sp:12s} N={len(a):7d}  raw median {np.median(a[:,0]):6.1f} mas  "
          f"head median {np.median(a[:,1]):6.1f} mas  ratio {np.median(a[:,1])/np.median(a[:,0]):.2f}")
