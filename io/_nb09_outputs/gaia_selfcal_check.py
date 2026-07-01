"""Self-calibrating Gaia DR3 absolute check for the joint-canonical positions.

Gaia DR3 positions are at epoch 2016.0; Rubin (DP1) and Euclid (Q1) were observed
~2024 and neither cleanly publishes the output reference epoch of its positions.
Rather than assume an epoch, we MEASURE it: at a wrong target epoch each matched
star's Gaia residual is linear in its proper motion, so a robust regression of
residual vs PM gives the effective epoch (slope = t_eff - 2016); at the correct
epoch the residual-vs-PM correlation vanishes. We report the frame tie and the
per-star floor at the naive (2016), the old (2025.0), and the self-calibrated epoch.

Positions: joint-canonical (vispeak) = data/astrometry_labels/joint_canonical_790_vispeak.pt
(VIS-pixel xy) projected to sky via each tile's VIS WCS.
"""
import sys
from pathlib import Path
import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree

ROOT = Path("/home/shemmati/Work/Projects/JAISP")
GAIA = ROOT / "data/gaia_ecdfs_astrometry_cache.npz"
LABELS = ROOT / "data/astrometry_labels/joint_canonical_790_vispeak.pt"
EDIR = ROOT / "data/euclid_tiles_all"
MATCH_ARCSEC = 0.3
GAIA_EPOCH = 2016.0

# ---- Gaia DR3 (2016.0) ----
g = np.load(GAIA, allow_pickle=True)
gra, gdec = g["ra"].astype(float), g["dec"].astype(float)
pmra, pmdec = g["pmra"].astype(float), g["pmdec"].astype(float)   # pmra = mu_alpha* (incl cos dec)
gmag = g["phot_g_mean_mag"].astype(float)
ok = np.isfinite(gra) & np.isfinite(gdec) & np.isfinite(pmra) & np.isfinite(pmdec)
gra, gdec, pmra, pmdec, gmag = gra[ok], gdec[ok], pmra[ok], pmdec[ok], gmag[ok]
print(f"Gaia DR3 usable stars: {ok.sum()}")

def gaia_at(epoch):
    dt = epoch - GAIA_EPOCH
    cosd = np.cos(np.deg2rad(gdec))
    ra = gra + (pmra/1000.0/3600.0)*dt / cosd   # mas/yr -> deg; undo cosdec for RA coord
    dec = gdec + (pmdec/1000.0/3600.0)*dt
    return ra, dec

# ---- joint-canonical positions -> sky ----
lab = torch.load(LABELS, map_location="cpu", weights_only=False)
lab = lab["labels"] if "labels" in lab else lab
cra, cdec, csnr = [], [], []
for stem, ent in lab.items():
    xy = np.asarray(ent["xy"], float); snr = np.asarray(ent["snr"], float)
    jok = np.asarray(ent["joint_ok"], bool)
    if jok.sum() == 0: continue
    ep = EDIR / f"{stem}_euclid.npz"
    if not ep.exists(): continue
    hdr = fits.Header.fromstring(np.load(ep, allow_pickle=True)["wcs_VIS"].item())
    w = WCS(hdr)
    r, d = w.all_pix2world(xy[jok,0], xy[jok,1], 0)
    cra.append(r); cdec.append(d); csnr.append(snr[jok])
cra = np.concatenate(cra); cdec = np.concatenate(cdec); csnr = np.concatenate(csnr)
print(f"joint-canonical sources: {len(cra)}")

# ---- match Gaia (near-epoch) to canonical, one-to-one nearest ----
cosd0 = np.cos(np.deg2rad(np.median(cdec)))
ctree = cKDTree(np.column_stack([cra*cosd0, cdec]))
gra0, gdec0 = gaia_at(2024.6)                       # nominal to get within match radius
dist, idx = ctree.query(np.column_stack([gra0*cosd0, gdec0]), k=1)
sel = dist*3600.0 <= MATCH_ARCSEC
mi = idx[sel]
mgra, mgdec = gra[sel], gdec[sel]
mpmra, mpmdec = pmra[sel], pmdec[sel]
mgmag = gmag[sel]
mcra, mcdec = cra[mi], cdec[mi]
cosd = np.cos(np.deg2rad(mcdec))
print(f"matched Gaia-canonical pairs (<= {MATCH_ARCSEC}\"): {sel.sum()}")

def resid(epoch):
    ra_e = mgra + (mpmra/1000.0/3600.0)*(epoch-GAIA_EPOCH)/cosd
    dec_e = mgdec + (mpmdec/1000.0/3600.0)*(epoch-GAIA_EPOCH)
    dra = (mcra - ra_e)*cosd*3600.0*1000.0     # mas
    ddec = (mcdec - dec_e)*3600.0*1000.0
    return dra, ddec

def robust_slope(x, y):
    # slope through origin-ish via robust (Theil-Sen-like) using median of pairwise not needed; use lstsq w/ clip
    A = np.column_stack([x, np.ones_like(x)])
    for _ in range(3):
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        r = y - (m*x+b); s = 1.4826*np.median(np.abs(r-np.median(r)))
        keep = np.abs(r-np.median(r)) < 3*s if s>0 else np.ones_like(r,bool)
        A2, y2, x2 = A[keep], y[keep], x[keep]
        m, b = np.linalg.lstsq(A2, y2, rcond=None)[0]
    return m, b

# ---- self-calibrate epoch from residual-vs-PM slope at the Gaia (2016) reference ----
dra16, ddec16 = resid(GAIA_EPOCH)
sl_ra, _ = robust_slope(mpmra, dra16)     # dra16 = pmra*(t_eff-2016) + bulk  -> slope = (t_eff-2016) yr per (mas/yr)
sl_de, _ = robust_slope(mpmdec, ddec16)
t_eff = GAIA_EPOCH + 0.5*(sl_ra+sl_de)
print(f"\nresidual-vs-PM slope (yr):  RA {sl_ra:+.2f}  Dec {sl_de:+.2f}  ->  effective epoch = {t_eff:.2f}")

def report(epoch, tag):
    dra, ddec = resid(epoch)
    bra, bde = np.median(dra), np.median(ddec)
    # PM-decorrelation check after this epoch's propagation
    s_ra,_ = robust_slope(mpmra, dra); s_de,_ = robust_slope(mpmdec, ddec)
    # per-star floor after bulk removal, bright
    rad = np.hypot(dra-bra, ddec-bde)
    bright = mgmag < 18.5
    floor_b = np.median(rad[bright]); floor_all = np.median(rad)
    print(f"[{tag:9s} @ {epoch:.2f}]  tie=({bra:+.1f},{bde:+.1f}) mas | "
          f"floor G<18.5={floor_b:.1f} all={floor_all:.1f} mas | "
          f"resid-PM slope RA={s_ra:+.2f} Dec={s_de:+.2f}")
    return dict(epoch=epoch, tie=(bra,bde), floor_bright=floor_b, floor_all=floor_all,
                slope=(s_ra,s_de), n=len(dra), nbright=int(bright.sum()))

print()
r_naive = report(GAIA_EPOCH, "naive2016")
r_old   = report(2025.0, "old2025.0")
r_self  = report(t_eff, "selfcal")

print("\n==> The self-cal epoch is where the residual no longer tracks PM (slopes ~0).")
print("    Compare the frame tie and floor across the three rows above.")
import json
json.dump({"t_eff":float(t_eff),"slope_ra":float(sl_ra),"slope_dec":float(sl_de),
           "naive":{k:(list(v) if isinstance(v,tuple) else v) for k,v in r_naive.items()},
           "old":{k:(list(v) if isinstance(v,tuple) else v) for k,v in r_old.items()},
           "selfcal":{k:(list(v) if isinstance(v,tuple) else v) for k,v in r_self.items()}},
          open(ROOT/"io/_nb09_outputs/gaia_selfcal_results.json","w"), indent=2)
print("saved io/_nb09_outputs/gaia_selfcal_results.json")
