"""Validate models.grism.sir_geometry against SIR location tables (LOCTAB h5).

For every location object on a detector the LOCTAB stores: FOV reference
positions (mm) per order, the pivot, the local CRV/IDS/PSF coefficients, the
pixel of the reference wavelength and the bounding rectangle (pixels at the
wavelength-range ends).  We recompute all of these from the calibration XMLs
and the frame header and report the residuals.

Usage: python experiments/grism/validate_geometry.py [--dets 23,13] [--dithers 0,1,2,3]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from models.grism.sir_geometry import FrameGeometry, ChebGlobal2D  # noqa: E402

BASE = REPO / "data" / "sir_edff"
CALIB = BASE / "calib" / "dr1_3025"
FRAMES = sorted(glob.glob(str(BASE / "products/3025/frames/EUC_SIR_W-SCIFRM_BKGSUB_*.fits.gz")))


def frame_for_dither(d):
    for f in FRAMES:
        if f"_3025_{13387 + d}_{d}_" in f:
            return f
    raise FileNotFoundError(d)


def loctab_for(dither, det):
    fs = sorted(glob.glob(str(BASE / f"products/3025/loctabs/EUC_SIR_W-LOCTAB_3025-{dither}-DET_{det}_*.h5")))
    return fs[-1]


def stats(name, diff, unit):
    diff = np.asarray(diff, float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        print(f"  {name:38s} no data"); return
    print(f"  {name:38s} n={diff.size:5d}  median|d|={np.median(np.abs(diff)):.2e} {unit}  "
          f"p99|d|={np.percentile(np.abs(diff), 99):.2e}  max|d|={np.max(np.abs(diff)):.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", default="23")
    ap.add_argument("--dithers", default="0,1,2,3")
    ap.add_argument("--max-objects", type=int, default=100000)
    args = ap.parse_args()

    for dither in [int(x) for x in args.dithers.split(",")]:
        frame = frame_for_dither(dither)
        geo = FrameGeometry.from_frame(frame, CALIB)
        print(f"\n=== dither {dither}: {Path(frame).name[:45]}  grism {geo.grism} tilt {geo.tilt}  "
              f"fov scale {geo.opt.wcs.wcs.cd.tolist()}")
        for det in args.dets.split(","):
            lt = h5py.File(loctab_for(dither, det), "r")
            A = dict(lt.attrs)
            # --- pointing / WCS check against the instantiated OPT in the LOCTAB
            cd_lt = np.array(lt["OPT/CD"]); crval_lt = np.array(lt["OPT/CRVAL"])
            print(f"\n-- DET {det}: LOCTAB real pointing {A['Real Pointing']} PA {A['Real PA']:.6f}; "
                  f"CD diff max {np.max(np.abs(cd_lt - geo.opt.wcs.wcs.cd)):.2e}  "
                  f"CRVAL diff {np.max(np.abs(crval_lt - geo.opt.wcs.wcs.crval)):.2e}")
            # global model check: LOCTAB json vs XML
            for grp, mine in (("CRV", geo.crv), ("IDS", geo.ids), ("PSF", geo.psf)):
                for o in (0, 1):
                    js = json.loads(lt[f"{grp}/Order_{o}"].attrs["XML"])
                    key = list(js.keys())[0]
                    m_lt = ChebGlobal2D.from_loctab_json(js[key][str(o)])
                    dmax = max(np.max(np.abs(a - b)) for a, b in zip(m_lt.matrices, mine[o].matrices))
                    print(f"   {grp} order {o}: global matrices max diff {dmax:.1e}; local range {m_lt.local_range}")
            wstart, wend = float(A["Wlen start"]), float(A["Wlen end"])
            geo.set_yshifts(dict(lt["Yshifts"].attrs))
            print(f"   yshift det {det}: {geo.yshift_mm[det] / geo.det.pixel_size:+.3f} px")

            # --- per object
            lo = lt["Location Objects"]
            ids = list(lo.keys())[: args.max_objects]
            ra = np.array([lo[i]["Astronomical Object"].attrs["RA"] for i in ids])
            dec = np.array([lo[i]["Astronomical Object"].attrs["Dec"] for i in ids])
            fov_lt = {0: [], 1: []}; piv_lt = []; refpix_lt = {0: [], 1: []}; bbox_lt = {0: [], 1: []}
            coef_lt = {0: {"CRV": [], "IDS": [], "PSF": []}, 1: {"CRV": [], "IDS": [], "PSF": []}}
            has = {0: [], 1: []}
            for i in ids:
                o = lo[i]
                fov = np.array(o["FOV"])
                fdict = {int(r["Order"]): (r["X pos"], r["Y pos"]) for r in fov}
                for order in (0, 1):
                    fov_lt[order].append(fdict.get(order, (np.nan, np.nan)))
                    g = o.get(f"Order_{order}")
                    has[order].append(g is not None)
                    if g is None:
                        refpix_lt[order].append((np.nan, np.nan)); bbox_lt[order].append(np.full((2, 2), np.nan))
                        for k in coef_lt[order]: coef_lt[order][k].append(None)
                        continue
                    refpix_lt[order].append(np.array(g["Reference Lambda Position"])[0])
                    bbox_lt[order].append(np.array(g["Bounding Rectangle"]))
                    for k in coef_lt[order]: coef_lt[order][k].append(np.array(g[k]["Coeffs"])[0])
                    if order == 1:
                        piv_lt.append(np.array(g["Pivot"])[0])
            piv_lt = np.array(piv_lt) if piv_lt else np.zeros((0, 2))

            # pivot
            piv = geo.opt.pivot_mm(ra, dec)
            if len(piv_lt):
                stats("pivot mm (objects with order 1)", piv[np.array(has[1])] - piv_lt, "mm")
            for order in (0, 1):
                m = np.array(has[order])
                ref = geo.opt.reference_mm(ra, dec, order, apply_tilt=False)
                fl = np.array(fov_lt[order], float)
                stats(f"order {order} FOV ref position (mm)", (ref - fl)[np.isfinite(fl[:, 0])], "mm")
                if m.sum() == 0:
                    continue
                # local coefficients from the global 2D model at the (SIR) FOV position
                for k, model in (("CRV", geo.crv[order]), ("IDS", geo.ids[order]), ("PSF", geo.psf[order])):
                    mine = model.local_coeffs(fl[m])
                    theirs = np.array([c for c in coef_lt[order][k] if c is not None])
                    stats(f"order {order} {k} local coeffs", mine - theirs, "")
                # SIR conventions: 'Reference Lambda Position' is the pixel of the (y-shifted) slit itself;
                # 'Bounding Rectangle' is the [min, max] box of the trace between wstart and wend.
                lams = np.array([wstart, wend])
                ref_mm, pos_mm, _ = geo.trace_mm(ra[m], dec[m], lams, order, slit_mm=fl[m], det_id=det)
                refpix = geo.det.mm_to_pix(ref_mm, det)
                pix = np.array([geo.det.mm_to_pix(p, det) for p in pos_mm])   # (n,2,2)
                rp = np.array(refpix_lt[order], float)[m]
                stats(f"order {order} ref pixel (SIR slit) x", refpix[:, 0] - rp[:, 0], "px")
                stats(f"order {order} ref pixel (SIR slit) y", refpix[:, 1] - rp[:, 1], "px")
                bb = np.array(bbox_lt[order], float)[m]
                stats(f"order {order} bbox min (x,y)", (pix.min(axis=1) - bb[:, 0]).ravel(), "px")
                stats(f"order {order} bbox max (x,y)", (pix.max(axis=1) - bb[:, 1]).ravel(), "px")
                # fully from sky coordinates through our OPT instead of SIR's stored FOV position
                ref_mm2, pos_mm2, _ = geo.trace_mm(ra[m], dec[m], lams, order, det_id=det)
                refpix2 = geo.det.mm_to_pix(ref_mm2, det)
                pix2 = np.array([geo.det.mm_to_pix(p, det) for p in pos_mm2])
                stats(f"order {order} ref pixel from sky (x)", refpix2[:, 0] - rp[:, 0], "px")
                stats(f"order {order} ref pixel from sky (y)", refpix2[:, 1] - rp[:, 1], "px")
                stats(f"order {order} bbox from sky (all)", np.concatenate([(pix2.min(axis=1) - bb[:, 0]).ravel(),
                                                                            (pix2.max(axis=1) - bb[:, 1]).ravel()]), "px")
            lt.close()


if __name__ == "__main__":
    main()
