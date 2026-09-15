"""Add image-derived NISP labels to the existing VIS labels, in an isolated run.

The MER catalogue is never used to construct labels. NISP Y/J/H are combined
with inverse-variance weights, smoothed, and extracted with SEP. The noise of
the smoothed image is estimated empirically to account for resampling
correlations. Existing VIS centroids are preserved exactly. Patch 25 is excluded
from the output, including all its augmentations.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
from functools import lru_cache

import numpy as np
import sep
import torch
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'models'))
from detection.masks import bright_star_saturation_mask, load_gaia_cache


@lru_cache(maxsize=1)
def _gaia():
    return load_gaia_cache(str(ROOT/'data/gaia_ecdfs_astrometry_cache.npz'))


def nir_labels(path, vis_xy, threshold=3.0, sigma=2.1, merge_px=5.0):
    """Return added normalized centroids and extraction diagnostics."""
    with np.load(path, allow_pickle=True) as ed:
        shape = ed['img_VIS'].shape
        artifact = bright_star_saturation_mask(ed['img_VIS'], str(ed['wcs_VIS']), _gaia())
        vw = WCS(fits.Header.fromstring(str(ed['wcs_VIS'])))
        yy, xx = np.indices((3, 3), dtype=float)
        xx *= (shape[1] - 1) / 2
        yy *= (shape[0] - 1) / 2
        ra, dec = vw.all_pix2world(xx.ravel(), yy.ravel(), 0)
        numerator = np.zeros(shape, np.float64)
        weight = np.zeros(shape, np.float64)
        coverage = np.ones(shape, bool)
        for band in ('Y', 'J', 'H'):
            nw = WCS(fits.Header.fromstring(str(ed[f'wcs_{band}'])))
            nx, ny = nw.all_world2pix(ra, dec, 0)
            if max(np.max(abs(nx-xx.ravel())), np.max(abs(ny-yy.ravel()))) > 0.01:
                raise ValueError(f'{path}: NISP and VIS grids do not coincide')
            im = np.asarray(ed[f'img_{band}'], np.float32)
            var = np.asarray(ed[f'var_{band}'], np.float32)
            if im.shape != shape or var.shape != shape:
                raise ValueError(f'{path}: inconsistent image dimensions')
            valid = np.isfinite(im) & np.isfinite(var) & (var > 0) & (var < 1e20)
            coverage &= valid
            clean = np.ascontiguousarray(np.where(valid, im, 0), dtype=np.float32)
            background = sep.Background(clean, mask=~valid)
            ivar = np.zeros(shape, np.float64)
            np.divide(1.0, var, out=ivar, where=valid)
            numerator += (clean-background.back()) * ivar
            weight += ivar
    z = np.divide(numerator, np.sqrt(weight), out=np.zeros(shape), where=weight > 0)
    smooth = np.ascontiguousarray(gaussian_filter(z, sigma=sigma), dtype=np.float32)
    valid = binary_erosion(coverage, iterations=int(np.ceil(4*sigma)))
    bkg = sep.Background(smooth, mask=~valid, bw=64, bh=64, fw=3, fh=3)
    err = np.ascontiguousarray(bkg.rms())
    if not np.isfinite(bkg.globalrms) or bkg.globalrms <= 0:
        raise ValueError(f'{path}: no measurable NISP noise')
    err = np.maximum(err, 0.25 * bkg.globalrms)
    catalog = sep.extract(np.ascontiguousarray(smooth-bkg.back()), threshold,
                          err=err, mask=~valid, minarea=10,
                          filter_kernel=None, deblend_nthresh=32,
                          deblend_cont=0.005, clean=True)
    xy = np.column_stack((catalog['x'], catalog['y']))
    inside = ((xy[:, 0] >= 10) & (xy[:, 0] < shape[1]-10)
              & (xy[:, 1] >= 10) & (xy[:, 1] < shape[0]-10))
    ix = np.clip(np.rint(xy[:, 0]).astype(int), 0, shape[1]-1)
    iy = np.clip(np.rint(xy[:, 1]).astype(int), 0, shape[0]-1)
    inside &= ~artifact[iy, ix]
    xy = xy[inside]
    flux = catalog['flux'][inside]
    # Merge with VIS before adding anything; retain the existing VIS coordinate.
    scale = np.array([shape[1]-1, shape[0]-1])
    if len(vis_xy) and len(xy):
        novel = cKDTree(np.asarray(vis_xy)*scale).query(xy)[0] > merge_px
        xy, flux = xy[novel], flux[novel]
    # Suppress NISP duplicates deterministically, brightest first.
    selected = []
    for i in np.argsort(-flux, kind='stable'):
        if not selected or np.min(np.linalg.norm(xy[selected]-xy[i], axis=1)) > merge_px:
            selected.append(int(i))
    added = (xy[selected]/scale).astype(np.float32)
    return added, dict(n_vis=len(vis_xy), n_nir_extracted=len(catalog),
                      n_nir_added=len(added), smoothed_noise=float(bkg.globalrms),
                      valid_fraction=float(valid.mean()))


def _one(job):
    tid, path, vis, threshold = job
    added, diag = nir_labels(path, vis, threshold=threshold)
    return tid, added, diag


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=3.0)
    p.add_argument('--limit', type=int, default=0, help='Training-only smoke test')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    labels = torch.load(ROOT/'data/cached_features_v11_q1/pseudo_labels_vis_sep.pt',
                        map_location='cpu', weights_only=False)['labels']
    train_ids = sorted(t for t in labels if not t.endswith('_patch_25'))
    if args.limit:
        train_ids = train_ids[::max(1, len(train_ids)//args.limit)][:args.limit]
    jobs = [(tid, ROOT/'data/euclid_tiles_all_q1'/f'{tid}_euclid.npz',
             labels[tid][0], args.threshold) for tid in train_ids]
    promoted, diagnostics = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (tid, added, diag) in enumerate(pool.map(_one, jobs)):
            promoted[tid], diagnostics[tid] = added, diag
            if (i+1) % 20 == 0 or i+1 == len(jobs):
                print(f'{i+1}/{len(jobs)} tiles; added {sum(len(x) for x in promoted.values())}', flush=True)
    config = dict(threshold=args.threshold, smooth_sigma_px=2.1, merge_px=5.0,
                  minarea=10, deblend_nthresh=32, deblend_cont=0.005,
                  excluded_patches=[25], catalog_supervision=False,
                  nir_artifact_mask='existing Gaia and VIS saturation mask',
                  stack='sum(background_subtracted_image/variance)/sqrt(sum(1/variance))',
                  error='SEP local RMS after smoothing', tiles=train_ids)
    torch.save(dict(promoted=promoted, demoted={}, config=config), args.out/'nir_extra_labels.pt')
    (args.out/'label_diagnostics.json').write_text(json.dumps(dict(config=config, tiles=diagnostics), indent=2)+'\n')
    print('Saved', args.out, flush=True)


if __name__ == '__main__':
    main()
