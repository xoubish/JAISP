"""Q1 native pixels + cached frozen v11 features. No catalog flux labels."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import minimize_scalar
from .core import BANDS, templates, fit_flux


def wcs(value):
    value = value.item() if isinstance(value, np.ndarray) else value
    return WCS(fits.Header(value) if isinstance(value, dict) else fits.Header.fromstring(str(value)))


def load_tile(pair, labels):
    name, rp, ep = pair
    bands = {}
    with np.load(rp, allow_pickle=True) as r, np.load(ep, allow_pickle=True) as e:
        for i, b in enumerate(BANDS):
            if b.startswith('rubin'):
                assert str(r['bands'][i]) == b[-1]
                im, var, wc = r['img'][i], r['var'][i].copy(), wcs(r['wcs_hdr'])
                # Verified bit assignments in io/ingest_tiles.py. DETECTED stays valid.
                bad = (r['mask'][i] & (1 | 2 | 256)) != 0
                var[bad] = np.nan
            else:
                short = b.split('_')[1]
                im, var, wc = e['img_'+short], e['var_'+short], wcs(e['wcs_'+short])
            bands[b] = (im.astype('float32'), var.astype('float32'), wc)
    h, width = bands['euclid_VIS'][0].shape
    xy = np.asarray(labels[name][0]) * [width-1, h-1]
    sky = np.column_stack(bands['euclid_VIS'][2].pixel_to_world_values(*xy.T))
    return dict(name=name, bands=bands, xy=xy, sky=sky)


def jacobian(wc, xy):
    """Local native pixel -> east/north tangent arcseconds, including rotation."""
    ra, dec = wc.pixel_to_world_values(*xy)
    matrix = np.empty((2, 2))
    for k in range(2):
        delta = np.eye(2)[k] * .1
        r1, d1 = wc.pixel_to_world_values(*(xy+delta))
        matrix[:, k] = [((r1-ra+180)%360-180)*np.cos(np.deg2rad(dec))*36000, (d1-dec)*36000]
    return matrix


def cut(im, var, xy, radius):
    origin = np.rint(xy).astype(int)-radius
    x, y = origin
    n = 2*radius+1
    if x < 0 or y < 0 or x+n > im.shape[1] or y+n > im.shape[0]:
        return None
    a, v = im[y:y+n,x:x+n].copy(), var[y:y+n,x:x+n].copy()
    mask = np.isfinite(a) & np.isfinite(v) & (v > 0)
    if mask.mean() < .95:
        return None
    return a, v, mask, origin


def calibrate_psf(tiles):
    """Approximate global circular PSF from isolated compact training detections.

    A lower-width VIS quartile reduces galaxy contamination but does not certify
    stars. Width scatter and fit chi² are saved; this is a pilot calibration.
    """
    initial = dict(zip(BANDS, [2.65,2.,1.95,1.9,1.925,1.975,.85,1.9,1.95,2.]))
    candidates = []
    def measure(tile, idx, band):
        im, var, wc = tile['bands'][band]
        xy = np.array(wc.world_to_pixel_values(*tile['sky'][idx]))
        stamp = cut(im,var,xy,10)
        if stamp is None:
            return None
        a,v,mask,origin=stamp
        position=torch.tensor((xy-origin)[None],dtype=torch.float32)
        def solve(sigma):
            t=templates(position,torch.zeros(1,2,2),torch.eye(2),sigma,a.shape)
            return fit_flux(torch.tensor(a),torch.tensor(v),t,torch.tensor(mask))
        result=minimize_scalar(lambda s:float(solve(s)['loss']),bounds=(.55,initial[band]*2),method='bounded',options={'xatol':.015})
        fit=solve(result.x)
        if float(fit['flux'][0]/fit['error'][0]) < 15 or result.x > initial[band]*1.95:
            return None
        return dict(sigma=float(result.x),reduced_chi2=float(fit['chi2']/fit['dof']))
    for tile in tiles:
        xy=tile['xy']
        distance=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
        np.fill_diagonal(distance,np.inf)
        for idx in np.flatnonzero(distance.min(1)>45)[:80]:
            fit=measure(tile,idx,'euclid_VIS')
            if fit is not None:
                candidates.append((fit['sigma'],tile,idx))
    if len(candidates)<8:
        raise ValueError('Insufficient isolated PSF calibrators; need at least 8')
    threshold=np.quantile([x[0] for x in candidates],.25)
    selected=[x for x in candidates if x[0]<=threshold]
    result={}
    for b in BANDS:
        measurements=[m for _,t,i in selected if (m:=measure(t,i,b)) is not None]
        if len(measurements)<3:
            raise ValueError(f'{b}: fewer than 3 compact PSF calibrators')
        values=[m['sigma'] for m in measurements]
        result[b]=dict(sigma_px=float(np.median(values)),n=len(values),
                       p16_p84_px=np.percentile(values,[16,84]).tolist(),
                       median_reduced_chi2=float(np.median([m['reduced_chi2'] for m in measurements])))
    return result


def initial_covariance(tile, indices, psf):
    """Positive VIS second moments, subtracting the fixed PSF covariance."""
    im,var,wc=tile['bands']['euclid_VIS']
    result=[]
    for idx in indices:
        xy=tile['xy'][idx]
        stamp=cut(im,var,xy,10)
        if stamp is None:
            result.append(np.eye(2)*.15**2)
            continue
        a,v,mask,origin=stamp
        yy,xx=np.indices(a.shape)
        d=np.stack((xx+origin[0]-xy[0],yy+origin[1]-xy[1]),-1)
        ann=np.linalg.norm(d,axis=-1)>8
        bg=np.median(a[ann & mask]) if np.any(ann & mask) else 0
        signal=np.where(mask,np.maximum(a-bg,0),0)*np.exp(-np.sum(d*d,axis=-1)/32)
        moment=np.einsum('hw,hwi,hwj->ij',signal,d,d)/max(signal.sum(),1e-12)
        moment-=np.eye(2)*psf['euclid_VIS']['sigma_px']**2
        p2s=jacobian(wc,xy)
        vals,vecs=np.linalg.eigh(p2s@moment@p2s.T)
        result.append((vecs*np.clip(vals,.025**2,.45**2))@vecs.T)
    return torch.tensor(np.asarray(result),dtype=torch.float32)


def make_scenes(tile, psf, cache_dir, max_scenes=6, occupied=None, ra_bounds=(-np.inf, np.inf)):
    """Fixed 12-arcsec native scenes; neighbors within 14 arcsec in sky.

    The neighbor radius covers rotated corners and a generous Gaussian-wing margin.
    This limits wing contamination, not undetected-source error.
    Scenes with >30 detections are skipped, never truncated. Disjoint footprints
    within and across overlapping tiles are enforced in tangent sky coordinates.
    """
    occupied=[] if occupied is None else occupied
    name=tile['name']
    cache=torch.load(Path(cache_dir)/(name+'_aug0.pt'),map_location='cpu',weights_only=False)
    if cache['tile_id'] != name or tuple(cache.get('aug_params', ())) != (0, False, False):
        raise ValueError('Mismatched feature cache')
    feature=cache['features'].float()
    if feature.ndim==3: feature=feature[None]
    h,w=tile['bands']['euclid_VIS'][0].shape
    scenes=[]
    for central in np.random.default_rng(42).permutation(len(tile['xy'])):
        ra,dec=tile['sky'][central]
        margin=14/(3600*np.cos(np.deg2rad(dec)))
        if not (ra-margin>ra_bounds[0] and ra+margin<ra_bounds[1]): continue
        if any(np.hypot((ra-r)*np.cos(np.deg2rad(dec)),dec-d)*3600<28 for r,d in occupied):
            continue
        offset=(tile['sky']-[ra,dec])*[np.cos(np.deg2rad(dec))*3600,3600]
        indices=np.flatnonzero(np.linalg.norm(offset,axis=1)<14)
        if len(indices)>30: continue
        data={}
        for b in BANDS:
            im,var,wc=tile['bands'][b]
            center=np.array(wc.world_to_pixel_values(ra,dec))
            p2s=jacobian(wc,center)
            scale=np.sqrt(abs(np.linalg.det(p2s)))
            # Require the entire neighbor search area to lie inside the tile.
            edge=np.min(np.r_[center, np.array([im.shape[1]-1,im.shape[0]-1])-center])*scale
            if edge<14:break
            stamp=cut(im,var,center,int(np.ceil(6/scale)))
            if stamp is None: break
            a,v,mask,origin=stamp
            positions=np.column_stack(wc.world_to_pixel_values(*tile['sky'][indices].T))-origin
            # Exclude templates that have essentially no footprint on this grid.
            data[b]=dict(image=torch.tensor(a),variance=torch.tensor(v),mask=torch.tensor(mask),
                         positions=torch.tensor(positions,dtype=torch.float32),
                         sky_to_pixel=torch.tensor(np.linalg.inv(p2s),dtype=torch.float32),
                         psf_sigma=psf[b]['sigma_px'])
        if len(data)!=10: continue
        # Same coordinate convention as astrometry2.vis_px_to_bottleneck_px.
        pos=tile['xy'][indices]*[feature.shape[-1]/w,feature.shape[-2]/h]
        off=np.stack(np.meshgrid(np.arange(-1,2),np.arange(-1,2)),axis=-1)
        grid=pos[:,None,None,:]+off
        grid=2*grid/[feature.shape[-1]-1,feature.shape[-2]-1]-1
        local=F.grid_sample(feature.expand(len(indices),-1,-1,-1),torch.tensor(grid,dtype=torch.float32),align_corners=True)
        scenes.append(dict(tile=name,central=int(np.flatnonzero(indices==central)[0]),
                           sky=tile['sky'][indices],features=local.detach(),
                           covariance=initial_covariance(tile,indices,psf),bands=data))
        occupied.append((ra,dec))
        if len(scenes)>=max_scenes: break
    return scenes
