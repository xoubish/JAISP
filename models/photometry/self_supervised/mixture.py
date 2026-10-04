"""Positive multiscale profiles with image-likelihood refinement and learned priors.

No catalog labels. The prior constrains morphology, never the total amplitude.
Signed total fluxes are measured again after selecting a normalized profile.
"""
import numpy as np
import torch
from scipy.optimize import nnls
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from .core import BANDS, templates, fit_flux

SCALES = np.array([.025, .06, .12, .22, .38, .65, 1.1])


def ellipse_from_pixels(scene):
    """Deweight VIS second moments before PSF subtraction; use circular faint prior.

    For a Gaussian weighted by exp(-r²/(2w²)), C = (M^-1 - I/w²)^-1.
    This is an initializer for ellipticity only: sizes are fitted with the bank.
    """
    d = scene['bands']['euclid_VIS']
    im, var = d['image'].numpy(), d['variance'].numpy()
    yy, xx = np.indices(im.shape)
    p2s = np.linalg.inv(d['sky_to_pixel'].numpy())
    output = []
    for xy in d['positions'].numpy():
        delta = np.stack((xx-xy[0], yy-xy[1]), -1)
        radius = np.linalg.norm(delta, axis=-1)
        valid = d['mask'].numpy()
        ann = (radius > 10) & (radius < 14) & valid
        if ann.sum() < 20:
            output.append(np.eye(2)); continue
        bg = np.median(im[ann])
        weight = np.exp(-radius**2 / (2*6**2)) * (radius < 14) * valid
        signal = np.where(valid, im-bg, 0) * weight
        total = signal.sum()
        error = np.sqrt((np.where(valid,var,0)*weight**2).sum())
        if total < 8*error:
            output.append(np.eye(2)); continue
        moment = np.einsum('hw,hwi,hwj->ij', signal, delta, delta) / total
        values, vectors = np.linalg.eigh(moment)
        values = np.clip(values, .2, .85*36)
        values = values/(1-values/36) - d['psf_sigma']**2 - 1/12
        intrinsic = (vectors*np.maximum(values, .1)) @ vectors.T
        sky = p2s @ intrinsic @ p2s.T
        eig, vec = np.linalg.eigh(sky)
        eig[0] = max(eig[0], .4**2*eig[1])
        sky = (vec*eig) @ vec.T
        output.append(sky / np.sqrt(np.linalg.det(sky)))
    return torch.tensor(np.array(output), dtype=torch.float32)


def dictionary(scene, band, ellipse=None):
    """[pixels, sources, scales], full-profile unit flux with native PSF/WCS.

    An optional scene['point_sources'] boolean mask fixes independently
    classified stars to the PSF. Their seven columns have the same zero-size
    intrinsic profile; regression weights cannot broaden these sources.
    """
    d = scene['bands'][band]
    ellipse = ellipse_from_pixels(scene) if ellipse is None else ellipse
    cov = ellipse[:, None] * torch.tensor(SCALES**2, dtype=torch.float32)[None,:,None,None]
    if 'point_sources' in scene:
        flags = np.asarray(scene['point_sources'])
        if flags.shape != (len(ellipse),) or flags.dtype != np.dtype(bool):
            raise ValueError('point_sources must be a boolean mask matching the source list')
        cov[torch.as_tensor(flags)] = 0
    positions = d['positions'][:, None].expand(-1, len(SCALES), -1).reshape(-1,2)
    if 'psf_kernels' in d:
        from .pixel_psf import convolved_profile
        kernels=np.asarray(d['psf_kernels'])
        if len(kernels)!=len(ellipse):raise ValueError('One PSF kernel per source required')
        jac=d['sky_to_pixel'].numpy()
        result=[]
        for i,xy in enumerate(d['positions'].numpy()):
            for j in range(len(SCALES)):
                covariance=jac@cov[i,j].numpy()@jac.T
                result.append(convolved_profile(d['image'].shape,xy,covariance,kernels[i]).ravel())
        return np.stack(result,axis=1).reshape(-1,len(ellipse),len(SCALES))
    with torch.no_grad():
        basis = templates(positions, cov.reshape(-1,2,2), d['sky_to_pixel'],
                          d['psf_sigma'], d['image'].shape)
    return basis.numpy().reshape(-1,len(ellipse),len(SCALES)).astype('float64')


def positive_profile_fit(d, basis, prior=None, strength=0., reference_flux=None, precision=None):
    """Actual NNLS with analytically profiled *unconstrained* constant sky.

    Penalty = strength * ||(I - prior 1ᵀ) coefficients / reference_flux||².
    Every amplitude times the prior has ZERO penalty: no flux target is imposed.
    """
    image, variance = d['image'].numpy().ravel(), d['variance'].numpy().ravel()
    valid = d['mask'].numpy().ravel() & np.isfinite(image) & np.isfinite(variance) & (variance>0)
    n, k = basis.shape[1:]
    matrix = basis.reshape(-1,n*k)
    active = (matrix[valid].sum(0)>1e-5)
    if not active.any(): raise ValueError('No source component intersects valid pixels')
    weight = 1/np.sqrt(variance[valid])
    a = matrix[valid][:,active]*weight[:,None]
    y = image[valid]*weight
    # Project the background out in whitened coordinates; it may be negative.
    wnorm = weight @ weight
    a -= weight[:,None]*((weight@a)/wnorm)[None,:]
    y -= weight*((weight@y)/wnorm)
    if prior is not None and strength>0:
        if reference_flux is None:raise ValueError('Morphology regularization needs an amplitude scale')
        reg=np.zeros((n*k,n*k))
        whitening=np.eye(k)
        if precision is not None:
            eigen,vectors=np.linalg.eigh(precision)
            whitening=(vectors*np.sqrt(np.maximum(eigen,0)))@vectors.T
        for i in range(n):
            projection=np.eye(k)-prior[i,:,None]*np.ones((1,k))
            reg[i*k:(i+1)*k,i*k:(i+1)*k]=np.sqrt(strength)*whitening@projection/max(reference_flux[i],1e-15)
        a=np.concatenate((a,reg[:,active]),axis=0)
        y=np.r_[y,np.zeros(n*k)]
    scale=np.linalg.norm(a,axis=0)
    if np.any(scale<=0):raise ValueError('Unidentifiable component')
    with threadpool_limits(limits=1):
        solution,_=nnls(a/scale,y,maxiter=100*len(scale))
    coeff=np.zeros(n*k);coeff[active]=solution/scale;coeff=coeff.reshape(n,k)
    total=coeff.sum(1)
    fallback=np.full((n,k),1/k) if prior is None else prior
    mixture=np.divide(coeff,total[:,None],out=fallback.copy(),where=total[:,None]>0)
    return mixture,total


def signed_measurement(d, basis, mixture, fixed_background=False):
    profile=np.einsum('pnk,nk->pn',basis,mixture)
    active=profile.sum(0)>1e-5
    background=None
    if fixed_background:
        from .amortised_scarlet import robust_background
        background=robust_background(d)
    if background is not None:
        fit=fit_flux(d['image']-background,d['variance'],torch.tensor(profile[:,active]),d['mask'],fit_background=False)
        fit['background']=background;fit['model']=fit['model']+background
    else:
        fit=fit_flux(d['image'],d['variance'],torch.tensor(profile[:,active]),d['mask'])
    n=len(mixture)
    flux=np.full(n,np.nan);error=flux.copy();footprint=np.zeros(n)
    flux[active]=fit['flux'].numpy();error[active]=fit['error'].numpy()
    footprint[active]=profile[:,active].sum(0)
    return dict(flux=flux,error=error,footprint=footprint,
                reduced_chi2=float(fit['chi2']/fit['dof']),chi2=float(fit['chi2']),dof=fit['dof'],
                model=fit['model'].numpy(),condition=float(fit['condition']),background=float(fit['background']),
                covariance=fit['covariance'].numpy(),source_indices=np.flatnonzero(active))


def amplitude_scale(d,basis,prior):
    fit=signed_measurement(d,basis,prior)
    # This only sets the profile-penalty scale, not a flux target.
    scale=np.maximum(np.abs(fit['flux']),5*fit['error'])
    finite=scale[np.isfinite(scale)&(scale>0)]
    fallback=np.median(finite) if len(finite) else 1.
    return np.where(np.isfinite(scale)&(scale>0),scale,fallback)


class FeaturePrior:
    """Small PCA/ridge morphology prior fitted to bright image-derived profiles.

    PCA/scaling use training sources only. Ridge strength is selected on the
    validation sources. Predictions are positive, normalized mixture weights.
    """
    def fit(self, features, targets, alpha=10., components=12):
        x=np.asarray(features,dtype='float64').reshape(len(features),-1)
        self.mean=x.mean(0);self.scale=x.std(0).clip(.01)
        x=(x-self.mean)/self.scale
        with threadpool_limits(limits=1):
            _,s,v=np.linalg.svd(x,full_matrices=False)
        q=min(components,len(x)-2)
        self.projection=v[:q].T/np.maximum(s[:q]/np.sqrt(len(x)),.01)
        z=np.c_[np.ones(len(x)),x@self.projection]
        target=np.log(np.maximum(targets,.01));target-=target.mean(1,keepdims=True)
        penalty=np.eye(z.shape[1])*alpha;penalty[0,0]=0
        self.beta=np.linalg.solve(z.T@z+penalty,z.T@target)
        return self

    def predict(self,features):
        x=np.asarray(features,dtype='float64').reshape(len(features),-1)
        z=np.c_[np.ones(len(x)),((x-self.mean)/self.scale)@self.projection]
        # Keep extrapolation finite; all sizes remain possible.
        return softmax(np.clip(z@self.beta,-8,8),axis=1)


def fit_multiband(scene, prior, strength=10., band_strength=100., banks=None, prior_precision=None, fixed_background=False):
    """VIS morphology posterior -> band-specific profile refinement -> signed flux.

    Both population and foundation controls use this EXACT image fitter.
    Independent band profiles allow color gradients; low-S/N bands revert to VIS.
    """
    ellipse=ellipse_from_pixels(scene)
    banks={} if banks is None else banks
    vis='euclid_VIS'
    if vis not in banks:banks[vis]=dictionary(scene,vis,ellipse)
    d=scene['bands'][vis]
    reference=amplitude_scale(d,banks[vis],prior)
    weights,_=positive_profile_fit(d,banks[vis],prior,strength,reference,precision=prior_precision)
    result={vis:signed_measurement(d,banks[vis],weights,fixed_background)}
    result[vis]['weights']=weights
    for band in BANDS:
        if band==vis:continue
        if band not in banks:banks[band]=dictionary(scene,band,ellipse)
        d=scene['bands'][band]
        ref=amplitude_scale(d,banks[band],weights)
        w,_=positive_profile_fit(d,banks[band],weights,band_strength,ref)
        result[band]=signed_measurement(d,banks[band],w,fixed_background)
        result[band]['weights']=w
    return result


class GroupedPrior:
    """VIS pixels plus separately compressed pretrained stem and multiband views.

    Separate PCAs prevent numerous positional/latent channels from displacing
    high-resolution image information. All transforms are fitted on training data.
    """
    grouped = True

    @staticmethod
    def views(item,kind):
        views=[np.asarray(item['image']).reshape(len(item['image']),-1)]
        if kind=='foundation':
            features=item['foundation']
            views += [features[:,256:].reshape(len(features),-1),features[:,:256].reshape(len(features),-1)]
        return views

    def fit(self,item,targets,kind='foundation',alpha=10.,components=12):
        self.kind=kind;self.transforms=[];latent=[]
        for i,x in enumerate(self.views(item,kind)):
            x=x.astype('float64');mean=x.mean(0);scale=x.std(0).clip(.01)
            x=(x-mean)/scale
            with threadpool_limits(limits=1):_,s,v=np.linalg.svd(x,full_matrices=False)
            q=min(components if i==0 else max(3,components//3),len(x)-2)
            projection=v[:q].T/np.maximum(s[:q]/np.sqrt(len(x)),.01)
            self.transforms.append((mean,scale,projection));latent.append(x@projection)
        z=np.c_[np.ones(len(targets)),np.concatenate(latent,axis=1)]
        target=np.log(np.maximum(targets,.01));target-=target.mean(1,keepdims=True)
        penalty=np.eye(z.shape[1])*alpha;penalty[0,0]=0
        self.beta=np.linalg.solve(z.T@z+penalty,z.T@target)
        return self

    def predict(self,item):
        latent=[]
        for x,(mean,scale,projection) in zip(self.views(item,self.kind),self.transforms):
            latent.append(((x-mean)/scale)@projection)
        z=np.c_[np.ones(len(latent[0])),np.concatenate(latent,axis=1)]
        return softmax(np.clip(z@self.beta,-8,8),axis=1)
