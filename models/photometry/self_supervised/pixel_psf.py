"""Pixelized-PSF convolution with intrinsic analytic Fourier profiles.

A delivered PSF stamp already includes the mosaic pixel response. Do not
integrate it over a second pixel. Normalize the supplied finite stamp once,
retain negative interpolation lobes, pad before convolution, then crop without
renormalizing the source footprint.
"""
import numpy as np
from scipy.fft import next_fast_len, rfft2, irfft2, rfftfreq, fftfreq


def normalize_kernel(stamp):
    stamp=np.asarray(stamp,dtype=float)
    if stamp.ndim!=2 or any(n%2!=1 for n in stamp.shape):raise ValueError('PSF must have odd 2D shape')
    if not np.isfinite(stamp).all() or stamp.sum()<=0:raise ValueError('Invalid PSF stamp')
    return stamp/stamp.sum()


def convolved_profile(shape,position,covariance,stamp,kind='gaussian',padding=None):
    """Unit-total-flux profile on output pixels; covariance in pixel².

    For an exponential, covariance specifies the elliptical exponential scale,
    rather than its second moment (which is three times larger).
    """
    kernel=normalize_kernel(stamp);cov=np.asarray(covariance,float)
    if np.linalg.eigvalsh(cov).min()<0:raise ValueError('Negative intrinsic covariance')
    radius=np.sqrt(np.linalg.eigvalsh(cov).max())
    pad=int(max(kernel.shape)//2+max(32,np.ceil((24 if kind=='exponential' else 9)*radius))) if padding is None else int(padding)
    fftshape=tuple(next_fast_len(n+2*pad) for n in shape)
    canvas=np.zeros(fftshape)
    cy,cx=np.array(fftshape)//2;hy,hx=np.array(kernel.shape)//2
    canvas[cy-hy:cy+hy+1,cx-hx:cx+hx+1]=kernel
    transfer=rfft2(np.fft.ifftshift(canvas))
    ky=fftfreq(fftshape[0])[:,None];kx=rfftfreq(fftshape[1])[None,:]
    q=cov[0,0]*kx*kx+2*cov[0,1]*kx*ky+cov[1,1]*ky*ky
    if kind=='gaussian':intrinsic=np.exp(-2*np.pi**2*q)
    elif kind=='exponential':intrinsic=(1+4*np.pi**2*q)**-1.5
    else:raise ValueError(kind)
    phase=np.exp(-2j*np.pi*(kx*(position[0]+pad)+ky*(position[1]+pad)))
    full=irfft2(transfer*intrinsic*phase,s=fftshape)
    return full[pad:pad+shape[0],pad:pad+shape[1]]


def gaussian_kernel(shape,sigma):
    """Pixel-integrated circular Gaussian for the PSF-core-width control."""
    from scipy.special import erf
    axes=[np.arange(n)-n//2 for n in shape]
    integrals=[.5*(erf((a+.5)/(np.sqrt(2)*sigma))-erf((a-.5)/(np.sqrt(2)*sigma))) for a in axes]
    return normalize_kernel(np.outer(*integrals))
