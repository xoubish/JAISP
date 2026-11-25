"""Differentiable grism forward model (torch) built on the SIR geometry.

A source is a morphology stamp M (ny, nx) on a sky grid (any pixel scale,
given by a WCS) times a spectrum f(lambda).  Its dispersed image on a NISP-S
detector is

    I(q) = sum_lambda f(lambda) * M( A^-1 (q - b(lambda)) )

where q is a detector pixel and (A, b(lambda)) is the affine map from stamp
pixels to detector pixels at wavelength lambda.  The map is the SIR chain
(sky -> OPT -> IDS/CRV -> detector WCS) linearised over the stamp, which is
exact to << 0.01 px for stamps of a few arcsec.  The lambda sum is done with
`grid_sample`, so gradients flow to f, M, and to the affine parameters
(per-source trace offsets) if they are made learnable.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from astropy.wcs import WCS

from .sir_geometry import FrameGeometry


def _affine_from_points(src, dst):
    """Least-squares affine (2x3) mapping src (N,2) -> dst (N,2)."""
    A = np.column_stack([src, np.ones(len(src))])
    sol, *_ = np.linalg.lstsq(A, dst, rcond=None)      # (3,2)
    return sol.T                                       # (2,3): dst = M[:, :2] @ src + M[:, 2]


class SourceTrace:
    """Linearised geometry for one source on one detector of one frame."""

    def __init__(self, geo: FrameGeometry, det_id: str, ra: float, dec: float,
                 stamp_wcs: WCS, stamp_shape, lambdas, order: int = 1):
        self.det_id = det_id
        self.order = order
        self.lambdas = np.asarray(lambdas, float)
        ny, nx = stamp_shape
        # sky positions of stamp corners/centre -> detector pixels at the reference wavelength
        px = np.array([[0, 0], [nx - 1, 0], [nx - 1, ny - 1], [0, ny - 1], [(nx - 1) / 2, (ny - 1) / 2]], float)
        sky = stamp_wcs.pixel_to_world_values(px[:, 0], px[:, 1])
        ref_mm = geo.opt.reference_mm(np.asarray(sky[0]), np.asarray(sky[1]), order, apply_tilt=False)
        ref_mm = ref_mm.copy(); ref_mm[:, 1] += geo.yshift_mm.get(det_id, 0.0)
        ref_pix = geo.det.mm_to_pix(ref_mm, det_id)
        self.A = _affine_from_points(px, ref_pix)                 # stamp px -> det px at delta_disp=0
        # dispersion shift along the trace for the source centre, in detector pixels
        _, pos_mm, _ = geo.trace_mm(ra, dec, self.lambdas, order, det_id=det_id)
        centre_mm = geo.opt.reference_mm(ra, dec, order, apply_tilt=False).copy()
        centre_mm[:, 1] += geo.yshift_mm.get(det_id, 0.0)
        shift_mm = pos_mm[0] - centre_mm[0]                       # (L,2) mm
        # mm -> pixel is linear per detector: use the detector CD (with SIR rotation handled by mm_to_pix)
        p0 = geo.det.mm_to_pix(centre_mm, det_id)[0]
        self.shift_px = geo.det.mm_to_pix(centre_mm + shift_mm, det_id) - p0   # (L,2)
        self.centre_px = p0
        self.ref_px = ref_pix[4]

    def bbox(self, stamp_shape, pad=3):
        ny, nx = stamp_shape
        corners = np.array([[0, 0], [nx - 1, 0], [nx - 1, ny - 1], [0, ny - 1]], float)
        det = corners @ self.A[:, :2].T + self.A[:, 2]
        lo = det.min(axis=0) + self.shift_px.min(axis=0) - pad
        hi = det.max(axis=0) + self.shift_px.max(axis=0) + pad
        return np.floor(lo).astype(int), np.ceil(hi).astype(int)


def two_gauss_kernel(sigma1, sigma2, f2, radius=None, device="cpu", dtype=torch.float32):
    """Normalised circular two-Gaussian kernel: (1-f2) G(sigma1) + f2 G(sigma2), sigmas in stamp px."""
    s1 = torch.as_tensor(sigma1, dtype=dtype, device=device); s2 = torch.as_tensor(sigma2, dtype=dtype, device=device)
    f2 = torch.as_tensor(f2, dtype=dtype, device=device)
    r = int(radius if radius is not None else max(3, np.ceil(3 * float(s2.detach()))))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    rr = x[:, None] ** 2 + x[None, :] ** 2
    g1 = torch.exp(-0.5 * rr / s1 ** 2); g1 = g1 / g1.sum()
    g2 = torch.exp(-0.5 * rr / s2 ** 2); g2 = g2 / g2.sum()
    return (1 - f2) * g1 + f2 * g2


def kernel_bank_from_table(lambdas, table_npz, device="cpu", dtype=torch.float32, radius=None):
    """Build (bank (K,1,kh,kw), idx (L,)) from a per-lambda (sigma1, sigma2, f2) table (stamp px)."""
    t = np.load(table_npz)
    lam_t = np.asarray(t["lam"], float)
    ks = [two_gauss_kernel(t["sigma1"][k], t["sigma2"][k], t["f2"][k], radius=radius, device=device, dtype=dtype)
          for k in range(len(lam_t))]
    r = max(k.shape[-1] for k in ks)
    bank = torch.zeros(len(ks), 1, r, r, device=device, dtype=dtype)
    for i, k in enumerate(ks):
        o = (r - k.shape[-1]) // 2
        bank[i, 0, o:o + k.shape[-1], o:o + k.shape[-1]] = k
    idx = torch.as_tensor(np.abs(np.asarray(lambdas, float)[:, None] - lam_t[None, :]).argmin(1), device=device)
    return bank, idx


def render_source(stamp: torch.Tensor, spectrum: torch.Tensor, trace: SourceTrace,
                  box_lo, box_hi, extra_shift=None, blur_sigma_px=None, kernels=None):
    """Render one source into a detector box [box_lo, box_hi) (x, y order).

    stamp:    (ny, nx) morphology (flux per stamp pixel, arbitrary normalisation)
    spectrum: (L,) flux multiplier per wavelength sample (same L as trace.lambdas)
    kernels:  optional (bank (K,1,kh,kw), idx (L,)) wavelength-dependent PSF kernels applied to the stamp
    returns:  (H, W) image on the detector box.
    """
    device = stamp.device
    ny, nx = stamp.shape
    W = int(box_hi[0] - box_lo[0]); H = int(box_hi[1] - box_lo[1])
    # detector pixel centres of the box
    xs = torch.arange(W, device=device, dtype=stamp.dtype) + float(box_lo[0])
    ys = torch.arange(H, device=device, dtype=stamp.dtype) + float(box_lo[1])
    qx, qy = torch.meshgrid(xs, ys, indexing="xy")               # (H, W)
    A = torch.as_tensor(trace.A, dtype=stamp.dtype, device=device)
    Ainv = torch.linalg.inv(A[:, :2])
    shift = torch.as_tensor(trace.shift_px, dtype=stamp.dtype, device=device)   # (L,2)
    if extra_shift is not None:
        shift = shift + extra_shift
    L = shift.shape[0]
    # stamp coordinates for every (lambda, q):  s = Ainv (q - b - shift_l)
    q = torch.stack([qx, qy], dim=-1).reshape(1, H * W, 2) - A[:, 2]        # (1, HW, 2)
    s = (q - shift[:, None, :]) @ Ainv.T                                    # (L, HW, 2)
    # normalised grid coords for grid_sample (align_corners=True: -1 -> pixel 0, +1 -> pixel n-1)
    gx = s[..., 0] / max(nx - 1, 1) * 2 - 1
    gy = s[..., 1] / max(ny - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).reshape(L, H, W, 2)
    img = stamp[None, None]
    if blur_sigma_px is not None and float(torch.as_tensor(blur_sigma_px).detach()) > 0.05:
        k = _gaussian_kernel(blur_sigma_px, device, stamp.dtype)
        img = F.conv2d(img, k, padding=k.shape[-1] // 2)
    if kernels is not None:
        bank, kidx = kernels
        img = F.conv2d(img, bank, padding=bank.shape[-1] // 2)        # (1, K, ny, nx)
        img = img.permute(1, 0, 2, 3)                                   # (K, 1, ny, nx)
    # integrate over the detector pixel footprint: a detector pixel covers ~1/scale stamp pixels per axis
    scale = float(torch.sqrt(torch.abs(torch.linalg.det(A[:, :2]))))
    box = int(round(1.0 / scale)) if scale < 1 else 1
    if box > 1:
        kb = torch.full((1, 1, box, box), 1.0 / (box * box), device=device, dtype=stamp.dtype)
        img = F.conv2d(F.pad(img, (box // 2, box - 1 - box // 2, box // 2, box - 1 - box // 2)), kb)
        # (flux per stamp pixel averaged over the footprint; multiplied back by the footprint area below)
    img = img[kidx] if kernels is not None else img.expand(L, 1, ny, nx)
    sampled = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)  # (L,1,H,W)
    # each detector pixel collects the stamp flux over its footprint (1/jac stamp pixels);
    # the box average above already divided by box^2, so rescale by box^2/jac
    jac = torch.abs(torch.linalg.det(A[:, :2]))
    return (sampled[:, 0] * spectrum[:, None, None]).sum(0) * (box * box) / jac


def render_source_windowed(stamp: torch.Tensor, spectrum: torch.Tensor, trace: SourceTrace,
                           box_lo, box_hi, extra_shift=None, kernels=None, win_pad=3):
    """Exact equivalent of `render_source` that samples, for every wavelength, only a small
    window around that wavelength's trace position (where the warped stamp is non-zero) and
    scatter-adds the windows into the box.  ~10x fewer samples than sampling the full box per
    wavelength; identical result up to window truncation (window = warped stamp extent + pad).
    """
    device, dtype = stamp.device, stamp.dtype
    ny, nx = stamp.shape
    W = int(box_hi[0] - box_lo[0]); H = int(box_hi[1] - box_lo[1])
    A = torch.as_tensor(trace.A, dtype=dtype, device=device)
    Ainv = torch.linalg.inv(A[:, :2])
    shift = torch.as_tensor(trace.shift_px, dtype=dtype, device=device)
    if extra_shift is not None:
        shift = shift + extra_shift
    L = shift.shape[0]
    img = stamp[None, None]
    if kernels is not None:
        bank, kidx = kernels
        img = F.conv2d(img, bank, padding=bank.shape[-1] // 2).permute(1, 0, 2, 3)   # (K,1,ny,nx)
    scale = float(torch.sqrt(torch.abs(torch.linalg.det(A[:, :2]))))
    box = int(round(1.0 / scale)) if scale < 1 else 1
    if box > 1:
        kb = torch.full((1, 1, box, box), 1.0 / (box * box), device=device, dtype=dtype)
        img = F.conv2d(F.pad(img, (box // 2, box - 1 - box // 2, box // 2, box - 1 - box // 2)), kb)
    img = img[kidx] if kernels is not None else img.expand(L, 1, ny, nx)             # (L,1,ny,nx)
    # window radius: warped stamp half-extent (+ kernel reach is already inside the stamp) + pad
    r = int(np.ceil(0.5 * max(ny, nx) * scale)) + win_pad
    # detector position of the stamp centre at each lambda (float), window origin = round(pos) - r
    centre_stamp = torch.tensor([(nx - 1) / 2, (ny - 1) / 2], device=device, dtype=dtype)
    pos = centre_stamp @ A[:, :2].T + A[:, 2] + shift                                   # (L,2) det px
    org = torch.round(pos.detach()) - r                                                 # (L,2) integer window origins
    wx = torch.arange(2 * r + 1, device=device, dtype=dtype); wy = wx
    gx_, gy_ = torch.meshgrid(wx, wy, indexing="xy")                                    # (2r+1, 2r+1)
    q = torch.stack([gx_, gy_], -1).reshape(1, -1, 2) + org[:, None, :]                 # (L, n, 2) det px
    s = (q - A[:, 2] - shift[:, None, :]) @ Ainv.T                                      # (L, n, 2) stamp px
    gx = s[..., 0] / max(nx - 1, 1) * 2 - 1; gy = s[..., 1] / max(ny - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], -1).reshape(L, 2 * r + 1, 2 * r + 1, 2)
    sampled = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]  # (L, 2r+1, 2r+1)
    jac = torch.abs(torch.linalg.det(A[:, :2]))
    sampled = sampled * spectrum[:, None, None] * (box * box) / jac
    # scatter-add windows into the box image
    oy = (org[:, 1] - float(box_lo[1])).long(); ox = (org[:, 0] - float(box_lo[0])).long()      # (L,)
    yy = oy[:, None, None] + torch.arange(2 * r + 1, device=device)[None, :, None]
    xx = ox[:, None, None] + torch.arange(2 * r + 1, device=device)[None, None, :]
    yy = yy.expand(L, 2 * r + 1, 2 * r + 1); xx = xx.expand(L, 2 * r + 1, 2 * r + 1)
    ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    out = torch.zeros(H, W, device=device, dtype=dtype)
    out.index_put_((yy[ok], xx[ok]), sampled[ok], accumulate=True)
    return out


def render_source_fast(stamp: torch.Tensor, spectrum: torch.Tensor, trace: SourceTrace,
                       box_lo, box_hi, extra_shift=None, kernels=None, win_pad=4):
    """Same model as `render_source`, computed as (stamp warped once onto the detector grid)
    convolved with the spectrum splatted along the trace path.  Exact for a constant affine
    map over the source (true for stamps of a few arcsec); ~20-50x faster and tiny memory.
    """
    device, dtype = stamp.device, stamp.dtype
    ny, nx = stamp.shape
    W = int(box_hi[0] - box_lo[0]); H = int(box_hi[1] - box_lo[1])
    A = torch.as_tensor(trace.A, dtype=dtype, device=device)
    # detector-resolution image of the stamp around the reference position (window of radius r)
    scale = float(torch.sqrt(torch.abs(torch.linalg.det(A[:, :2]))))
    r = int(np.ceil(0.5 * max(ny, nx) * scale)) + win_pad
    cx, cy = float(trace.ref_px[0]), float(trace.ref_px[1])
    xs = torch.arange(-r, r + 1, device=device, dtype=dtype); ys = xs
    qx, qy = torch.meshgrid(xs, ys, indexing="xy")                           # offsets from the reference pixel
    Ainv = torch.linalg.inv(A[:, :2])
    ref_q = torch.tensor([cx, cy], device=device, dtype=dtype)
    s = (torch.stack([qx, qy], -1).reshape(-1, 2) + ref_q - A[:, 2]) @ Ainv.T       # stamp coords
    gx = s[:, 0] / max(nx - 1, 1) * 2 - 1; gy = s[:, 1] / max(ny - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], -1).reshape(1, 2 * r + 1, 2 * r + 1, 2)
    img = stamp[None, None]
    if kernels is not None:
        bank, kidx = kernels
        img = F.conv2d(img, bank, padding=bank.shape[-1] // 2).permute(1, 0, 2, 3)   # (K,1,ny,nx)
    box = int(round(1.0 / scale)) if scale < 1 else 1
    if box > 1:
        kb = torch.full((1, 1, box, box), 1.0 / (box * box), device=device, dtype=dtype)
        img = F.conv2d(F.pad(img, (box // 2, box - 1 - box // 2, box // 2, box - 1 - box // 2)), kb)
    K = img.shape[0]
    Wimg = F.grid_sample(img, grid.expand(K, -1, -1, -1), mode="bilinear", padding_mode="zeros", align_corners=True)  # (K,1,2r+1,2r+1)
    jac = torch.abs(torch.linalg.det(A[:, :2]))
    Wimg = Wimg * (box * box) / jac
    # spectrum splatted along the trace: positions of each lambda sample relative to the reference pixel
    shift = torch.as_tensor(trace.shift_px, dtype=dtype, device=device)
    if extra_shift is not None:
        shift = shift + extra_shift
    ctr = torch.tensor([float(trace.centre_px[0]), float(trace.centre_px[1])], device=device, dtype=dtype)
    pos = ctr + shift - torch.tensor([float(box_lo[0]), float(box_lo[1])], device=device, dtype=dtype)   # (L,2) in box px
    # the warped stamp is centred on ref_px, the trace shift is relative to centre_px: include the difference
    pos = pos + (ref_q - ctr)
    L = pos.shape[0]
    line = torch.zeros(K, H + 2 * r + 2, W + 2 * r + 2, device=device, dtype=dtype)   # padded line image
    px = pos[:, 0] + r + 1; py = pos[:, 1] + r + 1
    x0 = torch.floor(px); y0 = torch.floor(py); fx = px - x0; fy = py - y0
    x0 = x0.long(); y0 = y0.long()
    kid = kidx if kernels is not None else torch.zeros(L, dtype=torch.long, device=device)
    for ddx, ddy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi = x0 + ddx; yi = y0 + ddy
        ok = (xi >= 0) & (xi < line.shape[2]) & (yi >= 0) & (yi < line.shape[1])
        line.index_put_((kid[ok], yi[ok], xi[ok]), spectrum[ok] * wgt[ok], accumulate=True)
    # convolve each group's line image with its warped stamp (correlation with flipped kernel = convolution)
    out = F.conv2d(line[None], torch.flip(Wimg, dims=(-1, -2)).permute(1, 0, 2, 3).reshape(1, K, 2 * r + 1, 2 * r + 1)
                   if False else torch.flip(Wimg, dims=(-1, -2)), groups=1) if False else None
    out = torch.zeros(H + 2, W + 2, device=device, dtype=dtype)
    for k in range(K):
        out = out + F.conv2d(line[k][None, None], torch.flip(Wimg[k], dims=(-1, -2))[None], padding=0)[0, 0]
    return out[1:H + 1, 1:W + 1]


def _gaussian_kernel(sigma, device, dtype):
    """Separable Gaussian kernel; `sigma` may be a tensor (gradients flow through it)."""
    s = sigma if torch.is_tensor(sigma) else torch.tensor(float(sigma), device=device, dtype=dtype)
    r = int(max(1, np.ceil(3 * float(s.detach()))))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    g = torch.exp(-0.5 * (x / s) ** 2); g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def lambda_grid(start=11900.0, stop=19000.0, step=13.4):
    return np.arange(start, stop + 0.5 * step, step)
