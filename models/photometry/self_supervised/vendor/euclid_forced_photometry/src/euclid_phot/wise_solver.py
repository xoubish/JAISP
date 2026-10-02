"""Accurate linear flux updates for one WISE image with a constant sky."""
from __future__ import annotations

import numpy as np
from scipy.linalg import lstsq
from scipy.sparse import diags
from scipy.sparse.linalg import lsqr
from tractor.lsqr_optimizer import LsqrOptimizer


class WiseFluxOptimizer(LsqrOptimizer):
    """Keep Tractor's templates and likelihood, but check the linear solve.

    Tractor's default forced-photometry path disables column scaling and
    accepts one LSQR update without checking its stop condition. In dense
    WISE scenes, this can leave the source amplitudes short of convergence.
    Normalize template columns before solving; use pivoted QR when the dense
    array is modest, otherwise a sparse solve with explicit stopping checks.
    """

    max_dense_elements = 20_000_000

    def __init__(self):
        self.diagnostics = None

    def getUpdateDirection(self, tractor, allderivs, damp=0., priors=False,
                           chiImages=None, shared_params=False, **kwargs):
        images = tractor.getImages()
        if len(images) != 1 or priors or shared_params or damp:
            raise ValueError("WISE flux optimizer requires one image, independent fluxes, and no priors or damping")
        matrix = super().getUpdateDirection(
            tractor, allderivs, damp=0., priors=False, chiImages=chiImages,
            shared_params=False, scale_columns=False, get_A_matrix=True)
        update = np.zeros(len(allderivs), dtype=float)
        if not hasattr(matrix, 'shape') or matrix.shape[1] == 0:
            raise ValueError("No weighted pixels constrain the WISE fit")
        matrix = matrix.astype(np.float64)
        norm = np.sqrt(np.asarray(matrix.power(2).sum(axis=0)).ravel())
        active = norm > 0
        if not active.any():
            raise ValueError("No weighted pixels constrain the WISE fit")
        scaled = matrix[:, active] @ diags(1. / norm[active])
        residual = (chiImages[0] if chiImages is not None
                    else tractor.getChiImage(img=images[0]))
        rhs = np.asarray(residual, dtype=np.float64).ravel()
        if scaled.shape[0] * scaled.shape[1] <= self.max_dense_elements:
            answer, _, rank, _ = lstsq(
                scaled.toarray(), rhs, lapack_driver='gelsy',
                cond=np.finfo(float).eps * max(scaled.shape))
            if rank < scaled.shape[1]:
                raise ValueError("WISE source templates are linearly dependent; individual fluxes cannot be separated")
            method = 'scaled QR'
            iterations = None
        else:
            result = lsqr(scaled, rhs, atol=1e-10, btol=1e-10,
                          conlim=1e12, iter_lim=max(1000, 10 * scaled.shape[1]))
            answer, stop, iterations = result[:3]
            if stop not in (0, 1, 2, 4, 5):
                raise RuntimeError(f"WISE linear solve did not converge (LSQR stop={stop})")
            method = 'scaled LSQR'
            rank = None
        step = np.zeros(matrix.shape[1], dtype=float)
        step[active] = answer / norm[active]
        update[:len(step)] = step
        gradient = scaled.T @ (rhs - scaled @ answer)
        self.diagnostics = dict(method=method, n_active=int(active.sum()), rank=rank,
                                iterations=iterations,
                                normalized_gradient_max=float(np.max(np.abs(gradient))))
        return update
