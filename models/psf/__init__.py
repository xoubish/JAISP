"""
models.psf — continuous, chromatic, spatially-varying PSF modelling.

Core components
---------------
- PSFField         : SIREN-based implicit PSF, 10-band, SED-conditioned
- SEDEncoder       : 10-band flux → SED embedding + scalar color
- DCRTerm          : per-Rubin-band color-dependent centroid shift
- analytic_optimal_flux, chi2_loss, normalise_psf : forward-model utilities
- BAND_ORDER, BAND_TO_IDX, BAND_PX_SCALE          : band metadata
"""

from .psf_field import (
    PSFField,
    SIREN,
    SineLayer,
    FourierFeatures,
    SEDEncoder,
    DCRTerm,
    analytic_optimal_flux,
    chi2_loss,
    normalise_psf,
    BAND_ORDER,
    BAND_TO_IDX,
    BAND_PX_SCALE,
    N_BANDS,
    N_RUBIN,
)
from .foundation_epsf_head import (
    FoundationEPSFHead,
    load_foundation_epsf_head,
    load_base_epsf_bank,
)

__all__ = [
    'PSFField',
    'SIREN',
    'SineLayer',
    'FourierFeatures',
    'SEDEncoder',
    'DCRTerm',
    'analytic_optimal_flux',
    'chi2_loss',
    'normalise_psf',
    'BAND_ORDER',
    'BAND_TO_IDX',
    'BAND_PX_SCALE',
    'N_BANDS',
    'N_RUBIN',
    'FoundationEPSFHead',
    'load_foundation_epsf_head',
    'load_base_epsf_bank',
]
