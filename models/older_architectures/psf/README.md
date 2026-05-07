Older PSF Architectures
======================

These files are preserved for reference, but they are no longer the active PSF
path. The active implementation lives in `models/psf/foundation_epsf_head.py`
and is trained by `models/psf/train_foundation_epsf_head.py` on Gaia-selected
stars with an analytic Gaussian/Moffat base ePSF.

Moved here in May 2026 after the empirical PCA/V4/PSFField attempts produced
poor ePSF reconstructions and the project switched to the foundation-conditioned
Gaia ePSF head.
