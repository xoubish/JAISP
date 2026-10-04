"""Supervised single-band flux regression without JAISP foundation features.

V2 combines training-calibrated native aperture sums with a CNN correction.
V1 preserves the original pure CNN. Both predict catalog flux from VIS stamps;
neither uses the detection, astrometry or PSF heads.
"""

__all__ = ["config", "catalog", "geometry", "check_inputs", "fetch_mer", "prepare",
           "data", "losses", "model", "baseline", "train", "evaluate", "archfig"]
