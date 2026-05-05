"""photutils.psf.EPSFModel adapter for the trained PSF v4 NN ePSF head.

Drop-in replacement: builds a callable that takes a (tile-normalised x, y, band)
position and returns a ``photutils.psf.EPSFModel`` ready for forced photometry,
PSF fitting, or any other photutils workflow.

Example::

    from psf.epsf_v4_photutils import PSFFieldV4ToEPSF
    helper = PSFFieldV4ToEPSF.from_checkpoint(
        'models/checkpoints/psf_field_v4/checkpoint_best.pt',
        device='cuda',
    )
    epsf_model = helper.epsf_at(pos_norm=(0.1, -0.3), band='euclid_VIS')
    # epsf_model is a photutils.psf.EPSFModel(data=..., oversampling=5)

The same ``helper.epsf_at`` can also produce a native-resolution stamp
directly via ``helper.stamp_at(pos_norm, band, frac_xy, stamp_size)`` —
useful if you want to skip photutils entirely and convolve into a model
image yourself.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

from psf.psf_field_v4 import PSFFieldV4, ALL_BANDS

try:
    from photutils.psf import EPSFModel
except ImportError:                 # pragma: no cover
    EPSFModel = None                # photutils is optional


class PSFFieldV4ToEPSF:
    """Wrap a trained PSFFieldV4 as a callable EPSFModel factory."""

    def __init__(self, model: PSFFieldV4, device: torch.device = None):
        self.model = model.eval()
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

    @classmethod
    def from_checkpoint(cls, ckpt_path: Union[str, Path],
                        device: Union[str, torch.device, None] = None
                        ) -> "PSFFieldV4ToEPSF":
        device = torch.device(device) if device else torch.device("cpu")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        model = PSFFieldV4(
            psf_size=cfg.get("psf_size", 47),
            oversampling=cfg.get("oversampling", 5),
            hidden_ch=cfg.get("hidden_ch", 128),
            n_freqs=cfg.get("n_freqs", 8),
            band_names=cfg.get("band_names", ALL_BANDS),
        )
        model.load_state_dict(ckpt["model"])
        return cls(model, device=device)

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    @torch.no_grad()
    def oversampled_psf(self, pos_norm: Tuple[float, float], band: str
                        ) -> np.ndarray:
        """Return the unit-flux oversampled PSF stamp as a numpy array."""
        x, y = pos_norm
        bi = self.model.band_to_idx[band]
        pos = torch.tensor([[x, y]], dtype=torch.float32, device=self.device)
        bidx = torch.tensor([bi], dtype=torch.long, device=self.device)
        psf = self.model(pos, bidx)
        return psf[0, 0].cpu().numpy()

    def epsf_at(self, pos_norm: Tuple[float, float], band: str
                ) -> "EPSFModel":
        """Return a photutils EPSFModel ready for fitting or rendering.

        The returned object has the right ``oversampling`` set so photutils
        renders to native pixels correctly.
        """
        if EPSFModel is None:
            raise ImportError(
                "photutils is not installed; install with `pip install photutils`."
            )
        psf = self.oversampled_psf(pos_norm, band)
        return EPSFModel(data=psf,
                         oversampling=(self.model.oversampling,
                                       self.model.oversampling))

    @torch.no_grad()
    def stamp_at(self, pos_norm: Tuple[float, float], band: str,
                 frac_xy: Tuple[float, float] = (0.0, 0.0),
                 stamp_size: int = 32) -> np.ndarray:
        """Render a unit-flux native-resolution stamp at the requested sub-pixel offset."""
        x, y = pos_norm
        fx, fy = frac_xy
        bi = self.model.band_to_idx[band]
        pos = torch.tensor([[x, y]], dtype=torch.float32, device=self.device)
        bidx = torch.tensor([bi], dtype=torch.long, device=self.device)
        frac = torch.tensor([[fx, fy]], dtype=torch.float32, device=self.device)
        psf_oversampled = self.model(pos, bidx)
        rendered = self.model.render_at_native(psf_oversampled, frac, stamp_size=stamp_size)
        return rendered[0, 0].cpu().numpy()


__all__ = ["PSFFieldV4ToEPSF"]
