"""Per-source morphology stamps from the JAISP foundation + A6_v4 photometry head.

For a sky position, find the prepared tile region (detcat `tile_inputs.npz`) that contains it,
the nearest JAISP detection, build the head's scene (central source + neighbours within 14"),
run the frozen foundation encoder + head, and return the central source's unit-flux VIS
template (PSF-convolved model on the 0.1"/px MER grid, neighbours removed) with a WCS.
Falls back to None when no detection matches within `match_arcsec`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS

REPO = Path(__file__).resolve().parents[2]
SS = REPO / "models/photometry/self_supervised"


class FoundationStamps:
    def __init__(self, device="cpu", run="A6_v4", region_index=REPO / "experiments/grism/region_index.json",
                 match_arcsec=0.4, band="euclid_VIS"):
        import sys
        sys.path.insert(0, str(REPO))
        from models.photometry.self_supervised.scene_features import SceneEncoder
        from models.photometry.self_supervised.truth_scarlet.model import TruthScarletHead, TruthScarlet
        from models.photometry.self_supervised.detcat.prepare import load_inputs, scene_for_source
        self._load_inputs, self._scene_for_source = load_inputs, scene_for_source
        self.device = torch.device(device)
        D = SS / "runs/truth_scarlet"
        st = torch.load(D / "runs" / run / "best.pt", map_location=self.device, weights_only=False); meta = st["metadata"]
        head = TruthScarletHead(per_band=True, perturbation=meta["perturbation"], eps_bound=meta["eps_bound"],
                                steps=meta["steps"], centroid=True).to(self.device)
        head.load_state_dict(st["head"]); head.eval()
        enc = SceneEncoder(REPO / "models/checkpoints/jaisp_v11_q1_soft/checkpoint_best.pt"); enc.encoder.to(self.device)
        self.model = TruthScarlet(head, enc, self.device)
        self.sigmas = {b: v["sigma_px"] for b, v in json.loads((SS / "runs/q1_all_bands/psf_calibration.json").read_text()).items()}
        self.regions = json.loads(Path(region_index).read_text())
        self.match_arcsec = match_arcsec
        self.band = band
        self._inputs = {}
        self.stats = dict(hit=0, nomatch=0, noregion=0, failed=0)

    def _inputs_for(self, region):
        if region not in self._inputs:
            self._inputs[region] = self._load_inputs(Path(region))
        return self._inputs[region]

    def _candidates(self, ra, dec):
        out = []
        for r in self.regions:
            if r["ra0"] - 0.002 <= ra <= r["ra1"] + 0.002 and r["dec0"] - 0.002 <= dec <= r["dec1"] + 0.002:
                out.append(r["region"])
        return out

    def stamp(self, ra, dec):
        """(unit-flux VIS template (H,W) float32, astropy WCS) for the central source, or (None, None)."""
        cands = self._candidates(ra, dec)
        if not cands:
            self.stats["noregion"] += 1; return None, None
        best = None
        for region in cands:
            inp = self._inputs_for(region)
            sky = inp["sky"]
            d = np.hypot((sky[:, 0] - ra) * np.cos(np.deg2rad(dec)), sky[:, 1] - dec) * 3600
            i = int(np.argmin(d))
            if d[i] <= self.match_arcsec and (best is None or d[i] < best[0]):
                best = (d[i], region, i)
        if best is None:
            self.stats["nomatch"] += 1; return None, None
        _, region, i = best
        inp = self._inputs_for(region)
        try:
            scene, info = self._scene_for_source(inp, i, self.sigmas)
            if scene is None:
                self.stats["failed"] += 1; return None, None
            with torch.enable_grad():
                out = self.model.run(scene)
            t = out["templates"][self.band]                      # (H*W, N) unit-flux templates
            H, W = scene["bands"][self.band]["image"].shape
            tmpl = t[:, 0].detach().reshape(H, W).cpu().numpy().astype(np.float32)
        except Exception as e:  # noqa: BLE001
            self.stats["failed"] += 1; return None, None
        # WCS of the scene crop: tile WCS shifted by the crop origin
        pos = inp[self.band + "__positions"][i]
        radius = (H - 1) // 2
        origin = np.rint(pos).astype(int) - radius
        hdr = inp[self.band + "__wcs"]
        w = WCS(fits.Header.fromstring(hdr) if isinstance(hdr, str) else hdr)
        w = w.deepcopy(); w.wcs.crpix = w.wcs.crpix - origin
        self.stats["hit"] += 1
        return np.clip(tmpl, 0, None), w
