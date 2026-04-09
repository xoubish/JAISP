"""High-resolution CenterNet-style detector on top of frozen V7 stems.

This variant avoids the coarse 0.8"/px bottleneck used by the existing
CenterNet detector. Instead it:

1. Runs the pretrained V7 per-band BandStem modules at native band resolution
2. Projects and fuses the stream features directly in the VIS frame
3. Predicts heatmap + offsets at full VIS resolution with a lightweight FPN

This keeps the foundation pretraining useful while preserving much more
localization detail for detection.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_HERE, _MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from jaisp_foundation_v7 import (
    ALL_BANDS,
    EUCLID_BANDS,
    RUBIN_BANDS,
    JAISPFoundationV7,
)

NISP_BANDS = [band for band in EUCLID_BANDS if band != "euclid_VIS"]


def _num_groups(ch: int) -> int:
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


def _head(in_ch: int, out_ch: int, hidden_ch: Optional[int] = None) -> nn.Sequential:
    hidden_ch = int(hidden_ch or max(16, in_ch))
    return nn.Sequential(
        nn.Conv2d(in_ch, hidden_ch, 3, padding=1, bias=False),
        nn.GroupNorm(_num_groups(hidden_ch), hidden_ch),
        nn.GELU(),
        nn.Conv2d(hidden_ch, out_ch, 1),
    )


class _ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: Optional[int] = None):
        super().__init__()
        out_ch = int(out_ch or in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act(y + self.skip(x))


class _DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_ch), out_ch),
            nn.GELU(),
            _ResidualBlock(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpFuseBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_ch), out_ch),
            nn.GELU(),
        )
        self.refine = _ResidualBlock(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.reduce(x)
        return self.refine(x)


class V7StemFeatureEncoder(nn.Module):
    """Fuse frozen V7 BandStem features directly in the VIS frame."""

    def __init__(
        self,
        foundation: JAISPFoundationV7,
        stream_ch: int = 16,
        fusion_ch: int = 32,
        freeze_stems: bool = True,
    ):
        super().__init__()
        self.stream_ch = int(stream_ch)
        self.fusion_ch = int(fusion_ch)
        self.freeze_stems = bool(freeze_stems)

        if not hasattr(foundation, "encoder") or not hasattr(foundation.encoder, "stems"):
            raise AttributeError("Expected JAISPFoundationV7 with encoder.stems")

        self.stem_ch = int(foundation.encoder.stem_ch)
        self.band_stems = foundation.encoder.stems
        if self.freeze_stems:
            for p in self.band_stems.parameters():
                p.requires_grad_(False)

        self.rubin_logits = nn.Parameter(torch.zeros(len(RUBIN_BANDS)))
        self.nisp_logits = nn.Parameter(torch.zeros(len(NISP_BANDS)))

        self.rubin_projs = nn.ModuleDict({
            band: nn.Conv2d(self.stem_ch, self.stream_ch, 1, bias=False)
            for band in RUBIN_BANDS
        })
        self.vis_proj = nn.Conv2d(self.stem_ch, self.stream_ch, 1, bias=False)
        self.nisp_projs = nn.ModuleDict({
            band: nn.Conv2d(self.stem_ch, self.stream_ch, 1, bias=False)
            for band in NISP_BANDS
        })

        self.stream_bias = nn.Parameter(torch.zeros(3, self.stream_ch, 1, 1))
        self.fuse_in = nn.Sequential(
            nn.Conv2d(self.stream_ch * 3, self.fusion_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(self.fusion_ch), self.fusion_ch),
            nn.GELU(),
            _ResidualBlock(self.fusion_ch),
        )

    @staticmethod
    def _target_hw(images: Dict[str, torch.Tensor]) -> Tuple[int, int]:
        if "euclid_VIS" in images:
            return images["euclid_VIS"].shape[-2:]
        for band in NISP_BANDS:
            if band in images:
                return images[band].shape[-2:]
        for band in RUBIN_BANDS:
            if band in images:
                return images[band].shape[-2:]
        raise ValueError("No bands available for stem feature encoder.")

    def _aggregate_stream(
        self,
        bands: list[str],
        images: Dict[str, torch.Tensor],
        rms: Dict[str, torch.Tensor],
        proj_map: nn.ModuleDict,
        logits: Optional[torch.Tensor],
        target_hw: Tuple[int, int],
        stream_idx: int,
    ) -> Optional[torch.Tensor]:
        available = [band for band in bands if band in images and band in rms]
        if not available:
            return None

        if logits is None:
            weights = [1.0 / len(available)] * len(available)
        else:
            indices = torch.tensor([bands.index(band) for band in available], device=logits.device)
            probs = torch.softmax(logits[indices], dim=0)
            weights = [probs[i] for i in range(len(available))]

        fused = None
        for w, band in zip(weights, available):
            feat = self.band_stems[band](images[band], rms[band])
            feat = proj_map[band](feat)
            if feat.shape[-2:] != target_hw:
                feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
            feat = feat * w
            fused = feat if fused is None else fused + feat

        return fused + self.stream_bias[stream_idx].view(1, self.stream_ch, 1, 1)

    def _vis_stream(
        self,
        images: Dict[str, torch.Tensor],
        rms: Dict[str, torch.Tensor],
        target_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        if "euclid_VIS" not in images or "euclid_VIS" not in rms:
            return None
        feat = self.band_stems["euclid_VIS"](images["euclid_VIS"], rms["euclid_VIS"])
        feat = self.vis_proj(feat)
        if feat.shape[-2:] != target_hw:
            feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
        return feat + self.stream_bias[1].view(1, self.stream_ch, 1, 1)

    def forward(
        self,
        images: Dict[str, torch.Tensor],
        rms: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        target_hw = self._target_hw(images)
        sample = next(iter(images.values()))
        bsz = sample.shape[0]
        device = sample.device
        dtype = sample.dtype

        def zeros() -> torch.Tensor:
            return torch.zeros(bsz, self.stream_ch, *target_hw, device=device, dtype=dtype)

        rubin = self._aggregate_stream(
            RUBIN_BANDS, images, rms, self.rubin_projs, self.rubin_logits, target_hw, stream_idx=0
        )
        vis = self._vis_stream(images, rms, target_hw)
        nisp = self._aggregate_stream(
            NISP_BANDS, images, rms, self.nisp_projs, self.nisp_logits, target_hw, stream_idx=2
        )

        x = torch.cat([
            rubin if rubin is not None else zeros(),
            vis if vis is not None else zeros(),
            nisp if nisp is not None else zeros(),
        ], dim=1)
        return self.fuse_in(x)


class StemCenterNetDetector(nn.Module):
    """VIS-frame CenterNet detector driven by native-resolution V7 stems."""

    def __init__(
        self,
        foundation: JAISPFoundationV7,
        stream_ch: int = 16,
        base_ch: int = 32,
        freeze_stems: bool = True,
        predict_profile: bool = False,
    ):
        super().__init__()
        self.stream_ch = int(stream_ch)
        self.base_ch = int(base_ch)
        self.freeze_stems = bool(freeze_stems)
        self.predict_profile = bool(predict_profile)

        self.encoder = V7StemFeatureEncoder(
            foundation=foundation,
            stream_ch=self.stream_ch,
            fusion_ch=self.base_ch,
            freeze_stems=self.freeze_stems,
        )

        self.stem_block = _ResidualBlock(self.base_ch)
        self.down1 = _DownBlock(self.base_ch, self.base_ch * 2)
        self.down2 = _DownBlock(self.base_ch * 2, self.base_ch * 4)
        self.up1 = _UpFuseBlock(self.base_ch * 4, self.base_ch * 2, self.base_ch * 2)
        self.up0 = _UpFuseBlock(self.base_ch * 2, self.base_ch, self.base_ch)
        self.out_block = _ResidualBlock(self.base_ch)

        self.hm_head = _head(self.base_ch, 1, self.base_ch)
        self.off_head = _head(self.base_ch, 2, self.base_ch)
        self.flux_head = _head(self.base_ch, 1, self.base_ch)
        self.profile_head = _head(self.base_ch, 4, self.base_ch) if self.predict_profile else None

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.constant_(self.hm_head[-1].bias, -2.19)
        nn.init.zeros_(self.off_head[-1].bias)

    def forward(
        self,
        images: Dict[str, torch.Tensor],
        rms: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        x0 = self.stem_block(self.encoder(images, rms))
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        u1 = self.up1(x2, x1)
        u0 = self.up0(u1, x0)
        feats = self.out_block(u0)

        out = {
            "heatmap": self.hm_head(feats).sigmoid(),
            "offset": self.off_head(feats),
            "log_flux": self.flux_head(feats).squeeze(1),
        }
        if self.profile_head is not None:
            out["profile"] = self.profile_head(feats)
        return out

    @torch.no_grad()
    def predict(
        self,
        images: Dict[str, torch.Tensor],
        rms: Dict[str, torch.Tensor],
        conf_threshold: float = 0.3,
        tile_hw: Optional[Tuple[int, int]] = None,
        nms_kernel: int = 7,
    ) -> Dict[str, torch.Tensor]:
        out = self(images, rms)
        hm = out["heatmap"][0, 0]
        H, W = hm.shape

        pad = nms_kernel // 2
        hm_max = F.max_pool2d(hm.unsqueeze(0).unsqueeze(0), nms_kernel, stride=1, padding=pad)[0, 0]
        keep = (hm == hm_max) & (hm > conf_threshold)

        if not keep.any():
            device = hm.device
            result = {
                "centroids": torch.zeros(0, 2, device=device),
                "classes": torch.zeros(0, dtype=torch.long, device=device),
                "scores": torch.zeros(0, device=device),
                "log_flux": torch.zeros(0, device=device),
            }
            if tile_hw is not None:
                result["positions_px"] = torch.zeros(0, 2, device=device)
            if self.profile_head is not None:
                result["profile"] = torch.zeros(0, 4, device=device)
            return result

        yi, xi = torch.where(keep)
        scores = hm[yi, xi]
        off = out["offset"][0]
        dx = off[0, yi, xi]
        dy = off[1, yi, xi]

        cx = ((xi.float() + dx) / max(W - 1, 1)).clamp(0.0, 1.0)
        cy = ((yi.float() + dy) / max(H - 1, 1)).clamp(0.0, 1.0)
        centroids = torch.stack([cx, cy], dim=1)
        flux = out["log_flux"][0, yi, xi]

        result = {
            "centroids": centroids,
            "classes": torch.zeros(len(scores), dtype=torch.long, device=hm.device),
            "scores": scores,
            "log_flux": flux,
        }
        if tile_hw is not None:
            ht, wt = tile_hw
            result["positions_px"] = centroids * centroids.new_tensor([wt - 1, ht - 1])
        if self.profile_head is not None:
            prof = out["profile"][0]
            result["profile"] = prof[:, yi, xi].T
        return result

    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "stream_ch": self.stream_ch,
            "base_ch": self.base_ch,
            "freeze_stems": self.freeze_stems,
            "predict_profile": self.predict_profile,
            "model_type": "stem_centernet",
        }, path)

    @classmethod
    def load(
        cls,
        path: str,
        foundation: JAISPFoundationV7,
        device: Optional[torch.device] = None,
    ) -> "StemCenterNetDetector":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(
            foundation=foundation,
            stream_ch=ckpt.get("stream_ch", 16),
            base_ch=ckpt.get("base_ch", 32),
            freeze_stems=ckpt.get("freeze_stems", True),
            predict_profile=ckpt.get("predict_profile", False),
        )
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            print(f"  [warn] Missing stem-detector keys: {missing}")
        if unexpected:
            print(f"  [warn] Unexpected stem-detector keys: {unexpected}")
        if device is not None:
            model = model.to(device)
        return model


def load_v7_foundation_from_checkpoint(
    checkpoint: str,
    device: Optional[torch.device] = None,
) -> JAISPFoundationV7:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = JAISPFoundationV7(
        band_names=cfg.get("band_names", ALL_BANDS),
        stem_ch=cfg.get("stem_ch", 64),
        hidden_ch=cfg.get("hidden_ch", 256),
        blocks_per_stage=cfg.get("blocks_per_stage", 2),
        transformer_depth=cfg.get("transformer_depth", 4),
        transformer_heads=cfg.get("transformer_heads", 8),
        fused_pixel_scale_arcsec=cfg.get("fused_pixel_scale_arcsec", 0.8),
    )
    missing, _ = model.load_state_dict(ckpt["model"], strict=False)
    enc_missing = [k for k in missing if not k.startswith(("encoder.skip_projs", "target_decoders"))]
    if enc_missing:
        print(f"  [warn] Missing foundation keys: {enc_missing}")
    model.eval()
    if device is not None:
        model = model.to(device)
    return model


__all__ = [
    "StemCenterNetDetector",
    "V7StemFeatureEncoder",
    "load_v7_foundation_from_checkpoint",
]
