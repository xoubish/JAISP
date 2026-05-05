"""Training script for JAISP Foundation v9 — symmetric concat fusion + adversarial masking.

Two changes from v8 (motivated by notebook 13's cross-instrument diagnosis):

1. **Architectural symmetry.** ``rubin_concat=True`` makes the Rubin StreamFuser
   use concat+project fusion, matching the Euclid stream. This restores
   per-Rubin-band identity in the bottleneck and removes the 1/6× gradient
   attenuation through Rubin's averaging operation.

2. **Adversarial masking.** With probability ``p_adversarial`` (default 0.25)
   the loss masks the target band PLUS one or two wavelength-adjacent
   within-instrument neighbours. The encoder cannot rely solely on the
   nearest within-instrument neighbours to reconstruct the target — it has
   to use far-wavelength and cross-instrument inputs.

Per-band reconstruction-loss telemetry is already tracked by the v7 trainer
(``band_losses = defaultdict(list)``); v9 inherits that without modification.

Architecturally identical to v8 (same params, same bottleneck resolution,
same downstream-head interface) plus ~25K parameters in the new Rubin
StreamFuser projection layer.

Usage::

    # Single GPU
    python train_jaisp_foundation_v9.py \\
        --rubin_dir  ../data/rubin_tiles_all \\
        --euclid_dir ../data/euclid_tiles_all \\
        --output_dir ./checkpoints/jaisp_v9 \\
        --fused_pixel_scale_arcsec 0.4 \\
        --crop_size_rubin 256 \\
        --p_adversarial 0.25 \\
        --epochs 100 --lr 3e-4 --accum_steps 4 \\
        --wandb_project JAISP-Foundation-v9 \\
        --wandb_name v9_concat_adv25

    # Multi-GPU
    torchrun --nproc_per_node=2 train_jaisp_foundation_v9.py \\
        --rubin_dir  ../data/rubin_tiles_all \\
        --euclid_dir ../data/euclid_tiles_all \\
        --output_dir ./checkpoints/jaisp_v9 \\
        --fused_pixel_scale_arcsec 0.4 \\
        --crop_size_rubin 256 \\
        --p_adversarial 0.25 \\
        --epochs 100 --lr 3e-4 --accum_steps 2
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import jaisp_foundation_v8 as _v8
from jaisp_dataset_v8 import random_crop_sample
from jaisp_dataset_v9 import adversarial_drop
from jaisp_dataset_v7 import sample_context_target_phaseB_mixed
from jaisp_dataset_v6 import sample_context_target
from train_jaisp_foundation_v8 import JAISPTrainerV8
from train_jaisp_foundation_v7 import build_argparser as build_v7_argparser, suppress_stdout

try:
    import wandb
except ImportError:
    wandb = None


class JAISPTrainerV9(JAISPTrainerV8):
    """V9 trainer: symmetric concat fusion + adversarial masking.

    Reuses the V8 trainer's __init__ (which builds a v8 model and crops
    samples), then replaces the model with a ``rubin_concat=True`` v9
    instance, and overrides ``_prepare_batch`` to apply adversarial drop
    after the standard Phase-B split.
    """

    def __init__(self, p_adversarial: float = 0.25, n_extra_max: int = 2, **kwargs):
        self.p_adversarial = float(p_adversarial)
        self.n_extra_max = int(n_extra_max)

        # Let v8 init build everything. It will create a v8 model (rubin_concat=False)
        # which we then replace with a v9 model (rubin_concat=True).
        super().__init__(**kwargs)

        if self.is_main_process:
            print(f"\nReplacing V8 model with V9 (rubin_concat=True, "
                  f"p_adversarial={self.p_adversarial})...")

        with suppress_stdout(not self.is_main_process):
            v9_model = _v8.JAISPFoundationV8(
                band_names=_v8.ALL_BANDS,
                stem_ch=self._v8_stem_ch,
                hidden_ch=self._v8_hidden_ch,
                blocks_per_stage=self._v8_blocks,
                transformer_depth=self._v8_tdepth,
                transformer_heads=self._v8_theads,
                fused_pixel_scale_arcsec=self._v8_fused,
                rubin_concat=True,
            ).to(self.device)

        if self.distributed:
            self.model = DDP(
                v9_model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                find_unused_parameters=True,
            )
        else:
            self.model = v9_model

        # Re-create optimizer + scheduler for the new params (extra Rubin
        # StreamFuser projection adds ~25K trainable parameters).
        from jaisp_foundation_v6 import create_optimizer, create_scheduler
        lr = kwargs.get('lr', 3e-4)
        wd = kwargs.get('weight_decay', 0.05)
        warmup = kwargs.get('warmup_epochs', 5)
        epochs = kwargs.get('epochs', 80)
        self.optimizer = create_optimizer(self.model, lr=lr, weight_decay=wd)
        self.scheduler = create_scheduler(self.optimizer, warmup, epochs)

        # Make sure the saved config records v9 markers so load_foundation
        # picks the right code path on resume.
        self.config['rubin_concat'] = True
        self.config['p_adversarial'] = self.p_adversarial
        self.config['n_extra_max']   = self.n_extra_max

        if self.is_main_process:
            n_params = sum(p.numel() for p in v9_model.parameters())
            print(f"  V9 params: {n_params/1e6:.1f}M  "
                  f"(adversarial masking: p={self.p_adversarial}, n_extra_max={self.n_extra_max})")

    def _prepare_batch(self, sample: dict, rng: np.random.RandomState, force_phase_b: bool = False) -> dict:
        """V9 batch prep: random crop → Phase-B split → adversarial drop."""
        if self.crop_size_rubin > 0:
            sample = random_crop_sample(sample, self.crop_size_rubin, rng)

        use_phase_b = (
            sample.get("has_euclid", False)
            and (
                force_phase_b
                or (self.cross_instrument_prob > 0.0 and rng.random() < self.cross_instrument_prob)
            )
        )
        if use_phase_b:
            split = sample_context_target_phaseB_mixed(sample, rng, n_targets=self.n_targets)
            # Only apply adversarial dropping in Phase B (cross-instrument samples).
            if split is not None:
                split = adversarial_drop(
                    split, rng,
                    p_adversarial=self.p_adversarial,
                    n_extra_max=self.n_extra_max,
                )
        else:
            split = sample_context_target(sample, rng, n_targets=self.n_targets)
        if split is None:
            return None

        ctx_img = {b: v.unsqueeze(0).to(self.device) for b, v in split["context_images"].items()}
        ctx_rms = {b: v.unsqueeze(0).to(self.device) for b, v in split["context_rms"].items()}
        targets = [
            {
                "band": t["band"],
                "image": t["image"].unsqueeze(0).to(self.device),
                "rms": t["rms"].unsqueeze(0).to(self.device),
            }
            for t in split["targets"]
        ]
        return {"ctx_img": ctx_img, "ctx_rms": ctx_rms, "targets": targets}


# ============================================================
# CLI
# ============================================================

def build_argparser():
    p = build_v7_argparser()
    p.description = ('Train JAISP Foundation v9 (symmetric concat fusion + '
                     'adversarial cross-instrument masking).')
    p.set_defaults(
        output_dir='./checkpoints/jaisp_v9',
        fused_pixel_scale_arcsec=0.4,
        wandb_project='JAISP-Foundation-v9',
    )
    p.add_argument(
        '--crop_size_rubin', type=int, default=256,
        help='Random crop side in Rubin pixels (default 256). 0 disables cropping.',
    )
    p.add_argument(
        '--p_adversarial', type=float, default=0.25,
        help='Per-step probability of applying adversarial cross-instrument masking '
             '(default 0.25). On adversarial steps the target is masked together '
             'with 1..n_extra_max wavelength-adjacent within-instrument neighbours, '
             'forcing the encoder to use far-wavelength and cross-instrument inputs.',
    )
    p.add_argument(
        '--n_extra_max', type=int, default=2,
        help='Max number of extra within-instrument neighbours to mask on an '
             'adversarial step (default 2).',
    )
    return p


if __name__ == '__main__':
    args = build_argparser().parse_args()
    trainer_kwargs = vars(args).copy()
    resume = trainer_kwargs.pop('resume', None)
    resume_weights_only = trainer_kwargs.pop('resume_weights_only', False)
    trainer = JAISPTrainerV9(**trainer_kwargs)
    trainer.train(resume_from=resume, resume_weights_only=resume_weights_only)
