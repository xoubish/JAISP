"""Self-training loop for the high-resolution stem-based detector.

Round 1: train on classical VIS pseudo-labels
Round 2: promote novel high-confidence 10-band detections, demote low-confidence
         pseudo-labels, and retrain on the refined label set
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_HERE, _MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from detection.dataset import TileDetectionDataset, _vis_bright_core_and_spike_mask
from detection.stem_centernet_detector import (
    StemCenterNetDetector,
    load_foundation_from_checkpoint,
)
from detection.train_stem_centernet import (
    _apply_extra_labels_inplace,
    _load_teacher_label_dict,
    _points_near,
    _teacher_for_tile,
)


def _run_training_round(
    encoder_ckpt: str,
    rubin_dir: str,
    euclid_dir: str,
    out_path: str,
    extra_labels: str = None,
    init_checkpoint: str = None,
    epochs: int = 60,
    batch_size: int = 1,
    lr: float = 1e-4,
    sigma: float = 2.0,
    nsig: float = 3.0,
    stream_ch: int = 16,
    base_ch: int = 32,
    unfreeze_stems: bool = False,
    teacher_labels: str = None,
    teacher_match_radius: float = 0.012,
    teacher_filter_mode: str = "bright",
    no_teacher_add_labels: bool = False,
    bright_ignore_radius: int = 0,
    teacher_spike_radius: int = 40,
    teacher_spike_width: float = 3.0,
    viz_spike_veto_radius: int = 0,
    viz_spike_veto_width: float = 0.0,
    device: torch.device = None,
    wandb_project: str = None,
    wandb_name: str = None,
    ddp_nproc: int = 0,
) -> None:
    use_ddp = ddp_nproc is not None and int(ddp_nproc) > 1
    if use_ddp:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node", str(ddp_nproc),
            str(_HERE / "train_stem_centernet.py"),
            "--ddp",
        ]
    else:
        cmd = [
            sys.executable, str(_HERE / "train_stem_centernet.py"),
        ]
    cmd += [
        "--encoder_ckpt", encoder_ckpt,
        "--rubin_dir", rubin_dir,
        "--out", out_path,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--sigma", str(sigma),
        "--nsig", str(nsig),
        "--stream_ch", str(stream_ch),
        "--base_ch", str(base_ch),
        "--viz_spike_veto_radius", str(viz_spike_veto_radius),
        "--viz_spike_veto_width", str(viz_spike_veto_width),
        "--teacher_match_radius", str(teacher_match_radius),
        "--teacher_filter_mode", teacher_filter_mode,
        "--bright_ignore_radius", str(bright_ignore_radius),
        "--teacher_spike_radius", str(teacher_spike_radius),
        "--teacher_spike_width", str(teacher_spike_width),
    ]
    if euclid_dir:
        cmd += ["--euclid_dir", euclid_dir]
    if extra_labels:
        cmd += ["--extra_labels", extra_labels]
    if teacher_labels:
        cmd += ["--teacher_labels", teacher_labels]
    if init_checkpoint:
        cmd += ["--init_checkpoint", init_checkpoint]
    if unfreeze_stems:
        cmd += ["--unfreeze_stems"]
    if no_teacher_add_labels:
        cmd += ["--no_teacher_add_labels"]
    if wandb_project:
        cmd += ["--wandb_project", wandb_project]
    if wandb_name:
        cmd += ["--wandb_run", wandb_name]
    if device is not None and not use_ddp:
        cmd += ["--device", str(device)]

    print(f'\n{"="*60}')
    print(f'Running: {" ".join(cmd)}')
    print(f'{"="*60}\n')
    subprocess.run(cmd, check=True)


def _batchify_item(item: dict, device: torch.device) -> tuple[dict, dict]:
    images = {b: v.unsqueeze(0).to(device) for b, v in item["images"].items()}
    rms = {b: v.unsqueeze(0).to(device) for b, v in item["rms"].items()}
    return images, rms


def _refine_labels(
    encoder_ckpt: str,
    detector_ckpt: str,
    rubin_dir: str,
    euclid_dir: str,
    nsig: float,
    extra_labels: str = None,
    promote_conf: float = 0.8,
    demote_conf: float = 0.3,
    match_radius: float = 0.01,
    promotion_spike_radius: int = 20,
    promotion_spike_width: float = 3.0,
    teacher_labels: str = None,
    teacher_match_radius: float = 0.012,
    require_teacher_for_promotion: bool = False,
    device: torch.device = None,
) -> tuple[dict, dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    foundation = load_foundation_from_checkpoint(encoder_ckpt, device=device)
    model = StemCenterNetDetector.load(detector_ckpt, foundation, device=device).eval()

    ds = TileDetectionDataset(
        rubin_dir=rubin_dir,
        euclid_dir=euclid_dir,
        nsig=nsig,
        max_sources=1000,
        use_all_bands=(euclid_dir is not None),
        augment=False,
    )
    _apply_extra_labels_inplace(ds, extra_labels)
    teacher = _load_teacher_label_dict(teacher_labels) if teacher_labels else {}

    promoted = {}
    demoted = {}
    total_promoted = 0
    total_demoted = 0
    total_artifact_vetoed = 0
    total_teacher_vetoed = 0

    for idx in range(len(ds)):
        item = ds[idx]
        images, rms = _batchify_item(item, device)
        tile_id = item["tile_id"]
        orig_labels = item["centroids"].cpu().numpy()
        teacher_xy = _teacher_for_tile(teacher, tile_id) if teacher else np.zeros((0, 2), dtype=np.float32)
        promotion_mask = None
        if promotion_spike_radius > 0 and "euclid_VIS" in item["images"]:
            vis_img = item["images"]["euclid_VIS"][0].cpu().numpy()
            _, _, promotion_mask = _vis_bright_core_and_spike_mask(
                vis_img,
                spike_radius=promotion_spike_radius,
                spike_width=promotion_spike_width,
                include_core=False,
            )

        with torch.no_grad():
            out = model(images, rms)

        hm = out["heatmap"][0, 0].detach().cpu().numpy()
        fH, fW = hm.shape

        if orig_labels.shape[0] > 0:
            fx = np.clip(np.round(orig_labels[:, 0] * (fW - 1)).astype(int), 0, fW - 1)
            fy = np.clip(np.round(orig_labels[:, 1] * (fH - 1)).astype(int), 0, fH - 1)
            confs = hm[fy, fx]
            demote_mask = confs < demote_conf
            if demote_mask.any():
                demoted[tile_id] = orig_labels[demote_mask]
                total_demoted += int(demote_mask.sum())

        hm_t = out["heatmap"]
        hm_max = F.max_pool2d(hm_t, 3, stride=1, padding=1)
        peaks = (hm_t == hm_max) & (hm_t > promote_conf)
        if peaks.any():
            yi, xi = torch.where(peaks[0, 0])
            off = out["offset"][0]
            dx = off[0, yi, xi]
            dy = off[1, yi, xi]
            pred_x = (xi.float() + dx) / max(fW - 1, 1)
            pred_y = (yi.float() + dy) / max(fH - 1, 1)
            pred_xy = torch.stack([pred_x, pred_y], dim=1).cpu().numpy()

            if promotion_mask is not None and len(pred_xy) > 0:
                if promotion_mask.shape != (fH, fW):
                    pm = torch.from_numpy(promotion_mask.astype(np.float32))
                    pm = pm.unsqueeze(0).unsqueeze(0)
                    pm = F.interpolate(pm, size=(fH, fW), mode="nearest")
                    promotion_mask = pm[0, 0].cpu().numpy() > 0
                px = np.clip(np.round(pred_xy[:, 0] * (fW - 1)).astype(int), 0, fW - 1)
                py = np.clip(np.round(pred_xy[:, 1] * (fH - 1)).astype(int), 0, fH - 1)
                keep = ~promotion_mask[py, px]
                total_artifact_vetoed += int((~keep).sum())
                pred_xy = pred_xy[keep]

            if require_teacher_for_promotion and len(pred_xy) > 0:
                keep = _points_near(pred_xy, teacher_xy, teacher_match_radius)
                total_teacher_vetoed += int((~keep).sum())
                pred_xy = pred_xy[keep]

            if orig_labels.shape[0] == 0:
                novel = pred_xy
            else:
                from scipy.spatial.distance import cdist
                d = cdist(pred_xy, orig_labels)
                novel = pred_xy[d.min(axis=1) > match_radius]

            if len(novel) > 0:
                promoted[tile_id] = novel.astype(np.float32)
                total_promoted += len(novel)

    print(f"\nLabel refinement across {len(ds)} tiles:")
    print(f"  Promoted: {total_promoted} novel high-confidence detections")
    print(f"  Demoted:  {total_demoted} low-confidence pseudo-labels")
    if promotion_spike_radius > 0:
        print(
            f"  Artifact-vetoed: {total_artifact_vetoed} high-confidence peaks "
            f"on thin bright-star spike masks "
            f"(radius={promotion_spike_radius}px, width={promotion_spike_width:.1f}px)"
        )
    if require_teacher_for_promotion:
        print(
            f"  Teacher-vetoed:  {total_teacher_vetoed} high-confidence peaks "
            f"outside fused CenterNet proposals (radius={teacher_match_radius})"
        )
    return promoted, demoted


def main():
    p = argparse.ArgumentParser(description="Self-training loop for the stem-based detector.")
    p.add_argument("--encoder_ckpt", required=True, help="Foundation v8/v9/v10 checkpoint")
    p.add_argument("--rubin_dir", required=True)
    p.add_argument("--euclid_dir", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--start_round", type=int, default=1,
                   help="Round number to start from (use 2 to continue from an existing round-1 checkpoint)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--nsig", type=float, default=3.0)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--stream_ch", type=int, default=16)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--promote_conf", type=float, default=0.8)
    p.add_argument("--demote_conf", type=float, default=0.3)
    p.add_argument("--match_radius", type=float, default=0.01)
    p.add_argument("--promotion_spike_radius", type=int, default=20,
                   help="Bright-star spike search radius for promoting novel detections; set 0 to disable")
    p.add_argument("--promotion_spike_width", type=float, default=3.0,
                   help="Thin spike veto half-width in VIS pixels for self-training promotion")
    p.add_argument("--teacher_labels", default=None,
                   help="Fused CenterNet label .pt file used as proposal teacher for StemCenterNet")
    p.add_argument("--teacher_match_radius", type=float, default=0.012,
                   help="Normalized radius for matching stem labels/promotions to teacher proposals")
    p.add_argument("--teacher_filter_mode", default="bright", choices=["none", "bright", "global"],
                   help="Where training labels require teacher support: none, bright-star zones, or globally")
    p.add_argument("--no_teacher_add_labels", action="store_true",
                   help="Do not merge teacher detections into stem training labels")
    p.add_argument("--require_teacher_for_promotion", action="store_true",
                   help="Only promote stem detections that are near fused CenterNet teacher proposals")
    p.add_argument("--bright_ignore_radius", type=int, default=0,
                   help="VIS-pixel radius around bright cores to ignore negative loss except on thin spikes")
    p.add_argument("--teacher_spike_radius", type=int, default=40,
                   help="VIS-pixel radial search length for thin spike hard-negative cleaning")
    p.add_argument("--teacher_spike_width", type=float, default=3.0,
                   help="VIS-pixel half-width for thin spike hard-negative cleaning")
    p.add_argument("--unfreeze_stems", action="store_true")
    p.add_argument("--init_checkpoint", default=None,
                   help="Existing stem-detector checkpoint to bootstrap from when start_round > 1")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--device", default="")
    p.add_argument("--ddp_nproc", type=int, default=0,
                   help="If >1, launch each training round with torchrun using this many processes.")
    args = p.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.start_round < 1 or args.start_round > args.rounds:
        raise ValueError(f"--start_round must be between 1 and --rounds ({args.rounds})")
    if args.start_round > 1 and not args.init_checkpoint:
        raise ValueError("--init_checkpoint is required when --start_round > 1")

    extra_labels_path = None
    init_checkpoint = args.init_checkpoint

    if args.start_round > 1:
        prior_round = args.start_round - 1
        print(f"\n--- Bootstrapping round {args.start_round} from {init_checkpoint} ---")
        promoted, demoted = _refine_labels(
            encoder_ckpt=args.encoder_ckpt,
            detector_ckpt=init_checkpoint,
            rubin_dir=args.rubin_dir,
            euclid_dir=args.euclid_dir,
            nsig=args.nsig,
            extra_labels=None,
            promote_conf=args.promote_conf,
            demote_conf=args.demote_conf,
            match_radius=args.match_radius,
            promotion_spike_radius=args.promotion_spike_radius,
            promotion_spike_width=args.promotion_spike_width,
            teacher_labels=args.teacher_labels,
            teacher_match_radius=args.teacher_match_radius,
            require_teacher_for_promotion=args.require_teacher_for_promotion,
            device=device,
        )
        extra_labels_path = str(out_dir / f"refined_labels_round{prior_round}.pt")
        torch.save({"promoted": promoted, "demoted": demoted}, extra_labels_path)
        print(f"Saved refined labels to {extra_labels_path}")

    ckpt_path = None
    for round_num in range(args.start_round, args.rounds + 1):
        print(f'\n{"#"*60}')
        print(f'# STEM SELF-TRAINING ROUND {round_num}/{args.rounds}')
        print(f'{"#"*60}')

        ckpt_path = str(out_dir / f"stem_centernet_round{round_num}.pt")
        _run_training_round(
            encoder_ckpt=args.encoder_ckpt,
            rubin_dir=args.rubin_dir,
            euclid_dir=args.euclid_dir,
            out_path=ckpt_path,
            extra_labels=extra_labels_path,
            init_checkpoint=init_checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            sigma=args.sigma,
            nsig=args.nsig,
            stream_ch=args.stream_ch,
            base_ch=args.base_ch,
            unfreeze_stems=args.unfreeze_stems,
            teacher_labels=args.teacher_labels,
            teacher_match_radius=args.teacher_match_radius,
            teacher_filter_mode=args.teacher_filter_mode,
            no_teacher_add_labels=args.no_teacher_add_labels,
            bright_ignore_radius=args.bright_ignore_radius,
            teacher_spike_radius=args.teacher_spike_radius,
            teacher_spike_width=args.teacher_spike_width,
            viz_spike_veto_radius=args.promotion_spike_radius,
            viz_spike_veto_width=args.promotion_spike_width,
            device=device,
            wandb_project=args.wandb_project,
            wandb_name=f"stem-round{round_num}",
            ddp_nproc=args.ddp_nproc,
        )
        init_checkpoint = None

        if round_num < args.rounds:
            print(f'\n--- Refining labels (promote > {args.promote_conf}, demote < {args.demote_conf}) ---')
            promoted, demoted = _refine_labels(
                encoder_ckpt=args.encoder_ckpt,
                detector_ckpt=ckpt_path,
                rubin_dir=args.rubin_dir,
                euclid_dir=args.euclid_dir,
                nsig=args.nsig,
                extra_labels=extra_labels_path,
                promote_conf=args.promote_conf,
                demote_conf=args.demote_conf,
                match_radius=args.match_radius,
                promotion_spike_radius=args.promotion_spike_radius,
                promotion_spike_width=args.promotion_spike_width,
                teacher_labels=args.teacher_labels,
                teacher_match_radius=args.teacher_match_radius,
                require_teacher_for_promotion=args.require_teacher_for_promotion,
                device=device,
            )
            extra_labels_path = str(out_dir / f"refined_labels_round{round_num}.pt")
            torch.save({"promoted": promoted, "demoted": demoted}, extra_labels_path)
            print(f"Saved refined labels to {extra_labels_path}")

    final = out_dir / "stem_centernet_best.pt"
    shutil.copy2(ckpt_path, final)
    print(f"\nDone. Final checkpoint: {final}")


if __name__ == "__main__":
    main()
