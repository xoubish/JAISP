from typing import Dict, Tuple

import torch


def _clamp_int(v: int, low: int, high: int) -> int:
    return max(low, min(high, v))


def _make_rect_mask(
    shape: Tuple[int, int],
    y0: int,
    x0: int,
    h: int,
    w: int,
    device: torch.device,
) -> torch.Tensor:
    height, width = int(shape[0]), int(shape[1])
    mask = torch.zeros(1, height, width, device=device, dtype=torch.float32)
    y1 = _clamp_int(y0 + h, 0, height)
    x1 = _clamp_int(x0 + w, 0, width)
    y0 = _clamp_int(y0, 0, height)
    x0 = _clamp_int(x0, 0, width)
    if y1 > y0 and x1 > x0:
        mask[:, y0:y1, x0:x1] = 1.0
    return mask


def random_mask(
    image: torch.Tensor,
    min_frac: float = 0.10,
    max_frac: float = 0.25,
) -> torch.Tensor:
    """Random rectangle mask."""
    _, height, width = image.shape
    device = image.device

    frac_h = torch.empty(1).uniform_(min_frac, max_frac).item()
    frac_w = torch.empty(1).uniform_(min_frac, max_frac).item()
    mh = max(8, int(height * frac_h))
    mw = max(8, int(width * frac_w))

    y0 = int(torch.randint(0, max(1, height - mh + 1), (1,), device=device).item())
    x0 = int(torch.randint(0, max(1, width - mw + 1), (1,), device=device).item())
    return _make_rect_mask((height, width), y0, x0, mh, mw, device)


def hard_mask(
    image: torch.Tensor,
    min_frac: float = 0.22,
    max_frac: float = 0.40,
) -> torch.Tensor:
    """Large or edge-touching random rectangle mask."""
    _, height, width = image.shape
    device = image.device

    frac_h = torch.empty(1).uniform_(min_frac, max_frac).item()
    frac_w = torch.empty(1).uniform_(min_frac, max_frac).item()
    mh = max(16, int(height * frac_h))
    mw = max(16, int(width * frac_w))

    edge_mode = int(torch.randint(0, 4, (1,), device=device).item())
    if edge_mode == 0:  # top edge
        y0 = 0
        x0 = int(torch.randint(0, max(1, width - mw + 1), (1,), device=device).item())
    elif edge_mode == 1:  # bottom edge
        y0 = max(0, height - mh)
        x0 = int(torch.randint(0, max(1, width - mw + 1), (1,), device=device).item())
    elif edge_mode == 2:  # left edge
        y0 = int(torch.randint(0, max(1, height - mh + 1), (1,), device=device).item())
        x0 = 0
    else:  # right edge
        y0 = int(torch.randint(0, max(1, height - mh + 1), (1,), device=device).item())
        x0 = max(0, width - mw)

    return _make_rect_mask((height, width), y0, x0, mh, mw, device)


def object_mask(
    image: torch.Tensor,
    min_frac: float = 0.10,
    max_frac: float = 0.20,
    snr_threshold: float = 2.5,
) -> torch.Tensor:
    """
    Source-aware mask via bright-pixel centering.
    Falls back to random mask when no strong candidate exists.
    """
    _, height, width = image.shape
    device = image.device

    img = image[0]
    finite = torch.isfinite(img)
    if finite.sum() < 32:
        return random_mask(image, min_frac=min_frac, max_frac=max_frac)

    vals = img[finite]
    med = torch.median(vals)
    std = vals.std(unbiased=False).clamp(min=1e-6)

    candidate = (img > med + snr_threshold * std) & finite
    ys, xs = torch.where(candidate)
    if ys.numel() == 0:
        return random_mask(image, min_frac=min_frac, max_frac=max_frac)

    pick = int(torch.randint(0, ys.numel(), (1,), device=device).item())
    cy, cx = int(ys[pick].item()), int(xs[pick].item())

    frac_h = torch.empty(1).uniform_(min_frac, max_frac).item()
    frac_w = torch.empty(1).uniform_(min_frac, max_frac).item()
    mh = max(8, int(height * frac_h))
    mw = max(8, int(width * frac_w))

    y0 = _clamp_int(cy - mh // 2, 0, max(0, height - mh))
    x0 = _clamp_int(cx - mw // 2, 0, max(0, width - mw))
    return _make_rect_mask((height, width), y0, x0, mh, mw, device)


def build_mask(
    image: torch.Tensor,
    probs: Dict[str, float],
) -> Tuple[torch.Tensor, str]:
    """
    Build a mask from mixed strategies.

    Args:
        image: [1,H,W]
        probs: e.g. {"random":0.5, "object":0.4, "hard":0.1}
    """
    p_random = float(probs.get("random", 0.5))
    p_object = float(probs.get("object", 0.4))
    p_hard = float(probs.get("hard", 0.1))

    total = p_random + p_object + p_hard
    if total <= 0:
        p_random, p_object, p_hard = 1.0, 0.0, 0.0
        total = 1.0

    p_random /= total
    p_object /= total
    p_hard /= total

    r = float(torch.rand(1).item())
    if r < p_random:
        return random_mask(image), "random"
    if r < p_random + p_object:
        return object_mask(image), "object"
    return hard_mask(image), "hard"
