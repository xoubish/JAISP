"""Architecture diagram of FluxCNN, drawn from the *current* config so it never goes
stale (input size from stamp/bin_factor, pooling branches from pool, head width from
head_hidden, target from flux_column). Mirrors model.FluxCNN: stride-2 5x5 stem to 16
channels, then three [Conv3x3+GN+GELU]x2 + AvgPool/2 blocks (32, 64, 128 channels).
"""
import numpy as np


def _dims(cfg):
    S = cfg.stamp // cfg.bin_factor
    sizes = [S // 2, S // 4, S // 8, S // 16]          # stem s2, then 3 avgpools
    chans = [16, 32, 64, 128]
    return S, list(zip(chans, sizes))


def _branches(pool):
    return {"avg": ["avg"], "max": ["max"], "avgmax": ["avg", "max"], "sum": ["sum"],
            "gatedsum": ["gated Σ"], "mix": ["avg", "max", "sum", "gated Σ"]}[pool]


def draw_architecture(cfg, stamp=None, path=None):
    if getattr(cfg, "model_version", 1) == 2:
        return draw_v2(cfg, path)
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle

    S, stages = _dims(cfg)
    br = _branches(cfg.pool)
    cf = stages[-1][0]
    head_in, H = cf * len(br), cfg.head_hidden
    pix = cfg.native_pixscale * cfg.bin_factor

    TAN, TAN_E = "#c9b79c", "#6f6047"
    POOL, POOL_E = "#cfe2f3", "#3b6ea5"
    CONCAT = "#b7a17f"
    OUT, OUT_E = "#f3c48a", "#d68a2e"
    NODE = ["#a9d0ec", "#5a9bd4", "#1f5fa6"]

    fig, ax = plt.subplots(figsize=(19, 8.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(50, 96.5, f"FluxCNN  —  VIS flux regressor   (label: {cfg.flux_column})",
            ha="center", va="center", fontsize=20, fontweight="bold")

    def stack(x, cy, w, h, n, dx=0.9, dy=1.2):
        for i in range(n - 1, -1, -1):
            ax.add_patch(Rectangle((x + i * dx, cy - h / 2 + i * dy), w, h,
                         facecolor=TAN, edgecolor=TAN_E, lw=1.2, zorder=n - i))
        return x + (n - 1) * dx + w

    def arrow(x0, y0, x1, y1, ls="-", lw=1.4, color="0.3", rad=0.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                     lw=lw, color=color, linestyle=ls, zorder=6, connectionstyle=f"arc3,rad={rad}"))

    CY = 60
    # ---- input ----
    ix, iw = 2.0, 11.0
    if stamp is not None:
        im = np.asarray(stamp, dtype=np.float32)
        if cfg.bin_factor > 1:
            f = cfg.bin_factor; h, w = im.shape
            im = im[:h // f * f, :w // f * f].reshape(h // f, f, w // f, f).sum((1, 3))
        ax.imshow(np.arcsinh(im / max(np.std(im), 1e-3)), extent=(ix, ix + iw, CY - 6, CY + 6),
                  cmap="gray", origin="lower", zorder=5, aspect="auto")
    else:
        ax.add_patch(Rectangle((ix, CY - 6), iw, 12, facecolor="0.1", zorder=5))
    ax.add_patch(Rectangle((ix, CY - 6), iw, 12, fill=False, edgecolor="k", lw=1.6, zorder=6))
    for i, d in enumerate((2.4, 1.2)):
        ax.add_patch(Rectangle((ix + d, CY - 6 + d * 0.9), iw, 12, fill=False, edgecolor="0.5", lw=1.0, zorder=3 - i))
    ax.text(ix + iw / 2 + 1.5, CY + 9.4, "VIS stamp", ha="center", fontsize=13, fontweight="bold")
    ax.text(ix + iw / 2 + 1.5, CY - 9.0, f"3 x {S} x {S}  @{pix:.1f}\"/px", ha="center", fontsize=11)
    ax.text(ix + iw / 2 + 1.5, CY - 12.0, "image · SNR · centre-prior", ha="center", fontsize=9.5, color="0.3")

    # ---- conv stack ----
    ops = ["Conv 5x5, s2", "Conv x2 + AvgPool", "Conv x2 + AvgPool", "Conv x2 + AvgPool"]
    xs, hss, ns = [20.0, 31.5, 42.0, 52.0], [9.0, 7.6, 6.2, 5.0], [4, 5, 7, 9]
    prev = ix + iw + 5.0
    for (ch, sp), x, hs, n, op in zip(stages, xs, hss, ns, ops):
        arrow(prev, CY, x - 0.5, CY, ls=(0, (4, 3)), lw=1.1, color="0.45")
        prev = stack(x, CY, hs * 1.15, hs * 2, n)
        ax.text(x + hs * 0.6, CY + hs + 5.5, op, ha="center", fontsize=10.5)
        ax.text(x + hs * 0.6, CY - hs - 5.5, f"{ch} x {sp} x {sp}", ha="center", fontsize=10, color="0.15")

    # ---- global pooling branches ----
    px = 66.5
    py = np.linspace(CY + 13.5, CY - 13.5, len(br)) if len(br) > 1 else [CY]
    for lab, y in zip(br, py):
        arrow(prev + 0.5, CY, px - 0.3, y, lw=1.2, color="0.35", rad=(y - CY) * 0.01)
        ax.add_patch(FancyBboxPatch((px, y - 3), 8.6, 6, boxstyle="round,pad=0.15,rounding_size=0.8",
                     facecolor=POOL, edgecolor=POOL_E, lw=1.4, zorder=6))
        ax.text(px + 4.3, y + 0.6, lab, ha="center", va="center", fontsize=12, fontweight="bold", zorder=10)
        ax.text(px + 4.3, y - 1.9, f"→ {cf}", ha="center", va="center", fontsize=8.5, color="0.3", zorder=10)
    cx = 80.5
    ax.add_patch(Rectangle((cx, CY - 16), 2.6, 32, facecolor=CONCAT, edgecolor="#5f533c", lw=1.3, zorder=6))
    for y in py:
        arrow(px + 8.6, y, cx - 0.2, y, lw=1.1, color="0.35")
    ax.text(cx + 1.3, CY + 18.7, "concat" if len(br) > 1 else "vector", ha="center", fontsize=11, fontweight="bold")
    ax.text(cx + 1.3, CY - 18.6, f"{head_in}", ha="center", fontsize=10)

    # ---- MLP head ----
    prevc = [(cx + 2.6, y) for y in np.linspace(CY + 13, CY - 13, 9)]
    for j, xc in enumerate([86.5, 90.5]):
        c = [(xc, y) for y in np.linspace(CY + 13, CY - 13, 8)]
        for (x, y) in c:
            ax.add_patch(Circle((x, y), 1.05, facecolor=NODE[j], edgecolor="w", lw=0.8, zorder=7))
        for (x0, y0) in prevc:
            for (x1, y1) in c:
                ax.plot([x0, x1], [y0, y1], color="0.75", lw=0.25, zorder=4)
        prevc = c
    on = (94.2, CY)
    ax.add_patch(Circle(on, 1.5, facecolor=NODE[2], edgecolor="w", lw=1, zorder=8))
    for (x0, y0) in prevc:
        ax.plot([x0, on[0]], [y0, on[1]], color="0.7", lw=0.3, zorder=4)
    ax.text(88.5, CY + 16.5, f"{head_in} → {H} → {H} → 1", ha="center", fontsize=10)

    # ---- output ----
    arrow(on[0] + 1.5, CY, 96.0, CY, lw=1.6, color="0.25")
    ax.add_patch(FancyBboxPatch((96.2, CY - 5.5), 3.6, 11, boxstyle="round,pad=0.15,rounding_size=0.8",
                 facecolor=OUT, edgecolor=OUT_E, lw=1.6, zorder=6))
    ax.text(98.0, CY + 2.6, "flux", ha="center", fontsize=12, fontweight="bold", zorder=10)
    ax.text(98.0, CY - 0.2, "[µJy]", ha="center", fontsize=10, zorder=10)
    ax.text(98.0, CY - 3.3, "exp(zσ+μ)" if getattr(cfg, "loss", "flux") == "mag" else "F0·sinh(zσ+μ)", ha="center", fontsize=7.2, color="0.25", zorder=10)

    def bracket(x0, x1, y, label, sub):
        ax.plot([x0, x0, x1, x1], [y + 1.2, y, y, y + 1.2], color="0.35", lw=1.3)
        ax.text((x0 + x1) / 2, y - 2.6, label, ha="center", fontsize=12.5, fontweight="bold")
        ax.text((x0 + x1) / 2, y - 5.2, sub, ha="center", fontsize=9.5, color="0.35")
    bracket(18, 63, 30, "Feature extraction", "conv + GroupNorm + GELU, AvgPool /2")
    bracket(65.5, 83.2, 30, f"Global pooling  ({cfg.pool})",
            "operators co-exist → concatenated" if len(br) > 1 else "single operator")
    bracket(84.5, 99.8, 30, "Regression head", "MLP  —  single value, no softmax")
    if getattr(cfg, "loss", "flux") == "mag":
        core = "dmag / MER sigma_mag" if cfg.mag_weighted else "dmag"
        loss_txt = (f"mean |{core}|^{cfg.mag_loss_power:g}" if getattr(cfg, "mag_loss_shape", "huber") == "power"
                    else f"Huber on {core}")
    else:
        loss_txt = "uncertainty-weighted Huber in flux"
    ax.text(50, 15.5, f"Target: {cfg.flux_column} (error: {cfg.fluxerr_column}).  Loss: {loss_txt}.  "
            f"Balanced sampling: {cfg.balanced_sampling}.  Dropout {cfg.dropout}, weight decay {cfg.weight_decay}, augment {cfg.augment}.",
            ha="center", fontsize=9.3, color="0.35")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def draw_v2(cfg, path=None):
    """Show the explicit native-flux bypass used by version 2."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set(xlim=(0, 13), ylim=(0, 5)); ax.axis("off")
    def box(x, y, text, color):
        ax.add_patch(FancyBboxPatch((x, y), 2.6, 1., boxstyle="round,pad=.1", facecolor=color))
        ax.text(x + 1.3, y + .5, text, ha="center", va="center", fontsize=10)
    def arrow(a, b):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.5))
    box(.2, 2, "VIS image + variance\nvalid mask + annulus sky", "#dde8ef")
    box(3.5, 3.2, f"Native aperture sums\n{cfg.aperture_radii_arcsec} arcsec", "#f0d6aa")
    box(3.5, .7, f"4-channel image context\nCNN + {cfg.pool} pooling", "#dde8ef")
    box(6.8, 3.2, "Linear flux calibration\nfitted on training only", "#f0d6aa")
    box(6.8, .7, "CNN + aperture features\nMLP correction (starts at 0)", "#dde8ef")
    box(10.1, 2, "baseline z + correction\n→ catalog flux", "#cde6c7")
    for a,b in [((2.8,2.7),(3.5,3.7)),((2.8,2.3),(3.5,1.2)),((6.1,3.7),(6.8,3.7)),
                ((6.1,1.2),(6.8,1.2)),((5.5,3.2),(7.3,1.7)),((9.4,3.7),(10.1,2.7)),
                ((9.4,1.2),(10.1,2.3))]: arrow(a,b)
    ax.set_title("Supervised aperture + CNN photometry — no foundation features")
    fig.tight_layout()
    if path: fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
