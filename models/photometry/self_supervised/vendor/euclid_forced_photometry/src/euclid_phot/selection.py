"""Chi^2 decision-tree model selection for Euclid VIS forced photometry.

An opt-in alternative to the MER-column-driven model construction in
``models.py``, porting the decision tree of The Farmer (Weaver et al. 2023,
ApJS 269, 20; Figures 3-4) onto the Tractor machinery used here.

``sufficient_thresh`` = 1.5 is the paper's chi^2_N bad-fit criterion;
``simplegalaxy_radius`` (0.30 arcsec), ``simplegalaxy_penalty`` (0.05) and
``exp_dev_similar_thresh`` (0.15) are Euclid tunings of the Farmer values
(0.45, 0.1, 0.2).

Deviations from Farmer: positions are frozen during the tree, with a hard
``max_pos_shift_arcsec`` clamp at the final joint re-fit; every source takes
the full ladder with the lowest chi-squared winning (``full_ladder``; the
published first-sufficient shortcut biases faint fluxes low); and a guarded
SersicGalaxy 6th tier (``include_sersic_tier``) extends the published
five-model ladder.

Public API: ``ModelSelector``, ``detect_blobs``, ``assign_mer_to_blobs``,
``run_model_selection``, ``reproduce_figure3``.
"""

from __future__ import annotations

import copy
import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
from tractor import (
    Image,
    NanoMaggies,
    PointSource,
    RaDecPos,
    Tractor,
)
from tractor.constrained_optimizer import ConstrainedOptimizer
from tractor.ellipses import EllipseE, EllipseESoft
from tractor.galaxy import (
    DevGalaxy,
    ExpGalaxy,
    FixedCompositeGalaxy,
    SoftenedFracDev,
)
from tractor.sersic import SersicGalaxy, SersicIndex

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Euclid-tuned defaults (see module docstring for rationale).
# ---------------------------------------------------------------------------
EUCLID_DEFAULTS = dict(
    sufficient_thresh=1.5,
    simplegalaxy_penalty=0.05,
    exp_dev_similar_thresh=0.15,
    simplegalaxy_radius=0.30,
    # Max centroid drift in the final joint re-fit, where position is thawed
    # to absorb sub-pixel MER-vs-VIS astrometry offsets. ~1 VIS pixel.
    max_pos_shift_arcsec=0.30,
    sep_threshold_sigma=1.5,
    sep_minarea=5,
    blob_dilate_pix=4,
    max_steps=50,
    dlnp_crit=1e-3,
    # A railed radius leaves a ring residual on large VIS galaxies, so use
    # the same 20 arcsec guard models.py applies to MER-seeded shapes.
    max_re_arcsec=20.0,
    # Every tier evaluated, lowest reduced chi-squared wins; the published
    # first-sufficient shortcut biases faint fluxes low.
    full_ladder=True,
    full_ladder_min_snr=0.0,
    per_source_psf=False,
    # Guarded SersicGalaxy 6th tier (extension beyond published Farmer);
    # on by default: accepted only when it lowers the total chi-squared on
    # the wing-wide window, both models scored there, by more than
    # sersic_dchi2_accept (a likelihood-ratio test for the one extra shape
    # parameter; 25 is a ~5 sigma preference).
    include_sersic_tier=True,
    sersic_min_snr=20.0,
    sersic_dchi2_accept=25.0,
    sersic_init_n=2.5,
)

MODEL_NAMES = ("PointSource", "SimpleGalaxy", "ExpGalaxy", "DevGalaxy",
               "CompositeGalaxy", "SersicGalaxy")


# ---------------------------------------------------------------------------
# SimpleGalaxy: Farmer's bridge tier, adapted to Tractor's standard
# parameter-freezing API.
# ---------------------------------------------------------------------------
class SimpleGalaxy(ExpGalaxy):
    """Exponential profile with a frozen circular shape at a fixed radius.

    Used as the "marginally resolved" bridge tier between PointSource and the
    free Exp/Dev fits. The frozen circular radius is set per instance via the
    ``radius_arcsec`` argument (the selector passes its ``simplegalaxy_radius``);
    when omitted it falls back to the Euclid-tuned class default (0.30 arcsec).

    The shape parameter is frozen via Tractor's standard ``freezeParam`` API
    so ``numberOfParams()`` and ``getDerivs()`` stay consistent under
    ``optimize_forced_photometry`` (Farmer's original ``isParamFrozen``
    override is incompatible with that path).
    """

    DEFAULT_RADIUS = EUCLID_DEFAULTS["simplegalaxy_radius"]

    def __init__(self, pos, brightness, radius_arcsec=None):
        r = SimpleGalaxy.DEFAULT_RADIUS if radius_arcsec is None else float(radius_arcsec)
        shape = EllipseE(r, 0.0, 0.0)
        super().__init__(pos, brightness, shape)
        self.freezeParam("shape")

    def getName(self):
        return "SimpleGalaxy"


def set_simplegalaxy_radius(radius_arcsec: float) -> None:
    """Set the class-default SimpleGalaxy shape radius (arcsec).

    Only affects instances built with no explicit ``radius_arcsec``; the
    fit path passes the selector's ``simplegalaxy_radius`` explicitly.
    """
    SimpleGalaxy.DEFAULT_RADIUS = float(radius_arcsec)


# ---------------------------------------------------------------------------
# Blob and History dataclasses.
# ---------------------------------------------------------------------------
@dataclass
class Blob:
    """A connected group of pixels and the MER sources that fall inside it.

    ``footprint`` is the dilated boolean mask over the VIS cutout that defines
    where chi^2 is evaluated for this blob (the rest of the image gets zero
    invvar). ``segments`` maps each member source's MER row index to its own
    per-source segment mask, used for the reduced chi^2 statistic.
    """

    blob_id: int
    footprint: np.ndarray
    member_mer_indices: list[int]
    segments: dict[int, np.ndarray] = field(default_factory=dict)
    sep_rows: dict[int, dict] = field(default_factory=dict)


@dataclass
class StageRecord:
    """Per-tier snapshot for the demo figure (history=True only)."""

    stage: int
    model_name: str
    source_idx: int
    chi2_red: float
    model_image: np.ndarray
    residual_image: np.ndarray
    accepted: bool = False
    flux_ujy: float = float("nan")
    chi2_wide: float = float("nan")


@dataclass
class FinalRecord:
    """Final joint-fit snapshot, shown on the top row of the demo figure."""

    data: np.ndarray
    model: np.ndarray
    residual: np.ndarray
    footprint: np.ndarray


@dataclass
class History:
    blob_id: int
    members: list[int]
    stages: list[StageRecord] = field(default_factory=list)
    final: FinalRecord | None = None


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _seed_shape_from_sep(sep_row: dict, pixscale_arcsec: float,
                         max_re_arcsec: float | None = None) -> EllipseESoft:
    """Build an initial Tractor shape from a SEP detection row.

    Mirrors Farmer's seeding; the radius cap defaults to
    ``EUCLID_DEFAULTS['max_re_arcsec']``.
    """
    a = max(float(sep_row["a"]), 0.5)
    b = max(float(sep_row["b"]), 0.5)
    theta = float(sep_row["theta"])

    r_eff = float(np.sqrt(a * b) * pixscale_arcsec)
    axis_ratio = float(b / a)
    pa_deg = 90.0 - float(np.rad2deg(theta))

    if max_re_arcsec is None:
        max_re_arcsec = EUCLID_DEFAULTS["max_re_arcsec"]
    # Per-source ceiling: without it, sources too faint to constrain a
    # shape rail at the global cap as razor-thin 20-arcsec needles. 5x the
    # SEP seed allows genuine growth; 2.72 arcsec stays the floor.
    per_source_cap = float(np.clip(5.0 * r_eff, 2.72, max_re_arcsec))
    shape = EllipseESoft.fromRAbPhi(r_eff, axis_ratio, pa_deg)
    shape.lowers = [-5.0, -np.inf, -np.inf]
    shape.uppers = [float(np.log(per_source_cap)), np.inf, np.inf]
    return shape


def _make_masked_image(tim: Image, footprint: np.ndarray, psf=None,
                       bbox=None) -> Image:
    """Clone a Tractor Image with invvar zeroed outside ``footprint``.

    The original ``tim`` is untouched. When ``psf`` is given, the clone uses
    that PSF instead of ``tim``'s (lets the caller swap in a per-blob /
    per-source CATALOG-PSF stamp). WCS/photocal/sky are still shared.

    ``bbox`` (y0, y1, x0, x1) crops the clone to that pixel window with a
    correspondingly shifted WCS; the likelihood is identical (zero weight
    outside the footprint) but rendering scales with the window area.
    """
    # Clone the PSF: PixelizedPSF lazily caches an FFT on the instance, and
    # two worker threads racing on a cold cache can return half-initialized
    # arrays.
    import copy as _copy
    psf_obj = psf if psf is not None else _copy.deepcopy(tim.getPsf())
    if bbox is None:
        invvar = tim.getInvvar().copy()
        invvar[~footprint] = 0.0
        data = tim.getImage()
        wcs = tim.getWcs()
    else:
        y0, y1, x0, x1 = bbox
        invvar = tim.getInvvar()[y0:y1, x0:x1].copy()
        invvar[~footprint[y0:y1, x0:x1]] = 0.0
        data = tim.getImage()[y0:y1, x0:x1]
        wcs = tim.getWcs().shifted(x0, y0)
        psf_obj = psf_obj.getShifted(x0, y0)
    new_tim = Image(
        data=data,
        invvar=invvar,
        psf=psf_obj,
        wcs=wcs,
        photocal=tim.getPhotoCal(),
        sky=tim.getSky(),
        name=getattr(tim, "name", "vis") + "[blob]",
    )
    new_tim.freezeAllParams()
    return new_tim


def _blob_bbox(footprint: np.ndarray, shape, margin: int = 20):
    """Bounding box (y0, y1, x0, x1) of ``footprint`` padded by ``margin``."""
    ys, xs = np.nonzero(footprint)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(int(shape[0]), int(ys.max()) + 1 + margin)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(int(shape[1]), int(xs.max()) + 1 + margin)
    return y0, y1, x0, x1


def _count_free_params(source) -> int:
    """Number of currently-thawed Tractor parameters on a source.

    A static per-class count would overcount the dof by 1-2 in stages where
    pos is frozen, biasing the reduced chi^2.
    """
    try:
        n = int(source.numberOfParams())
    except Exception:
        n = len(source.getParams())
    return max(n, 1)


def _build_initial_source(mer_row, model_class, sep_row, pixscale_arcsec,
                          *, simplegalaxy_radius=None):
    """Construct an initial Tractor source for a given model class.

    Seeds brightness from the MER flux; the tree decides morphology.
    """
    pos = RaDecPos(float(mer_row["ra"]), float(mer_row["dec"]))
    flux_uJy = float(mer_row["flux_vis_sersic"])
    if not np.isfinite(flux_uJy) or flux_uJy <= 0:
        flux_uJy = 1.0
    brightness = NanoMaggies(**{"VIS": flux_uJy / 3.631})

    if model_class is PointSource:
        return PointSource(pos, brightness)
    if model_class is SimpleGalaxy:
        return SimpleGalaxy(pos, brightness, radius_arcsec=simplegalaxy_radius)

    if sep_row is not None:
        shape = _seed_shape_from_sep(sep_row, pixscale_arcsec)
    else:
        shape = EllipseESoft.fromRAbPhi(0.3, 1.0, 0.0)
        # Same per-source ceiling rule as the SEP-seeded path.
        shape.lowers = [-5.0, -np.inf, -np.inf]
        shape.uppers = [1.0, np.inf, np.inf]
    if model_class is ExpGalaxy:
        return ExpGalaxy(pos, brightness, shape)
    if model_class is DevGalaxy:
        return DevGalaxy(pos, brightness, shape)
    if model_class is FixedCompositeGalaxy:
        return FixedCompositeGalaxy(pos, brightness, SoftenedFracDev(0.5), shape, shape)
    if model_class is SersicGalaxy:
        return SersicGalaxy(pos, brightness, shape,
                            SersicIndex(EUCLID_DEFAULTS["sersic_init_n"]))
    raise ValueError(f"Unsupported model class: {model_class}")


def _classname(src) -> str:
    if isinstance(src, PointSource):
        return "PointSource"
    if isinstance(src, SimpleGalaxy):
        return "SimpleGalaxy"
    if isinstance(src, FixedCompositeGalaxy):
        return "CompositeGalaxy"
    if isinstance(src, SersicGalaxy):
        return "SersicGalaxy"
    if isinstance(src, DevGalaxy):
        return "DevGalaxy"
    if isinstance(src, ExpGalaxy):
        return "ExpGalaxy"
    return type(src).__name__


# ---------------------------------------------------------------------------
# ModelSelector: orchestrates the decision tree on one blob at a time.
# ---------------------------------------------------------------------------
class ModelSelector:
    """Run the chi^2 decision tree (Weaver et al. 2023) on Euclid VIS data.

    The defaults below are Euclid-tuned (see module docstring). Pass keyword
    overrides to compare against Farmer's published COSMOS-Web defaults.
    """

    def __init__(
        self,
        sufficient_thresh: float = EUCLID_DEFAULTS["sufficient_thresh"],
        simplegalaxy_penalty: float = EUCLID_DEFAULTS["simplegalaxy_penalty"],
        exp_dev_similar_thresh: float = EUCLID_DEFAULTS["exp_dev_similar_thresh"],
        simplegalaxy_radius: float = EUCLID_DEFAULTS["simplegalaxy_radius"],
        max_pos_shift_arcsec: float = EUCLID_DEFAULTS["max_pos_shift_arcsec"],
        max_steps: int = EUCLID_DEFAULTS["max_steps"],
        dlnp_crit: float = EUCLID_DEFAULTS["dlnp_crit"],
        full_ladder: bool = EUCLID_DEFAULTS["full_ladder"],
        full_ladder_min_snr: float = EUCLID_DEFAULTS["full_ladder_min_snr"],
        per_source_psf: bool = EUCLID_DEFAULTS["per_source_psf"],
        include_sersic_tier: bool = EUCLID_DEFAULTS["include_sersic_tier"],
        sersic_min_snr: float = EUCLID_DEFAULTS["sersic_min_snr"],
        sersic_dchi2_accept: float = EUCLID_DEFAULTS["sersic_dchi2_accept"],
    ):
        self.sufficient_thresh = float(sufficient_thresh)
        self.simplegalaxy_penalty = float(simplegalaxy_penalty)
        self.exp_dev_similar_thresh = float(exp_dev_similar_thresh)
        self.simplegalaxy_radius = float(simplegalaxy_radius)
        self.max_pos_shift_arcsec = float(max_pos_shift_arcsec)
        self.max_steps = int(max_steps)
        self.dlnp_crit = float(dlnp_crit)
        self.full_ladder = bool(full_ladder)
        self.full_ladder_min_snr = float(full_ladder_min_snr)
        self.per_source_psf = bool(per_source_psf)
        # 6th tier beyond the published ladder: a free-Sersic fit on bright
        # resolved sources whose accepted model leaves a core residual.
        self.include_sersic_tier = bool(include_sersic_tier)
        self.sersic_min_snr = float(sersic_min_snr)
        self.sersic_dchi2_accept = float(sersic_dchi2_accept)

        # Sync the shared SimpleGalaxy radius with this selector's value.
        set_simplegalaxy_radius(self.simplegalaxy_radius)

    # ---- inner machinery ------------------------------------------------
    def _optimize(self, tractor: Tractor) -> None:
        """Joint optimization of the active free parameters in ``tractor``.

        Failures are swallowed and logged, as in Farmer with
        IGNORE_FAILURES=True.
        """
        for _step in range(self.max_steps):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dlnp, _, _ = tractor.optimize()
            except Exception as exc:
                logger.debug("optimize() raised %s; continuing.", exc)
                return
            if dlnp < self.dlnp_crit:
                return

    def _fit_stage(self, tim_blob, sources_by_idx, free_indices, segments_by_idx,
                   eval_segments=None):
        """Build a Tractor for the blob with the given source list and fit it.

        ``free_indices`` lists which source idx's have their morphology params
        thawed (the rest stay frozen at their locked-in model). ``segments``
        is a dict mapping source idx -> per-source segmentation mask, used for
        the per-source reduced chi^2.

        ``eval_segments`` (optional dict idx -> dilated mask) additionally
        scores those members on a wider window. Full-ladder mode uses this:
        the winning tier must be the one that damages the source's
        surroundings least, not just its own segment.

        Returns ``(chi2, chi2_wide, tractor)``; ``chi2_wide`` only holds the
        members present in ``eval_segments``.
        """
        all_sources = [sources_by_idx[i] for i in sorted(sources_by_idx)]
        idx_in_order = sorted(sources_by_idx)
        tractor = Tractor([tim_blob], all_sources, optimizer=ConstrainedOptimizer())

        for idx, src in zip(idx_in_order, all_sources, strict=False):
            if idx in free_indices:
                src.thawAllParams()
                src.freezeParam("pos")  # positions stay at MER for forced phot
                if isinstance(src, SimpleGalaxy):
                    # Preserve SimpleGalaxy's frozen circular shape.
                    src.freezeParam("shape")
            else:
                # Deviation from Farmer (which keeps locked sources' fluxes
                # variable at later tiers): freeze them entirely and recover
                # the joint solution in the final all-source re-fit.
                src.freezeAllParams()

        self._optimize(tractor)

        chi_img = tractor.getChiImage(imgi=0)

        def _rchi2(mask, n_free):
            npix = int(mask.sum())
            return float(np.sum(chi_img[mask] ** 2) / max(npix - n_free, 1))

        chi2 = {}
        chi2_wide = {}
        for idx in free_indices:
            nf = _count_free_params(sources_by_idx[idx])
            chi2[idx] = _rchi2(segments_by_idx[idx], nf)
            if eval_segments and idx in eval_segments:
                chi2_wide[idx] = _rchi2(eval_segments[idx], nf)
        return chi2, chi2_wide, tractor

    # ---- public entry point --------------------------------------------
    def fit_blob(
        self,
        blob: Blob,
        tim_vis: Image,
        mer_cat,
        pixscale_arcsec: float,
        history: bool = False,
        psf_stamp: np.ndarray | None = None,
        psf_by_member: dict | None = None,
    ):
        """Walk the decision tree for every source in ``blob``.

        Returns ``(sources_dict, history_or_None)`` where ``sources_dict`` maps
        each MER row index to its chosen, fit Tractor source. ``history`` is a
        :class:`History` instance when ``history=True``, otherwise ``None``.

        ``psf_stamp`` (HxW float array) overrides ``tim_vis``'s PSF for this
        blob. This is an explicit constant-PSF approximation for all its
        members; omit it to retain the image's spatial PSF.

        ``psf_by_member`` (dict[mer_idx -> stamp]) supplies local samples
        for a spatial PSF. Each member uses its own nearest sample while
        all members remain in the joint fit. ``psf_stamp`` explicitly
        overrides this with a constant PSF when supplied.
        """
        member_psf = None
        if psf_by_member is not None and psf_stamp is None:
            from .spatial_psf import SpatialPixelizedPSF
            members = list(blob.member_mer_indices)
            member_psf = SpatialPixelizedPSF({
                'stamps': np.stack([psf_by_member[i] for i in members]),
                'ra': np.array([float(mer_cat[i]['ra']) for i in members]),
                'dec': np.array([float(mer_cat[i]['dec']) for i in members]),
            }, tim_vis.getWcs())

        # Fit on the blob bounding box (identical likelihood, much cheaper
        # rendering); the history path keeps the full frame because
        # reproduce_figure3 draws in full-image coordinates.
        if history:
            bbox = None
            segments = blob.segments
        else:
            bbox = _blob_bbox(blob.footprint, tim_vis.shape)
            y0, y1, x0, x1 = bbox
            segments = {k: v[y0:y1, x0:x1] for k, v in blob.segments.items()}
        if psf_stamp is not None:
            from tractor.psf import PixelizedPSF
            blob_psf = PixelizedPSF(np.asarray(psf_stamp, dtype=np.float32))
            tim_blob = _make_masked_image(tim_vis, blob.footprint,
                                          psf=blob_psf, bbox=bbox)
        else:
            tim_blob = _make_masked_image(tim_vis, blob.footprint,
                                          psf=member_psf, bbox=bbox)
        idx_list = list(blob.member_mer_indices)
        hist = History(blob_id=blob.blob_id, members=idx_list) if history else None

        # Full-ladder deferral set: these members skip the early sufficiency
        # locks so every tier is evaluated.
        defer = set()
        wide_segments = {}
        if self.full_ladder:
            from scipy import ndimage as _ndi
            for _i in idx_list:
                if (_i in segments and
                        self._segment_snr(tim_blob, segments[_i])
                        >= self.full_ladder_min_snr):
                    defer.add(_i)
                    # Window = segment grown ~2x its radius (10-40 px): wing
                    # damage counts, sky does not drown the signal.
                    _npix = int(segments[_i].sum())
                    _grow = int(np.clip(2.0 * np.sqrt(_npix / np.pi), 10, 40))
                    wide_segments[_i] = (
                        _ndi.distance_transform_edt(~segments[_i]) <= _grow)

        chi2_tracker: dict[int, dict[int, float]] = {s: {} for s in range(1, 6)}
        solved: dict[int, bool] = {i: False for i in idx_list}
        locked_source: dict[int, object] = {}

        sources_by_idx: dict[int, object] = {}

        # --- Stage 1: SimpleGalaxy for all members --------------------------
        for idx in idx_list:
            row = mer_cat[idx]
            sep_row = blob.sep_rows.get(idx)
            sources_by_idx[idx] = _build_initial_source(
                row, SimpleGalaxy, sep_row, pixscale_arcsec,
                simplegalaxy_radius=self.simplegalaxy_radius)

        chi2_stage1, chi2w_stage1, t1 = self._fit_stage(
            tim_blob, sources_by_idx, idx_list, segments,
            eval_segments=wide_segments)
        chi2_tracker[1] = chi2_stage1
        chi2w_tracker = {1: chi2w_stage1}
        if history:
            mod_full = t1.getModelImage(0)
            for idx in idx_list:
                hist.stages.append(self._record(1, sources_by_idx[idx], idx,
                                                chi2_stage1[idx], tim_blob, mod_full,
                                                chi2_wide=chi2w_stage1.get(idx, float("nan"))))

        # --- Stage 2: PointSource vs SimpleGalaxy ---------------------------
        for idx in idx_list:
            row = mer_cat[idx]
            sources_by_idx[idx] = _build_initial_source(row, PointSource, None, pixscale_arcsec)
        unsolved = [i for i in idx_list if not solved[i]]
        chi2_stage2, chi2w_tracker[2], t2 = self._fit_stage(
            tim_blob, sources_by_idx, unsolved, segments,
            eval_segments=wide_segments)
        chi2_tracker[2] = chi2_stage2

        if history:
            mod_full = t2.getModelImage(0)
            for idx in unsolved:
                hist.stages.append(self._record(2, sources_by_idx[idx], idx,
                                                chi2_stage2[idx], tim_blob, mod_full,
                                                chi2_wide=chi2w_tracker[2].get(idx, float("nan"))))

        for idx in unsolved:
            # Deferral starts after this tier: a source that wins
            # PointSource here is a star and locks now; dragging it through
            # the galaxy tiers lets an extended model win the local window
            # while rendering puffy wings globally.
            ps = chi2_stage2[idx]
            sg = chi2_tracker[1][idx]
            delta = ps - (sg + self.simplegalaxy_penalty)
            if (ps > self.sufficient_thresh) and (sg > self.sufficient_thresh):
                continue  # neither wins; advance to Exp/Dev
            if (delta <= 0) and (ps <= self.sufficient_thresh):
                locked_source[idx] = sources_by_idx[idx]  # PointSource wins
                solved[idx] = True
                if history:
                    self._mark_accepted(hist, idx, 2)

        # --- Stage 3: ExpGalaxy for unsolved --------------------------------
        for idx in idx_list:
            if solved[idx]:
                sources_by_idx[idx] = locked_source[idx]
            else:
                row = mer_cat[idx]
                sep_row = blob.sep_rows.get(idx)
                sources_by_idx[idx] = _build_initial_source(row, ExpGalaxy, sep_row, pixscale_arcsec)
        unsolved = [i for i in idx_list if not solved[i]]
        chi2_stage3, chi2w_tracker[3], t3 = self._fit_stage(
            tim_blob, sources_by_idx, unsolved, segments,
            eval_segments=wide_segments)
        chi2_tracker[3] = chi2_stage3

        if history:
            mod_full = t3.getModelImage(0)
            for idx in unsolved:
                hist.stages.append(self._record(3, sources_by_idx[idx], idx,
                                                chi2_stage3[idx], tim_blob, mod_full,
                                                chi2_wide=chi2w_tracker[3].get(idx, float("nan"))))

        # --- Stage 4: DevGalaxy for unsolved + Exp/Dev decision -------------
        exp_snapshot = {i: copy.deepcopy(sources_by_idx[i]) for i in unsolved}
        for idx in idx_list:
            if solved[idx]:
                sources_by_idx[idx] = locked_source[idx]
            else:
                row = mer_cat[idx]
                sep_row = blob.sep_rows.get(idx)
                sources_by_idx[idx] = _build_initial_source(row, DevGalaxy, sep_row, pixscale_arcsec)
        chi2_stage4, chi2w_tracker[4], t4 = self._fit_stage(
            tim_blob, sources_by_idx, unsolved, segments,
            eval_segments=wide_segments)
        chi2_tracker[4] = chi2_stage4
        dev_snapshot = {i: copy.deepcopy(sources_by_idx[i]) for i in unsolved}

        if history:
            mod_full = t4.getModelImage(0)
            for idx in unsolved:
                hist.stages.append(self._record(4, sources_by_idx[idx], idx,
                                                chi2_stage4[idx], tim_blob, mod_full,
                                                chi2_wide=chi2w_tracker[4].get(idx, float("nan"))))

        still_unsolved = []
        for idx in unsolved:
            if idx in defer:
                still_unsolved.append(idx)
                continue  # full ladder: evaluate Composite too
            exp_chi2 = chi2_tracker[3][idx]
            dev_chi2 = chi2_tracker[4][idx]
            sg_chi2 = chi2_tracker[1][idx]
            if (exp_chi2 > self.sufficient_thresh) and (dev_chi2 > self.sufficient_thresh):
                still_unsolved.append(idx)
                continue
            if (sg_chi2 <= exp_chi2) and (sg_chi2 <= dev_chi2) and (sg_chi2 <= self.sufficient_thresh):
                row = mer_cat[idx]
                locked_source[idx] = _build_initial_source(
                    row, SimpleGalaxy, None, pixscale_arcsec,
                    simplegalaxy_radius=self.simplegalaxy_radius)
                solved[idx] = True
                if history:
                    self._mark_accepted(hist, idx, 1)
                continue
            if abs(exp_chi2 - dev_chi2) < self.exp_dev_similar_thresh:
                still_unsolved.append(idx)
                continue
            if (exp_chi2 < dev_chi2) and (exp_chi2 <= self.sufficient_thresh):
                locked_source[idx] = exp_snapshot[idx]
                solved[idx] = True
                if history:
                    self._mark_accepted(hist, idx, 3)
                continue
            if (dev_chi2 < exp_chi2) and (dev_chi2 <= self.sufficient_thresh):
                locked_source[idx] = dev_snapshot[idx]
                solved[idx] = True
                if history:
                    self._mark_accepted(hist, idx, 4)
                continue
            still_unsolved.append(idx)

        # --- Stage 5: CompositeGalaxy for the remainder ---------------------
        if still_unsolved:
            for idx in idx_list:
                if solved[idx]:
                    sources_by_idx[idx] = locked_source[idx]
                else:
                    row = mer_cat[idx]
                    sep_row = blob.sep_rows.get(idx)
                    sources_by_idx[idx] = _build_initial_source(
                        row, FixedCompositeGalaxy, sep_row, pixscale_arcsec)
            chi2_stage5, chi2w_tracker[5], t5 = self._fit_stage(
                tim_blob, sources_by_idx, still_unsolved, segments,
                eval_segments=wide_segments)
            chi2_tracker[5] = chi2_stage5

            if history:
                mod_full = t5.getModelImage(0)
                for idx in still_unsolved:
                    hist.stages.append(self._record(5, sources_by_idx[idx], idx,
                                                    chi2_stage5[idx], tim_blob, mod_full,
                                                    chi2_wide=chi2w_tracker[5].get(idx, float("nan"))))

            for idx in still_unsolved:
                exp_chi2 = chi2_tracker[3].get(idx, np.inf)
                dev_chi2 = chi2_tracker[4].get(idx, np.inf)
                comp_chi2 = chi2_tracker[5].get(idx, np.inf)
                # Farmer's Stage-5 selection is a preference ladder (Exp,
                # then Dev, then Composite while sufficient), with an argmin
                # over all five only when every galaxy model is bad; ties
                # break toward the simpler model. Full-ladder members take
                # the pure argmin instead.
                if (idx not in defer
                        and (exp_chi2 <= comp_chi2)
                        and (exp_chi2 <= self.sufficient_thresh)):
                    locked_source[idx] = exp_snapshot[idx]
                    accepted_stage = 3
                elif (idx not in defer and (dev_chi2 <= comp_chi2)
                        and (dev_chi2 <= self.sufficient_thresh)):
                    locked_source[idx] = dev_snapshot[idx]
                    accepted_stage = 4
                elif idx not in defer and comp_chi2 <= self.sufficient_thresh:
                    locked_source[idx] = sources_by_idx[idx]  # Composite
                    accepted_stage = 5
                else:
                    # Every galaxy model is bad: argmin over all five.
                    # Full-ladder members argmin over the wide-window chi2
                    # so wing damage outside the segment counts.
                    if idx in defer:
                        chis = {
                            1: chi2w_tracker.get(2, {}).get(idx, np.inf),  # PointSource
                            2: chi2w_tracker.get(1, {}).get(idx, np.inf),  # SimpleGalaxy
                            3: chi2w_tracker.get(3, {}).get(idx, np.inf),
                            4: chi2w_tracker.get(4, {}).get(idx, np.inf),
                            5: chi2w_tracker.get(5, {}).get(idx, np.inf),
                        }
                    else:
                        chis = {
                            1: chi2_tracker[2].get(idx, np.inf),  # PointSource
                            2: chi2_tracker[1].get(idx, np.inf),  # SimpleGalaxy
                            3: exp_chi2, 4: dev_chi2, 5: comp_chi2,
                        }
                    best_stage = min(chis, key=chis.get)
                    row = mer_cat[idx]
                    if best_stage == 1:
                        locked_source[idx] = _build_initial_source(row, PointSource, None, pixscale_arcsec)
                        accepted_stage = 2
                    elif best_stage == 2:
                        locked_source[idx] = _build_initial_source(
                            row, SimpleGalaxy, None, pixscale_arcsec,
                            simplegalaxy_radius=self.simplegalaxy_radius)
                        accepted_stage = 1
                    elif best_stage == 3:
                        locked_source[idx] = exp_snapshot[idx]
                        accepted_stage = 3
                    elif best_stage == 4:
                        locked_source[idx] = dev_snapshot[idx]
                        accepted_stage = 4
                    else:
                        locked_source[idx] = sources_by_idx[idx]
                        accepted_stage = 5
                solved[idx] = True
                if history:
                    self._mark_accepted(hist, idx, accepted_stage)

        # --- Final joint re-fit -------------------------------------------
        # Position is thawed here (as in Farmer's force_models): locking pos
        # at the MER centroid leaves a dipole residual whenever MER and VIS
        # astrometry differ by a fraction of a pixel.
        all_sources_final = [locked_source[i] for i in idx_list]
        tractor_final = Tractor([tim_blob], all_sources_final, optimizer=ConstrainedOptimizer())
        for src in all_sources_final:
            src.thawAllParams()
            if isinstance(src, SimpleGalaxy):
                src.freezeParam("shape")
        self._optimize(tractor_final)

        # --- Optional Stage 6: guarded SersicGalaxy refit -------------------
        # Free-n Sersic for Exp/Dev/Composite sources bright enough to
        # constrain n; kept only if the total chi^2 drops significantly.
        if self.include_sersic_tier:
            self._try_sersic_tier(tim_blob, blob, idx_list, locked_source,
                                  all_sources_final, mer_cat, pixscale_arcsec,
                                  hist if history else None,
                                  segments=segments)
            tractor_final = Tractor([tim_blob], all_sources_final, optimizer=ConstrainedOptimizer())
            for src in all_sources_final:
                src.thawAllParams()
                if isinstance(src, SimpleGalaxy):
                    src.freezeParam("shape")
            self._optimize(tractor_final)

        # Clamp runaway centroids back to the MER prior and re-fit
        # flux-only; runs last so no later refit reintroduces drift past
        # the cap.
        self._clamp_positions(tractor_final, all_sources_final, idx_list,
                              mer_cat)

        if history:
            model = tractor_final.getModelImage(0)
            data = tim_blob.getImage()
            hist.final = FinalRecord(
                data=data.copy(),
                model=model.copy(),
                residual=(data - model).copy(),
                footprint=blob.footprint.copy(),
            )

        result = {idx: src for idx, src in zip(idx_list, all_sources_final, strict=False)}
        return result, hist

    # ---- Position-drift clamp -----------------------------------------
    def _clamp_positions(self, tractor_final, sources, idx_list, mer_cat):
        """Reset any source that drifted past ``max_pos_shift_arcsec`` from its
        MER prior, then re-fit flux-only for the clamped ones.

        Keeps the sub-pixel astrometry refinement for well-behaved sources
        while preventing faint/blended centroids from wandering off the prior.
        """
        cap = self.max_pos_shift_arcsec
        clamped = []
        for src, idx in zip(sources, idx_list, strict=False):
            try:
                p = src.getPosition()
                ra0 = float(mer_cat[idx]["ra"])
                dec0 = float(mer_cat[idx]["dec"])
            except Exception:
                continue
            cosd = np.cos(np.radians(dec0))
            shift = np.hypot((p.ra - ra0) * cosd, p.dec - dec0) * 3600.0
            if shift > cap:
                src.setPosition(RaDecPos(ra0, dec0))
                clamped.append(src)
        if not clamped:
            return
        # Brightness only: freeze position so the reset sticks and shape so
        # the fitted morphology is preserved.
        for src in sources:
            src.freezeAllBut("brightness")
        self._optimize(tractor_final)

    # ---- Sersic-tier extension ----------------------------------------
    def _try_sersic_tier(self, tim_blob, blob, idx_list, locked_source,
                         all_sources_final, mer_cat, pixscale_arcsec,
                         hist, segments=None):
        """For each Exp/Dev/Composite source bright enough to constrain Sérsic
        index, try a free-n SersicGalaxy and keep it only if the total chi^2
        on the wing-wide window improves by more than ``sersic_dchi2_accept``.
        Updates ``locked_source``, ``all_sources_final`` (in place), and ``hist``.
        """
        eligible_classes = (ExpGalaxy, DevGalaxy, FixedCompositeGalaxy)
        # Don't catch SimpleGalaxy via ExpGalaxy isinstance:
        for pos_in_list, idx in enumerate(idx_list):
            src = locked_source[idx]
            if isinstance(src, SimpleGalaxy):
                continue
            if not isinstance(src, eligible_classes):
                continue

            # segments is bbox-cropped to tim_blob's grid; blob.segments is
            # the full-frame original (history path).
            seg = (segments if segments is not None else blob.segments)[idx]
            snr = self._segment_snr(tim_blob, seg)
            if snr < self.sersic_min_snr:
                logger.debug("Skip Sersic for src#%s: SNR=%.1f < %s",
                             idx, snr, self.sersic_min_snr)
                continue

            # Judge on the dilated segment: the failure this tier fixes is a
            # wing-mismatch ring outside the SEP segment (which hugs the
            # core and is blind to the ring). Both candidates score on the
            # same window with the same frozen neighbors.
            from scipy import ndimage as _ndi
            try:
                re_pix = float(src.shape.re) / max(pixscale_arcsec, 1e-6)
            except Exception:
                re_pix = 10.0
            grow = int(np.clip(2.0 * re_pix, 10, 40))
            eval_seg = _ndi.distance_transform_edt(~seg) <= grow

            # Score the locked model with the free-parameter count of its
            # accepted tier (pos frozen, shape+brightness free), captured
            # before the freezeAllParams below; a frozen source counts 0
            # params, making the dof inconsistent with the Sersic candidate.
            src.thawAllParams()
            src.freezeParam("pos")
            n_free_prev = _count_free_params(src)
            tractor_cur = Tractor([tim_blob], all_sources_final, optimizer=ConstrainedOptimizer())
            for s in all_sources_final:
                s.freezeAllParams()
            chi_prev = tractor_cur.getChiImage(imgi=0)
            npix_eval = int(eval_seg.sum())
            prev_tot = float(np.sum(chi_prev[eval_seg] ** 2))
            prev_chi2 = prev_tot / max(npix_eval - n_free_prev, 1)

            # Two starts, n=1 and n=4: the Sersic likelihood is bimodal for
            # a disk+core galaxy. Each start gets a fresh seed shape since
            # the first fit mutates it.
            saved = all_sources_final[pos_in_list]
            best = None   # (total chi2, reduced chi2, segment chi2, source, n)

            def _fresh_seed(idx=idx):
                sep_row = blob.sep_rows.get(idx)
                if sep_row is not None:
                    return _seed_shape_from_sep(sep_row, pixscale_arcsec)
                sh = EllipseESoft.fromRAbPhi(0.3, 1.0, 0.0)
                sh.lowers = [-5.0, -np.inf, -np.inf]
                sh.uppers = [1.0, np.inf, np.inf]
                return sh

            for n_init in (1.0, 4.0):
                new_src = SersicGalaxy(
                    src.getPosition(),
                    NanoMaggies(VIS=float(src.brightness.getFlux("VIS"))),
                    _fresh_seed(), SersicIndex(n_init))
                all_sources_final[pos_in_list] = new_src
                try:
                    tractor_try = Tractor([tim_blob], all_sources_final,
                                          optimizer=ConstrainedOptimizer())
                    for s in all_sources_final:
                        if s is new_src:
                            s.thawAllParams()
                            s.freezeParam("pos")
                        else:
                            s.freezeAllParams()
                    self._optimize(tractor_try)
                    chi_try = tractor_try.getChiImage(imgi=0)
                    n_free_new = _count_free_params(new_src)
                    cand_tot = float(np.sum(chi_try[eval_seg] ** 2))
                    cand_chi2 = cand_tot / max(npix_eval - n_free_new, 1)
                    # Segment-window chi^2 for the figure, so the stage-6
                    # row is comparable with the earlier tiers.
                    cand_seg = (float(np.sum(chi_try[seg] ** 2))
                                / max(int(seg.sum()) - n_free_new, 1))
                except Exception as exc:
                    logger.debug("Sersic refit (n_init=%s) raised %s for "
                                 "src#%s.", n_init, exc, idx)
                    continue
                if best is None or cand_tot < best[0]:
                    best = (cand_tot, cand_chi2, cand_seg, new_src,
                            float(new_src.sersicindex.val))

            if best is None:
                all_sources_final[pos_in_list] = saved
                continue
            new_tot, new_chi2, new_seg_chi2, new_src, n_fit = best
            all_sources_final[pos_in_list] = new_src

            improved = (prev_tot - new_tot) > self.sersic_dchi2_accept

            if hist is not None:
                # Render the best candidate, not whichever start ran last.
                tractor_best = Tractor([tim_blob], all_sources_final,
                                       optimizer=ConstrainedOptimizer())
                rec = StageRecord(
                    stage=6,
                    model_name=f"SersicGalaxy(n={n_fit:.2f})",
                    source_idx=idx,
                    chi2_red=new_seg_chi2,
                    chi2_wide=new_chi2,
                    flux_ujy=float(new_src.brightness.getFlux("VIS")) * 3.631,
                    model_image=tractor_best.getModelImage(0).copy(),
                    residual_image=(tim_blob.getImage()
                                    - tractor_best.getModelImage(0)).copy(),
                    accepted=bool(improved),
                )
                hist.stages.append(rec)
                # Unmark the previously accepted tier so the figure shows
                # stage 6 winning.
                if improved:
                    for r in hist.stages:
                        if (r.source_idx == idx and r.stage != 6 and r.accepted):
                            r.accepted = False

            if improved:
                locked_source[idx] = new_src
                logger.debug("Sersic accepted for src#%s: dchi2=%.1f "
                             "(chi2red %.3f -> %.3f), n=%.2f",
                             idx, prev_tot - new_tot, prev_chi2, new_chi2, n_fit)
            else:
                logger.debug("Sersic rejected for src#%s: dchi2=%.1f "
                             "(needs > %.1f), n=%.2f",
                             idx, prev_tot - new_tot,
                             self.sersic_dchi2_accept, n_fit)
                all_sources_final[pos_in_list] = saved

    @staticmethod
    def _segment_snr(tim_blob: Image, segment: np.ndarray) -> float:
        """Approximate segment S/N: matched-filter value for a model that
        is constant inside the segment."""
        data = tim_blob.getImage()
        invvar = tim_blob.getInvvar()
        mask = segment & (invvar > 0)
        if not mask.any():
            return 0.0
        signal = float(np.sum(data[mask]))
        noise = float(np.sqrt(np.sum(1.0 / invvar[mask])))
        return signal / max(noise, 1e-12)

    # ---- history helpers -----------------------------------------------
    def _record(self, stage, src, idx, chi2_red, tim_blob, mod_full,
                chi2_wide=float("nan")):
        try:
            flux_ujy = float(src.brightness.getFlux("VIS")) * 3.631
        except Exception:
            flux_ujy = float("nan")
        return StageRecord(
            stage=stage,
            model_name=_classname(src),
            source_idx=idx,
            chi2_red=chi2_red,
            model_image=mod_full.copy(),
            residual_image=(tim_blob.getImage() - mod_full).copy(),
            accepted=False,
            flux_ujy=flux_ujy,
            chi2_wide=chi2_wide,
        )

    @staticmethod
    def _mark_accepted(hist: History, idx: int, stage: int) -> None:
        for rec in hist.stages:
            if rec.source_idx == idx and rec.stage == stage:
                rec.accepted = True


# ---------------------------------------------------------------------------
# SEP-based blob detection and MER cross-matching.
# ---------------------------------------------------------------------------
# SEP's shipped default kernel (3x3 Gaussian, FWHM ~2 pix, byte-identical
# to SExtractor's default.conv); filter_kernel=None means no filtering.
SEP_DEFAULT_KERNEL = np.array([[1.0, 2.0, 1.0],
                               [2.0, 4.0, 2.0],
                               [1.0, 2.0, 1.0]])


def detect_blobs(
    vis_data: np.ndarray,
    vis_invvar: np.ndarray,
    *,
    threshold_sigma: float = EUCLID_DEFAULTS["sep_threshold_sigma"],
    minarea: int = EUCLID_DEFAULTS["sep_minarea"],
    dilate_pix: int = EUCLID_DEFAULTS["blob_dilate_pix"],
    deblend_cont: float = 0.005,
    deblend_nthresh: int = 32,
    bkg_box: int = 64,
    filter_kernel: np.ndarray | None = SEP_DEFAULT_KERNEL,
    filter_type: str = "matched",
    clean: bool = True,
) -> tuple[list[Blob], np.ndarray, np.ndarray]:
    """Run SEP on the VIS cutout and return (blobs, segmap, sep_objects).

    ``segmap`` is the raw SEP segmentation (per-source label image); ``blobs``
    contains the dilated/merged groups. ``sep_objects`` is the structured array
    SEP returns (a, b, theta, x, y, flux, ...).

    Parameters
    ----------
    threshold_sigma : float
        Detection threshold in units of the per-pixel noise (SExtractor
        ``DETECT_THRESH``). Raise it for a cleaner, higher-purity source list.
    minarea : int
        Minimum number of connected pixels above threshold (``DETECT_MINAREA``).
    dilate_pix : int
        Radius (pixels) by which SEP segments are grown before grouping into
        blobs; controls how readily neighboring sources share a joint fit.
    deblend_cont : float
        SEP deblend contrast (``DEBLEND_MINCONT``). Smaller splits blends more
        aggressively; 0.005 is the SExtractor default.
    deblend_nthresh : int
        Number of deblending sub-thresholds (``DEBLEND_NTHRESH``).
    bkg_box : int
        Background-mesh box size in pixels for ``sep.Background``.
    filter_kernel : ndarray or None
        Detection convolution kernel. Defaults to SEP's shipped 3x3
        FWHM~2 pix kernel (a good match to the ~1.6 pix VIS PSF FWHM);
        ``None`` disables filtering. The Farmer's own config uses a
        5x5 FWHM=2 Gaussian; COSMOS2020 detection used 7x7 FWHM=4 on
        seeing-limited ground-based stacks.
    filter_type : {'matched', 'conv'}
        ``'matched'`` (SEP default) weights the kernel by the local noise,
        which matters on coadds with spatially varying depth.
    clean : bool
        SExtractor-style cleaning of spurious detections near bright
        sources (SEP default True; the Farmer ships clean=False).

    Notes
    -----
    The detection threshold (1.5 sigma) and ``minarea`` (5) match Farmer's
    config. The deblend and grouping defaults are the SExtractor/SEP
    defaults plus a 4-pixel dilation, gentler than Farmer's shipped values
    (``DEBLEND_CONT=1e-10``, ``DEBLEND_NTHRESH=256``, 0.2" dilation), which
    over-split Euclid VIS blends.
    """
    import sep
    from scipy import ndimage

    # SEP wants a contiguous float32 array.
    data = np.ascontiguousarray(vis_data.astype(np.float32))
    err = np.zeros_like(data)
    # Q1 stores masked/no-coverage pixels with a huge RMS (~1e16), i.e. a
    # tiny but positive inverse-variance; a bare invvar > 0 test would pass
    # them. Floor the test relative to the median real weight.
    finite_iv = vis_invvar[np.isfinite(vis_invvar) & (vis_invvar > 0)]
    iv_floor = float(np.median(finite_iv)) * 1e-6 if finite_iv.size else 0.0
    good = np.isfinite(vis_invvar) & (vis_invvar > iv_floor)
    err[good] = 1.0 / np.sqrt(vis_invvar[good])
    err[~good] = err[good].max() if good.any() else 1.0
    mask = ~good

    try:
        bkg = sep.Background(data, mask=mask, bw=int(bkg_box), bh=int(bkg_box))
        data_sub = data - bkg.back()
    except Exception as exc:
        logger.warning("SEP background failed (%s); using raw data.", exc)
        data_sub = data

    # SEP's pixel buffer defaults to 300k active pixels; extract() aborts
    # with a pixstack overflow on a deep multi-arcmin field. Size it to the
    # image.
    sep.set_extract_pixstack(max(300_000, min(int(data.size // 5), 5_000_000)))

    objects, segmap = sep.extract(
        data_sub,
        thresh=threshold_sigma,
        err=err,
        mask=mask,
        minarea=int(minarea),
        filter_kernel=filter_kernel,
        filter_type=filter_type,
        deblend_nthresh=int(deblend_nthresh),
        deblend_cont=deblend_cont,
        clean=clean,
        segmentation_map=True,
    )

    if dilate_pix > 0:
        struct = np.ones((2 * dilate_pix + 1, 2 * dilate_pix + 1), dtype=bool)
        dilated = ndimage.binary_dilation(segmap > 0, structure=struct)
    else:
        dilated = segmap > 0

    blob_labels, n_blobs = ndimage.label(dilated)

    blobs: list[Blob] = []
    for blob_id in range(1, n_blobs + 1):
        footprint = blob_labels == blob_id
        members = np.unique(segmap[footprint])
        members = members[members > 0]
        segments = {}
        sep_rows = {}
        for sep_label in members:
            seg = segmap == sep_label
            segments[int(sep_label) - 1] = seg  # 0-based index into ``objects``
            sep_rows[int(sep_label) - 1] = {k: objects[int(sep_label) - 1][k]
                                            for k in objects.dtype.names}
        blobs.append(Blob(
            blob_id=blob_id,
            footprint=footprint,
            member_mer_indices=list(segments.keys()),  # placeholder; filled by assign
            segments=segments,
            sep_rows=sep_rows,
        ))
    return blobs, segmap, objects


def assign_mer_to_blobs(mer_cat, blobs: list[Blob], wcs) -> list[Blob]:
    """Assign each MER row to the blob whose footprint contains it.

    Returns a new list of Blob objects with ``member_mer_indices`` (and
    ``segments`` keyed by MER idx) populated. Sources not falling inside any
    blob are dropped; sources inside a blob but outside any SEP segment are
    attached to the geometrically nearest SEP segment in that blob.
    """
    projected = []
    if blobs:
        for mer_idx, row in enumerate(mer_cat):
            try:
                px, py = wcs.positionToPixel(RaDecPos(float(row["ra"]), float(row["dec"])))
            except Exception:
                continue
            projected.append((mer_idx, px, py))
    assigned: list[Blob] = []
    for blob in blobs:
        mer_indices = []
        new_segments: dict[int, np.ndarray] = {}
        new_sep_rows: dict[int, dict] = {}
        sep_labels = list(blob.segments.keys())  # original 0-based SEP indices

        for mer_idx, px, py in projected:
            xi, yi = int(round(px)), int(round(py))
            H, W = blob.footprint.shape
            if not (0 <= yi < H and 0 <= xi < W):
                continue
            if not blob.footprint[yi, xi]:
                continue
            owning_label = None
            for sl in sep_labels:
                if blob.segments[sl][yi, xi]:
                    owning_label = sl
                    break
            if owning_label is None and sep_labels:
                # Nearest SEP centroid in this blob.
                d_best = np.inf
                for sl in sep_labels:
                    seg_yx = np.argwhere(blob.segments[sl])
                    if not len(seg_yx):
                        continue
                    cy, cx = seg_yx.mean(axis=0)
                    d = (cy - yi) ** 2 + (cx - xi) ** 2
                    if d < d_best:
                        d_best = d
                        owning_label = sl
            if owning_label is None:
                continue
            mer_indices.append(mer_idx)
            new_segments[mer_idx] = blob.segments[owning_label]
            new_sep_rows[mer_idx] = blob.sep_rows[owning_label]

        if not mer_indices:
            continue
        assigned.append(Blob(
            blob_id=blob.blob_id,
            footprint=blob.footprint,
            member_mer_indices=mer_indices,
            segments=new_segments,
            sep_rows=new_sep_rows,
        ))
    return assigned


def unlisted_sep_detections(mer_cat, tim_vis, vis_data, vis_invvar, *,
                            min_separation_pix: float = 3.0,
                            min_snr: float = 5.0,
                            mask_margin_pix: int = 3,
                            **detect_kwargs) -> dict:
    """SEP detections that share a blob with a catalog row but match none.

    Runs the same detection and assignment as :func:`run_model_selection`.
    Inside every blob that contains at least one ``mer_cat`` row, the SEP
    segments no row was attached to are returned as candidate neighbours;
    blobs without a catalog row are ignored. A candidate is skipped when it
    lies closer than ``min_separation_pix`` to a catalog row, when its
    segment flux is below ``min_snr`` times the pixel noise summed in
    quadrature over the segment, or when the segment comes within
    ``mask_margin_pix`` of a zero-weight pixel (bright-star mask, flagged
    or missing data), where detections are treated as the masked object's
    light.

    Returns a dict of arrays: ``ra``, ``dec`` (deg), ``flux`` (SEP
    isophotal flux in image counts), ``snr``, ``x``, ``y`` (pixels).
    """
    from scipy import ndimage

    blobs_raw, _, _ = detect_blobs(vis_data, vis_invvar, **detect_kwargs)
    wcs = tim_vis.getWcs()
    finite_iv = vis_invvar[np.isfinite(vis_invvar) & (vis_invvar > 0)]
    iv_floor = float(np.median(finite_iv)) * 1e-6 if finite_iv.size else 0.0
    good = np.isfinite(vis_invvar) & (vis_invvar > iv_floor)
    near_mask = ~good
    if mask_margin_pix > 0 and near_mask.any():
        near_mask = ndimage.binary_dilation(
            near_mask, iterations=int(mask_margin_pix))
    assigned = {b.blob_id: b for b in assign_mer_to_blobs(mer_cat, blobs_raw, wcs)}
    listed = []
    for row in mer_cat:
        try:
            listed.append(wcs.positionToPixel(
                RaDecPos(float(row["ra"]), float(row["dec"]))))
        except Exception:
            continue
    listed = np.asarray(listed, dtype=float).reshape(-1, 2)

    out = {k: [] for k in ("ra", "dec", "flux", "snr", "x", "y")}
    for raw in blobs_raw:
        blob = assigned.get(raw.blob_id)
        if blob is None:
            continue
        # assign_mer_to_blobs re-keys the segment arrays without copying
        # them, so identity tells which SEP segments received a row.
        taken = {id(seg) for seg in blob.segments.values()}
        for label, seg in raw.segments.items():
            if id(seg) in taken:
                continue
            sep_row = raw.sep_rows[label]
            x, y = float(sep_row["x"]), float(sep_row["y"])
            if listed.size and np.hypot(listed[:, 0] - x,
                                        listed[:, 1] - y).min() < min_separation_pix:
                continue
            if np.any(seg & near_mask):
                continue
            pix = seg & good
            noise = float(np.sqrt(np.sum(1.0 / vis_invvar[pix]))) if pix.any() else np.inf
            snr = float(sep_row["flux"]) / noise if noise > 0 else 0.0
            if not snr >= min_snr:
                continue
            pos = wcs.pixelToPosition(x, y)
            out["ra"].append(float(pos.ra))
            out["dec"].append(float(pos.dec))
            out["flux"].append(float(sep_row["flux"]))
            out["snr"].append(snr)
            out["x"].append(x)
            out["y"].append(y)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Convenience: run the tree over every blob and return a list[Source].
# ---------------------------------------------------------------------------
def _fit_one_blob(args):
    """Worker for parallel Farmer fits: one blob, runnable in a thread."""
    selector, blob, tim_vis, mer_cat, pixscale, psf_by_member = args
    chosen, _ = selector.fit_blob(blob, tim_vis, mer_cat, pixscale,
                                  history=False, psf_by_member=psf_by_member)
    return blob, chosen


def _pack_blob(blob: Blob) -> dict:
    """Bit-pack a Blob's full-frame boolean masks to shrink what is sent
    to worker processes.

    A 2000x2000 footprint is 4 MB as bool and 0.5 MB packed; a field of a
    few hundred blobs would otherwise copy gigabytes to each worker process.
    """
    pack = lambda m: (np.packbits(m), m.shape)  # noqa: E731
    return {
        "blob_id": blob.blob_id,
        "footprint": pack(blob.footprint),
        "member_mer_indices": list(blob.member_mer_indices),
        "segments": {k: pack(v) for k, v in blob.segments.items()},
        "sep_rows": blob.sep_rows,
    }


def _unpack_blob(d: dict) -> Blob:
    def unpack(p):
        bits, shape = p
        n = int(shape[0]) * int(shape[1])
        return np.unpackbits(bits, count=n).astype(bool).reshape(shape)
    return Blob(
        blob_id=d["blob_id"],
        footprint=unpack(d["footprint"]),
        member_mer_indices=d["member_mer_indices"],
        segments={k: unpack(v) for k, v in d["segments"].items()},
        sep_rows=d["sep_rows"],
    )


def _fit_blob_chunk(args):
    """Worker for process-parallel Farmer fits: a chunk of packed blobs."""
    selector, packed, tim_vis, mer_cat, pixscale, psf_by_member = args
    out = []
    for pb in packed:
        blob = _unpack_blob(pb)
        chosen, _ = selector.fit_blob(blob, tim_vis, mer_cat, pixscale,
                                      history=False,
                                      psf_by_member=psf_by_member)
        out.append(chosen)
    return out


def run_model_selection(
    mer_cat,
    tim_vis: Image,
    vis_data: np.ndarray,
    vis_invvar: np.ndarray,
    *,
    pixscale_arcsec: float,
    selector: ModelSelector | None = None,
    psf_data: dict | None = None,
    n_workers: int = 1,
    **detect_kwargs,
):
    """Detect blobs and run :meth:`ModelSelector.fit_blob` over each.

    Parameters
    ----------
    n_workers : int, default 1
        If >1, fit blobs in a process pool (blob fits are independent, and
        separate processes scale better here than threads, since the
        optimizer runs single-threaded internally). Falls back to threads
        if a blob cannot be sent to a separate process for an unusual
        source class.

    Returns ``(sources_list, summary_dict)`` where ``sources_list`` is aligned
    1:1 with ``mer_cat`` (sources outside any blob get ``None``) and
    ``summary_dict`` reports how many MER sources landed at each model tier.
    """
    selector = selector or ModelSelector()
    blobs_raw, segmap, _ = detect_blobs(vis_data, vis_invvar, **detect_kwargs)
    blobs = assign_mer_to_blobs(mer_cat, blobs_raw, tim_vis.getWcs())

    psf_by_member = None
    if (psf_data is not None and selector.per_source_psf
            and not getattr(tim_vis.getPsf(), 'is_spatial', False)
            and psf_data.get("stamps") is not None
            and len(psf_data["stamps"]) > 0):
        from .spatial_psf import SpatialPixelizedPSF
        tim_vis = _make_masked_image(
            tim_vis, np.ones(tim_vis.shape, dtype=bool),
            psf=SpatialPixelizedPSF(psf_data, tim_vis.getWcs()))

    sources_list: list[object | None] = [None] * len(mer_cat)
    counts = {name: 0 for name in MODEL_NAMES}
    counts["Unblobbed"] = 0

    if n_workers <= 1 or len(blobs) <= 1:
        work = [(selector, b, tim_vis, mer_cat, pixscale_arcsec,
                 psf_by_member) for b in blobs]
        chosen_list = [c for _, c in (_fit_one_blob(w) for w in work)]
    else:
        import concurrent.futures
        n = min(n_workers, len(blobs))
        # Interleave so the brightest (usually first) blobs spread across
        # workers; each chunk pickles tim_vis once.
        chunks = [(selector, [_pack_blob(b) for b in blobs[i::n]],
                   tim_vis, mer_cat, pixscale_arcsec, psf_by_member)
                  for i in range(n)]
        try:
            with concurrent.futures.ProcessPoolExecutor(n) as exec_:
                chosen_list = [c for sub in exec_.map(_fit_blob_chunk, chunks)
                               for c in sub]
        except Exception as exc:
            logger.warning(
                "process-parallel blob fits failed (%r); retrying with "
                "threads.", exc)
            work = [(selector, b, tim_vis, mer_cat, pixscale_arcsec,
                     psf_by_member) for b in blobs]
            with concurrent.futures.ThreadPoolExecutor(n) as exec_:
                chosen_list = [c for _, c in exec_.map(_fit_one_blob, work)]

    for chosen in chosen_list:
        for mer_idx, src in chosen.items():
            sources_list[mer_idx] = src
            counts[_classname(src)] += 1

    counts["Unblobbed"] = sum(1 for s in sources_list if s is None)
    return sources_list, counts


# ---------------------------------------------------------------------------
# Figure reproduction (Weaver+2023 Fig. 3).
# ---------------------------------------------------------------------------
def reproduce_figure3(
    blob: Blob,
    history: History,
    *,
    save_path: str | None = None,
    figsize_per_panel: float = 2.8,
    vlim_sigma: float = 5.0,
):
    """Plot the Weaver+2023 Fig. 3-style per-tier residual panels.

    Row 0: VIS data cutout and final residual (linear, ±vlim_sigma·σ); one
    row per (source, stage) trial with the model image (log stretch) and
    residual, annotated with the reduced chi^2 and a check on the accepted
    tier. σ is the sigma-clipped residual RMS outside the blob footprint.
    """
    import matplotlib.pyplot as plt
    from astropy.stats import sigma_clipped_stats
    from matplotlib.colors import LogNorm, Normalize

    if history.final is None:
        raise ValueError("History has no final record; call fit_blob(..., history=True).")

    final = history.final
    H, W = final.data.shape

    # Sigma-clipped residual stats outside the blob footprint, so core
    # residual structure does not inflate the stretch.
    outside = ~blob.footprint
    if outside.sum() > 100:
        _, _, sigma = sigma_clipped_stats(final.residual[outside], sigma=3.0)
    else:
        sigma = float(np.std(final.residual[blob.footprint]))
    sigma = float(sigma)
    if sigma <= 0:
        sigma = 1e-3
    # The paper uses ±3σ; ±5σ here because bright Q1 galaxies carry a few
    # 3-5σ pixels (astrometry, PSF stamp imperfections, substructure) that
    # saturate a ±3σ stretch.
    vlim = float(vlim_sigma) * sigma

    ymin, ymax = np.argwhere(blob.footprint).min(0), np.argwhere(blob.footprint).max(0)
    pad = 5
    y0 = max(0, int(ymin[0]) - pad)
    y1 = min(H, int(ymax[0]) + pad + 1)
    x0 = max(0, int(ymin[1]) - pad)
    x1 = min(W, int(ymax[1]) + pad + 1)

    def crop(a):
        return a[y0:y1, x0:x1]

    footprint_crop = crop(blob.footprint)

    norm_lin = Normalize(vmin=-vlim, vmax=vlim)

    # Log stretch from the final model so all tier rows share one scale.
    model_in = crop(final.model)[footprint_crop]
    if model_in.size and np.any(model_in > 0):
        positive = model_in[model_in > 0]
        v_hi = float(np.nanpercentile(positive, 99.5))
        v_lo = float(max(sigma, 1e-6))
        if v_hi <= v_lo:
            v_hi = v_lo * 10
        norm_log = LogNorm(vmin=v_lo, vmax=v_hi)
    else:
        norm_log = LogNorm(vmin=sigma, vmax=max(sigma * 100, 1e-3))

    cmap_img = "RdBu_r"
    cmap_mod = "Grays"
    n_rows = 1 + len(history.stages)
    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(2 * figsize_per_panel, n_rows * figsize_per_panel),
                             squeeze=False)

    axes[0, 0].imshow(crop(final.data),
                       origin="lower", cmap=cmap_img, norm=norm_lin)
    axes[0, 0].set_title(f"VIS data  (±{vlim_sigma:.0f}σ, σ={sigma:.3g})", fontsize=13)
    axes[0, 1].imshow(crop(final.residual),
                       origin="lower", cmap=cmap_img, norm=norm_lin)
    axes[0, 1].set_title(f"Final residual  (±{vlim_sigma:.0f}σ)", fontsize=13)

    for r, rec in enumerate(history.stages, start=1):
        mod = crop(rec.model_image)
        res = crop(rec.residual_image)
        mod_clip = np.clip(mod, norm_log.vmin, None)
        check = "  (accepted)" if rec.accepted else ""
        title_color = "darkgreen" if rec.accepted else "black"

        axes[r, 0].imshow(mod_clip, origin="lower", cmap=cmap_mod, norm=norm_log)
        axes[r, 0].set_title(f"src#{rec.source_idx} = {rec.model_name}{check}\n"
                              f"(joint blob model)", fontsize=12,
                              color=title_color)
        axes[r, 1].imshow(res, origin="lower", cmap=cmap_img, norm=norm_lin)
        axes[r, 1].set_title(f"joint residual   χ²ν(src#{rec.source_idx}) = "
                              f"{rec.chi2_red:.2f}", fontsize=12,
                              color=title_color)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Farmer-style decision tree on blob #{blob.blob_id} "
                 f"({len(history.members)} source{'s' if len(history.members) != 1 else ''})",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes
