"""Re-implementation of the Euclid SIR spectra-location geometry.

Maps a sky position (RA, Dec) to the pixel trace of its 0th / 1st order
spectrum on a NISP-S dispersed frame, using only the SIR calibration
products (OPT, CRV, IDS, PSF models and the detector metrology).  No SIR
binaries are needed.  The conventions follow SIR 6.0.1:

  sky --TAN WCS (adjusted pointing)--> undistorted FOV mm
      --OPT 6x6 polynomials--> reference-wavelength FOV mm (per order)
      --IDS/CRV local Chebyshev models (coefficients from a 2D Chebyshev
        expansion over the FOV)--> FOV mm along the trace
      --optional extra-tilt rotation about the pivot-->
      --detector linear WCS (SIR rotated layout)--> 0-based pixel on DETxx

Everything is plain numpy so it can later be ported to torch.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.polynomial import chebyshev as _cheb
from numpy.polynomial import polynomial as _poly
from astropy.io import fits
from astropy.wcs import WCS

DET_IDS = ["11", "21", "31", "41", "12", "22", "32", "42",
           "13", "23", "33", "43", "14", "24", "34", "44"]
DET_SIZE = 2041          # used by SIR's layout rotation
NX = NY = 2040           # science pixels per detector
PIXEL_OFFSET = 0.5


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _floats(text: str) -> np.ndarray:
    return np.fromstring(text.strip(), dtype=float, sep=" ")


def _matrix(node) -> np.ndarray:
    shape = np.fromstring(node.findtext("Shape"), dtype=int, sep=" ")
    return _floats(node.findtext("Data")).reshape(tuple(shape))


def _grism_block(root, grism: str, tilt: int):
    """Return the <GRISM> block of a SIR calibration XML for a given tilt."""
    data = root.find(".//Data")
    for el in data.findall(grism):
        if int(float(el.findtext("GWATilt"))) == int(tilt):
            return el
    raise KeyError(f"No {grism} block with GWATilt {tilt}")


def _normalize(v, lo, hi):
    return (2.0 * v - lo - hi) / (hi - lo)


# --------------------------------------------------------------------------- #
# optical model: sky -> FOV mm
# --------------------------------------------------------------------------- #
@dataclass
class OpticalModel:
    grism: str
    tilt: int
    extra_tilt: float
    ref_x: np.ndarray            # 1st order, 6x6
    ref_y: np.ndarray
    ref_lambda: float
    pivot_x: np.ndarray
    pivot_y: np.ndarray
    disp: dict = field(default_factory=dict)   # order -> dict(X, Y, lambda)
    wcs: WCS | None = None

    # ---- loading --------------------------------------------------------- #
    @classmethod
    def from_xml(cls, path, grism: str, tilt: int) -> "OpticalModel":
        root = ET.parse(str(path)).getroot()
        el = _grism_block(root, grism, tilt)
        ref = el.find("Reference")
        piv = el.find("Pivot")
        disp = {}
        for d in el.findall("Displacements"):
            disp[int(d.findtext("Order"))] = dict(
                X=_matrix(d.find("OptXModel")), Y=_matrix(d.find("OptYModel")),
                ref_lambda=float(d.findtext("ReferenceLambda")))
        return cls(grism=grism, tilt=int(tilt),
                   extra_tilt=float(el.findtext("ExtraTilt") or 0.0),
                   ref_x=_matrix(ref.find("OptXModel")), ref_y=_matrix(ref.find("OptYModel")),
                   ref_lambda=float(ref.findtext("ReferenceLambda")),
                   pivot_x=_matrix(piv.find("OptXModel")), pivot_y=_matrix(piv.find("OptYModel")),
                   disp=disp)

    # ---- pointing -------------------------------------------------------- #
    def set_pointing(self, ra: float, dec: float, pa: float, fov_scale_as_per_mm: float,
                     rotated: bool = True):
        """TAN WCS whose 'pixels' are undistorted FOV millimetres (SIR setPointingFromImage)."""
        theta = np.deg2rad(-pa + 90.0)
        cdelt = fov_scale_as_per_mm / 3600.0
        cd = np.array([[cdelt * np.cos(theta), -cdelt * np.sin(theta)],
                       [cdelt * np.sin(theta), cdelt * np.cos(theta)]])
        if rotated and self.grism != "RGS270":
            cd = np.array([[0, -1], [1, 0]]) @ cd
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crpix = [1.0, 1.0]
        w.wcs.crval = [ra, dec]
        w.wcs.cd = cd
        w.wcs.cunit = ["deg", "deg"]
        self.wcs = w
        return self

    def set_wcs_from_arrays(self, crval, cd):
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crpix = [1.0, 1.0]
        w.wcs.crval = list(crval)
        w.wcs.cd = np.asarray(cd, dtype=float)
        w.wcs.cunit = ["deg", "deg"]
        self.wcs = w
        return self

    @property
    def extra_tilt_matrix(self):
        c, s = np.cos(np.deg2rad(self.extra_tilt)), np.sin(np.deg2rad(self.extra_tilt))
        return np.array([[c, -s], [s, c]])

    # ---- evaluation ------------------------------------------------------ #
    def undistorted_mm(self, ra, dec) -> np.ndarray:
        ra = np.atleast_1d(np.asarray(ra, float)); dec = np.atleast_1d(np.asarray(dec, float))
        xy = self.wcs.wcs_world2pix(np.column_stack([ra, dec]), 0)
        return np.asarray(xy, float)

    def pivot_mm(self, ra, dec) -> np.ndarray:
        u = self.undistorted_mm(ra, dec)
        return np.column_stack([_poly.polyval2d(u[:, 0], u[:, 1], self.pivot_x),
                                _poly.polyval2d(u[:, 0], u[:, 1], self.pivot_y)])

    def reference_mm(self, ra, dec, order: int = 1, apply_tilt: bool = True) -> np.ndarray:
        """FOV position (mm) of the reference wavelength of `order`."""
        u = self.undistorted_mm(ra, dec)
        if order == 1:
            X, Y = self.ref_x, self.ref_y
            pos = np.column_stack([_poly.polyval2d(u[:, 0], u[:, 1], X),
                                   _poly.polyval2d(u[:, 0], u[:, 1], Y)])
        elif order == 0:
            X, Y = self.disp[0]["X"], self.disp[0]["Y"]
            pos = np.column_stack([_poly.polyval2d(u[:, 0], u[:, 1], X),
                                   _poly.polyval2d(u[:, 0], u[:, 1], Y)])
        else:  # higher orders: displacement from the 1st-order absolute position
            ab = self.reference_mm(ra, dec, 1, apply_tilt=False)
            X, Y = self.disp[order]["X"], self.disp[order]["Y"]
            pos = ab + np.column_stack([_poly.polyval2d(ab[:, 0], ab[:, 1], X),
                                        _poly.polyval2d(ab[:, 0], ab[:, 1], Y)])
        if apply_tilt and self.extra_tilt != 0.0:
            piv = self.pivot_mm(ra, dec)
            pos = (pos - piv) @ self.extra_tilt_matrix.T + piv
        return pos

    def reference_lambda(self, order: int) -> float:
        return self.ref_lambda if order == 1 else self.disp[order]["ref_lambda"]


# --------------------------------------------------------------------------- #
# detector layout: FOV mm <-> detector pixel
# --------------------------------------------------------------------------- #
@dataclass
class DetectorLayout:
    cd: dict      # det id -> 2x2 (mm per pixel)
    crpix: dict   # det id -> (2,) FITS 1-based
    crval: dict   # det id -> (2,) mm
    rotated: bool
    pixel_size: float
    grism: str = "RGS000"

    @classmethod
    def from_csv(cls, path, rotated: bool = True, grism: str = "RGS000") -> "DetectorLayout":
        rows = [l for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]
        header = [h.strip() for h in rows[0].split(",")]
        cd, crpix, crval = {}, {}, {}
        for line in rows[1:]:
            vals = dict(zip(header, [v.strip() for v in line.split(",")]))
            det = vals["SCE_POS"]
            m = np.array([[float(vals["CD1_1"]), float(vals["CD1_2"])],
                          [float(vals["CD2_1"]), float(vals["CD2_2"])]])
            cp = np.array([float(vals["CRPIX1"]) - 4.0, float(vals["CRPIX2"]) - 4.0])  # prescan
            cv = np.array([float(vals["CRVAL1"]), float(vals["CRVAL2"])])
            if rotated:
                m, cp = cls._rotate_wcs(m, cp, det, grism)
            cd[det], crpix[det], crval[det] = m, cp, cv
        px = float(np.mean([np.sqrt(abs(np.linalg.det(m))) for m in cd.values()]))
        return cls(cd=cd, crpix=crpix, crval=crval, rotated=rotated, pixel_size=px, grism=grism)

    @staticmethod
    def _rotate_wcs(cd, crpix, det_id, grism):
        cd = cd.copy(); crpix = crpix.copy()
        if det_id[0] in ("3", "4"):          # 180 deg flip of the upper detector rows
            cd *= -1
            crpix = DET_SIZE - crpix
        if grism == "RGS270":
            return cd, crpix
        cd = cd @ np.array([[0, -1], [1, 0]])
        crpix = np.array([[0, 1], [-1, 0]]) @ crpix
        crpix[1] += DET_SIZE
        return cd, crpix

    def mm_to_pix(self, xy_mm, det_id: str) -> np.ndarray:
        """FOV mm -> 0-based pixel on detector `det_id` (may fall outside)."""
        w = np.atleast_2d(np.asarray(xy_mm, float)).copy()
        if self.rotated:
            w = np.column_stack([-w[:, 1], w[:, 0]])   # SIR FOV -> original SIM FOV
        inv = np.linalg.inv(self.cd[det_id])
        pix = (w - self.crval[det_id]) @ inv.T + self.crpix[det_id]
        return pix - 1.0

    def pix_to_mm(self, xy_pix, det_id: str) -> np.ndarray:
        p = np.atleast_2d(np.asarray(xy_pix, float)) + 1.0
        w = (p - self.crpix[det_id]) @ self.cd[det_id].T + self.crval[det_id]
        if self.rotated:
            w = np.column_stack([w[:, 1], -w[:, 0]])
        return w

    def corners_mm(self, det_id: str) -> np.ndarray:
        c = np.array([[-PIXEL_OFFSET, -PIXEL_OFFSET], [NX - 1 + PIXEL_OFFSET, -PIXEL_OFFSET],
                      [NX - 1 + PIXEL_OFFSET, NY - 1 + PIXEL_OFFSET], [-PIXEL_OFFSET, NY - 1 + PIXEL_OFFSET]])
        return self.pix_to_mm(c, det_id)

    @staticmethod
    def inside(pix) -> np.ndarray:
        p = np.atleast_2d(pix)
        return ((p[:, 0] >= -PIXEL_OFFSET) & (p[:, 0] <= NX - 1 + PIXEL_OFFSET) &
                (p[:, 1] >= -PIXEL_OFFSET) & (p[:, 1] <= NY - 1 + PIXEL_OFFSET))

    def detector_of(self, xy_mm) -> np.ndarray:
        """Detector id ('' if none) for each FOV position."""
        xy = np.atleast_2d(xy_mm)
        out = np.array([""] * len(xy), dtype=object)
        for det in DET_IDS:
            m = self.inside(self.mm_to_pix(xy, det)) & (out == "")
            out[m] = det
        return out


# --------------------------------------------------------------------------- #
# 2D Chebyshev "global" model giving a local 1D Chebyshev model along the trace
# --------------------------------------------------------------------------- #
@dataclass
class ChebGlobal2D:
    matrices: list            # k -> (xdeg+1, ydeg+1) matrix for local coefficient k
    xrange: tuple
    yrange: tuple
    local_range: tuple

    @classmethod
    def from_xml(cls, path, grism: str, tilt: int, order: int) -> "ChebGlobal2D":
        root = ET.parse(str(path)).getroot()
        el = _grism_block(root, grism, tilt)
        for so in el.findall("SpectralOrder"):
            if int(so.findtext("Order")) == int(order):
                break
        else:
            raise KeyError(f"no SpectralOrder {order} in {path}")
        deg = int(so.findtext("LocalModelDeg"))
        mats = [None] * (deg + 1)
        for m in so.findall("Model"):
            mats[int(m.findtext("Degree"))] = _matrix(m.find("Matrix"))
        gr = so.find("GlobalRanges")
        return cls(matrices=mats,
                   xrange=tuple(_floats(gr.findtext("DoubleRow1"))),
                   yrange=tuple(_floats(gr.findtext("DoubleRow2"))),
                   local_range=tuple(_floats(so.findtext("LocalRanges"))))

    @classmethod
    def from_loctab_json(cls, d: dict) -> "ChebGlobal2D":
        mats = [np.asarray(d["Matrices"][str(k)]["_value_"]) for k in range(len(d["Matrices"]))]
        return cls(matrices=mats, xrange=tuple(d["GlobalRangesX"]["_value_"]),
                   yrange=tuple(d["GlobalRangesY"]["_value_"]), local_range=tuple(d["LocalRanges"]["_value_"]))

    def local_coeffs(self, xy_mm) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy_mm, float))
        xn = _normalize(xy[:, 0], *self.xrange)
        yn = _normalize(xy[:, 1], *self.yrange)
        return np.column_stack([_cheb.chebval2d(xn, yn, M) for M in self.matrices])

    def eval_local(self, coeffs, t) -> np.ndarray:
        """Evaluate local models (rows of `coeffs`) at `t` (same length or broadcast)."""
        tn = _normalize(np.asarray(t, float), *self.local_range)
        coeffs = np.atleast_2d(coeffs)
        if tn.ndim == 1 and tn.shape[0] == coeffs.shape[0] and coeffs.shape[0] != 1:
            return np.array([_cheb.chebval(tn[i], coeffs[i]) for i in range(len(tn))])
        return np.array([_cheb.chebval(tn, c) for c in coeffs])


# --------------------------------------------------------------------------- #
# full frame geometry
# --------------------------------------------------------------------------- #
@dataclass
class FrameGeometry:
    grism: str
    tilt: int
    opt: OpticalModel
    det: DetectorLayout
    crv: dict      # order -> ChebGlobal2D
    ids: dict
    psf: dict
    wlen_range: tuple = (11900.0, 19000.0)
    yshift_mm: dict = field(default_factory=dict)   # det id -> cross-dispersion slit shift (SIR issue 28974)

    def set_yshifts(self, shifts: dict):
        self.yshift_mm = {str(k): float(v) for k, v in shifts.items()}
        return self

    @classmethod
    def from_frame(cls, frame_path, calib_dir, orders=(0, 1), fov_scale_as_per_mm: float | None = None,
                   mdb_xml=None) -> "FrameGeometry":
        calib_dir = Path(calib_dir)
        hdr = fits.getheader(str(frame_path), 0)
        grism = hdr["GWA_POS"]; tilt = int(round(float(hdr["SIR_TILT"])))
        rotated = bool(hdr.get("SIR_ROT", False))
        xmls = {k: next((calib_dir / "xml").glob(p)) for k, p in
                dict(opt="SIR_Calib_Opt_*", crv="SIR_Calib_Crv_*crv_cal*", psf="SIR_Calib_Crv_*psf_cal*",
                     ids="SIR_Calib_Ids_*").items()}
        opt = OpticalModel.from_xml(xmls["opt"], grism, tilt)
        if fov_scale_as_per_mm is None:
            fov_scale_as_per_mm = cls.fov_scale_from_mdb(mdb_xml or next((calib_dir / "xml").glob("EUC_MDB_*")))
        opt.set_pointing(float(hdr["SIR_RA"]), float(hdr["SIR_DEC"]), float(hdr["SIR_PA"]),
                         fov_scale_as_per_mm, rotated=rotated)
        metro = ET.parse(str(xmls["opt"])).getroot().findtext(".//Parameter[Key='SIR_DetectorsMetrology']/Files/FileName")
        det = DetectorLayout.from_csv(calib_dir / "data" / metro, rotated=rotated, grism=grism)
        crv = {o: ChebGlobal2D.from_xml(xmls["crv"], grism, tilt, o) for o in orders}
        ids = {o: ChebGlobal2D.from_xml(xmls["ids"], grism, tilt, o) for o in orders}
        psf = {o: ChebGlobal2D.from_xml(xmls["psf"], grism, tilt, o) for o in orders}
        return cls(grism=grism, tilt=tilt, opt=opt, det=det, crv=crv, ids=ids, psf=psf)

    @staticmethod
    def fov_scale_from_mdb(mdb_xml) -> float:
        """SIR's FOV scale in arcsec per 'mm' (DpdMdbDataBase.get_nisp_fov_scale_mm):
        TelescopeNISPPlatescale [arcsec/um] / NISPPlateScaleNominal * 1000."""
        txt = Path(mdb_xml).read_text()
        def val(key):
            m = re.search(r'title="%s">.*?<DoubleValue>([^<]+)<' % re.escape(key), txt, re.S)
            return float(m.group(1))
        return val("SpaceSegment.PLM.TelescopeNISPPlatescale") / \
            val("SpaceSegment.Instrument.NISP.NISPPlateScaleNominal") * 1000.0

    # ---- trace ----------------------------------------------------------- #
    def _sign(self):
        if self.grism in ("RGS000", "BGS000"):
            return +1.0, +1.0
        if self.grism == "RGS180":
            return -1.0, -1.0
        raise NotImplementedError(self.grism)

    def trace_mm(self, ra, dec, lambdas, order: int = 1, slit_mm=None, det_id=None):
        """FOV mm positions along the trace.

        Returns (ref_mm (N,2), pos_mm (N,L,2), sigma_mm (N,L)).  `slit_mm`
        optionally replaces the reference positions (N,2) (SIR's virtual-slit rows).
        `det_id` (str or array of str) applies that detector's Y-shift to the slit.
        """
        ref = self.opt.reference_mm(ra, dec, order, apply_tilt=False) if slit_mm is None else np.atleast_2d(slit_mm)
        ref = np.array(ref, float, copy=True)
        if det_id is not None and self.yshift_mm:
            dets = np.atleast_1d(np.asarray(det_id, dtype=object))
            if dets.shape[0] == 1 and len(ref) > 1:
                dets = np.repeat(dets, len(ref))
            ref[:, 1] += np.array([self.yshift_mm.get(str(d), 0.0) for d in dets])
        lam = np.atleast_1d(np.asarray(lambdas, float))
        ids_c = self.ids[order].local_coeffs(ref)
        crv_c = self.crv[order].local_coeffs(ref)
        psf_c = self.psf[order].local_coeffs(ref)
        ddisp = self.ids[order].eval_local(ids_c, lam)              # (N, L) mm along dispersion
        dx = np.array([self.crv[order].eval_local(crv_c[i:i + 1], ddisp[i])[0] for i in range(len(ref))])
        sig = np.array([self.psf[order].eval_local(psf_c[i:i + 1], ddisp[i])[0] for i in range(len(ref))])
        sx, sy = self._sign()
        pos = np.stack([ref[:, None, 0] + sx * ddisp, ref[:, None, 1] + sy * dx], axis=-1)
        if self.opt.extra_tilt != 0.0:
            piv = self.opt.pivot_mm(ra, dec)[:, None, :]
            pos = (pos - piv) @ self.opt.extra_tilt_matrix.T + piv
        return ref, pos, sig

    def trace_pix(self, ra, dec, lambdas, order: int = 1, det_id: str | None = None):
        """Pixel trace on `det_id` (or the detector holding the reference position)."""
        ref0 = self.opt.reference_mm(ra, dec, order, apply_tilt=False)
        if det_id is None:
            det_id = self.det.detector_of(ref0)
        else:
            det_id = np.array([det_id] * len(ref0), dtype=object)
        ref, pos, sig = self.trace_mm(ra, dec, lambdas, order, det_id=det_id)
        out = np.full(pos.shape, np.nan)
        for i, d in enumerate(det_id):
            if d:
                out[i] = self.det.mm_to_pix(pos[i], d)
        return det_id, out, sig / self.det.pixel_size
