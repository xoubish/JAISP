#!/usr/bin/env python
"""Regenerate the paper figures with enlarged labels for the talk.

Re-executes each figure notebook in paper/paper_figures with a patch cell
prepended: at savefig time every text object on the figure (titles, axis
labels, tick labels, legends, annotations) is scaled by TALKFIG_SCALE
(default 1.5) and the output is redirected to talk/roman_ipac_2026/figures/.
The notebooks' hard-coded fontsize= literals make an rcParams bump
insufficient, hence the walk over Text objects. paper/figures/ is untouched.

Usage:  python regen_talk_figures.py [notebook_stem ...]
        TALKFIG_SCALE=1.7 python regen_talk_figures.py figure_8

Notebooks are executed with their existing caches (detector/reconstruction
caches under paper_figures/), so no GPU-heavy stage reruns unless a cache
is missing.
"""
import json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NBDIR = REPO / "paper" / "paper_figures"
OUTDIR = Path(__file__).resolve().parent / "figures"
SCALE = float(os.environ.get("TALKFIG_SCALE", "1.5"))

# notebook stem -> figure file(s) it saves
NOTEBOOKS = {
    "figure_1": ["fig1_data.png"],
    "figure_2": ["fig2_architecture.png"],
    "figure_3_reconstruction": ["fig3_reconstruction.png"],
    "figure_4_probe": ["fig4_probe.png"],
    "figure_5_detection": ["fig5_detection.png"],
    "figure_6_injection": ["fig6_detection_performance.png"],
    "figure_8": ["fig8_head_correction.png"],
    "figure_9_concordance": ["fig10_concordance_field.png"],
    "fig10_ood": ["fig11_ood.png"],
}

PATCH = f"""
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.figure as _mf, matplotlib.text as _mt
from pathlib import Path as _P
_SCALE, _OUT = {SCALE}, _P({str(OUTDIR)!r})
_OUT.mkdir(parents=True, exist_ok=True)
_orig_savefig = _mf.Figure.savefig
def _talk_savefig(self, fname, *a, **k):
    if not getattr(self, '_talk_scaled', False):
        for _t in self.findobj(_mt.Text):
            try: _t.set_fontsize(_t.get_fontsize() * _SCALE)
            except Exception: pass
        self._talk_scaled = True
    return _orig_savefig(self, _OUT / _P(str(fname)).name, *a, **k)
_mf.Figure.savefig = _talk_savefig
print('[talkfig] savefig patched: scale', _SCALE, '->', _OUT)
"""

def run(stem):
    nb = json.load(open(NBDIR / f"{stem}.ipynb"))
    cell = {"cell_type": "code", "metadata": {}, "outputs": [],
            "execution_count": None, "source": PATCH}
    nb["cells"].insert(0, cell)
    tmp = NBDIR / f"_talkregen_{stem}.ipynb"
    json.dump(nb, open(tmp, "w"))
    t0 = time.time()
    r = subprocess.run(
        ["jupyter", "nbconvert", "--to", "notebook", "--execute",
         "--stdout", "--ExecutePreprocessor.timeout=3600", str(tmp)],
        cwd=NBDIR, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    tmp.unlink(missing_ok=True)
    ok = r.returncode == 0 and all((OUTDIR / f).exists() for f in NOTEBOOKS[stem])
    print(f"{'OK  ' if ok else 'FAIL'} {stem:26s} {time.time()-t0:6.0f}s  -> {', '.join(NOTEBOOKS[stem])}")
    if not ok:
        print(r.stderr[-3000:])
    return ok

if __name__ == "__main__":
    stems = sys.argv[1:] or list(NOTEBOOKS)
    fails = [s for s in stems if not run(s)]
    print("\nall done" if not fails else f"\nFAILED: {', '.join(fails)}")
    sys.exit(len(fails))
