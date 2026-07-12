#!/usr/bin/env python
"""
Regenerate talk figures with enlarged fonts, WITHOUT touching the paper figures.

For each target it re-runs the relevant cells of the paper-figure notebook, then
(via a monkeypatched Figure.savefig) scales every text element in the figure by a
per-figure factor and writes the PNG into talk/roman_ipac_2026/figures/ instead of
paper/figures/. The paper's committed figures are never modified.

Run from anywhere with the anaconda python that has matplotlib:
    /opt/anaconda3/bin/python talk/roman_ipac_2026/regen_figs.py [fig4] [fig6] ...
With no args it regenerates every target whose input data is present, and skips
(with a clear message) the ones whose gitignored checkpoints/data are absent.
"""
import os, sys, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text

REPO = Path(__file__).resolve().parents[2]
NBDIR = REPO / "paper" / "paper_figures"
OUTDIR = Path(__file__).resolve().parent / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# notebook, cells to run (in order), font scale factor
TARGETS = {
    "fig4":  ("figure_4_probe.ipynb",       [2, 3],    1.55),
    "fig6":  ("figure_6_injection.ipynb",   [1, 2],    1.45),
    "fig8":  ("figure_8.ipynb",             [1, 2],    1.45),
    "fig10": ("figure_9_concordance.ipynb", [1, 2],    1.45),
    "fig11": ("fig10_ood.ipynb",            [1, 2],    1.45),
}

_orig_savefig = Figure.savefig


def make_scaled_savefig(scale):
    def scaled_savefig(self, fname, *args, **kwargs):
        # ensure tick labels etc. are instantiated before we scale
        try:
            self.canvas.draw()
        except Exception:
            pass
        for t in self.findobj(Text):
            fs = t.get_fontsize()
            if fs:
                t.set_fontsize(fs * scale)
        base = os.path.basename(str(fname))
        out = OUTDIR / base
        kwargs.pop("fname", None)
        return _orig_savefig(self, str(out), *args, **kwargs)
    return scaled_savefig


def run_cells(nb_path, cell_idxs):
    nb = json.load(open(nb_path))
    cells = nb["cells"]  # index by ORIGINAL notebook position
    ns = {"__name__": "__main__"}
    for i in cell_idxs:
        assert cells[i]["cell_type"] == "code", f"cell {i} is not code"
        src = "".join(cells[i]["source"])
        # strip IPython magics / shell escapes / show()
        lines = []
        for ln in src.splitlines():
            s = ln.lstrip()
            if s.startswith("%") or s.startswith("!") or s.startswith("get_ipython"):
                continue
            lines.append(ln)
        exec(compile("\n".join(lines), f"<{nb_path.name}:cell{i}>", "exec"), ns)


def main():
    want = [a for a in sys.argv[1:] if not a.startswith("-")] or list(TARGETS)
    os.chdir(NBDIR)  # notebooks resolve repo root from cwd
    for key in want:
        if key not in TARGETS:
            print(f"[skip] unknown target {key!r}"); continue
        nb, cells, scale = TARGETS[key]
        Figure.savefig = make_scaled_savefig(scale)
        try:
            run_cells(NBDIR / nb, cells)
            print(f"[ok]   {key}: {nb} -> talk/roman_ipac_2026/figures/ (x{scale})")
        except (FileNotFoundError, AssertionError) as e:
            print(f"[MISS] {key}: input data absent ({e}). "
                  f"Run where the checkpoints/data live.")
        except Exception as e:
            print(f"[ERR]  {key}: {type(e).__name__}: {e}")
        finally:
            Figure.savefig = _orig_savefig
            plt.close("all")


if __name__ == "__main__":
    main()
