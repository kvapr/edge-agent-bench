#!/usr/bin/env python3
"""Build paper/main.pdf from paper/main.tex.

Exports every docs/figures/*.svg to paper/figures/<name>.pdf, then runs
Tectonic. The figures are drawn in DejaVu Sans, the font whose advance widths
analysis/figures.py assumes when it places labels. Needs CairoSVG
(pip install cairosvg) and Tectonic (https://tectonic-typesetting.github.io).

    python3 paper/build.py [--tectonic PATH] [--figures-only]
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

import cairosvg

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
SVG_DIR = os.path.join(REPO, "docs", "figures")
PDF_DIR = os.path.join(HERE, "figures")
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def export_figures():
    os.makedirs(PDF_DIR, exist_ok=True)
    svgs = sorted(glob.glob(os.path.join(SVG_DIR, "*.svg")))
    for svg in svgs:
        text = open(svg, encoding="utf-8").read().replace(FONT, "DejaVu Sans")
        out = os.path.join(PDF_DIR, os.path.basename(svg)[:-4] + ".pdf")
        cairosvg.svg2pdf(bytestring=text.encode("utf-8"), write_to=out)
    return len(svgs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tectonic", default=shutil.which("tectonic") or os.path.expanduser("~/.local/bin/tectonic"))
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    print(f"exported {export_figures()} figures to {os.path.relpath(PDF_DIR, REPO)}")
    if a.figures_only:
        return 0
    r = subprocess.run([a.tectonic, "--keep-logs", "main.tex"], cwd=HERE)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
