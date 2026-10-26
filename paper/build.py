#!/usr/bin/env python3
"""Build paper/main.pdf from paper/main.tex.

Exports every docs/figures/*.svg to paper/figures/<name>.pdf, then runs
Tectonic. The figures are drawn in DejaVu Sans, the font whose advance widths
analysis/figures.py assumes when it places labels. Needs CairoSVG
(pip install cairosvg) and Tectonic (https://tectonic-typesetting.github.io).

    python3 paper/build.py [--tectonic PATH] [--figures-only] [--arxiv]

--arxiv also writes paper/arxiv-submission.tar.gz: main.tex, the main.bbl that
BibTeX produced (arXiv does not run BibTeX), refs.bib and the figure PDFs,
which arXiv compiles with pdflatex. The author's email is kept out of the
repository: the package's main.tex gets it from EAB_AUTHOR_EMAIL or from
~/edge-agent-bench-private/arxiv-author-email.txt.
"""
import argparse
import glob
import io
import os
import shutil
import subprocess
import sys
import tarfile

import cairosvg

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
SVG_DIR = os.path.join(REPO, "docs", "figures")
PDF_DIR = os.path.join(HERE, "figures")
ARXIV = os.path.join(HERE, "arxiv-submission.tar.gz")
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def export_figures():
    os.makedirs(PDF_DIR, exist_ok=True)
    svgs = sorted(glob.glob(os.path.join(SVG_DIR, "*.svg")))
    for svg in svgs:
        text = open(svg, encoding="utf-8").read().replace(FONT, "DejaVu Sans")
        out = os.path.join(PDF_DIR, os.path.basename(svg)[:-4] + ".pdf")
        cairosvg.svg2pdf(bytestring=text.encode("utf-8"), write_to=out)
    return len(svgs)


def author_email():
    email = os.environ.get("EAB_AUTHOR_EMAIL", "")
    path = os.path.expanduser("~/edge-agent-bench-private/arxiv-author-email.txt")
    if not email and os.path.exists(path):
        email = open(path).read().strip()
    if not email:
        raise SystemExit("set EAB_AUTHOR_EMAIL or write the address to " + path)
    return email


def arxiv_package():
    tex = open(os.path.join(HERE, "main.tex"), encoding="utf-8").read()
    blank = "\\newcommand{\\authorcontact}{}"
    if blank not in tex:
        raise SystemExit("main.tex no longer defines an empty \\authorcontact")
    tex = tex.replace(blank, "\\newcommand{\\authorcontact}{\\\\\\texttt{" + author_email() + "}}")
    members = ["main.tex", "main.bbl", "refs.bib"] + sorted(
        os.path.relpath(p, HERE) for p in glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    with tarfile.open(ARXIV, "w:gz") as tar:
        for m in members:
            if m == "main.tex":
                data = tex.encode("utf-8")
                info = tarfile.TarInfo("main.tex")
                info.size = len(data)
                info.mtime = int(os.path.getmtime(os.path.join(HERE, m)))
                tar.addfile(info, io.BytesIO(data))
            else:
                tar.add(os.path.join(HERE, m), arcname=m)
    return members


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tectonic", default=shutil.which("tectonic") or os.path.expanduser("~/.local/bin/tectonic"))
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--arxiv", action="store_true", help="also write the arXiv source package")
    a = ap.parse_args()
    print(f"exported {export_figures()} figures to {os.path.relpath(PDF_DIR, REPO)}")
    if a.figures_only:
        return 0
    cmd = [a.tectonic, "--keep-logs"] + (["--keep-intermediates"] if a.arxiv else []) + ["main.tex"]
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode == 0 and a.arxiv:
        members = arxiv_package()
        print(f"wrote {os.path.relpath(ARXIV, REPO)} with {len(members)} files")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
