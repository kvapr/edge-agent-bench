# Paper draft (LaTeX)

`main.tex` and `refs.bib` hold a condensed version of docs/REPORT.md for arXiv.
Every number in `main.tex` is taken from docs/REPORT.md or
analysis/out/report_tables.md, and `refs.bib` is converted from Appendix J of
the report.

`python3 paper/build.py` exports each docs/figures/*.svg to paper/figures/ as
PDF and compiles `main.pdf` with Tectonic. It needs CairoSVG
(`pip install cairosvg`) and the `tectonic` program; `--figures-only` stops
after the export, for building with `pdflatex main && bibtex main && pdflatex
main && pdflatex main` instead.
