# arXiv submission sheet

Everything the arXiv submission form asks for, in the order it asks.

## Before submitting

- [ ] Work approval for publication.
- [ ] The GitHub repository is public (the paper links to it).
- [x] Author block (name and email, no affiliation) and the acknowledgment with the
      AI-assistance disclosure.
- [ ] The repository URL in the reproducibility statement (the last `TODO(Krish)` in
      `main.tex`), once the repository is public.
- [ ] Krish has read the final PDF; the acknowledgment states that he checked all results and text.
- [ ] An endorsement for the primary category (cs.CL), unless arXiv endorses the account automatically.
- [ ] A fresh package: `python3 paper/build.py --arxiv` (needs CairoSVG and Tectonic) writes
      `paper/arxiv-submission.tar.gz` with `main.tex`, `main.bbl` (arXiv does not run BibTeX),
      `refs.bib` and the figure PDFs. arXiv compiles it with pdflatex; check its preview PDF
      before announcing.

## Form fields

- **License:** CC BY 4.0 (lets others reuse the text and figures with attribution; matches the
  Apache-2.0 code).
- **Primary category:** cs.CL (Computation and Language).
- **Cross-lists:** cs.LG (Machine Learning), cs.PF (Performance), cs.AR (Hardware Architecture).
- **Title:** Tool calling through the deployed server on a Qualcomm QCS9075: 13 small models on
  three serving paths
- **Authors:** Krish Viswanathan
- **Comments:** 43 pages, 7 figures, 22 tables. Code, data and the full technical report:
  https://github.com/kvapr/edge-agent-bench
- **Abstract** (plain text, 1,567 of 1,920 characters):

Can a small model, served through the vendor's on-device stack, drive a tool-calling agent? Thirteen catalog models ran on a Qualcomm QCS9075 through GenieX v0.8.0 on up to three paths: one llama.cpp model file (GGUF) on the CPU and the NPU, and compiled NPU bundles. The 24 configurations that loaded each answered 390 single-turn requests offering four tools, graded on the deployed server's structured call. The serving stack, not the model, decided 14 of 33 planned steps: nine models did not load on the llama.cpp NPU path, two GGUF templates and a compiled prompt builder dropped the tools, and one model's calls came back as text. Three models reach 89.2 to 92.7% on the CPU at about 50 s per turn and cannot be ordered. The best NPU configuration, compiled Qwen3-4B-Instruct-2507, scores 80.0% (95% CI 75.0 to 84.6) at 6.71 s per turn, 11.0 percentage points (pp) below its CPU run for a turn 7.9 times shorter. No request reused a prompt prefix; a model of prefix reuse, not a measurement, cuts that speed-up to 1.22 to 1.51. The same GGUF on the NPU scored Qwen3-1.7B within 0.4 pp of the CPU, while compiled bundles scored significantly below the CPU GGUF for three of five Qwen3 models; for compiled Qwen3-1.7B and Qwen3-8B most failures are text tool calls one closing brace short. Rephrasing cost the 11 configurations with at least 50% on canonical requests 15.7 pp on average, and temperature 0 did not give repeatable replies on the llama.cpp paths. A later check under GenieX v0.9.3 found 10 of the 14 stack-decided steps still decided by the stack.
