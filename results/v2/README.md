# Results, version 2 (QCS9075, October 2026)

These are the raw results of the version 2 campaign, run between 2026-10-14
and 2026-10-17 on one Qualcomm Dragonwing IQ-9075 EVK (QCS9075, Qualcomm Linux
1.6) with GenieX v0.8.0 (QAIRT 2.45, llama.cpp 9425611) serving all three
paths. The protocol is described in docs/METHODOLOGY.md.

Each directory is one model, and each subdirectory one serving path: `cpu`
(llama.cpp on the CPU), `npu_llama` (the same GGUF on the llama.cpp Hexagon
backend) and `compiled` (the Qualcomm AI Hub W4A16 bundle on the NPU). A path
directory holds up to four result files and a record of the server.

- `canonical.json` and `paraphrase.json` are tool-calling runs (65 cases with
  3 repetitions, and 195 paraphrases with 1 repetition). Each repetition
  keeps the reply (up to 4,000 characters), token counts, server timings and
  its grading; the system prompt and tool subset are in the provenance block.
- `probe_agent.json` and `probe_chat.json` are throughput probes at agent-like
  prompt lengths and at a chat-like shape, with 5 repetitions each.
- `server.json` records the server command, runtime version, artifact hash
  and the runtime configuration the server logged.
- `load_failure.json` replaces the result files when the model did not load,
  with the load check and the relevant server log lines.
- `diag_tools_in_prompt.json` is a diagnostic run of the canonical suite with
  the tool definitions written into the system prompt (see the methodology).
  It exists only for configurations whose native prompts lacked the tools and
  for two calibration configurations, and it is not part of the headline
  numbers.
- `qwen3-4b-2507/npu_llama/workaround_tests.json` records the runtime settings
  tried when that model failed to load on the llama.cpp NPU path.

The board address in request URLs is replaced with `<board>`, the client
hostname with `client`, and the home directory with `~`, by
scripts/import_results.py; nothing else was altered, and the analysis gives
identical numbers on the raw and the imported copy. Three records of load
failures caused by a campaign-runner bug, fixed before the affected runs were
repeated, are not included.

Model replies are kept verbatim. Some failing replies invent content, such as
addresses in private network ranges and the names of network equipment
vendors; this is model output, not a detail of the test setup, and the report
does not quote it.

The tables and statistics are produced from this directory alone. The base
statistics come from analysis/analyze.py, and every table and figure in the
report from analysis/report.py and analysis/figures.py:

    python3 analysis/analyze.py results/v2 --md analysis.md --json analysis.json
    python3 analysis/report.py results/v2 --json analysis/out/report.json --md analysis/out/report_tables.md
    python3 analysis/figures.py analysis/out/report.json docs/figures
