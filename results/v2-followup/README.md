# Results, follow-up on GenieX v0.9.3 (QCS9075, 2026-10-21)

These are the results of the follow-up to the version 2 campaign, run on
2026-10-21 on the same Qualcomm Dragonwing IQ-9075 EVK (QCS9075, Qualcomm Linux
1.6) with GenieX v0.9.3 (QAIRT 2.50, llama.cpp e467e5b), the same harness,
suites and model files. Section 18 of docs/REPORT.md reports them, and
analysis/report_followup.py turns them into analysis/out/followup_tables.md.
The phases are defined in campaign/v2_followup.json.

- `triage/<model>/<path>/` holds one step of the triage: the 14 steps the
  stack decided under v0.8.0, three readable anchors and the GPU path for two
  models. `triage.json` is a tool-calling run of the first six canonical cases
  with one repetition, in the format of the version 2 runs; `server.json` and
  `load_failure.json` are as in results/v2. `triage_summary.json` is the
  runner's verdict per step under the report's readable rule.
- `lengthsweep/<model>/compiled/sweep_length.json` is a run of
  harness/sweep_prompt_length.py: for the first canonical case of each level,
  one tool, a compact and a benchmark series of prompts padded to
  server-reported lengths up to about 2,800 tokens, and the full four-tool
  request, each request graded. The compiled Gemma-4-E4B-it is measured, with
  the compiled Qwen3-4B-Instruct-2507 as a control.

Six requests show whether a configuration works under v0.9.3, not how
accurate it is; no step was run with the full version 2 protocol.

The board address in request URLs is replaced with `<board>`, the client
hostname with `client`, and the home directory with `~`, by
scripts/import_results.py; nothing else was altered, and
analysis/report_followup.py gives identical tables on the raw and the imported
copy. Model replies are kept verbatim.
