# Harness hygiene upgrade: notes and calibration (2026-09-10)

Purpose: make every published number defensible. Four parts: repetitions with
spread, provenance stamps per result, a readiness gate, and one-harness
comparisons. Implemented in bench_toolcall.py / soak_npu.py (originals kept
as *.pre-hygiene-20260910). Self-test: harness_selftest.py (fake endpoint,
no board needed).

## Thermal calibration (QCS9075 EVK, QLI 1.6, fan on, production qwen)

Six bench requests (~47 s of NPU work) against the production container:

| moment | cpu max | nsp (NPU) max | all zones max |
|---|---|---|---|
| idle before | 47.9 C | 46.5 C | 48.2 C |
| right after burst | 50.5 C | 49.7 C | 50.5 C |
| +30 s | 48.3 C | 47.6 C | 49.4 C |
| +90 s | 47.9 C | 47.2 C | 49.1 C |

Conclusion: at bench duty cycle the board barely heats and recovers in ~30 s.
The readiness gate therefore targets abnormal states (fan failure, stray
process, hot board after a long soak), not routine warmth. Default threshold
set to 55 C (idle + ~7). CPU-busy threshold 0.30 (idle measures ~0.02; the
July incident that produced a bogus 0.25 tok/s had a stray process at ~250%
CPU, which this catches).

Not yet measured: steady-state temperature under a 40-minute 5-repetition
bench or the 300-request soak. Record it on the first hygiene-mode run.

## Cold-start evidence for --warmup

Same burst, per-request latency: 12.15, 7.38, 6.08, 6.82, 6.48, 7.45 s.
The first request after idle is ~1.7x the rest. Earlier single-run results
included this outlier in their medians. Recommended: --warmup 1 for all
published runs.

## Repetition design

Whole-suite repeats (rep 1 all cases, rep 2 all cases, ...), not per-case
back-to-back, so thermal drift and order effects spread evenly and a case's
stability across reps is measured under realistic sequencing. Matches the
earlier seed-stability runs (three shuffled full-suite runs, all 96.9%).
Report per-rep accuracy list + mean + sample std, per-case latency
mean + std, and the list of unstable cases.

## CORRECTION (same day): sustained load runs far hotter than the burst

During the first hygiene-mode run (5 reps x 65 cases, continuous), the NPU
zones read 66.5 C a few minutes into rep 1, versus the 49.7 C peak after
the 47-second calibration burst. Short bursts underestimate sustained
temperature by ~17 C on this board. Implications:

- The 55 C gate is a between-repetition cool-down gate in practice: after
  each ~8-minute rep the run pauses until the board returns below 55 C.
  That is the intended behavior (a run starts cool every time), but the
  600 s timeout must comfortably exceed the cool-down time or the run
  aborts and loses its data. Measure the cool-down from 66 C on this run.
- The gate threshold should be justified against the board's actual
  throttling trip points, not against idle temperature. See trip-point
  readout below.
- Per-rep latency (from each case's reps list) tells whether the board
  throttled during a rep: rep 1 heats up from cool; if later cases in a rep
  are slower than early ones, that is thermal, not the model.

## Trip points (read during the run)

CPU and NPU thermal zones have passive throttling trip points at 105 C
(second trip 105-116 C). All cooling devices (idle-cpu1..7, GPU devfreq)
were at state 0 during the run: no throttling at 66 C, and ~39 C of headroom.

Conclusion: on this board with its fan, the readiness gate is NOT a
throttle-avoidance mechanism; it is a repeatability mechanism, making every
repetition start from the same cool state. 55 C is a fine value for that.
Cool-down observed: an NPU zone read 66.5 C under load and 52.7 C a few
minutes later, so the between-rep pause is short and the 600 s timeout is
safe. Boards without a fan (or in an enclosure) would need this re-measured.

## First hygiene-mode run: what the upgrade showed (2026-09-10)

Production Qwen3-4B via GenieX v0.4.0 NPU, 5 reps, warmup 1, 55 C gate.
- Accuracy: 92.3% e2e in all five reps (std 0), 100% tool selection, zero
  unstable cases. At temperature 0 on this stack the suite is fully
  deterministic across reps.
- Latency: per-rep medians 7.34-7.39 s; per-case std across reps median
  0.036 s, max 0.071 s. Run-to-run noise is ~0.5% of the value, so latency
  differences above ~0.2 s between configurations are real on this setup.
- Gate behavior: warmup at 48.1 C, rep 1 at 49.4 C, then the gate waited
  for cool-down and released reps 2-5 at 54.8/54.1/54.3/54.3 C. After the
  run: NPU 54.5 C. Reps starting at 49 C and at 54 C produced identical
  latencies, so temperature in this range does not affect results.
- Within-rep slope: the first 10 cases average 6.9 s and the last 10 average
  7.6 s, identically in every rep, so it is case content (later levels carry
  heavier prompts), not thermal drift. A drifting board would show the slope
  growing across reps.
- Cold start: warmup absorbed it; no 12 s outlier in the data.
Reference file: results/qwen3-4b-4k-geniex040-hygiene-5rep-20260910.json.
