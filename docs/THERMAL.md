# Thermal behaviour on the QCS9075 (as of 2026-09)

Board: Qualcomm QCS9075, Dragonwing IQ-9075 EVK, Qualcomm Linux 1.6, active
fan. Every result file carries board thermal in its provenance: a snapshot
before the run, one after, and one before each repetition through the
readiness gate. This note summarises what those recordings show. All figures
are the hottest thermal zone in degrees Celsius.

## Headroom

The CPU and NPU zones have passive throttling trip points at 105 C, with a
second trip between 105 and 116 C. Through every run the cooling devices
stayed at state 0, so nothing throttled. The hottest reading observed under
load was 66.5 C, which leaves about 39 C of headroom. The board is not
thermally constrained for this workload.

## Bursts hide the real temperature

A short calibration burst is misleading. Six bench requests, about 47 seconds
of NPU work, peaked at 49.7 C on the NPU zones and recovered to idle within 30
seconds. The first sustained run then read 66.5 C on the same zones a few
minutes into its first repetition. A short burst underestimates sustained
temperature by roughly 17 C on this board, so thermal claims must come from
full runs, not warm-up probes.

## Each repetition starts cool

The readiness gate holds each repetition until the hottest zone is below 55 C,
so every repetition begins from a similar state. The recorded rep-start
temperatures:

    run (5 reps each)        warmup rep1 rep2 rep3 rep4 rep5  after
    Qwen3-4B-2507  NPU comp.  48.1  49.4 54.8 54.1 54.3 54.3   55.9
    Qwen3-8B       CPU        54.8  54.9 54.3 54.7 54.8 54.8   59.2
    Qwen3-4B-2507  CPU        54.6  54.4 54.8 54.8 54.7 54.8   59.2
    Qwen3.5-2B     CPU        54.0  53.7 54.6 54.5 54.6 54.6   59.2

Two things stand out. The NPU run started its first repetition cool (49.4 C)
because the board was fresh, then settled near 54 C once the gate began
pausing for cool-down. The CPU runs sit at the 55 C ceiling at every boundary:
the long CPU load keeps the board warm, and the gate releases each repetition
as soon as it dips just under the threshold. The CPU runs also finish hotter,
59.2 C right after the last repetition versus 55.9 C for the NPU run.

## Temperature in this range does not move the numbers

On the NPU run, repetitions that started at 49 C and at 54 C produced
identical per-case latencies. Within every repetition the first ten cases
averaged 6.9 s and the last ten 7.6 s, the same in each pass, which tracks
case content, since later levels carry heavier prompts, not thermal drift. A
board that was throttling would show that slope grow across repetitions, and
it does not. So the readiness gate is a repeatability mechanism, making every
repetition start from the same cool state, not a throttle-avoidance measure.

## What is not yet recorded

Only the rep-boundary snapshots and the before and after snapshots are logged,
plus the single 66.5 C reading noted by hand during the first NPU run. The
peak temperature during a repetition, when the CPU is fully loaded for the
whole 65-case pass, is not sampled for any of the CPU runs, which run the CPU
hotter and for longer than the NPU path. The next step is a lightweight
sampler that logs the hottest zone every 15 to 30 seconds for the duration of
a run, so a future row carries a continuous under-load trace rather than only
its cool endpoints.

Sources: docs/HYGIENE_NOTES.md and the provenance blocks of the 2026-09-10,
2026-09-14, 2026-09-15 and 2026-09-16 result files.
