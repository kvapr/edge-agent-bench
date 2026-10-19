# 2026-09-15: Qwen3-4B CPU re-measurement (controlled, 5 reps, public schema)

Board: Qualcomm QCS9075 (Dragonwing IQ-9075 EVK), Qualcomm Linux 1.6, fan
cooled. Board addresses replaced with <board-ip>.

This re-measures the 4B CPU serving path so it is directly comparable to the
8B row (results/2026-09-14). The 2026-09-10 4B CPU figure used 2 repetitions
and the dev tool import; this run uses 5 repetitions and the public tool schema
(suites/tools_nested.json), matching the 8B run exactly.

## Files

- qwen3-4b-2507-q4_0-cpu-llamacpp-hygiene-5rep-20260915: 65-case agent bench,
  GenieX 0.5.1-alpha CLI, llama.cpp CPU, Q4_0 GGUF, 5 reps.
- phase-qwen3-4b-2507-cpu-llamacpp-q4_0-20260915: prefill/decode throughput on
  the same path, streaming method, 5 reps, 256 to 2800 token prompts.

## Results

Tool-calling accuracy, 65 cases, temperature 0:

- End to end 98.5% in all 5 reps (std 0.0), tool selection 100%, no unstable
  cases. Fully deterministic on this path at temperature 0.
- Latency 52.5 s median per turn, 52.8 s mean, 56.1 s p95.

Per-turn phase throughput (tokens per second, mean of 5 reps):

    prompt(tok)   prefill   decode
    256           74.5      17.1
    512           72.8      16.5
    1024          69.1      15.4
    2048          60.7      13.5
    2800          55.4      12.3

## Controlled comparison, 4B vs 8B on CPU

Same harness, 5 reps, public tool schema, temperature 0:

    model       e2e     tool_sel   median/turn   unstable
    Qwen3-4B    98.5%   100.0%     52.5 s        0
    Qwen3-8B    97.2%   98.8%      77.7 s        1

On this tool-calling workload the larger model is a net loss on CPU: about 1.5x
the latency, no accuracy gain, and less run-to-run stability. This is the
controlled form of the proposal's thesis that a bigger model buys nothing here.
The 4B CPU path (98.5%) also exceeds the 4B production compiled NPU path (92.3%,
results/2026-09-10), the serving-path capability gap.

The client hostname in provenance is replaced with client.
