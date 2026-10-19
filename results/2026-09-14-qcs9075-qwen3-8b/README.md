# 2026-09-14: Qwen3-8B on QCS9075, CPU served, NPU paths do not fit

Board: Qualcomm QCS9075 (Dragonwing IQ-9075 EVK), Qualcomm Linux 1.6, fan
cooled. Runtime versions are in each file's provenance block. Board addresses
replaced with <board-ip>.

This is the second model row of the matrix. The CPU serving path is measured
under the full hygiene harness (5 repetitions, mean and Bessel standard
deviation, warmup, readiness gate, provenance stamps). Both NPU serving paths
were attempted and neither loads the 8B on this board; the evidence is
recorded below and stands as a result, not a gap.

## Files

- qwen3-8b-q4_0-cpu-llamacpp-hygiene-5rep-20260914: 65-case agent bench on the
  board CPU (GenieX 0.5.1-alpha CLI, llama.cpp backend, Q4_0 GGUF), 5 reps,
  public tool schema passed with --tools-file suites/tools_nested.json.
- phase-qwen3-8b-cpu-llamacpp-q4_0-20260914: prefill/decode throughput on the
  same CPU path, streaming method, 5 reps, prompt lengths 256 to 2800 tokens.

The Q4_0 GGUF was quantized locally from the unsloth Qwen3-8B BF16 GGUF using
the same llama.cpp tree that serves the board (llama.cpp runtime hash 6ba5ef2),
because no upstream Q4_0 GGUF of Qwen3-8B is published; only Q4_K_M, which the
Hexagon backend cannot offload.

## CPU serving results

Tool-calling accuracy, 65 cases, temperature 0:

- End to end 97.2% (mean of 5 reps, sample std 0.72), per rep 96.9, 96.9,
  96.9, 96.9, 98.5.
- Tool selection 98.8% (std 0.67). Strict and lenient parameter accuracy both
  equal the end-to-end figure.
- By level: levels 1 to 3 at 100%, level 4 (multi-parameter) at 13.2 of 15.
- Latency 77.7 s median per turn, 77.5 s mean, 80.2 s p95.

Per-turn phase throughput (tokens per second, mean of 5 reps):

    prompt(tok)   prefill   decode   ttft(s)
    256           45.6      9.67     5.7
    512           44.7      9.43     11.6
    1024          43.4      9.11     23.6
    2048          40.0      8.36     51.1
    2800          37.7      7.98     74.1

One unstable case across reps: idx 63, a level-4 request ("assign interface 2
to vlan 30"). In four reps the model returned the command name as the tool; in
the fifth it returned the correct nested call. Unlike the compiled qairt path,
which was fully deterministic across reps at temperature 0, this GGUF CPU path
is not bit-deterministic, so accuracy carries a small run-to-run spread.

## NPU serving paths: neither loads the 8B on this board

Both NPU paths fail at model load against the same wall, the FastRPC
per-protection-domain memory map limit (roughly 1 to 1.5 GB) on this silicon.
The 4B fits (its buffers are about 344 MB); the 8B does not.

- Compiled W4A16 (production qairt path, GenieX v0.4.0). Qualcomm's public AI
  Hub bundle for qcs9075 (release v0.62.2) is a 5-way split whose largest part
  is 1.55 GB. The loader maps parts 0 to 3, including a 1.245 GB embedding
  part, then fails on the 1.55 GB final part with "Could not create context
  from binary for context index = 4 : err 1002" followed by "Create From
  Binary FAILED". Re-exporting at a finer split does not help by itself:
  for W4A16 the AI Hub export recompiles the pre-quantized, pre-split 5-part
  checkpoint unchanged, so the split count only changes if the model is
  re-quantized from full precision with AIMET, a separate effort with its own
  accuracy caveats.

- llama.cpp Hexagon (Q4_0 GGUF, the same file used for the CPU leg). Load
  fails while allocating a single roughly 1.008 GB repacked-weights buffer:
  "HTP0 buffer mapping failed : domain_id 3 ... error 0x00000001" then
  "unable to allocate HTP0-REPACK buffer". Two Hexagon sessions
  (GGML_HEXAGON_NDEV=2) and a smaller max buffer size (GGML_HEXAGON_MBUF=512)
  did not resolve it.

Partial offload with a reduced layer count would let the 8B run, but it would
make the NPU leg a CPU and NPU hybrid rather than a full offload, so it is not
comparable to the 4B NPU results and was not measured.

## Comparison to the 4B row (controlled)

Re-measured on 2026-09-15 under the same harness, 5 repetitions, and the public
tool schema (results/2026-09-15-qcs9075-qwen3-4b-cpu-controlled), the 4B CPU
path scores 98.5% end to end at 52.5 s per turn, fully deterministic across
reps. The 8B is slower (77.7 s per turn, about 1.5x) and slightly less accurate
(97.2% versus 98.5%), with one unstable case against the 4B's zero. On this
tool-calling workload a larger model is a net loss on CPU: more latency, no
accuracy gain, and less run-to-run stability.

The client hostname in provenance is replaced with client.
