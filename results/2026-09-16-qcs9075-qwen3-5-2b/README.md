# 2026-09-16: Qwen3.5-2B, cautionary note (not a matrix row)

Board: Qualcomm QCS9075 (Dragonwing IQ-9075 EVK), Qualcomm Linux 1.6.
Board addresses replaced with <board-ip>.

This directory is a cautionary note, not a serving-path matrix row. Qwen3.5-2B
was selected before checking Qualcomm's AI Hub catalog, and it should not have
been. It is kept as CPU data plus this note because it demonstrates why model
selection must be gated on the catalog.

## Why it does not belong in the matrix

Qwen3.5-2B has no compiled W4A16 bundle for the QCS9075 in Qualcomm's catalog;
it ships q4_0 GGUF only (same for Qwen3.5-0.8B). With no compiled bundle there
is no first-class NPU path, and its Gated DeltaNet layers have no Hexagon kernel
so the llama.cpp NPU path only partially offloads. The genuinely NPU-supported
edge LLMs on this board are the Qwen3 family (0.6B, 1.7B, 4B, 4B-Instruct-2507,
8B), each carrying a qcs9075 W4A16 bundle.

Selection rule going forward: an NPU row requires a `w4a16` entry under
`chipset_assets` for `qualcomm-qcs9075` in the model's release assets.

## Two things the CPU run showed

- The serving path does not surface this model's tool calls. Through GenieX's
  OpenAI endpoint the 2B emits calls as Hermes-style text in the message content
  (`<function=...><parameter=...>`), not as structured `tool_calls`. The harness
  reads the structured field, so it scored the run 0% end to end. A real agent
  on this endpoint would likewise get no usable call. The 4B and 8B do not have
  this problem; their calls arrive structured.
- Even re-parsed generously, the model is a weak tool-caller. Extracting the
  text calls from the saved output, only about 70% of case-reps produced a
  parseable call, and end-to-end accuracy is roughly 21% (tool selection ~67%).
  This parse is approximate; the point is that the model is not competitive on
  the task, so the 0% is not hiding a strong result.

## CPU throughput (valid regardless of the grading issue)

The one clean measurement here is CPU phase throughput, 5 reps, streaming:

    prompt(tok)   prefill   decode
    256           163       29.7
    512           169       29.5
    1024          165       29.1
    2048          161       28.2
    2800          159       27.1

Faster than the 4B on CPU (the model is half the size), 20.2 s median per
65-case turn. Files: the CPU phase probe and the CPU 65-case bench (the latter
scored 0% for the format reason above; kept for the raw output).

The client hostname in provenance is replaced with client.
