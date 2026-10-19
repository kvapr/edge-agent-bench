# 2026-09-10: Qwen3-4B on QCS9075, CPU vs NPU, controlled

Board: Qualcomm QCS9075 (Dragonwing IQ-9075 EVK), Qualcomm Linux 1.6, fan
cooled. Runtime versions are in each file's provenance block.

Files:
- phase2-*-npu-qairt-w4a16: production compiled path (GenieX v0.4.0, qairt
  W4A16 bundle), streaming method, 5 reps. 2800-token row missing: with max_tokens
  ignored, the probe's long answer overflowed the 4096-token context and the
  server returned 400 (not a prompt-length limit; short answers at 2.9K
  tokens succeed). max_tokens is not honored by this
  server version, so decode rates are averages over 2048-token generations.
- phase2-*-cpu-llamacpp-q4_0: board CPU, llama.cpp backend (GenieX
  0.5.1-alpha CLI), Q4_0 GGUF, streaming, 5 reps.
- phase2-*-npu-llamacpp-q4_0: SAME GGUF and SAME binary on the NPU
  (llama.cpp Hexagon backend), streaming, 5 reps. Together with the CPU file
  this is the controlled pair: only the compute flag differs.
- qwen3-4b-2507-q4_0-{cpu,npu}-llamacpp-hygiene-2rep: 65-case agent bench on
  the two GGUF paths (2 reps each): 98.5% both, 52.5 s vs 14.5 s per turn.
- qwen3-4b-4k-geniex040-hygiene-5rep: 65-case bench on the production
  compiled path, 5 reps: 92.3%, 7.37 s per turn.

Headline (prompt 2044 tokens): prefill CPU 60 / NPU-llama.cpp 386 / compiled
NPU 1020 tok/s; time to first token 33.8 s / 5.3 s / 2.0 s; decode 13.4 /
7.9 / 13.1 tok/s. Accuracy: 98.5% on both GGUF paths, 92.3% on the compiled
W4A16 path.

Board addresses replaced with <board-ip> and the client hostname with client; the development tool-source label in
provenance is shown as dev-import, and file paths are given relative to the repo. Scripts that ran the matrix are in
scripts/ (lab-specific; read as an example of serialized legs with a
production-restore trap).
