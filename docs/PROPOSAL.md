# Proposal and project history

This page began as the project's one-page proposal, dated 2026-08-25. It now
records the question the project set out to answer, how the plan was carried
out, and how the proposal's candidate findings fared once they were measured
under one protocol. The results themselves are in [REPORT.md](REPORT.md) and
the protocol in [METHODOLOGY.md](METHODOLOGY.md).

## The question

The proposal asked whether a small model can drive a real agent on the device,
through the stack the device actually ships with. Four choices followed from
that question. Quality is scored through the deployed serving path on the
hardware rather than on a datacenter GPU. The NPU path is included from the
start. The serving path is a variable of the benchmark rather than a fixed
backdrop. The metrics are the ones an agent depends on, tool selection,
parameter accuracy and robustness to wording, measured beside throughput.

The proposal was prompted by the launch, on 2026-08-24, of an on-device
benchmark that measures speed on phones and laptops but takes quality from
reference runs elsewhere and includes no NPU path. Section 16 of the report
places edge-agent-bench among that and other benchmarks.

## The plan and what was done

The plan had four steps: bring the harness to a standard of repeated
measurement with standard deviations, recorded runtime versions and a thermal
readiness gate; re-measure CPU and NPU throughput under one harness; fill a
matrix of four to six small models on two serving paths; and write a
methodology and a report.

Version 1 ran from 2026-09-10 to 2026-09-16 on one Qualcomm QCS9075 board. It
measured Qwen3-4B-Instruct-2507, Qwen3-8B and Qwen3.5-2B, with the compiled NPU
path and the llama.cpp paths served by different GenieX builds. Its report
remained an unpublished draft, and its results are in the directories under
results/2026-09-*/.

Version 2 ran from 2026-10-14 to 2026-10-17 on the same board. It replaced the
hand-picked matrix with every model that Qualcomm AI Hub release 0.64.0
supports on this chip, 13 in all, served every path from one GenieX v0.8.0
binary, measured every model on each serving path the catalog supports for it
(three paths where the proposal had planned two), added the 195 paraphrases to
the canonical suite, and reported bootstrap intervals, paired tests and rank
intervals. Its results are in results/v2/ and its report is
[REPORT.md](REPORT.md).

## The candidate findings, measured

The proposal named three candidate findings. Each is restated here with the
version 2 evidence that now decides it; every number comes from
analysis/out/report_tables.md and is scoped to the QCS9075, GenieX v0.8.0 and
October 2026.

The first was that the shape of the workload decides whether the NPU matters.
It holds. For Qwen3-4B-Instruct-2507 at 2,044 prompt tokens, the compiled NPU
path read the prompt 16.27 times as fast as the CPU while decoding at 0.998 of
the CPU's rate. An agent turn of about 2,700 prompt tokens took 52.93 s on the
CPU and 6.71 s compiled, while a chat turn of about 100 tokens in and 300 out
took 18.37 s and 17.30 s. Two parts of the original statement are withdrawn:
its remark that public leaderboards rank only decode speed, which was not
established, and its throughput figures for the llama.cpp NPU path, which came
from version 1 and have no version 2 counterpart, since that model did not load
on that path under GenieX v0.8.0. Version 2 also found that the server re-read
the whole prompt on every request, so the agent-turn comparison holds for
requests without prompt-prefix reuse.

The second was that the serving path changes capability, not only speed. It
holds, in a more specific form. Moving a GGUF file from the CPU to the NPU
through llama.cpp did not measurably change accuracy (Qwen3-1.7B, 68.5% against
68.1%). Changing the serving stack did: Gemma-4-E4B-it scored 92.7% as a CPU
GGUF and 0.0% as a compiled bundle whose prompts never carried the tools;
Qwen3-4B scored 89.2% on the CPU and 14.6% compiled; and Ministral-3-3B scored
0.0% as served because the server returned its correct calls as text. The
original figure for this finding, a swing from 0% to 100% for the same
weights, matches no stored result and is withdrawn.

The third was that canonical phrasing overstates field accuracy. It holds.
Over the 11 configurations with canonical accuracy of at least 50%,
paraphrases cost a mean of 15.7 percentage points; Qwen3-8B on the CPU, for
example, scored 95.9% on canonical requests and 72.3% on paraphrases. The
original figure for this finding, a drop from 96.9% to 77.4%, matches no stored
result and is withdrawn.

The proposal also recorded a caveat that its throughput comparison mixed
serving stacks. Version 2 clears it for the comparison of the CPU with the
compiled path, measured with one server binary, five repetitions with standard
deviations, the answering binary's version recorded and decode timed over the
same 128 tokens on both paths. It is not cleared for the version 1 llama.cpp
NPU figures, which have no version 2 counterpart.

## Decisions

The benchmark is published independently, under the Apache-2.0 licence with
copyright held by Krish (decided 2026-10-15). It is hosted as a public
repository with the report and, later, a results page. Engagement with related
benchmarks is limited to sharing the NPU-path findings once the report is out
(decided 2026-08-26).
