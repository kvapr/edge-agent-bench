# Methodology

This document describes how the measurements in this benchmark are taken, so
that any number in a report can be reproduced and its uncertainty is visible.
This is version 2, as run from 2026-10-14 to 2026-10-17 and revised 2026-10-18. Numbers quoted here come
from that campaign on the reference setup below (QCS9075, GenieX v0.8.0,
this four-tool suite, single-turn requests). Version 1 (2026-09-10 to
2026-09-16, results under `results/2026-09-*/`, written up only in an
unpublished draft) is summarised under "Changes from version 1".

## What is measured

The benchmark asks whether a small language model, served on the device
through the software stack the device ships with, can act as a tool-calling
agent. Each test case is one single-turn request: a system prompt, four tool
definitions and one user instruction. The model must answer with a structured
tool call, which an OpenAI-compatible server returns in the `tool_calls` field
of its reply. A case passes end to end when the first returned call has the
expected tool, the expected command, the target device and every expected
parameter; tool selection, command choice and parameter accuracy are also
reported on their own. Further calls in the same reply are ignored, and their
number is not recorded. Every case expects exactly one call: the suites hold no
request whose right answer is to call no tool, and none that needs several
calls. The suites, their labelling conventions and the grading rules are
described in [suites/README.md](../suites/README.md).

Each request asks for temperature 0 and at most 512 output tokens, with any
reasoning mode switched off. Temperature 0 is chosen for repeatability; it
differs from the sampling settings some model cards recommend. When the tools
reach the model, a request is about 2,700 prompt tokens by the server's own
count (medians of 2,710 to 2,864 where the model's chat template renders the
tools, and 3,248 for the compiled SmolLM2-1.7B-Instruct bundle, where the
server added them), of which the four tool definitions are about 2,230.
Answers in those configurations are short, a median of 41 to 70 tokens.

A request is served in two phases. In prefill the server reads the whole
prompt, many tokens at a time, and fills its key-value cache; in decode it
generates the answer one token at a time. Prefill is bound mainly by compute
and decode mainly by memory bandwidth, so a processor that speeds up one need
not speed up the other, and the ratio of prompt to answer decides which phase
dominates a turn. Across the configurations whose prompts carried the tools,
the per-request median prefill share of an agent turn was 87.6 to 93.7% on the
CPU, 53.3 to 54.0% on NPU (llama.cpp) and 41.5 to 48.0% on NPU (compiled).

Alongside accuracy the benchmark records the latency of every request, the
prompt and output token counts the server reports, and the server's own
prefill and decode timing for each response. Separate throughput runs measure
time to first token, prefill rate and decode rate across prompt lengths.

## Models

Models are chosen from the vendor's catalog rather than by hand. For the
Qualcomm QCS9075 that means Qualcomm AI Hub, release 0.64.0, checked on
2026-10-14, which gave thirteen models (`campaign/v2_models.json`). The seven
with a compiled bundle for the QCS9075 are measured on all three serving
paths; the six that the catalog supports only through its llama.cpp route are
measured on the two llama.cpp paths, 33 planned model and path steps in all.
Models with an optional reasoning mode are run with it switched off, as an
agent would be for latency. Models are named after their artifacts, so the
catalog's Granite-4.0-Micro entry, the hybrid model with architecture
`granitehybrid` in its GGUF header, is named Granite-4.0-H-Micro.

The llama.cpp paths read a GGUF file in Q4_0, llama.cpp's basic 4-bit format
(weights in blocks of 32, each block with one 16-bit scale). How each Q4_0 file
was made differs by model, as `campaign/v2_models.json` and
`analysis/data/artifacts.json` record. Nine were quantised with an importance
matrix (imatrix), which weights rounding error by activation statistics from
sample text: Qwen3-1.7B, Qwen3-4B, Qwen3-4B-Instruct-2507, Qwen3.5-0.8B,
Qwen3.5-2B, Granite-4.0-H-Micro, Ministral-3-3B-Instruct-2512,
Phi-4-mini-instruct and SmolLM2-1.7B-Instruct, the last a community file
because the catalog lists no GGUF for it. Qwen3-0.6B is a plain Q4_0 without
an importance matrix. The two Gemma 4 files carry no importance matrix and,
by Google's public model card, are its quantisation-aware-trained (QAT) Q4_0
release. The catalog publishes no Q4_0 for Qwen3-8B, so that file was
quantised for this benchmark from a public BF16 GGUF, without an importance
matrix. Comparisons between models on the llama.cpp paths, and between a GGUF
and its compiled bundle, therefore include differences of quantisation recipe.
The compiled bundles are W4A16 (4-bit weights, 16-bit activations), compiled
ahead of time for the NPU with QAIRT 2.45, split into parts, with a 4,096-token
context.

## Serving paths

Every path is served by the same binary, GenieX v0.8.0, so the server version
is not a variable between paths. The binary holds two runtimes, llama.cpp and
Qualcomm's QAIRT, each with its own chat-template and tool handling, so how a
prompt is rendered can still differ between the llama.cpp paths and NPU
(compiled). CPU (llama.cpp) runs the Q4_0 GGUF with llama.cpp on the CPU cores.
NPU (llama.cpp) runs the same file with the server's default `npu` compute
setting, so llama.cpp offloads the operations it supports to the Hexagon NPU.
NPU (compiled) runs the W4A16 bundle on QAIRT. The NPU of this chip is built
from two Hexagon tensor processors (HTP), which the CPU drives through FastRPC,
and under these defaults each NPU path used one: the llama.cpp backend logged
a single device (HTP0) in every configuration that loaded, and the bundles'
backend configuration names one core. GenieX also offers a `hybrid` llama.cpp
setting, which its documentation calls the fast path on Snapdragon,
and a GPU path; neither was measured.

The llama.cpp context is set to 4,096 tokens to match the compiled bundles
(`--nctx 4096 --keepalive 3600` on every path). The NPU power mode is left at
the server default, which GenieX documents as burst, and the bundles' backend
configuration requests the burst profile. The runner writes that label into
each server record, but the server logs no power-mode line, so the mode in use
was not independently observed. Each model and path gets a freshly started
server, which serves its four runs in turn (canonical, paraphrase and the two
throughput runs); diagnostic runs, made after the planned runs, get a server of their
own. Before any NPU server starts, the NPU's tensor processors are reset and
the board's resident container, which also uses the NPU, is stopped, because the NPU serves
one model at a time; during the CPU runs that container stays up (see "Readiness
gate and thermal trace"). A step resumed after an interruption starts a new
server, and the server record then describes the latest instance; in version 2
this applies to Qwen3-4B on the CPU, whose tool-calling runs were served by an
earlier instance with the same binary and options. The CPU path runs at the
server's default llama.cpp CPU settings as logged, untuned and without core
pinning; the CPU frequency governor is not recorded.

The runner's load check waits up to 900 s for a first reply. A model that does
not load gets a load-failure record, holding the check's reply and the relevant
server log lines, in place of results; each recorded failure is one attempt
under default settings, and the settings tried as workarounds for
Qwen3-4B-Instruct-2507 are recorded separately. Under v0.8.0 defaults NPU
(llama.cpp) loaded GGUF files up to 1,056,782,912 bytes and none from
1,214,873,856 bytes up; most failures were logged while mapping a buffer to the
NPU, and the mechanism is not resolved.

## Repetitions and items

Each configuration runs the 65 canonical cases three times in sequence and the
195 paraphrases once, in suite order. Repetitions run the whole suite before
starting again, so order and thermal effects spread evenly. An item is a
canonical case, scored by its pass fraction over the three repetitions, or a
paraphrase, scored by its single outcome; the pooled score weights the 260
items equally.

The split was chosen from version 1, where accuracy at temperature 0 was
identical across repetitions in every configuration but one. In version 2 that
holds only for NPU (compiled): six of the seven bundles repeated their reply
exactly in every canonical case, and the seventh (Gemma-4-E4B-it, whose prompts
lacked the tools) varied only among failing replies, so no compiled case
changed grade. On CPU (llama.cpp) and NPU (llama.cpp), every request asked for
temperature 0, yet the three replies were identical (same tool name, arguments
with keys sorted, and message text) in only 58.0% and 33.1% of canonical cases,
and per-repetition accuracy varied in most configurations, above all the weaker
ones; the two llama.cpp paths hold different models, and for the same GGUF
they were equally repeatable. The mechanism is not established. Differing text
replies usually differ from their first character, which is consistent with
sampling rather than with numerical drift that accumulates over a reply; it
would also follow from non-deterministic prefill numerics that flip a
near-tied first token, which the records cannot exclude. The bundles'
configuration files declare a sampled default with a fixed seed, and the server
does not record the sampler it applied. A rerun with greedy decoding forced,
for example by top-k 1 or a fixed seed, has not been made.

Three repetitions and one paraphrase pass mean different things on the two
kinds of path. On NPU (compiled) the repetitions repeat the reply (exactly,
for six of the seven bundles), so the canonical score rests on one draw per
case and the repetitions add information only about latency. On the llama.cpp
paths the canonical score averages three draws per case while each paraphrase
is a single draw, so paraphrase and pooled scores carry run-to-run variation
that a rerun would change. The item bootstrap below resamples items with the
outcomes observed in this run, so rerun variation enters it only as part of
the variation between items; it does not estimate how far a rerun of the same
items would move a score. Paired tests use a canonical case's majority outcome
over its three repetitions. Latency repeats closely. On the reference configuration,
Qwen3-4B-Instruct-2507, the median per-case coefficient of variation across
repetitions was 0.19% on the CPU and 0.80% on NPU (compiled).

## Warm-up and request uniqueness

Each run discards one warm-up request, because the first request after loading
is slower; the warm-up uses an input other than the first timed case. During
development, GenieX v0.8.0's llama.cpp server was seen to answer a request
byte-identical to the one before it with an empty reply, and the behaviour is
publicly reported against v0.9.0 (qualcomm/GenieX, issue 1554). The harness
therefore never sends the same request twice in a row: consecutive cases
differ, and so do the last case of one repetition and the first of the next. A
reply with no call, no text and zero token counts would be retried once after a
different spacer request and flagged, and recorded as an error if it persisted;
it is never graded. No such reply occurred in the campaign: none of its 10,530
tool-calling requests was retried or returned an error. Throughput runs add a
request number to every prompt for the same reason.

## Throughput

Throughput runs stream each response and time every token. Synthetic prompts
of about 256, 512, 1,024, 2,048 and 2,800 tokens, the last close to the agent
prompt, each run five times with 128 generated tokens; a chat-shaped run uses a
prompt of about 100 tokens and 300 generated tokens. The filler is calibrated
to each target length with untimed requests. Time to first token is measured at
the client and includes request overhead, 22 to 85 ms beyond the server's
prompt time at the longest prompt. Decode rate is measured over exactly the
requested number of tokens, so it stays comparable even on a server that
generates past the limit, which the harness flags; every version 2 probe
repetition kept to the limit. Prefill and decode rates reported by the server
are kept beside the client measurements as a cross-check. Averaged per
configuration, client and server decode rates agreed within 0.1% on every
llama.cpp configuration and within 0.33% on the compiled bundles (largest
single repetition 1.15%).

Agent turn time is the client latency of a tool-calling request, so it holds a
full prefill whenever the server reuses no cached part of the prompt. In
llama.cpp's server the processed prompt count (`timings.prompt_n`) falls below
the prompt length when a cached prefix is reused, and OpenAI-style usage can
report cached tokens; whether GenieX follows the same convention was not
verified.
Neither field showed reuse in any request (0 of 9,360 tool-calling, 0 of 1,170
diagnostic and 0 of 720 probe requests), so every
measured turn re-reads the whole system prompt and tool definitions. A cause
has been attributed to GenieX's public source code but not verified, and any
turn time with a cached prefix is a model, not a measurement. Output lengths
also differ between paths, so a ratio of turn times combines the rates with the
answer lengths.

## Statistics

Accuracy is the mean over items as defined above. Its 95% confidence interval
is a percentile bootstrap over items, 10,000 resamples with the fixed seed
20261014, as every bootstrap here. The four items built on one base request (a
canonical case and its three paraphrases) are not independent, so pooled and
paraphrase scores also carry a cluster bootstrap interval that resamples the
65 base requests. A difference between two configurations is the mean over
shared items of the per-item difference, with a paired percentile bootstrap
that resamples items, or base requests, jointly. Two configurations are
compared with an exact McNemar test on each item's majority outcome, and
p values are Holm-corrected within the families of `analysis/analyze.py`:
serving paths within a model (28 suite-level and 14 pooled comparisons) and
models within a path (210 and 105). These families hold every measured
configuration, readable or not. Analyses with a family of their own (steps
along the frontier, paraphrase styles, command names called as tools) name it
in their tables.

Rank uncertainty comes from a joint cluster bootstrap: each resample draws 65
base requests with replacement and scores every readable configuration on the
same draw. A configuration's P(rank 1) is the share of resamples in which it
scores highest, a tie counting for each tied configuration, and its rank
interval holds the central 95% of its resampled ranks, tied scores sharing the
best rank; the same is done among the NPU configurations alone. Where a
difference is read as small or absent, the minimum detectable paired
difference is given: the difference a two-sided test would detect with 80%
power, from the normal approximation to McNemar and the observed rate of
discordant items, at alpha 0.05 and at the strictest Holm threshold of the
pair's family, 0.05/m. Because the pooled score weights paraphrases three to
one against canonical cases, the ranking is recomputed with the two suites
weighted equally and with each suite alone, and compared with the headline
ranking by Kendall tau-b.

The paraphrase effect is measured per base request, as the mean of its three
paraphrase outcomes minus the canonical pass fraction, with a bootstrap over
the 65 base requests and an exact sign test, Holm-adjusted across the
functional pool (readable configurations with canonical accuracy of at least
50%, a threshold set after the campaign) and, for comparison,
across every configuration whose prompts carried the tools.

Latency is summarised by the median of per-case mean latencies over the
canonical cases, with a bootstrap interval, and by nearest-rank 5th and 95th
percentiles. Speed-up over the CPU is computed per canonical case and
summarised by its median with a bootstrap interval, beside the ratio of the two
median turn times with a paired bootstrap interval. Seconds per correct action
divides the median turn time by pooled accuracy. The frontier of accuracy
against turn time holds the readable configurations that no other readable
configuration beats on one measure without losing on the other; each step along
it is tested with exact McNemar on pooled majority outcomes, Holm-adjusted over
the steps.

The analysis code is `analysis/analyze.py`, which writes per-file statistics
and the comparison families to `analysis/out/v2.json`, and the report modules
`analysis/report_*.py`, which reuse it through `analysis/report_common.py`. The
analysis needs the Python standard library and PyYAML, which the harness
grader it imports uses to read the suites.

## Reading a score as model capability

An as-served score measures the model and the serving stack together. Before a
score is read as the model's capability, the analysis checks that the stack
delivered the tools and returned the model's calls, and it reports the
diagnostics below separately. None of them replaces the as-served score, which
is the result.

The tools count as delivered when the median server-reported prompt of both
native suites reaches 1,500 tokens (`TOOLS_DELIVERED_MIN_PROMPT_TOKENS` in
`analysis/analyze.py`). The system prompt alone is about 470 tokens and the
tool definitions add about 2,230, so a shorter prompt cannot have held them. A
configuration is unreadable, under `report_common.unreadable_reason`, when its
tools were not delivered, or when none of its replies is a structured call
while more than half hold a text tool call in the model's own format that the
server did not convert. Unreadable configurations are reported with their
as-served score and the reason, and are left out of ranks, rank intervals, the
frontier and the functional pool.

A reply that writes its tool call as text in the message body fails grading,
because an agent reading the structured field would receive nothing. As a
diagnostic, `analyze.parse_text_tool_call` reads such text in four forms (the
Hermes `<tool_call>` JSON form, the XML-parameter form, and the Mistral-family
forms `[TOOL_CALLS]name[ARGS]{...}` and `[TOOL_CALLS][{...}]`); the recovered
call is graded by the harness grader and reported as intent accuracy. The
parser repairs nothing, so a call that is not valid as written still fails. For
Ministral-3-3B-Instruct-2512, whose correct calls in its own `[TOOL_CALLS]`
form came back as text, intent accuracy is the only reading of the model, and
it remains a diagnostic.

Three further readings were defined after the outputs had been seen, so they
are post hoc sensitivity analyses that show how much of a score rests on the
form of the reply. Brace repair accepts a text call that is valid JSON as
written or after exactly one appended closing brace, because almost all Qwen3
text calls are valid calls one closing brace short, mostly on the compiled
bundles; whether the model, the W4A16 numerics or the runtime causes this is a
hypothesis. Rename repair maps a call whose function name is a documented
command to that command's tool. Semantic regrading applies the grader audit's
rules to structured calls (port-number forms, an @ in the device name,
misspelt device keys, flat parameters, command synonyms and command-named
functions); text calls still fail.

The tools-in-prompt diagnostic (`harness/bench_tools_in_prompt.py`) runs the
canonical suite with the tool definitions written into the system prompt as
JSON lines, an instruction to answer in `<tool_call>` form, and no `tools`
field; cases, grader and provenance are the harness's own. It ran on the four
configurations whose prompts lacked the tools and, for calibration, on two
where native calls work: Qwen3-1.7B on NPU (llama.cpp) and
Qwen3-4B-Instruct-2507 on NPU (compiled). It shows whether a model can make
the calls when it sees the tools. Its score is not a bound on the native score
in either direction: as recorded, both calibration configurations scored below
their native 90.8% (83.6 and 76.9%), but the wrapper's parser rejects a reply
with one surplus closing brace, and with a tolerant JSON decoder that accepts
it, Qwen3-4B-Instruct-2507 on NPU (compiled) scores 100.0%.

## Provenance

Every result file records the harness version, the hashes of the harness, the
suite and the tool schema, the full system prompt and its hash, the request
configuration, the version string of the server binary that answered the
requests (`provenance.server_version`), and the endpoint's model list. The
board snapshot also carries a version field for the resident container's
runtime; that is not the answering server and is not used. Each model and path
also records the exact server command and options, the power-mode label, the
runtime configuration lines the server logged, the artifact file with its
SHA-256 and source, and, for compiled bundles, the context size and largest
part; `analysis/data/artifacts.json` adds what the GGUF headers and bundle
listings show (`analysis/inspect_artifacts.py`). Diagnostic runs record the
prompt actually sent and the wrapper's hash. The files also record the board's
OS, kernel, thermal zones and CPU load before and after the run. Two numbers
are comparable only when these records match on everything except the
variable under study. In version 2 every recorded hash and version string is
single-valued within its run type across the 102 run files (two harness hashes,
for tool-calling and probe runs, and two system-prompt hashes, for native and
diagnostic runs), and the hashes matched the repository's harness, suites and
prompts when the report was built.
`scripts/import_results.py` replaces the board address, client hostname and
home directory with placeholders when results enter the repository and alters
nothing else; the analysis gives identical numbers on the raw and the imported
copies.

## Readiness gate and thermal trace

Before the warm-up and before each repetition of a tool-calling run, and
before the warm-up and each prompt length of a throughput run, the harness
waits until the hottest thermal zone is below 55 C and the CPU is less than 30%
busy, polling every 10 s; a board that is not ready within 600 s aborts the run
without writing a result. Every version 2 gate released below both thresholds.
On the reference board the CPU and NPU zones throttle at 105 C and the hottest
reading of the campaign was 75.6 C, so the gate serves repeatability rather
than throttle avoidance. Throughout every run a sampler records the hottest
zone, the hottest CPU and NPU zones, and the load average every 20 seconds over
ssh, so each result carries an under-load temperature trace and any outside
load on the board is visible. [THERMAL.md](THERMAL.md) holds the version 1
thermal calibration.

The sampler is not free on the CPU path. The harness does not record request
times on the sampler's clock, so they were reconstructed by fitting the
load-average trace. On that fitted timeline a decode that overlaps a sampler
ssh session is a median 4.0% (205 ms) slower than the fastest repetition of its
case, against 0.1% for decodes 5 s or more from one, and each session inside a
prefill adds a median 60 ms, about 0.3% of CPU compute time at one session
every 20.6 s. The resident container that stays up during the CPU runs adds little:
setting every CPU prefill to the fastest repetition of its case moves the
median turn time by a median 0.3% (at most 0.9%) over the 13 CPU runs, which
bounds the variable part of what the container and the sampler do to prefill;
a load present in every repetition would not show.

## Validation

`harness/harness_selftest.py` runs the harness against a local fake server and
checks the output contract, including the cases above: a tool call written as
text, a server that ignores the output limit, and a server that returns empty
replies to repeated requests. `analysis/check_suites.py` checks the suites'
labels. The campaign runner, `campaign/run_campaign.py`, skips finished outputs,
so an interrupted campaign resumes where it stopped.

`analysis/report.py` runs the report modules on `results/v2` and writes every
number, table and figure input of the report to `analysis/out/report.json` and
`analysis/out/report_tables.md`; `analysis/figures.py` draws the figures from
that file without recomputing anything. Before publication each module's
output was recomputed from the result files by separately written code; the
disagreements found were fixed in the modules or traced to differences of
definition, which the table notes state, or to errors in earlier analyses.

The grader was audited on a stratified random sample (seed 20261014) of 126
failing and 57 passing repetitions from the configurations whose prompts
carried the tools, judged blind from the request and the reply by one rater
(`analysis/data/grader_audit.json`). Its rates come with Wilson intervals,
unweighted and reweighted by inverse sampling probability, and the agreement
of the semantic rules with the verdicts is given as Cohen's kappa. With one
rater the audit measures consistency with that rater, not agreement between
raters. `analysis/data/label_review.json` lists paraphrases whose expected
label is debatable, with the plausible alternatives; they are regraded
accepting the alternatives as a sensitivity check, the headline scores keep
the suite labels, and the flagged labels have not been adjudicated
independently.

## Reference setup (as of 2026-10)

The reference setup is a Qualcomm QCS9075 on the Dragonwing IQ-9075 EVK,
running Qualcomm Linux 1.6 (ver.1.2.1) and fan cooled, with GenieX v0.8.0
(QAIRT 2.45, llama.cpp 9425611) as the same binary on every path. The compiled
bundles come from Qualcomm AI Hub release 0.64.0, built with QAIRT
2.45.0.260326, and the client ran Python 3.12.3. The campaign ran from
2026-10-14 to 2026-10-17, about 63 board hours. GenieX v0.9.0, released on
2026-10-16 during the campaign, includes changes aimed at the NPU (llama.cpp)
load failures and at tools being dropped on the QAIRT vision-language path;
both changes first appeared in the pre-release v0.8.1-rc.4 on 2026-10-14.
Neither release was measured.

## Changes from version 1

Version 1 ran NPU (compiled) on GenieX v0.4.0 inside the board's resident
container and the llama.cpp paths on a 0.5.1 pre-release (0.5.1-alpha.2), so
path comparisons also crossed server versions; version 2 uses one server
binary, run outside that container. The v0.4.0 server ignored the output
limit, so its decode rates were averaged over 2,048 generated tokens, and its
2,800-token probe failed; version 2 measures decode over a fixed number of
tokens on every path. Version 1 recorded the runtime version of the container
rather than of the server that answered, so the llama.cpp revision it reported
(6ba5ef2) was the container's; version 2 records the answering binary. Version
1 left the llama.cpp context at its default, recorded no GGUF hash, and named
no bundle release in its result files; version 2 sets a 4,096-token context on
every path and records each artifact's SHA-256 and catalog release.

Version 1 measured three models chosen by hand; version 2 takes thirteen from
the catalog. Version 1 used two or five repetitions of the canonical suite
only; version 2 uses three repetitions plus the paraphrases. Version 1 kept a
reply's text only when it held no structured call, cut at 200 characters, and
stored no prompt token counts or server timings; version 2 keeps every reply up
to 4,000 characters with both. The 2026-09-10 runs used a development copy of
the tool schema and the later ones `suites/tools_nested.json`, which version 2
uses throughout. The system prompt no longer names a persona or a vendor, and
it now states, with an example, that the device name is passed without its @
prefix, because the earlier wording ("pass it as device_name") could be read
either way; in the version 1 4B and 8B runs every structured call (1,235 of
1,235) already omitted the @. The grader is unchanged: re-grading the 1,560
stored version 1 repetitions with it changes no outcome.

Version 2 adds the warm-up and empty-reply safeguards, request numbers in
throughput prompts, a measured chat-shaped probe (version 1 extrapolated the
chat turn from its short-prompt rates) and the 20-second thermal trace.
Version 1 reported means and standard deviations; version 2 adds item and
base-request bootstrap intervals, paired tests with Holm correction, rank
intervals, minimum detectable differences and the suite-weighting check,
together with the tools-delivered check, the readable rule, the intent and
tools-in-prompt diagnostics, the post hoc sensitivity analyses, the grader
audit and the label review.
