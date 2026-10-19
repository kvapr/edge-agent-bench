#!/usr/bin/env python3
"""Version 1 against version 2 on the configurations both campaigns measured.

Version 1 ran from 2026-09-10 to 2026-09-16 (results/2026-09-*/), version 2
from 2026-10-14 to 2026-10-17 (the results directory passed in, results/v2),
on the same QCS9075 board. Version 1 files predate the labelled format, so
each is mapped to its configuration in V1_RUNS and read with the differences
noted where it is read. Computed:

- Canonical accuracy (65 cases) of each overlapping configuration in each
  version: the mean over cases of each case's pass fraction across
  repetitions, with a percentile bootstrap CI over cases. Cases are matched by
  index after checking that the request text is identical.
- The exact McNemar test on per-case majority outcomes across versions, Holm
  across the cross-version tests, and a paired bootstrap CI of the version 2
  minus version 1 difference.
- Identical modal answers: the most frequent structured (tool, arguments)
  answer of each case in each version, compared as stored and with an empty
  "parameters" object treated as absent.
- Median turn time: the median over cases of each case's mean latency, with a
  bootstrap CI of the median.
- Phase throughput at the prompt lengths both versions probed, as mean and
  sample sd over 5 repetitions (analyze.probe_stats), the compiled to CPU
  ratios of each version, and the chat-shaped turn, which version 1
  extrapolated and version 2 measured.
- The version 1 claims that version 2 can re-test, with the same tests.
  Version 2 Holm values use the four comparison families of analyze.py over
  every version 2 configuration, so they agree with analysis/out/v2.json.
- Protocol and provenance: what each file records about server, tools, cases,
  request settings and board, and a re-grade of every stored version 1
  repetition with the harness grader.
- A status for each version 1 finding (the unpublished version 1 draft report,
  docs/PROPOSAL.md and the version 1 run READMEs): holds, changed or
  superseded, with the version 2 numbers that decide it.

Bootstraps use SEED and B from report_common.
"""
import collections
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_common as rc  # noqa: E402

analyze = rc.analyze
V1_ROOT = os.path.join(rc.REPO, "results")
TARGETS = (256, 512, 1024, 2048, 2800)
ALPHA = "GenieX 0.5.1-alpha.2"  # the binary scripts/*.sh launch for the version 1 llama.cpp legs

# Version 1 files by the configuration they measured. The files carry no
# model or path label and do not record the server that answered: the
# llama.cpp legs ran on the alpha CLI (scripts/, run READMEs), the compiled
# leg on the resident GenieX v0.4.0 container (its provenance.board).
V1_RUNS = [
    {"key": "v1 4B-2507 CPU 09-15", "model": "qwen3-4b-2507", "path": "cpu", "date": "2026-09-15",
     "server": ALPHA, "tools": "public schema",
     "bench": "2026-09-15-qcs9075-qwen3-4b-cpu-controlled/qwen3-4b-2507-q4_0-cpu-llamacpp-hygiene-5rep-20260915.json",
     "probe": "2026-09-15-qcs9075-qwen3-4b-cpu-controlled/phase-qwen3-4b-2507-cpu-llamacpp-q4_0-20260915.json"},
    {"key": "v1 4B-2507 CPU 09-10", "model": "qwen3-4b-2507", "path": "cpu", "date": "2026-09-10",
     "server": ALPHA, "tools": "development copy",
     "bench": "2026-09-10-qcs9075-qwen3-4b/qwen3-4b-2507-q4_0-cpu-llamacpp-hygiene-2rep-20260910.json",
     "probe": "2026-09-10-qcs9075-qwen3-4b/phase2-qwen3-4b-cpu-llamacpp-q4_0-20260910.json"},
    {"key": "v1 4B-2507 NPU llama.cpp 09-10", "model": "qwen3-4b-2507", "path": "npu_llama", "date": "2026-09-10",
     "server": ALPHA, "tools": "development copy",
     "bench": "2026-09-10-qcs9075-qwen3-4b/qwen3-4b-2507-q4_0-npu-llamacpp-hygiene-2rep-20260910.json",
     "probe": "2026-09-10-qcs9075-qwen3-4b/phase2-qwen3-4b-npu-llamacpp-q4_0-20260910.json"},
    {"key": "v1 4B-2507 compiled 09-10", "model": "qwen3-4b-2507", "path": "compiled", "date": "2026-09-10",
     "server": "GenieX v0.4.0", "tools": "development copy",
     "bench": "2026-09-10-qcs9075-qwen3-4b/qwen3-4b-4k-geniex040-hygiene-5rep-20260910.json",
     "probe": "2026-09-10-qcs9075-qwen3-4b/phase2-qwen3-4b-npu-qairt-w4a16-20260910.json"},
    {"key": "v1 8B CPU 09-14", "model": "qwen3-8b", "path": "cpu", "date": "2026-09-14",
     "server": ALPHA, "tools": "public schema",
     "bench": "2026-09-14-qcs9075-qwen3-8b/qwen3-8b-q4_0-cpu-llamacpp-hygiene-5rep-20260914.json",
     "probe": "2026-09-14-qcs9075-qwen3-8b/phase-qwen3-8b-cpu-llamacpp-q4_0-20260914.json"},
    {"key": "v1 Qwen3.5-2B CPU 09-16", "model": "qwen3.5-2b", "path": "cpu", "date": "2026-09-16",
     "server": ALPHA, "tools": "public schema",
     "bench": "2026-09-16-qcs9075-qwen3-5-2b/qwen3-5-2b-q4_0-cpu-llamacpp-hygiene-5rep-20260916.json",
     "probe": "2026-09-16-qcs9075-qwen3-5-2b/phase-qwen3-5-2b-cpu-llamacpp-q4_0-20260916.json"},
]
# Version 1 attempts that left no result file, as the run READMEs record them.
V1_UNMEASURED = [
    {"model": "qwen3-8b", "path": "compiled", "outcome": "did not load (context for the fifth of five parts failed)",
     "source": "results/2026-09-14-qcs9075-qwen3-8b/README.md"},
    {"model": "qwen3-8b", "path": "npu_llama", "outcome": "did not load (HTP buffer mapping failed)",
     "source": "results/2026-09-14-qcs9075-qwen3-8b/README.md"},
    {"model": "qwen3.5-2b", "path": "npu_llama", "outcome": "loaded with partial offload, not benchmarked",
     "source": "results/2026-09-16-qcs9075-qwen3-5-2b/README.md"},
]
CPU_0915, CPU_0910, NPU_0910, COMP_0910, CPU8_0914, Q35_0916 = (r["key"] for r in V1_RUNS)
PATH_SHORT = {"cpu": "on the CPU", "npu_llama": "on llama.cpp NPU", "compiled": "compiled"}
TRIP_C = 105.0  # CPU and NPU passive throttling trip point (docs/THERMAL.md)
# GGUF sizes come from the model files, which a reader of the repository does
# not have; analysis/inspect_artifacts.py records them here.
ARTIFACTS = os.path.join(rc.HERE, "data", "artifacts.json")


def rel(path):
    return os.path.relpath(path, rc.REPO)


def label(model, path):
    return f"{rc.MODEL_NAMES.get(model, model)}, {rc.PATH_NAMES[path]}"


# ------------------------------------------------------------ per-file reads
def file_items(data, suite="canonical", grade_fn=analyze.e2e_ok):
    return rc.items({suite: data}, grade_fn=grade_fn, suites_=(suite,))


def mean_ci(vals):
    vals = list(vals)
    return statistics.mean(vals), list(analyze.bootstrap_ci(vals))


def _answer(rep, drop_empty_params):
    args = rep.get("got_args")
    if drop_empty_params and isinstance(args, dict) and args.get("parameters") == {}:
        args = {k: v for k, v in args.items() if k != "parameters"}
    return json.dumps([rep.get("got_tool"), args], sort_keys=True)


def answers(data, drop_empty_params=False):
    """{case idx: structured answer of each repetition}, argument keys sorted."""
    return {c["idx"]: [_answer(r, drop_empty_params) for r in c["reps"]] for c in data["results"] if c.get("reps")}


def modal(ans):
    return {i: collections.Counter(a).most_common(1)[0][0] for i, a in ans.items()}


def modal_command(answer):
    """(tool, command_name) of a stored answer."""
    tool, args = json.loads(answer)
    return tool, args.get("command_name") if isinstance(args, dict) else None


def bench_summary(data, suite="canonical"):
    """Accuracy, latency and stability of one tool-calling file. Version 1
    files keep no prompt token counts and no server timings, so those fields
    are None for them."""
    it = file_items(data, suite)
    e2e, e2e_ci = mean_ci(it.values())
    tool = statistics.mean(file_items(data, suite, lambda r, c: bool(r.get("ok_tool"))).values())
    cases = [c for c in data["results"] if c.get("reps")]
    reps = [r for c in cases for r in c["reps"]]
    lat = {c["idx"]: [r["latency_s"] for r in c["reps"] if isinstance(r.get("latency_s"), (int, float))] for c in cases}
    case_mean = [statistics.mean(v) for v in lat.values() if v]
    cv = [statistics.stdev(v) / statistics.mean(v) for v in lat.values() if len(v) > 1]
    slow = sum(x > 1.05 * statistics.median(v) for v in lat.values() for x in v if len(v) > 1)
    pt = [r["prompt_tokens"] for r in reps if isinstance(r.get("prompt_tokens"), (int, float)) and r["prompt_tokens"] > 0]
    ct = [r["completion_tokens"] for r in reps if isinstance(r.get("completion_tokens"), (int, float))]
    ans = answers(data)
    return {
        "suite": suite, "n_cases": len(it), "repeats": data["summary"].get("repeats"), "n_requests": len(reps),
        "e2e": e2e, "e2e_ci": e2e_ci, "tool_sel": tool,
        "no_structured_call_rate": sum(not r.get("got_tool") for r in reps) / len(reps),
        "turn_median_s": statistics.median(case_mean),
        "turn_median_ci": list(analyze.bootstrap_ci(case_mean, stat=statistics.median)),
        "latency_cv_median_pct": 100 * statistics.median(cv) if cv else None,
        "slow_requests": slow,
        "prompt_tokens_median": statistics.median(pt) if pt else None,
        "completion_tokens_median": statistics.median(ct) if ct else None,
        "failing_idx": sorted(int(k.split(":")[1]) for k, v in it.items() if v <= 0.5),
        "text_reply_idx": sorted(c["idx"] for c in cases if not any(r.get("got_tool") for r in c["reps"])),
        "outcome_unstable_idx": sorted(int(k.split(":")[1]) for k, v in it.items() if 0 < v < 1),
        "answer_unstable_idx": sorted(i for i, a in ans.items() if len(set(a)) > 1),
        "command_as_tool_requests": sum(1 for r in reps if r.get("got_tool") and r["got_tool"] not in rc.SUBSET),
    }


def poe_for_port_state(data):
    """Failing cases whose modal answer uses the PoE mode command where the
    port enable or disable command was expected."""
    m = modal(answers(data))
    out = []
    for c in data["results"]:
        if c["idx"] in m and c["expected_command"] in ("enable_port", "disable_port") \
                and modal_command(m[c["idx"]])[1] == "configure_poe_mode":
            out.append(c["idx"])
    return out


# ------------------------------------------------------ cross-version tests
def matched(d1, d2):
    """Item keys of the canonical cases whose request text is identical in both files."""
    in1 = {c["idx"]: c["input"] for c in d1["results"]}
    in2 = {c["idx"]: c["input"] for c in d2["results"]}
    return {f"canonical:{i}" for i in in1 if in2.get(i) == in1[i]}


def compare(d1, d2):
    """Paired comparison of two canonical files (first, second): McNemar on
    majority outcomes, paired bootstrap of second minus first, and agreement
    of modal answers."""
    keys = matched(d1, d2)
    a = {k: v for k, v in file_items(d1).items() if k in keys}
    b = {k: v for k, v in file_items(d2).items() if k in keys}
    only_a, only_b, p = rc.mcnemar_items(a, b)
    diff, ci = rc.paired_diff_ci(a, b)
    idx = lambda ks: sorted(int(k.split(":")[1]) for k in ks)  # noqa: E731
    out = {"n_cases": len(keys), "only_first": only_a, "only_second": only_b, "p_exact": p,
           "only_first_idx": idx(k for k in a if a[k] > 0.5 >= b[k]),
           "only_second_idx": idx(k for k in a if b[k] > 0.5 >= a[k]),
           "both_fail_idx": idx(k for k in a if a[k] <= 0.5 and b[k] <= 0.5),
           "diff_second_minus_first": diff, "diff_ci": list(ci)}
    for name, drop in (("identical_modal_answers", False), ("identical_modal_answers_empty_params_absent", True)):
        m1, m2 = modal(answers(d1, drop)), modal(answers(d2, drop))
        out[name] = sum(m1[i] == m2.get(i) for i in m1 if f"canonical:{i}" in keys)
    return out


def v2_tests(cfgs):
    """Exact McNemar for every pair analyze.comparisons forms on a version 2
    directory, Holm-adjusted within its family. Returns a lookup
    (family, suite, (model, path), (model, path)) -> result, both orientations."""
    runs = [(c["model"], c["path"], s, rc.items(c, suites_=(s,))) for c in cfgs
            for s in ("canonical", "paraphrase") if c.get(s)]
    pooled = [(c["model"], c["path"], "all", rc.items(c)) for c in cfgs if c.get("canonical") or c.get("paraphrase")]
    out, sizes = {}, {}
    for fam, rows, key in (("paths_within_model", runs, lambda r: (r[0], r[2])),
                           ("models_within_path", runs, lambda r: (r[1], r[2])),
                           ("paths_within_model_all_items", pooled, lambda r: r[0]),
                           ("models_within_path_all_items", pooled, lambda r: r[1])):
        groups = {}
        for r in rows:
            groups.setdefault(key(r), []).append(r)
        tests = [(g[i], g[j], rc.mcnemar_items(g[i][3], g[j][3]))
                 for g in groups.values() for i in range(len(g)) for j in range(i + 1, len(g))]
        for (a, b, (oa, ob, p)), h in zip(tests, rc.holm([t[2][2] for t in tests])):
            out[(fam, a[2], a[:2], b[:2])] = {"only_first": oa, "only_second": ob, "p_exact": p, "p_holm": h}
            out[(fam, a[2], b[:2], a[:2])] = {"only_first": ob, "only_second": oa, "p_exact": p, "p_holm": h}
        sizes[fam] = len(tests)
    return out, sizes


# ------------------------------------------------------------- throughput
def probe_rows(path, version):
    """{target: row} of one phase-probe file. Version 1 files record no server
    timings and no max_tokens_honored flag (analyze.probe_stats then reports
    True), so for them the flag comes from the number of tokens generated."""
    p = analyze.probe_stats(path)
    out = {}
    for r in p["rows"]:
        if "error" in r:
            out[r["target"]] = {"target": r["target"], "error": str(r["error"])}
            continue
        row = {"target": r["target"], "prompt_tokens": r["prompt_tokens"], "n": r["n"], "ttft_s": list(r["ttft_s"]),
               "prefill": list(r["prefill_client"]), "decode": list(r["decode_client"]),
               "e2e_s": list(r["e2e_s"]), "decode_tokens": r["n_out"]}
        if version == 1:
            row["max_tokens_honored"] = r["n_out"] <= p["gen_tokens"]
        else:
            row.update(prefill_server=list(r["prefill_server"]), decode_server=list(r["decode_server"]),
                       max_tokens_honored=r["max_tokens_honored"])
        out[r["target"]] = row
    return out


def change_pct(a, b):
    return None if a in (None, 0) or b is None else 100 * (b - a) / a


def throughput_pair(v1rows, v2rows):
    rows = []
    for t in TARGETS:
        x, y = v1rows.get(t), v2rows.get(t)
        row = {"target": t, "v1": x, "v2": y}
        if x and y and "error" not in x and "error" not in y:
            row.update(ttft_change_pct=change_pct(x["ttft_s"][0], y["ttft_s"][0]),
                       prefill_change_pct=change_pct(x["prefill"][0], y["prefill"][0]),
                       decode_change_pct=change_pct(x["decode"][0], y["decode"][0]))
        rows.append(row)
    return rows


def ratios(cpu, npu):
    """Compiled (or NPU) over CPU at each target both rows exist."""
    out = []
    for t in TARGETS:
        c, n = cpu.get(t), npu.get(t)
        if not c or not n or "error" in c or "error" in n:
            out.append({"target": t, "missing": (n or {}).get("error") or (c or {}).get("error") or "not probed"})
            continue
        srv = n.get("prefill_server", [None])[0], c.get("prefill_server", [None])[0]
        out.append({"target": t, "prefill": n["prefill"][0] / c["prefill"][0],
                    "prefill_server": srv[0] / srv[1] if srv[0] and srv[1] else None,
                    "decode": n["decode"][0] / c["decode"][0], "ttft_speedup": c["ttft_s"][0] / n["ttft_s"][0]})
    return out


def chat_estimate(rows):
    """Version 1's chat turn: 100 prompt tokens and 300 generated at the rates
    of its shortest (256-token) probe row."""
    r = rows[256]
    return 100 / r["prefill"][0] + 300 / r["decode"][0]


def chat_measured(path):
    r = analyze.probe_stats(path)["rows"][0]
    return {"prompt_tokens": r["prompt_tokens"], "decode_tokens": r["n_out"], "n": r["n"], "e2e_s": list(r["e2e_s"])}


def compiled_turn_change(d1, d2):
    """Per-case mean latency and completion tokens of two compiled runs,
    matched by case; least-squares slope of the latency change on the token
    change, and version 2's median split of a request by server timings."""
    keys = matched(d1, d2)
    c1 = {c["idx"]: c for c in d1["results"] if f"canonical:{c['idx']}" in keys}
    c2 = {c["idx"]: c for c in d2["results"] if c["idx"] in c1}
    mean = lambda c, k: statistics.mean(r[k] for r in c["reps"])  # noqa: E731
    dl = [mean(c1[i], "latency_s") - mean(c2[i], "latency_s") for i in sorted(c2)]
    dt = [mean(c1[i], "completion_tokens") - mean(c2[i], "completion_tokens") for i in sorted(c2)]
    mx, my = statistics.mean(dt), statistics.mean(dl)
    slope = sum((x - mx) * (y - my) for x, y in zip(dt, dl)) / sum((x - mx) ** 2 for x in dt)
    split = {"prefill": [], "decode": [], "other": []}
    for c in d2["results"]:
        for r in c["reps"]:
            t = (r.get("server_extra") or {}).get("timings") or {}
            if "prompt_ms" in t and "predicted_ms" in t:
                split["prefill"].append(t["prompt_ms"] / 1000)
                split["decode"].append(t["predicted_ms"] / 1000)
                split["other"].append(r["latency_s"] - (t["prompt_ms"] + t["predicted_ms"]) / 1000)
    return {"n_cases": len(dl), "mean_latency_saved_s": my, "mean_tokens_fewer": mx,
            "slope_s_per_token": slope, "intercept_s": my - slope * mx,
            "v1_completion_tokens_median": statistics.median(mean(c, "completion_tokens") for c in c1.values()),
            "v2_completion_tokens_median": statistics.median(mean(c, "completion_tokens") for c in c2.values()),
            "v2_request_median_s": {k: statistics.median(v) for k, v in split.items()}}


# ------------------------------------------------------------ Qwen3.5-2B
_FUNC = re.compile(r"<function=([A-Za-z0-9_]+)>")
_PARAM = re.compile(r"<parameter=([A-Za-z0-9_]+)>\s*(.*?)\s*</parameter>", re.S)
_MARKER = ("<tool_call>", "<function=")


def lenient_call(content):
    """<function=NAME> and every closed <parameter> block after it, so a reply
    cut off before </function> still yields a call."""
    m = _FUNC.search(content or "")
    if not m:
        return None, None
    args = {}
    for pm in _PARAM.finditer(content[m.end():]):
        try:
            args[pm.group(1)] = json.loads(pm.group(2))
        except json.JSONDecodeError:
            args[pm.group(1)] = pm.group(2)
    return m.group(1), args


def text_call_score(data, parser):
    per_case, tool, parsed = [], [], 0
    for c in data["results"]:
        oks = []
        for r in c["reps"]:
            name, args = parser(r.get("content") or "")
            g = analyze.grade(c, name, args) if name else {}
            parsed += bool(name)
            oks.append(analyze.e2e_ok(g))
            tool.append(bool(g.get("ok_tool")))
        per_case.append(sum(oks) / len(oks))
    e2e, ci = mean_ci(per_case)
    return {"parsed": parsed, "parsed_rate": parsed / len(tool), "e2e": e2e, "e2e_ci": ci,
            "tool_sel": sum(tool) / len(tool)}


def qwen35_reparse(d1, d2):
    """What version 1's stored Qwen3.5-2B replies support. Version 1 kept only
    the first 200 characters of each reply, and only for text replies."""
    reps1 = [r for c in d1["results"] for r in c["reps"]]
    cap = max(len(r.get("content") or "") for r in reps1)
    reps2 = [r for c in d2["results"] for r in c["reps"]]
    reply = lambda r: json.dumps([r.get("got_tool"), r.get("got_args"), r.get("content")], sort_keys=True)  # noqa: E731
    return {
        "v1_requests": len(reps1), "v1_content_cap_chars": cap,
        "v1_at_cap": sum(len(r.get("content") or "") >= cap for r in reps1),
        "v1_with_call_marker": sum(any(m in (r.get("content") or "") for m in _MARKER) for r in reps1),
        "v1_strict": text_call_score(d1, analyze.parse_text_tool_call),
        "v1_lenient": text_call_score(d1, lenient_call),
        "v1_cases_raw_reply_varies": sum(len({r.get("content") for r in c["reps"]}) > 1 for c in d1["results"]),
        "v2_requests": len(reps2), "v2_without_structured_call": sum(not r.get("got_tool") for r in reps2),
        "v2_text_tool_calls": sum(bool(r.get("text_tool_call")) for r in reps2),
        "v2_cases_reply_varies": sum(len({reply(r) for r in c["reps"]}) > 1 for c in d2["results"]),
    }


# -------------------------------------------------------------- provenance
def regrade(datas):
    """Re-grade every stored repetition with the harness grader; count
    repetitions whose stored flags differ from the re-grade."""
    n = differ = 0
    for d in datas:
        for c in d["results"]:
            for r in c.get("reps") or []:
                g = analyze.grade(c, r.get("got_tool"), r.get("got_args"))
                n += 1
                differ += any(bool(g[k]) != bool(r.get(k)) for k in g)
    return n, differ


def device_names(datas):
    """Structured calls, and how many name the target device without its @."""
    calls = same = 0
    for d in datas:
        for c in d["results"]:
            want = re.match(r"@(\S+)", c["input"])
            for r in c.get("reps") or []:
                if isinstance(r.get("got_args"), dict):
                    calls += 1
                    same += bool(want) and r["got_args"].get("device_name") == want.group(1)
    return calls, same


def prov(d):
    p = d.get("provenance", {})
    b = p.get("board") or {}
    return {"tools_source": p.get("tools_source"),
            "tools_sha256": (p.get("tools_sha256") or p.get("tool_definitions_sha256") or "")[:8],
            "cases_sha256": (p.get("cases_sha256") or "")[:8], "harness_sha256": (p.get("harness_sha256") or "")[:8],
            "system_prompt_sha256": (p.get("system_prompt_sha256") or "")[:8] or None,
            "request_config": p.get("request_config"),
            "server_version": (p.get("server_version") or "").replace("\n", "; ") or None,
            "container_version_field": "; ".join(b.get("geniex_version") or b.get("gx_container_geniex_version") or []) or None,
            "resident_container_up": bool(b.get("containers")),
            "os": (b.get("os") or {}).get("PRETTY_NAME"), "kernel": b.get("kernel")}


def thermal(cfgs):
    """Hottest traced reading over every version 2 run, and over CPU-path runs."""
    rows = []
    for c in cfgs:
        for name in ("canonical", "paraphrase", "probe_agent", "probe_chat", "diag_tools_in_prompt"):
            d = c.get(name)
            th = d and ((d.get("summary") or {}).get("thermal") or d.get("thermal"))
            if isinstance(th, dict) and th.get("max_all_c") is not None:
                rows.append((th["max_all_c"], c["model"], c["path"], name))
    hottest = max(rows)
    cpu = max(r for r in rows if r[2] == "cpu")
    return {"files_traced": len(rows), "max_c": hottest[0], "max_at": f"{hottest[1]}/{hottest[2]}/{hottest[3]}.json",
            "cpu_path_max_c": cpu[0], "cpu_path_max_at": f"{cpu[1]}/{cpu[2]}/{cpu[3]}.json",
            "trip_c": TRIP_C, "margin_c": round(TRIP_C - hottest[0], 1)}


def npu_llama_outcomes(cfgs):
    """Version 2 llama.cpp NPU load outcomes, with the failed mapping size the
    server log reports where it does."""
    out = []
    for c in cfgs:
        if c["path"] != "npu_llama":
            continue
        lf = c.get("load_failure")
        row = {"model": c["model"], "loaded": not lf and bool(c.get("canonical"))}
        if lf:
            m = re.search(r"buffer mapping failed : domain_id \d+ size (\d+)", lf.get("server_log_tail") or "")
            row["failed_mapping_bytes"] = int(m.group(1)) if m else None
            check = lf.get("load_check") or {}
            row["load_check"] = check.get("error") or f"HTTP {check.get('http_status')}"
        out.append(row)
    return out


def gguf_boundary(outcomes):
    """Largest GGUF that loaded and smallest that failed on llama.cpp NPU, by
    the sizes in analysis/data/artifacts.json; None when that file is absent."""
    art = rc.load_json(ARTIFACTS)
    if not art:
        return None
    size = {m: (v.get("gguf") or {}).get("bytes") for m, v in art["models"].items()}
    ok = [size[o["model"]] for o in outcomes if o["loaded"] and size.get(o["model"])]
    bad = [size[o["model"]] for o in outcomes if not o["loaded"] and size.get(o["model"])]
    return {"largest_loaded_bytes": max(ok), "smallest_failed_bytes": min(bad), "source": rel(ARTIFACTS)}


# ------------------------------------------------------------------- main
class Runs:
    """Both versions' parsed files, with bench summaries computed once."""

    def __init__(self, results_dir):
        self.dir = results_dir
        self.cfgs = rc.configs(results_dir)
        self.v2 = {(c["model"], c["path"]): c for c in self.cfgs}
        self.v1 = {r["key"]: {**r, "data": rc.load_json(os.path.join(V1_ROOT, r["bench"])),
                              "probe_rows": probe_rows(os.path.join(V1_ROOT, r["probe"]), 1)} for r in V1_RUNS}
        self._sum = {}

    def file(self, model, path, name="canonical"):
        return os.path.join(self.dir, model, path, name + ".json")

    def data(self, model, path, name="canonical"):
        return (self.v2.get((model, path)) or {}).get(name)

    def s1(self, key):
        if key not in self._sum:
            self._sum[key] = bench_summary(self.v1[key]["data"])
        return self._sum[key]

    def s2(self, model, path, suite="canonical"):
        k = (model, path, suite)
        if k not in self._sum:
            d = self.data(model, path, suite)
            self._sum[k] = bench_summary(d, suite) if d else None
        return self._sum[k]


def overlap_section(R):
    """Every version 1 run beside its version 2 counterpart, the control
    comparison, and the version 1 attempts that left no file."""
    rows, ps = [], []
    for run in V1_RUNS:
        key, m, p = run["key"], run["model"], run["path"]
        d2 = R.data(m, p)
        row = {"configuration": label(m, p), "model": m, "path": p, "v1_run": key, "v1_date": run["date"],
               "v1_server": run["server"], "v1_tools": run["tools"], "v1_file": "results/" + run["bench"],
               "v1": R.s1(key), "v2_file": rel(R.file(m, p)) if d2 else None,
               "v2": R.s2(m, p), "v2_paraphrase": R.s2(m, p, "paraphrase"),
               "v2_load_failure": rel(R.file(m, p, "load_failure")) if R.data(m, p, "load_failure") else None}
        if d2:
            row["test"] = compare(R.v1[key]["data"], d2)
            ps.append(row["test"]["p_exact"])
        rows.append(row)
    control = compare(R.v1[COMP_0910]["data"], R.data("qwen3-4b", "compiled"))
    adj = rc.holm(ps + [control["p_exact"]])
    for row, h in zip([r for r in rows if "test" in r], adj):
        row["test"]["p_holm_cross_version"] = h
    control["p_holm_cross_version"] = adj[-1]
    unmeasured = [{**u, "configuration": label(u["model"], u["path"]),
                   "v2": R.s2(u["model"], u["path"]), "v2_paraphrase": R.s2(u["model"], u["path"], "paraphrase"),
                   "v2_load_check": (R.data(u["model"], u["path"], "load_failure") or {}).get("load_check")}
                  for u in V1_UNMEASURED]
    return rows, control, unmeasured, len(adj)


def identity_section(R):
    """Which version 2 build the version 1 compiled run ('qualcomm/qwen3-4b-4k')
    behaves like: modal-answer agreement with an empty parameters object
    treated as absent, and where its failing answers recur."""
    ref = modal(answers(R.v1[COMP_0910]["data"], True))
    others = {f"v2 {m}/{p}": modal(answers(R.data(m, p), True)) for m, p in
              (("qwen3-4b-2507", "compiled"), ("qwen3-4b", "compiled"), ("qwen3-4b-2507", "cpu"), ("qwen3-4b", "cpu"))}
    others.update({k: modal(answers(R.v1[k]["data"], True)) for k in (CPU_0915, CPU_0910)})
    failures = {}
    for i in R.s1(COMP_0910)["failing_idx"]:
        failures[str(i)] = {"answer": modal_command(ref[i]), "also_in": [k for k, o in others.items() if o.get(i) == ref[i]]}
    return {"v1_file": "results/" + R.v1[COMP_0910]["bench"],
            "identical_modal_answers": {k: sum(ref[i] == o.get(i) for i in ref) for k, o in others.items()},
            "v1_failures": failures,
            "failures_shared_with_2507_bundle": sorted(int(i) for i, f in failures.items()
                                                       if "v2 qwen3-4b-2507/compiled" in f["also_in"]),
            "failures_only_in_2507_bundle": sorted(int(i) for i, f in failures.items()
                                                   if f["also_in"] == ["v2 qwen3-4b-2507/compiled"])}


def throughput_section(R):
    """Probe rows of both versions at each target, compiled over CPU ratios
    per version, and the version 2 rows without a version 1 counterpart."""
    out = []
    for run in V1_RUNS:
        has = R.data(run["model"], run["path"], "probe_agent")
        v2rows = probe_rows(R.file(run["model"], run["path"], "probe_agent"), 2) if has else {}
        out.append({"configuration": label(run["model"], run["path"]), "v1_run": run["key"], "v1_date": run["date"],
                    "v1_file": "results/" + run["probe"],
                    "v2_file": rel(R.file(run["model"], run["path"], "probe_agent")) if has else None,
                    "rows": throughput_pair(R.v1[run["key"]]["probe_rows"], v2rows)})
    v2rows = {(m, p): probe_rows(R.file(m, p, "probe_agent"), 2) for m, p in
              (("qwen3-4b-2507", "cpu"), ("qwen3-4b-2507", "compiled"), ("qwen3-8b", "compiled"),
               ("qwen3-1.7b", "cpu"), ("qwen3-1.7b", "npu_llama"), ("qwen3-1.7b", "compiled"))}
    v1r = lambda k: R.v1[k]["probe_rows"]  # noqa: E731
    ratio = {"v1_compiled_over_cpu": ratios(v1r(CPU_0910), v1r(COMP_0910)),
             "v1_npu_llama_over_cpu": ratios(v1r(CPU_0910), v1r(NPU_0910)),
             "v2_compiled_over_cpu": ratios(v2rows[("qwen3-4b-2507", "cpu")], v2rows[("qwen3-4b-2507", "compiled")]),
             "v2_qwen3_1_7b_npu_llama_over_cpu": ratios(v2rows[("qwen3-1.7b", "cpu")], v2rows[("qwen3-1.7b", "npu_llama")])}
    v2_only = {f"{m}/{p}": {"configuration": label(m, p), "file": rel(R.file(m, p, "probe_agent")),
                            "rows": [rows[t] for t in TARGETS if t in rows]}
               for (m, p), rows in v2rows.items() if m != "qwen3-4b-2507"}
    honoured = all(r["max_tokens_honored"] for rows in v2rows.values() for r in rows.values() if "error" not in r)
    tokens = sorted({r["decode_tokens"] for rows in v2rows.values() for r in rows.values() if "error" not in r})
    return out, ratio, v2_only, honoured, tokens


def turn_section(R):
    """Turn-time ratios and the chat-shaped turn in each version."""
    tm = lambda s: s["turn_median_s"]  # noqa: E731
    c4, p4 = R.s2("qwen3-4b-2507", "cpu"), R.s2("qwen3-4b-2507", "compiled")
    c8, p8 = R.s2("qwen3-8b", "cpu"), R.s2("qwen3-8b", "compiled")
    turn = {"v1_cpu_over_compiled": tm(R.s1(CPU_0915)) / tm(R.s1(COMP_0910)),
            "v2_cpu_over_compiled": tm(c4) / tm(p4), "v2_8b_cpu_over_compiled": tm(c8) / tm(p8),
            "v1_8b_over_4b_cpu": tm(R.s1(CPU8_0914)) / tm(R.s1(CPU_0915)),
            "v2_8b_over_4b_cpu": tm(c8) / tm(c4), "v2_8b_over_4b_compiled": tm(p8) / tm(p4)}
    chat = {"v1_estimate_s": {k: chat_estimate(R.v1[k]["probe_rows"]) for k in (CPU_0910, NPU_0910, COMP_0910, CPU8_0914, Q35_0916)},
            "v2_measured": {f"{m}/{p}": chat_measured(R.file(m, p, "probe_chat"))
                            for m in ("qwen3-4b-2507", "qwen3-8b", "qwen3.5-2b", "qwen3-1.7b", "qwen3-0.6b")
                            for p in rc.PATHS if R.data(m, p, "probe_chat")}}
    return turn, chat


# Version 2 re-tests of version 1 claims: (name, analyze.py family, first, second).
V2_RETESTS = (
    ("4B-2507 CPU against compiled", "paths_within_model", ("qwen3-4b-2507", "cpu"), ("qwen3-4b-2507", "compiled")),
    ("4B (base) CPU against compiled", "paths_within_model", ("qwen3-4b", "cpu"), ("qwen3-4b", "compiled")),
    ("1.7B CPU against NPU llama.cpp", "paths_within_model", ("qwen3-1.7b", "cpu"), ("qwen3-1.7b", "npu_llama")),
    ("4B-2507 against 8B, CPU", "models_within_path", ("qwen3-4b-2507", "cpu"), ("qwen3-8b", "cpu")),
    ("4B-2507 against 8B, compiled", "models_within_path", ("qwen3-4b-2507", "compiled"), ("qwen3-8b", "compiled")),
)
V1_RETESTS = (("4B-2507 CPU against compiled", CPU_0915, COMP_0910),
              ("4B-2507 CPU against NPU llama.cpp", CPU_0910, NPU_0910),
              ("4B-2507 against 8B, CPU", CPU_0915, CPU8_0914))


def within_section(R, tests):
    """Version 1 claims tested inside each version; differences are first
    minus second."""
    out = []
    for name, a, b in V1_RETESTS:
        da, db = R.v1[a]["data"], R.v1[b]["data"]
        r = compare(da, db)
        lo, hi = r.pop("diff_ci")
        out.append({"comparison": name, "version": 1, "suite": "canonical", "first": a, "second": b,
                    "files": ["results/" + R.v1[a]["bench"], "results/" + R.v1[b]["bench"]], "n_items": r["n_cases"],
                    "first_e2e": mean_ci(file_items(da).values()), "second_e2e": mean_ci(file_items(db).values()),
                    "diff_first_minus_second": -r.pop("diff_second_minus_first"), "diff_ci": [-hi, -lo], **r})
    for name, fam, a, b in V2_RETESTS:
        for suite in ("canonical", "paraphrase", "all"):
            chosen = ("canonical", "paraphrase") if suite == "all" else (suite,)
            ia, ib = rc.items(R.v2[a], suites_=chosen), rc.items(R.v2[b], suites_=chosen)
            diff, ci = rc.paired_diff_ci(ib, ia)
            family = fam + ("_all_items" if suite == "all" else "")
            row = {"comparison": name, "version": 2, "suite": suite, "first": "/".join(a), "second": "/".join(b),
                   "family": family, "n_items": len(set(ia) & set(ib)),
                   "first_e2e": mean_ci(ia.values()), "second_e2e": mean_ci(ib.values()),
                   "diff_first_minus_second": diff, "diff_ci": list(ci), **tests[(family, suite, a, b)]}
            if suite == "canonical":
                m1, m2 = modal(answers(R.data(*a))), modal(answers(R.data(*b)))
                row["identical_modal_answers"] = sum(m1[i] == m2.get(i) for i in m1)
            out.append(row)
    return out


def spread_section(R):
    """Widest within-model canonical spread across paths in version 2, over
    all configurations and over those whose prompts carried the tools."""
    by_model = {}
    for c in R.cfgs:
        if c.get("canonical"):
            s = R.s2(c["model"], c["path"])
            delivered = (s["prompt_tokens_median"] or 0) >= analyze.TOOLS_DELIVERED_MIN_PROMPT_TOKENS
            by_model.setdefault(c["model"], []).append((s["e2e"], c["path"], delivered))
    out = {}
    for scope in ("all", "tools_delivered"):
        best = None
        for m, runs in sorted(by_model.items()):
            vals = [r[:2] for r in runs if scope == "all" or r[2]]
            if len(vals) > 1 and (best is None or max(vals)[0] - min(vals)[0] > best["spread"]):
                best = {"model": m, "high": max(vals), "low": min(vals), "spread": max(vals)[0] - min(vals)[0]}
        out[scope] = best
    return out


OVERLAP_V2 = (("qwen3-4b-2507", "cpu"), ("qwen3-4b-2507", "compiled"), ("qwen3-8b", "cpu"),
              ("qwen3-8b", "compiled"), ("qwen3.5-2b", "cpu"))


def provenance_section(R):
    """Per-file provenance of the overlapping runs and the checks drawn from it."""
    files = {k: {"file": "results/" + R.v1[k]["bench"], **prov(R.v1[k]["data"])} for k in R.v1}
    for m, p in OVERLAP_V2:
        srv = R.data(m, p, "server") or {}
        files[f"v2 {m}/{p}"] = {"file": rel(R.file(m, p)), **prov(R.data(m, p)),
                                "server_options": srv.get("server_options"),
                                "artifact_sha256": (srv.get("artifact_sha256") or "")[:12],
                                "served_model_id": srv.get("served_model_id"), "bundle": srv.get("bundle"),
                                "load_and_first_reply_s": (srv.get("load_check") or {}).get("load_and_first_reply_s")}
    n_regraded, n_differ = regrade([r["data"] for r in R.v1.values()])
    calls, same_device = device_names([R.v1[k]["data"] for k in R.v1 if k != Q35_0916])
    checks = {
        "cases_sha256_identical": len({f["cases_sha256"] for f in files.values()}) == 1,
        "request_config_identical": len({json.dumps(f["request_config"], sort_keys=True) for f in files.values()}) == 1,
        "os_identical": len({f["os"] for f in files.values()}) == 1,
        "kernel_identical": len({f["kernel"] for f in files.values()}) == 1,
        "v1_regraded_requests": n_regraded, "v1_regrade_disagreements": n_differ,
        "v1_structured_calls": calls, "v1_calls_with_device_name_without_at": same_device,
        "v2_max_content_chars": max(len(r.get("content") or "") for m, p in OVERLAP_V2
                                    for c in R.data(m, p)["results"] for r in c["reps"]),
        "v1_compiled_decode_tokens": sorted({r["decode_tokens"] for r in R.v1[COMP_0910]["probe_rows"].values() if "error" not in r}),
        "v2_catalog_release": (rc.spec() or {}).get("catalog_release"),
    }
    return files, checks


def compute(results_dir):
    R = Runs(results_dir)
    tests, family_sizes = v2_tests(R.cfgs)
    overlap, control, unmeasured, n_cross = overlap_section(R)
    throughput, ratio, v2_only, honoured, decode_tokens = throughput_section(R)
    turn, chat = turn_section(R)
    files, checks = provenance_section(R)
    checks.update(v2_probe_rows_max_tokens_honored=honoured, v2_probe_decode_tokens=decode_tokens)
    data = {
        "scope": "QCS9075 (Dragonwing IQ-9075 EVK, Qualcomm Linux 1.6), with version 1 measured from 2026-09-10 to "
                 "2026-09-16 and version 2 from 2026-10-14 to 2026-10-17 on GenieX v0.8.0 (QAIRT 2.45, llama.cpp 9425611)",
        "method": {"bootstrap": "percentile, over cases (canonical) or items", "resamples": rc.B, "seed": rc.SEED,
                   "test": "exact two-sided McNemar on per-case majority outcome",
                   "cross_version_holm_tests": n_cross, "v2_family_sizes": family_sizes},
        "overlap": overlap, "control_v1_compiled_vs_v2_base_4b_compiled": control, "unmeasured": unmeasured,
        "compiled_identity": identity_section(R),
        "throughput": throughput, "throughput_v2_only": v2_only, "ratios": ratio, "turn": turn, "chat": chat,
        "compiled_turn_change": compiled_turn_change(R.v1[COMP_0910]["data"], R.data("qwen3-4b-2507", "compiled")),
        "within_version": within_section(R, tests), "v2_widest_within_model_spread": spread_section(R),
        "qwen35": qwen35_reparse(R.v1[Q35_0916]["data"], R.data("qwen3.5-2b", "cpu")),
        "poe_for_port_state_idx": {**{k: poe_for_port_state(R.v1[k]["data"]) for k in (CPU_0915, COMP_0910)},
                                   **{f"v2 qwen3-4b-2507/{p}": poe_for_port_state(R.data("qwen3-4b-2507", p))
                                      for p in ("cpu", "compiled")}},
        "npu_llama_v2": npu_llama_outcomes(R.cfgs),
        "npu_llama_gguf_boundary": gguf_boundary(npu_llama_outcomes(R.cfgs)),
        "workaround_tests": [{k: a.get(k) for k in ("setting", "loaded", "failed_buffer_bytes")}
                             for a in (R.data("qwen3-4b-2507", "npu_llama", "workaround_tests") or {}).get("attempts", [])],
        "thermal": thermal(R.cfgs),
        "provenance": files, "checks": checks,
    }
    data["protocol"] = protocol(data)
    data["findings"] = findings(data)
    return data


# --------------------------------------------------------------- formatting
def P(x):
    return "-" if x is None else f"{100 * x:.1f}"


def PCI(ci):
    return f"{P(ci[0])}-{P(ci[1])}"


def PP(x):
    # adding 0.0 turns a negative zero into "+0.0"
    return "-" if x is None else f"{round(100 * x, 1) + 0.0:+.1f}"


def FP(p):
    return "-" if p is None else (f"{p:.3f}" if p >= 0.01 else f"{p:.1e}")


def S(x):
    return "-" if x is None else f"{x:.2f}"


def R(x, nd=None):
    """A rate: nearest integer from 100 up, one decimal below (nd overrides)."""
    if x is None:
        return "-"
    nd = (0 if x >= 100 else 1) if nd is None else nd
    return f"{x:,.{nd}f}"


def MS(ms, fmt=None):
    """mean +/- sd, both at the mean's precision (seconds when fmt is S)."""
    if not ms or ms[0] is None:
        return "-"
    if fmt is S:
        return f"{S(ms[0])} +/- {S(ms[1])}"
    nd = 0 if ms[0] >= 100 else 1
    return f"{R(ms[0], nd)} +/- {R(ms[1], nd)}"


def CH(x):
    return "-" if x is None else f"{round(x, 1) + 0.0:+.1f}%"


def V(s):
    """A recorded version string on one line, runs of spaces collapsed."""
    return re.sub(r"\s+", " ", s or "-")


def _within(data, name, suite, version=2):
    for r in data["within_version"]:
        if r["comparison"] == name and r["suite"] == suite and r["version"] == version:
            return r


def _ratio(rows, target):
    return next(r for r in rows if r["target"] == target)


def _tp(data, key, target):
    for t in data["throughput"]:
        if t["v1_run"] == key:
            return next(r for r in t["rows"] if r["target"] == target)


# ---------------------------------------------------------------- protocol
def protocol(d):
    """Protocol and server differences between the overlapping runs; each
    entry says where its content comes from."""
    f, c = d["provenance"], d["checks"]
    v2cpu, v2comp = f["v2 qwen3-4b-2507/cpu"], f["v2 qwen3-4b-2507/compiled"]
    tools_v1 = sorted({f"{f[k]['tools_source']} ({f[k]['tools_sha256']})" for k in f if k.startswith("v1")})
    harness_v1 = sorted({f[k]["harness_sha256"] for k in f if k.startswith("v1")})
    reps_v1 = sorted({r["v1"]["repeats"] for r in d["overlap"]})
    s1 = {r["v1_run"]: {**r["v1"], "v2": r["v2"]} for r in d["overlap"]}
    same = ("cases_sha256_identical", "request_config_identical", "os_identical", "kernel_identical")
    up = lambda x: "running" if x else "stopped"  # noqa: E731
    return [
        {"item": "Server, llama.cpp paths",
         "v1": f"{ALPHA} command-line server; its llama.cpp revision is not recorded. The files carry only the "
               f"resident container's version string ({V(f[CPU_0915]['container_version_field'])})",
         "v2": f"{V(v2cpu['server_version'])}, recorded from the answering binary",
         "source": "scripts/*.sh (binary path), run READMEs; provenance.board.geniex_version; provenance.server_version"},
        {"item": "Server, compiled path",
         "v1": f"resident container: {V(f[COMP_0910]['container_version_field'])}; it ignored max_tokens "
               f"(probe generated {c['v1_compiled_decode_tokens'][0]} tokens per request)",
         "v2": f"{V(v2comp['server_version'])}; max_tokens "
               f"{'honoured in every probe row' if c['v2_probe_rows_max_tokens_honored'] else 'not honoured in some probe rows'}",
         "source": "provenance.board.geniex_version; provenance.server_version; lengths[].reps[].n_out"},
        {"item": "Context length",
         "v1": "compiled bundle 4k; llama.cpp default (the scripts pass no context option)",
         "v2": f"server options '{v2cpu['server_options']}' on every path",
         "source": "scripts/*.sh; server.json server_options"},
        {"item": "Compiled bundle",
         "v1": "served as 'qualcomm/qwen3-4b-4k'; release not recorded in the file (AI Hub 0.62.2 for the 8B attempt)",
         "v2": f"served as '{v2comp['served_model_id']}', AI Hub {c['v2_catalog_release']}, "
               f"{v2comp['bundle']['parts']} parts, largest {v2comp['bundle']['largest_part_gb']} GB",
         "source": "summary.model; results/2026-09-14-qcs9075-qwen3-8b/README.md; server.json; campaign/v2_models.json"},
        {"item": "GGUF identity",
         "v1": "no hash recorded",
         "v2": f"SHA-256 recorded per artifact (4B-2507 {v2cpu['artifact_sha256']}, 8B {f['v2 qwen3-8b/cpu']['artifact_sha256']})",
         "source": "server.json artifact_sha256"},
        {"item": "System prompt",
         "v1": "not recorded in the files; the version 1 harness (file hash 22d5833f) named a persona and a vendor and "
               f"said to 'pass it as device_name'. {c['v1_calls_with_device_name_without_at']:,} of the "
               f"{c['v1_structured_calls']:,} structured calls of the 4B and 8B runs named the device without the @",
         "v2": f"neutral; the device name is passed without the @, with an example (SHA-256 {v2cpu['system_prompt_sha256']})",
         "source": "version 1 harness file (hash 22d5833f); provenance.system_prompt; results[].reps[].got_args"},
        {"item": "Tool schema", "v1": "; ".join(tools_v1),
         "v2": f"{v2cpu['tools_source']} ({v2cpu['tools_sha256']})", "source": "provenance.tools_sha256 or tool_definitions_sha256"},
        {"item": "Cases, request settings, board",
         "v1": f"cases {f[CPU_0915]['cases_sha256']}; temperature 0, max_tokens 512, thinking off; {f[CPU_0915]['os']}",
         "v2": "identical in every overlapping file" if all(c[k] for k in same) else
               "differs in: " + ", ".join(k.replace("_identical", "") for k in same if not c[k]),
         "source": "provenance.cases_sha256, request_config, board.os, board.kernel"},
        {"item": "Harness and grader",
         "v1": f"harness {', '.join(harness_v1)} (1839a393 matches no known harness version)",
         "v2": f"harness {v2cpu['harness_sha256']}; re-grading the {c['v1_regraded_requests']:,} stored version 1 "
               f"repetitions with the current grader changes {c['v1_regrade_disagreements']} flags",
         "source": "provenance.harness_sha256; harness/bench_toolcall.py grade()"},
        {"item": "Repetitions",
         "v1": f"{' or '.join(str(x) for x in reps_v1)} per canonical case; no paraphrases",
         "v2": f"{d['overlap'][0]['v2']['repeats']} per canonical case, plus 195 paraphrases x 1",
         "source": "summary.repeats"},
        {"item": "Stored reply",
         "v1": f"text only for replies without a structured call, cut at {d['qwen35']['v1_content_cap_chars']} characters; "
               "no prompt token counts, no server timings",
         "v2": f"every reply (longest stored here {c['v2_max_content_chars']} characters), token counts and server timings",
         "source": "results[].reps[].content, prompt_tokens, server_extra"},
        {"item": "Phase probes",
         "v1": f"5 reps; 128 decode tokens on llama.cpp, {c['v1_compiled_decode_tokens'][0]} on compiled; no chat probe",
         "v2": f"5 reps; {', '.join(str(x) for x in c['v2_probe_decode_tokens'])} decode tokens on both paths; chat probe of "
               "about 100 prompt and 300 generated tokens",
         "source": "lengths[].reps[].n_out; probe_chat.json"},
        {"item": "Resident container",
         "v1": f"{up(f[CPU_0915]['resident_container_up'])} during CPU legs, {up(f[NPU_0910]['resident_container_up'])} "
               "during the llama.cpp NPU leg; CPU requests more than 5% above their case median: 4B "
               f"{s1[CPU_0915]['slow_requests']} of {s1[CPU_0915]['n_requests']}, 8B {s1[CPU8_0914]['slow_requests']} of "
               f"{s1[CPU8_0914]['n_requests']}",
         "v2": f"{up(v2cpu['resident_container_up'])} during CPU legs, {up(v2comp['resident_container_up'])} during "
               f"NPU legs; slow CPU requests 4B {s1[CPU_0915]['v2']['slow_requests']} of {s1[CPU_0915]['v2']['n_requests']}, 8B "
               f"{s1[CPU8_0914]['v2']['slow_requests']} of {s1[CPU8_0914]['v2']['n_requests']}",
         "source": "provenance.board.containers; results[].reps[].latency_s"},
        {"item": "Thermal record",
         "v1": "readiness gate and snapshots at repetition boundaries; one in-run reading",
         "v2": f"readiness gate plus a 20 s trace in {d['thermal']['files_traced']} files",
         "source": "readiness; thermal_trace; summary.thermal"},
    ]


# ---------------------------------------------------------------- findings
def findings(d):
    """Each version 1 claim, its status in version 2 and the deciding numbers."""
    o = {r["v1_run"]: r for r in d["overlap"]}
    rv1 = d["ratios"]["v1_compiled_over_cpu"]
    rv2 = d["ratios"]["v2_compiled_over_cpu"]
    r17 = _ratio(d["ratios"]["v2_qwen3_1_7b_npu_llama_over_cpu"], 2048)
    v2c, v2p = o[CPU_0915]["v2"], o[COMP_0910]["v2"]
    tp_cpu, tp_comp = _tp(d, CPU_0910, 2048), _tp(d, COMP_0910, 2048)
    q17 = {p: next(r for r in d["throughput_v2_only"][f"qwen3-1.7b/{p}"]["rows"] if r["target"] == 2048)
           for p in ("cpu", "npu_llama")}
    chat = d["chat"]
    cm = chat["v2_measured"]
    w = lambda n, s, v=2: _within(d, n, s, v)  # noqa: E731
    cpu_comp = {s: w("4B-2507 CPU against compiled", s) for s in ("canonical", "paraphrase", "all")}
    b4 = {s: w("4B-2507 against 8B, CPU", s) for s in ("canonical", "paraphrase", "all")}
    c48 = w("4B-2507 against 8B, compiled", "all")
    u17 = {s: w("1.7B CPU against NPU llama.cpp", s) for s in ("canonical", "all")}
    base = w("4B (base) CPU against compiled", "canonical")
    v1cc = w("4B-2507 CPU against compiled", "canonical", 1)
    v1cn = w("4B-2507 CPU against NPU llama.cpp", "canonical", 1)
    v1b = w("4B-2507 against 8B, CPU", "canonical", 1)
    un = {(u["model"], u["path"]): u for u in d["unmeasured"]}
    q8c, q8n, q35u = un[("qwen3-8b", "compiled")], un[("qwen3-8b", "npu_llama")], un[("qwen3.5-2b", "npu_llama")]
    f8 = d["provenance"]["v2 qwen3-8b/compiled"]
    npu = d["npu_llama_v2"]
    loaded = [rc.MODEL_NAMES[x["model"]] for x in npu if x["loaded"]]
    sizes = [x["failed_mapping_bytes"] for x in npu if x.get("failed_mapping_bytes")]
    q = d["qwen35"]
    q35 = o[Q35_0916]
    sp = d["v2_widest_within_model_spread"]
    th = d["thermal"]
    det = {k: o[k]["v1"] for k in (CPU_0915, COMP_0910, CPU8_0914)}
    poe = d["poe_for_port_state_idx"]
    pooled = lambda r: f"{P(r['first_e2e'][0])} against {P(r['second_e2e'][0])}"  # noqa: E731
    t = d["turn"]
    wt = d["workaround_tests"]
    gb = d["npu_llama_gguf_boundary"]
    bound = (f"; GGUF files up to {gb['largest_loaded_bytes']:,} bytes loaded and none from "
             f"{gb['smallest_failed_bytes']:,} bytes up did ({gb['source']})") if gb else ""
    rows = [
        ("Version 1 draft (unpublished), finding 1; proposal, candidate finding 1",
         "On the compiled path the NPU reads a prompt about 17 times faster than the CPU and writes at about the same "
         "rate (2,044-token prompt: prefill 60 against 1,020 tok/s, decode 13.4 against 13.1 tok/s).",
         "holds",
         f"With one server binary, 5 reps and 128 decode tokens on both paths, prefill is {R(tp_cpu['v2']['prefill'][0])} "
         f"against {R(tp_comp['v2']['prefill'][0])} tok/s, a ratio of {_ratio(rv2, 2048)['prefill']:.2f}x by client timing and "
         f"{_ratio(rv2, 2048)['prefill_server']:.2f}x by server timing (version 1 {_ratio(rv1, 2048)['prefill']:.2f}x), and "
         f"decode is {tp_cpu['v2']['decode'][0]:.2f} against {tp_comp['v2']['decode'][0]:.2f} tok/s, a ratio of "
         f"{_ratio(rv2, 2048)['decode']:.3f} (version 1 {_ratio(rv1, 2048)['decode']:.3f}, compiled decode averaged over "
         f"{tp_comp['v1']['decode_tokens']} tokens). At 2,795 tokens the prefill ratio is {_ratio(rv2, 2800)['prefill']:.2f}x; "
         "version 1 has no compiled row there.",
         ["v1 4B-2507 CPU 09-10", "v1 4B-2507 compiled 09-10", "results/v2/qwen3-4b-2507/{cpu,compiled}/probe_agent.json"]),
        ("Version 1 draft (unpublished), finding 1",
         "An agent turn takes 7.4 s on the compiled path against 52.5 s on the CPU, about 7 times faster.",
         "holds",
         f"The median turn takes {S(v2c['turn_median_s'])} s on the CPU against {S(v2p['turn_median_s'])} s compiled "
         f"({t['v2_cpu_over_compiled']:.2f}x; version 1 {S(o[CPU_0915]['v1']['turn_median_s'])} against "
         f"{S(o[COMP_0910]['v1']['turn_median_s'])} s, {t['v1_cpu_over_compiled']:.2f}x), and for Qwen3-8B "
         f"{S(o[CPU8_0914]['v2']['turn_median_s'])} against {S(q8c['v2']['turn_median_s'])} s ({t['v2_8b_cpu_over_compiled']:.2f}x).",
         ["v1 4B-2507 CPU 09-15", "v1 4B-2507 compiled 09-10", "results/v2/qwen3-4b-2507/{cpu,compiled}/canonical.json"]),
        ("Version 1 draft (unpublished), finding 1; proposal, candidate finding 1",
         "A chat turn of 100 prompt and 300 generated tokens is a tie, about 19 s on the CPU and on the compiled path, and "
         "about 25 s on the llama.cpp NPU path (extrapolated from the rates at the shortest prompt).",
         "holds",
         f"Measured with about {cm['qwen3-4b-2507/cpu']['prompt_tokens']} prompt and "
         f"{cm['qwen3-4b-2507/cpu']['decode_tokens']} generated tokens (n = 5), the turn takes "
         f"{MS(cm['qwen3-4b-2507/cpu']['e2e_s'], S)} s on the CPU against "
         f"{MS(cm['qwen3-4b-2507/compiled']['e2e_s'], S)} s compiled (version 1 estimates "
         f"{chat['v1_estimate_s'][CPU_0910]:.1f} and {chat['v1_estimate_s'][COMP_0910]:.1f} s). The 4B no longer loads on "
         f"llama.cpp NPU; for Qwen3-1.7B the chat turn takes {MS(cm['qwen3-1.7b/cpu']['e2e_s'], S)} s on the CPU, "
         f"{MS(cm['qwen3-1.7b/npu_llama']['e2e_s'], S)} s on llama.cpp NPU and {MS(cm['qwen3-1.7b/compiled']['e2e_s'], S)} s compiled.",
         ["v1 4B-2507 CPU 09-10", "v1 4B-2507 compiled 09-10", "results/v2/*/*/probe_chat.json"]),
        ("Version 1 draft (unpublished), finding 1 and table 2",
         "The llama.cpp NPU path is about 6 times faster than the CPU at prefill but slower at decode (386 against "
         "60 tok/s; 7.9 against 13.4 tok/s).",
         "holds",
         f"Version 1 measured {_ratio(d['ratios']['v1_npu_llama_over_cpu'], 2048)['prefill']:.2f}x prefill and a decode "
         f"ratio of {_ratio(d['ratios']['v1_npu_llama_over_cpu'], 2048)['decode']:.2f} at 2,044 tokens. The 4B cannot be "
         "re-measured, since it no longer loads on this path; for Qwen3-1.7B at 2,041 tokens prefill is "
         f"{R(q17['npu_llama']['prefill'][0])} against {R(q17['cpu']['prefill'][0])} tok/s ({r17['prefill']:.1f}x) and decode "
         f"{q17['npu_llama']['decode'][0]:.1f} against {q17['cpu']['decode'][0]:.1f} tok/s (ratio {r17['decode']:.2f}).",
         ["v1 4B-2507 NPU llama.cpp 09-10", "results/v2/qwen3-1.7b/{cpu,npu_llama}/probe_agent.json"]),
        ("Version 1 draft (unpublished), finding 2",
         "Changing only the compute unit (same GGUF, same binary) changes nothing: 98.5% on the CPU and on the NPU, with "
         "the same failing case and the same wrong answer.",
         "holds",
         f"Version 1 has {v1cn['only_first']} against {v1cn['only_second']} discordant cases and "
         f"{v1cn['identical_modal_answers']}/65 identical answers. The 4B cannot be re-tested on llama.cpp NPU. In version 2 "
         f"Qwen3-1.7B scores {pooled(u17['canonical'])}% canonical ({u17['canonical']['only_first']} against "
         f"{u17['canonical']['only_second']} discordant, {u17['canonical']['identical_modal_answers']}/65 identical answers) "
         f"and {pooled(u17['all'])}% over all 260 items (p = {FP(u17['all']['p_exact'])}).",
         ["v1 4B-2507 CPU 09-10", "v1 4B-2507 NPU llama.cpp 09-10", "results/v2/qwen3-1.7b/{cpu,npu_llama}/*.json"]),
        ("Version 1 draft (unpublished), finding 2; 2026-09-15 README",
         "The serving stack changes capability: the compiled build of the same checkpoint scores 92.3% against 98.5% "
         "for the GGUF.",
         "holds",
         f"The gap becomes significant only with version 2's paraphrases. CPU-only against compiled-only passes are "
         f"{v1cc['only_first']} against {v1cc['only_second']} in version 1 (p = {FP(v1cc['p_exact'])}); in version 2 canonical "
         f"{cpu_comp['canonical']['only_first']} against {cpu_comp['canonical']['only_second']}, p = "
         f"{FP(cpu_comp['canonical']['p_exact'])} (Holm {FP(cpu_comp['canonical']['p_holm'])}); paraphrase "
         f"{cpu_comp['paraphrase']['only_first']} against {cpu_comp['paraphrase']['only_second']}, p = "
         f"{FP(cpu_comp['paraphrase']['p_exact'])} (Holm {FP(cpu_comp['paraphrase']['p_holm'])}); all items "
         f"{cpu_comp['all']['only_first']} against {cpu_comp['all']['only_second']}, p = {FP(cpu_comp['all']['p_exact'])} "
         f"(Holm {FP(cpu_comp['all']['p_holm'])}), pooled {pooled(cpu_comp['all'])}%. The base Qwen3-4B scores "
         f"{pooled(base)}% canonical, {base['only_first']} against {base['only_second']}.",
         ["v1 4B-2507 CPU 09-15", "v1 4B-2507 compiled 09-10", "results/v2/qwen3-4b-2507/*/*.json", "results/v2/qwen3-4b/*/canonical.json"]),
        ("Version 1 draft (unpublished), finding 2",
         "On the compiled path tool selection stays at 100%, so every error is the wrong command within the right tool.",
         "changed",
         f"Version 2 compiled tool selection is {P(v2p['tool_sel'])}% canonical and {P(o[COMP_0910]['v2_paraphrase']['tool_sel'])}% "
         f"paraphrase; {P(v2p['no_structured_call_rate'])}% of canonical requests return no structured call (cases "
         f"{', '.join(map(str, v2p['text_reply_idx']))} answer with text in every repetition), against "
         f"{P(o[COMP_0910]['v1']['tool_sel'])}% tool selection in version 1.",
         ["v1 4B-2507 compiled 09-10", "results/v2/qwen3-4b-2507/compiled/{canonical,paraphrase}.json"]),
        ("Version 1 draft (unpublished), finding 2",
         "Both builds err at the boundary between a port's administrative state and its PoE power, and the compiled "
         "build crosses it more often.",
         "holds",
         f"The PoE mode command answers a request to enable or disable a port in {len(poe[CPU_0915])} case on the CPU in "
         f"version 1 and {len(poe['v2 qwen3-4b-2507/cpu'])} in version 2, and in {len(poe[COMP_0910])} compiled cases "
         f"({', '.join(map(str, poe[COMP_0910]))}) against {len(poe['v2 qwen3-4b-2507/compiled'])} "
         f"({', '.join(map(str, poe['v2 qwen3-4b-2507/compiled']))}).",
         ["v1 4B-2507 CPU 09-15", "v1 4B-2507 compiled 09-10", "results/v2/qwen3-4b-2507/{cpu,compiled}/canonical.json"]),
        ("Version 1 draft (unpublished), finding 3; 2026-09-14 and 2026-09-15 READMEs",
         "A larger model is a net loss on the CPU: Qwen3-8B is about 1.5 times slower than Qwen3-4B-Instruct-2507 and "
         "slightly less accurate (97.2% against 98.5%).",
         "holds",
         f"The deficit is now significant. Version 1 has {v1b['only_first']} against {v1b['only_second']} discordant cases "
         f"(p = {FP(v1b['p_exact'])}); version 2 has canonical {b4['canonical']['only_first']} against {b4['canonical']['only_second']}, p = "
         f"{FP(b4['canonical']['p_exact'])}; paraphrase {b4['paraphrase']['only_first']} against {b4['paraphrase']['only_second']}, "
         f"p = {FP(b4['paraphrase']['p_exact'])} (Holm {FP(b4['paraphrase']['p_holm'])}); all items {pooled(b4['all'])}%, "
         f"{b4['all']['only_first']} against {b4['all']['only_second']}, p = {FP(b4['all']['p_exact'])} (Holm "
         f"{FP(b4['all']['p_holm'])}). The 8B takes {t['v1_8b_over_4b_cpu']:.3f} times the 4B's turn time in version 1 and "
         f"{t['v2_8b_over_4b_cpu']:.3f} times in version 2.",
         ["v1 4B-2507 CPU 09-15", "v1 8B CPU 09-14", "results/v2/qwen3-4b-2507/cpu/*.json", "results/v2/qwen3-8b/cpu/*.json"]),
        ("Version 1 draft (unpublished), finding 3",
         "The 8B makes a structural error the 4B never makes, returning a command name as the tool name, and picks the "
         "PoE mode command for 'configure poe power 15W port 1' in every repetition.",
         "holds",
         f"The 8B names a command as the tool in {o[CPU8_0914]['v1']['command_as_tool_requests']} of "
         f"{o[CPU8_0914]['v1']['n_requests']} requests in version 1, and in {o[CPU8_0914]['v2']['command_as_tool_requests']} of "
         f"{o[CPU8_0914]['v2']['n_requests']} canonical and {o[CPU8_0914]['v2_paraphrase']['command_as_tool_requests']} of "
         f"{o[CPU8_0914]['v2_paraphrase']['n_requests']} paraphrase requests in version 2; the 4B-2507 on the CPU does so in "
         f"{o[CPU_0915]['v1']['command_as_tool_requests']} and {v2c['command_as_tool_requests']}. The 8B fails cases "
         f"{', '.join(map(str, o[CPU8_0914]['v1']['failing_idx']))} in version 1 and "
         f"{', '.join(map(str, o[CPU8_0914]['v2']['failing_idx']))} in version 2, with the PoE mode answer to case 57 in both.",
         ["v1 8B CPU 09-14", "results/v2/qwen3-8b/cpu/{canonical,paraphrase}.json"]),
        ("Version 1 draft (unpublished), finding 3",
         "Because the 8B cannot use the NPU, its best turn on this board is 77.7 s, against 7.4 s for the 4B on the "
         "compiled path.",
         "superseded",
         f"The compiled 8B serves at {S(q8c['v2']['turn_median_s'])} s per turn ({P(q8c['v2']['e2e'])}% canonical, "
         f"{P(c48['second_e2e'][0])}% over all items), {t['v2_8b_over_4b_compiled']:.2f} times the compiled 4B-2507's "
         f"{S(v2p['turn_median_s'])} s, with no detectable accuracy difference ({pooled(c48)}%, {c48['only_first']} against "
         f"{c48['only_second']}, p = {FP(c48['p_exact'])}).",
         ["results/v2/qwen3-8b/compiled/*.json", "results/v2/qwen3-4b-2507/compiled/*.json"]),
        ("Version 1 draft (unpublished), finding 4",
         "The 8B does not run on this board's NPU by either path, so the largest model the NPU can serve lies between "
         "4B and 8B parameters.",
         "superseded",
         f"On the compiled path a bundle with the same layout ({f8['bundle']['parts']} parts, largest "
         f"{f8['bundle']['largest_part_gb']} GB) loads in {f8['load_and_first_reply_s']} s to first reply under GenieX v0.8.0 "
         f"and scores {P(q8c['v2']['e2e'])}% ({PCI(q8c['v2']['e2e_ci'])}) canonical. The claim still holds for llama.cpp NPU, "
         f"where the 8B load check ends in {q8n['v2_load_check'].get('error', q8n['v2_load_check'])} at the 900 s limit.",
         ["results/2026-09-14-qcs9075-qwen3-8b/README.md", "results/v2/qwen3-8b/compiled/server.json",
          "results/v2/qwen3-8b/npu_llama/load_failure.json"]),
        ("2026-09-14 README; version 1 report, finding 4",
         "Both NPU load failures come from the per-domain memory-map limit of this silicon (roughly 1 to 1.5 GB).",
         "superseded",
         "The compiled 8B now loads with the same part sizes, so a fixed limit of the silicon does not explain the version 1 "
         "failure. The change cannot be attributed to one cause, since the server (v0.4.0 to v0.8.0), the rebuilt bundle (AI Hub 0.62.2 to "
         f"{d['checks']['v2_catalog_release']}) and NPU co-residency (version 1 loaded inside the resident container) all "
         f"changed. On llama.cpp NPU the {len(sizes)} failed mappings in the version 2 load logs range from "
         f"{min(sizes) / 1e9:.3f} to {max(sizes) / 1e9:.3f} GB; the mechanism is unresolved.",
         ["results/2026-09-14-qcs9075-qwen3-8b/README.md", "results/v2/*/npu_llama/load_failure.json"]),
        ("Version 1 report, table 1; 2026-09-10 README",
         "Qwen3-4B-Instruct-2507 runs on the llama.cpp NPU path at 98.5% and 14.5 s per turn, and its buffers load "
         "without difficulty.",
         "superseded",
         f"Under GenieX v0.8.0 it fails at load in all {len(wt)} settings tried ({'; '.join(a['setting'] for a in wt)}). The "
         f"path loads {len(loaded)} of {len(npu)} models ({', '.join(loaded)}){bound}. Version 1's figures "
         f"({P(o[NPU_0910]['v1']['e2e'])}%, {S(o[NPU_0910]['v1']['turn_median_s'])} s) cannot be reproduced on current "
         "software. This is probably a regression, but identical artifact and configuration are not proven: version 1 "
         "recorded no GGUF hash, and version 2 sets a 4,096-token context that version 1 did not.",
         ["v1 4B-2507 NPU llama.cpp 09-10", "results/v2/qwen3-4b-2507/npu_llama/{load_failure,workaround_tests}.json"]),
        ("Version 1 draft (unpublished), finding 5; 2026-09-16 README",
         "Qwen3.5-2B returns its tool calls as text that the endpoint does not turn into structured calls, so it scores "
         "0% end to end.",
         "superseded",
         f"GenieX v0.8.0 returns structured calls, and the model scores {P(q35['v2']['e2e'])}% ({PCI(q35['v2']['e2e_ci'])}) canonical with "
         f"{q['v2_text_tool_calls']} text calls in {q['v2_requests']} requests; {q35['test']['only_first']} against "
         f"{q35['test']['only_second']} discordant cases, p = {FP(q35['test']['p_exact'])}. Version 1 recorded no GGUF hash, "
         "so a changed file cannot be ruled out.",
         ["v1 Qwen3.5-2B CPU 09-16", "results/v2/qwen3.5-2b/cpu/canonical.json"]),
        ("Version 1 draft (unpublished), finding 5; 2026-09-16 README",
         "Re-parsed leniently, about 70% of attempts give a usable call and about 21% pass end to end, so the model is "
         "weak even with a parser.",
         "superseded",
         f"The figures reproduce only with a parser that tolerates truncation ({P(q['v1_lenient']['parsed_rate'])}% parsed, "
         f"{P(q['v1_lenient']['e2e'])}% ({PCI(q['v1_lenient']['e2e_ci'])}) end to end), because {q['v1_at_cap']} of "
         f"{q['v1_requests']} stored replies were cut at {q['v1_content_cap_chars']} characters; the analysis parser recovers "
         f"{P(q['v1_strict']['e2e'])}%. Version 2 measures {P(q35['v2']['e2e'])}% canonical and "
         f"{P(q35['v2_paraphrase']['e2e'])}% paraphrase.",
         ["v1 Qwen3.5-2B CPU 09-16", "results/v2/qwen3.5-2b/cpu/{canonical,paraphrase}.json"]),
        ("2026-09-16 README; version 1 report, finding 5",
         "On the llama.cpp NPU path Qwen3.5-2B offloads only part of the model.",
         "changed",
         f"Under GenieX v0.8.0 it does not load (load check {q35u['v2_load_check'].get('http_status')}, mapping failure "
         f"logged at {next(x['failed_mapping_bytes'] for x in npu if x['model'] == 'qwen3.5-2b') / 1e9:.3f} GB).",
         ["results/2026-09-16-qcs9075-qwen3-5-2b/README.md", "results/v2/qwen3.5-2b/npu_llama/load_failure.json"]),
        ("Proposal, candidate finding 2",
         "The same weights swing from 0% to 100% tool-calling accuracy depending on how they are served.",
         "changed",
         f"No version 1 result in the repository shows this. In version 2 the widest within-model canonical spread among "
         f"configurations whose prompts carried the tools is {rc.MODEL_NAMES[sp['tools_delivered']['model']]}, "
         f"{P(sp['tools_delivered']['high'][0])}% {PATH_SHORT[sp['tools_delivered']['high'][1]]} against "
         f"{P(sp['tools_delivered']['low'][0])}% {PATH_SHORT[sp['tools_delivered']['low'][1]]}; counting configurations "
         f"that never received the tools, {rc.MODEL_NAMES[sp['all']['model']]} spans {P(sp['all']['high'][0])}% "
         f"{PATH_SHORT[sp['all']['high'][1]]} to {P(sp['all']['low'][0])}% {PATH_SHORT[sp['all']['low'][1]]}.",
         ["results/v2/*/*/canonical.json"]),
        ("Proposal, candidate finding 3",
         "Accuracy of 96.9% on clean phrasing drops to 77.4% on real-world paraphrases for the same model.",
         "changed",
         "The figures match no result in the repository. Version 2 measures, canonical to paraphrase, "
         + "; ".join(f"{r['configuration']} {P(r['v2']['e2e'])} to {P(r['v2_paraphrase']['e2e'])}%"
                     for r in [o[CPU_0915], o[COMP_0910], o[CPU8_0914], o[Q35_0916]])
         + f"; {q8c['configuration']} {P(q8c['v2']['e2e'])} to {P(q8c['v2_paraphrase']['e2e'])}%.",
         ["results/v2/*/*/{canonical,paraphrase}.json"]),
        ("Proposal, work to do (data caveat)",
         "The mixed-stack throughput caveat was cleared on 2026-09-10 by re-measuring under one harness with 5 "
         "repetitions, standard deviation and pinned versions.",
         "changed",
         f"The 2026-09-10 data used two server binaries (v0.4.0 compiled, {ALPHA} llama.cpp), averaged compiled decode over "
         f"{d['checks']['v1_compiled_decode_tokens'][0]} tokens against 128, lacked the 2,800-token compiled row and stamped the "
         "llama.cpp legs with the container's version. Version 2 clears the caveat for CPU against compiled (one binary, "
         f"max_tokens {'honoured in every row' if d['checks']['v2_probe_rows_max_tokens_honored'] else 'not always honoured'}, "
         "all five lengths present), "
         "not for the version 1 llama.cpp NPU figures, which have no version 2 counterpart.",
         ["results/2026-09-10-qcs9075-qwen3-4b/phase2-*.json", "results/v2/qwen3-4b-2507/{cpu,compiled}/probe_agent.json"]),
        ("Version 1 report, limitations",
         "Temperature 0 gave identical results across repetitions for every 4B configuration; one 8B case changed outcome.",
         "changed",
         f"The structured answer differs across repetitions in {len(det[COMP_0910]['answer_unstable_idx'])} compiled 4B-2507 "
         f"cases in version 1 and {len(v2p['answer_unstable_idx'])} in version 2; CPU 4B-2507 {len(det[CPU_0915]['answer_unstable_idx'])} "
         f"and {len(v2c['answer_unstable_idx'])} (case {', '.join(map(str, v2c['answer_unstable_idx']))}); 8B CPU "
         f"{len(det[CPU8_0914]['answer_unstable_idx'])} and {len(o[CPU8_0914]['v2']['answer_unstable_idx'])}, of which "
         f"{len(det[CPU8_0914]['outcome_unstable_idx'])} and {len(o[CPU8_0914]['v2']['outcome_unstable_idx'])} change outcome "
         f"(cases {', '.join(map(str, o[CPU8_0914]['v2']['outcome_unstable_idx']))} in version 2). Version 2 has 3 "
         "repetitions against 5.",
         ["v1 4B-2507 CPU 09-15", "v1 4B-2507 compiled 09-10", "v1 8B CPU 09-14", "results/v2/*/*/canonical.json"]),
        ("Version 1 report, measurement discipline",
         "Run-to-run latency noise on the reference configuration is about 0.5% of the value.",
         "holds",
         f"The median per-case coefficient of variation is {o[COMP_0910]['v1']['latency_cv_median_pct']:.2f}% compiled in "
         f"version 1 and {v2p['latency_cv_median_pct']:.2f}% in version 2, and {o[CPU_0915]['v1']['latency_cv_median_pct']:.2f}% "
         f"and {v2c['latency_cv_median_pct']:.2f}% on the CPU.",
         ["v1 4B-2507 CPU 09-15", "v1 4B-2507 compiled 09-10", "results/v2/qwen3-4b-2507/{cpu,compiled}/canonical.json"]),
        ("Version 1 report, thermal headroom",
         "The board has ample headroom: its zones throttle at 105 C and the one in-run reading was 66.5 C.",
         "holds",
         f"Traced every 20 s in {th['files_traced']} files, the hottest reading is {th['max_c']:.1f} C ({th['max_at']}), "
         f"{th['margin_c']:.1f} C below the trip; the hottest CPU-path run reaches {th['cpu_path_max_c']:.1f} C.",
         ["results/v2/*/*/*.json (thermal)"]),
        ("Version 1 report, what was measured",
         "One agent request is about 2,700 prompt tokens and the median answer is 51 tokens.",
         "holds",
         f"The median request has {v2c['prompt_tokens_median']:,.0f} prompt tokens on the CPU and "
         f"{v2p['prompt_tokens_median']:,.0f} compiled, with median completion tokens {v2c['completion_tokens_median']:.0f} and {v2p['completion_tokens_median']:.0f}. Version 1 "
         f"recorded no prompt counts; its compiled server reported {d['compiled_turn_change']['v1_completion_tokens_median']:.0f} "
         "completion tokens per answer.",
         ["results/v2/qwen3-4b-2507/{cpu,compiled}/canonical.json", "v1 4B-2507 compiled 09-10"]),
        ("Version 1 report, board and software",
         "Both llama.cpp paths run on the GenieX 0.5.1-alpha server with llama.cpp runtime 6ba5ef2.",
         "superseded",
         f"Version 1 files record 6ba5ef2 only in the resident container's version field ('{V(d['provenance'][CPU_0915]['container_version_field'])}'); "
         f"the alpha server's llama.cpp revision is not recorded. Version 2 records '{V(d['provenance']['v2 qwen3-4b-2507/cpu']['server_version'])}' "
         "from the answering binary.",
         ["v1 4B-2507 CPU 09-15", "results/v2/qwen3-4b-2507/cpu/canonical.json"]),
        ("Version 1 report, limitations",
         "Paraphrases are not measured, the compiled path has not been run on the public schema, the 2,800-token compiled "
         "phase row is missing and the llama.cpp NPU path ran two repetitions.",
         "superseded",
         f"Version 2 measures the paraphrases (195 items x 1), runs the compiled path on the public schema at "
         f"{P(v2p['e2e'])}% ({PCI(v2p['e2e_ci'])}) and has the 2,795-token compiled row "
         f"({R(_tp(d, COMP_0910, 2800)['v2']['prefill'][0])} tok/s). The llama.cpp NPU "
         "limitation remains, since the 4B cannot be re-run there.",
         ["results/v2/qwen3-4b-2507/compiled/*.json"]),
        ("Version 1 report, limitations",
         "The CPU path scores 98.5% under both the development tool copy and the public schema, so the two grade alike.",
         "holds",
         f"Version 2 gives the same modal answer as the 2026-09-15 run (public schema) on "
         f"{o[CPU_0915]['test']['identical_modal_answers']}/65 cases and as the 2026-09-10 run (development copy) on "
         f"{o[CPU_0910]['test']['identical_modal_answers']}/65, with {o[CPU_0915]['test']['only_first']} against "
         f"{o[CPU_0915]['test']['only_second']} discordant cases.",
         ["v1 4B-2507 CPU 09-15", "v1 4B-2507 CPU 09-10", "results/v2/qwen3-4b-2507/cpu/canonical.json"]),
    ]
    return [{"id": f"F{i:02d}", "source": src, "claim": claim, "status": status, "evidence": ev, "files": files}
            for i, (src, claim, status, ev, files) in enumerate(rows, 1)]


# ------------------------------------------------------------------ tables
def tables_md(data):
    m = data["method"]
    L = ["## Version 1 against version 2: canonical accuracy on overlapping configurations\n",
         f"All figures are for the {data['scope']}. Each version 1 run is compared with the version 2 run of the same model and path on "
         "the 65 canonical cases, matched by index with identical request text.\n",
         "| Configuration | Version 1 run | Reps v1 / v2 | End to end v1 | End to end v2 | v2 paraphrase | "
         "Only v1 / only v2 passes | Exact McNemar p (Holm) | v2 minus v1, pp (95% CI) | Identical modal answers | "
         "Median s/turn v1 / v2 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in data["overlap"]:
        a, b, t = r["v1"], r["v2"], r.get("test")
        run = f"{r['v1_date']}, {r['v1_server']}, {r['v1_tools']}"
        if b:
            L.append(f"| {r['configuration']} | {run} | {a['repeats']} / {b['repeats']} | {P(a['e2e'])} ({PCI(a['e2e_ci'])}) | "
                     f"{P(b['e2e'])} ({PCI(b['e2e_ci'])}) | {P(r['v2_paraphrase']['e2e'])} ({PCI(r['v2_paraphrase']['e2e_ci'])}) | "
                     f"{t['only_first']} / {t['only_second']} | {FP(t['p_exact'])} ({FP(t['p_holm_cross_version'])}) | "
                     f"{PP(t['diff_second_minus_first'])} ({PP(t['diff_ci'][0])} to {PP(t['diff_ci'][1])}) | "
                     f"{t['identical_modal_answers']}/{t['n_cases']} ({t['identical_modal_answers_empty_params_absent']}) | "
                     f"{S(a['turn_median_s'])} / {S(b['turn_median_s'])} |")
        else:
            L.append(f"| {r['configuration']} | {run} | {a['repeats']} / - | {P(a['e2e'])} ({PCI(a['e2e_ci'])}) | "
                     f"did not load | - | - | - | - | - | {S(a['turn_median_s'])} / - |")
    for u in data["unmeasured"]:
        b = u["v2"]
        v2txt = f"{P(b['e2e'])} ({PCI(b['e2e_ci'])})" if b else "did not load"
        para = f"{P(u['v2_paraphrase']['e2e'])} ({PCI(u['v2_paraphrase']['e2e_ci'])})" if u["v2_paraphrase"] else "-"
        L.append(f"| {u['configuration']} | {u['outcome']} | - / {b['repeats'] if b else '-'} | - | {v2txt} | {para} | - | - | - | - | "
                 f"- / {S(b['turn_median_s']) if b else '-'} |")
    c, ci = data["control_v1_compiled_vs_v2_base_4b_compiled"], data["compiled_identity"]
    L.append(f"\nAs a control, the version 1 compiled run against version 2's base Qwen3-4B bundle gives {c['only_first']} against "
             f"{c['only_second']} discordant cases (p = {FP(c['p_exact'])}, Holm {FP(c['p_holm_cross_version'])}) and "
             f"{c['identical_modal_answers_empty_params_absent']}/65 identical answers, against "
             f"{ci['identical_modal_answers']['v2 qwen3-4b-2507/compiled']}/65 for the Qwen3-4B-Instruct-2507 bundle, "
             f"which repeats the version 1 compiled answers on failing cases "
             f"{', '.join(map(str, ci['failures_shared_with_2507_bundle']))}; no 4B CPU build gives the same answer on "
             f"{' or '.join(map(str, ci['failures_only_in_2507_bundle']))}. The version 1 bundle therefore behaves as the "
             "Instruct-2507 build (behavioural evidence; version 1 recorded no bundle hash).\n")
    L.append(f"End to end in %, n = 65 cases (v2 paraphrase n = 195 items, 1 rep), 95% percentile bootstrap CI over cases "
             f"or items ({m['resamples']:,} resamples, seed {m['seed']}). Exact two-sided McNemar on per-case majority "
             f"outcome; Holm across the {m['cross_version_holm_tests']} cross-version tests, the control included. The "
             "difference is a paired bootstrap over cases. Identical modal answers compare the most frequent structured "
             "answer per case as stored, and in brackets with an empty parameters object treated as absent. Median turn is "
             "the median over cases of the mean latency across repetitions. Sources are in the last section.\n")

    L += ["## Version 1 against version 2: phase throughput at shared prompt lengths\n",
          "| Configuration | Target | Prompt tok v1 / v2 | TTFT s v1 | TTFT s v2 | Prefill tok/s v1 | Prefill tok/s v2 [server] | "
          "Change | Decode tok/s v1 | Decode tok/s v2 [server] | Change | Decode tokens v1 / v2 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in data["throughput"]:
        for r in t["rows"]:
            x, y = r["v1"], r["v2"]
            if not y:
                if r["target"] == 2048:
                    L.append(f"| {t['configuration']}, v1 {t['v1_date']} | {r['target']} | {x['prompt_tokens']} / - | "
                             f"{MS(x['ttft_s'], S)} | did not load | {MS(x['prefill'])} | - | - | {MS(x['decode'])} | - | - | "
                             f"{x['decode_tokens']} / - |")
                continue
            if "error" in x:
                L.append(f"| {t['configuration']}, v1 {t['v1_date']} | {r['target']} | - / {y['prompt_tokens']} | "
                         f"{x['error']} | {MS(y['ttft_s'], S)} | - | {MS(y['prefill'])} [{R(y['prefill_server'][0])}] | - | - | "
                         f"{MS(y['decode'])} [{R(y['decode_server'][0])}] | - | - / {y['decode_tokens']} |")
                continue
            L.append(f"| {t['configuration']}, v1 {t['v1_date']} | {r['target']} | {x['prompt_tokens']} / {y['prompt_tokens']} | "
                     f"{MS(x['ttft_s'], S)} | {MS(y['ttft_s'], S)} | {MS(x['prefill'])} | {MS(y['prefill'])} "
                     f"[{R(y['prefill_server'][0])}] | {CH(r['prefill_change_pct'])} | {MS(x['decode'])} | {MS(y['decode'])} "
                     f"[{R(y['decode_server'][0])}] | {CH(r['decode_change_pct'])} | {x['decode_tokens']} / {y['decode_tokens']} |")
    v8 = data["throughput_v2_only"]["qwen3-8b/compiled"]
    r8 = next(r for r in v8["rows"] if r["target"] == 2048)
    L.append(f"| {v8['configuration']} | 2048 | - / {r8['prompt_tokens']} | did not load | {MS(r8['ttft_s'], S)} | - | "
             f"{MS(r8['prefill'])} [{R(r8['prefill_server'][0])}] | - | - | {MS(r8['decode'])} [{R(r8['decode_server'][0])}] | - | "
             f"- / {r8['decode_tokens']} |")
    L.append("\nMean +/- sample sd over 5 repetitions per row (client timing, streaming); version 2 server-reported "
             "rates in brackets (version 1 recorded none). Change is version 2 relative to version 1. Rows for a "
             "configuration measured in one version only are shown at the 2,048 target.\n")
    L += ["### Compiled over CPU, Qwen3-4B-Instruct-2507\n",
          "| Target | Prefill ratio v1 | Prefill ratio v2 client [server] | Decode ratio v1 | Decode ratio v2 | TTFT speed-up v1 / v2 |",
          "|---|---|---|---|---|---|"]
    for t in TARGETS:
        a, b = _ratio(data["ratios"]["v1_compiled_over_cpu"], t), _ratio(data["ratios"]["v2_compiled_over_cpu"], t)
        if "missing" in a:
            L.append(f"| {t} | missing ({a['missing']}) | {b['prefill']:.2f} [{b['prefill_server']:.2f}] | - | {b['decode']:.3f} | "
                     f"- / {b['ttft_speedup']:.1f} |")
        else:
            L.append(f"| {t} | {a['prefill']:.2f} | {b['prefill']:.2f} [{b['prefill_server']:.2f}] | {a['decode']:.3f} | "
                     f"{b['decode']:.3f} | {a['ttft_speedup']:.1f} / {b['ttft_speedup']:.1f} |")
    L.append("\nVersion 1 ratios use the 2026-09-10 CPU and compiled probes; its compiled decode is an average over "
             f"{data['checks']['v1_compiled_decode_tokens'][0]} generated tokens against 128 on the CPU. Version 2 decodes "
             "128 tokens on both paths with one server binary.\n")
    tn = data["turn"]
    ch = data["chat"]
    L += ["### Turn times\n",
          "| Configuration | Median s/turn v1 (95% CI) | Median s/turn v2 (95% CI) | Chat turn v1, estimated s | Chat turn v2, measured s |",
          "|---|---|---|---|---|"]
    est = ch["v1_estimate_s"]
    for r in data["overlap"]:
        a, b = r["v1"], r["v2"]
        ckey = f"{r['model']}/{r['path']}"
        meas = ch["v2_measured"].get(ckey) if b else None
        e = est.get(r["v1_run"])
        L.append(f"| {r['configuration']}, v1 {r['v1_date']} | {S(a['turn_median_s'])} ({S(a['turn_median_ci'][0])}-{S(a['turn_median_ci'][1])}) | "
                 + (f"{S(b['turn_median_s'])} ({S(b['turn_median_ci'][0])}-{S(b['turn_median_ci'][1])})" if b else "did not load")
                 + f" | {S(e) if e else '-'} | {MS(meas['e2e_s'], S) + ' (n = ' + str(meas['n']) + ')' if meas else '-'} |")
    q8 = next(u for u in data["unmeasured"] if u["model"] == "qwen3-8b" and u["path"] == "compiled")
    L.append(f"| {q8['configuration']} | did not load | {S(q8['v2']['turn_median_s'])} ({S(q8['v2']['turn_median_ci'][0])}-"
             f"{S(q8['v2']['turn_median_ci'][1])}) | - | {MS(ch['v2_measured']['qwen3-8b/compiled']['e2e_s'], S)} (n = 5) |")
    L.append(f"\nTurn time is the median over the 65 cases of the mean latency across repetitions, with a bootstrap CI of "
             f"the median. The CPU over compiled turn-time ratio for the 4B-2507 is {tn['v1_cpu_over_compiled']:.2f}x in version 1, {tn['v2_cpu_over_compiled']:.2f}x "
             f"in version 2, and the 8B over 4B-2507 ratio on the CPU {tn['v1_8b_over_4b_cpu']:.3f} and "
             f"{tn['v2_8b_over_4b_cpu']:.3f}. For the chat turn, version 1 applied its 256-token rates to 100 prompt and 300 generated tokens; version 2 measured about "
             "100 prompt and 300 generated tokens, mean +/- sd. "
             f"The compiled turn is {data['compiled_turn_change']['mean_latency_saved_s']:.2f} s shorter in version 2 on "
             f"average while answers are {data['compiled_turn_change']['mean_tokens_fewer']:.1f} completion tokens shorter "
             f"(slope {data['compiled_turn_change']['slope_s_per_token']:.3f} s per token over "
             f"{data['compiled_turn_change']['n_cases']} matched cases).\n")

    L += ["## Version 1 against version 2: protocol and server differences\n",
          "| Item | Version 1 | Version 2 | Source |", "|---|---|---|---|"]
    for p in data["protocol"]:
        L.append(f"| {p['item']} | {p['v1']} | {p['v2']} | {p['source']} |")

    L += ["\n## Version 1 claims re-tested within each version\n",
          "| Comparison | Version | Suite | n | First / second end to end | Only first / only second | Exact McNemar p (Holm) | "
          "First minus second, pp (95% CI) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in data["within_version"]:
        holm = f" ({FP(r['p_holm'])})" if "p_holm" in r else ""
        L.append(f"| {r['comparison']} | {r['version']} | {r['suite']} | {r['n_items']} | {P(r['first_e2e'][0])} / "
                 f"{P(r['second_e2e'][0])} | {r['only_first']} / {r['only_second']} | {FP(r['p_exact'])}{holm} | "
                 f"{PP(r['diff_first_minus_second'])} ({PP(r['diff_ci'][0])} to {PP(r['diff_ci'][1])}) |")
    fs = m["v2_family_sizes"]
    L.append(f"\nVersion 1 p-values are unadjusted. Version 2 Holm values are within the analyze.py families: paths within a "
             f"model per suite ({fs['paths_within_model']} tests), models within a path per suite ({fs['models_within_path']}), "
             f"and the same over all items ({fs['paths_within_model_all_items']} and {fs['models_within_path_all_items']}). "
             "Differences are 95% percentile paired bootstraps over cases or items.\n")

    L += ["## Status of version 1 findings\n",
          "Claims are taken from the unpublished version 1 draft report, from docs/PROPOSAL.md and from the version 1 "
          "run READMEs. A claim holds when version 2 supports it, is changed when its substance survives with different "
          "numbers or qualifiers, and is superseded when version 2 contradicts or replaces it. Intervals are 95% "
          "percentile bootstrap intervals over cases or items; p-values are exact McNemar, Holm-adjusted where marked.\n",
          "| # | Version 1 claim | Source | Status | Version 2 evidence |", "|---|---|---|---|---|"]
    for f in data["findings"]:
        L.append(f"| {f['id']} | {f['claim']} | {f['source']} | {f['status']} | {f['evidence']} |")

    L += ["\n## Sources of the version 1 numbers\n", "| Run | File | Format notes |", "|---|---|---|"]
    for r in data["overlap"]:
        L.append(f"| {r['v1_run']} | {r['v1_file']} | no label, server or prompt token counts; tools "
                 f"{data['provenance'][r['v1_run']]['tools_sha256']} |")
    for t in data["throughput"]:
        L.append(f"| {t['v1_run']} (probe) | {t['v1_file']} | no server timings or max_tokens flag; decode tokens "
                 f"{t['rows'][0]['v1'].get('decode_tokens')} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    print(tables_md(compute(sys.argv[1] if len(sys.argv) > 1 else os.path.join(rc.REPO, "results", "v2"))))
