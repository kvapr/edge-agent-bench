#!/usr/bin/env python3
"""Serving stack and measurement quality, for the version 2 report. Standard
library, plus PyYAML through report_common. Everything is computed from a
results directory laid out as results/v2, the suites, the code and
analysis/data/artifacts.json (written from the GGUF headers and the bundle listings
by analysis/inspect_artifacts.py).

- Coverage. The 13 models of campaign/v2_models.json by the three paths. A
  step is planned when its result directory exists; the compiled path is not
  offered for models without a catalog bundle. A measured step is unreadable
  when the median server-reported prompt of either suite is below analyze's
  tools-delivered threshold (the tools never reached the model), or when no
  reply was a structured call while most replies held a text tool call (the
  model's native format, returned unparsed). This is the rule the speed
  module uses.
- Load boundary on the llama.cpp NPU path. GGUF size from artifacts.json
  against the load outcome; the failing buffer is parsed from the server log
  kept in load_failure.json, and set beside the f16 KV cache at the served
  context (bytes per token from the GGUF header times --nctx). The workaround
  tests are read from workaround_tests.json. The boundary is reported as
  observed; the mechanism is not inferred.
- Tools delivered. Median prompt tokens per configuration and suite, the
  threshold, and how the artifact's prompt construction uses tools.
- Text-call forms. For configurations unreadable because calls came back as
  text: marker counts, parse success with analyze.parse_text_tool_call, calls
  per reply (a call is [TOOL_CALLS]name[ARGS], the template's form), and the
  intent score (the harness grader applied to the parsed call), with a
  percentile bootstrap CI over items.
- Provenance. Distinct values and counts of every recorded hash and version
  across run files and server records, and whether the hashes equal the
  harness, suites and system prompts in this checkout (the diagnostic prompt
  rebuilt with bench_tools_in_prompt.prompt_with_tools from the tool subset).
- Measurement quality. Thermal traces by path (main matrix; diagnostics
  apart), readiness gates, the outside-load check and the thermal-sampler
  check, client overhead, board time, errors and retries.

  The outside-load and sampler checks need each request's position on the
  sampler clock, which the harness does not record. For CPU-path runs it is
  reconstructed by fitting the 1-minute load average: while a request runs,
  llama.cpp keeps n tasks active (n from the deconvolved trace), so
  L(t) = L0 e^(-t/60) + b (1 - e^(-t/60)) + (n - b) * sum of block responses,
  with the gate durations and the warm-up as free parameters (coordinate
  search, several starts) and the run end constrained to fall between the last
  sample and the next. Residual load is the deconvolved observation minus the
  fitted own load. The compiled path, with the resident container stopped
  and little CPU use of its own, is the contrast, as the deviation from each
  run's median. The npu_llama legs also ran with the container stopped, but their
  own CPU load swings within a run, so the deviation from the median does not
  isolate outside load; their figures are kept only to show that. Prefill
  slowdown is each request's prompt_ms over the fastest of the three
  repetitions of its case. The sampler check sets each decode against the
  sampler's ssh sessions (which end at each sample time), with a placebo
  shifted by half a period, and is repeated with every fitted request moved
  0.5 s earlier and later to show how far it rests on the fit.

  Gate waits are not recorded. The canonical run's time beyond its requests
  (its start to the paraphrase run's start, minus the summed latency and one
  warm-up) bounds its four gates, and the CPU timeline fit estimates single
  gates. Board hours run from each server's start to the end of the last run
  it served; bench files stamp a run's start, probe files its end, and the
  thermal trace spans the run. Runs that start before any recorded server
  start belong to one unrecorded instance, counted from the first of them.
  Percentiles are analyze.pct (nearest rank).

Every bootstrap uses report_common.SEED and report_common.B.

Usage:
  python3 analysis/report_stack.py [results_dir]
"""
import collections
import hashlib
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import report_common as rc  # noqa: E402
from report_common import analyze  # noqa: E402
import bench_toolcall  # noqa: E402  (harness; report_common puts it on the path)
import bench_tools_in_prompt  # noqa: E402  (harness, the diagnostic wrapper)

ARTIFACTS = os.path.join(HERE, "data", "artifacts.json")
SCOPE = ("QCS9075 (Dragonwing IQ-9075 EVK, Qualcomm Linux 1.6), GenieX v0.8.0 (QAIRT 2.45, llama.cpp 9425611), "
         "2026-10-14 to 2026-10-17")
SUITES = ("canonical", "paraphrase")
TASKS = ("canonical", "paraphrase", "diag_tools_in_prompt", "probe_agent", "probe_chat")
HEAD_FILES = ("harness/bench_toolcall.py", "harness/phase_probe.py", "harness/bench_tools_in_prompt.py",
              "suites/test_cases.yaml", "suites/test_cases_paraphrase.yaml", "suites/tools_nested.json")
GIB = 1 << 30
MIB = 1 << 20
TRIP_C = 105.0                      # passive trip of the CPU and NPU zones (docs/THERMAL.md)
GATE_MAX_C, GATE_MAX_BUSY = 55.0, 0.30   # readiness thresholds the campaign ran with (harness defaults)
LOAD_CHECK_TIMEOUT_S = 900          # campaign/run_campaign.py load_check()
TAU = 60.0                          # time constant of the Linux 1-minute load average, s
SAMPLER_SLEEP_S = 20.0              # the thermal sampler sleeps this long between ssh sessions
MIN_GATE_S = 1.5                    # shortest readiness gate the timeline fit allows
SLOW = 0.03                         # a request is slowed when 3% slower than the fastest repetition
EXCESS = 0.5                        # residual active tasks counted as outside load
MIN_GROUP = 20                      # smallest group used for the per-sampler-session prefill cost
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_MAPPING = re.compile(r"buffer mapping failed : domain_id (\d+) size (\d+)")
_LOG_TIME = re.compile(r"^([A-Z][a-z]{2} +\d+ \d\d:\d\d:\d\d\.\d+)")
# one Mistral-form call, as the GGUF template writes it; a malformed reply can repeat [TOOL_CALLS] without [ARGS]
_TEXT_CALL = re.compile(r"\[TOOL_CALLS\]\s*[\w.-]+\s*\[ARGS\]")
SHIFTS_S = (-0.5, 0.5)              # timeline shifts for the sensitivity of the sampler result


# ------------------------------------------------------------------ helpers
def _epoch(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _timings(r):
    return (r.get("server_extra") or {}).get("timings") or {}


def _reps(cfg, suites=SUITES):
    return [r for s in suites if cfg.get(s) for c in cfg[s]["results"] for r in c.get("reps") or []]


def _median_prompt(cfg, suite):
    toks = [r["prompt_tokens"] for r in _reps(cfg, (suite,)) if (r.get("prompt_tokens") or 0) > 0]
    return statistics.median(toks) if toks else None


def _nctx(server):
    return int(re.search(r"--nctx (\d+)", server["server_options"]).group(1))


def _runs(cfgs):
    """Every tool-calling and probe run as {model, path, task, kind, diag, d}."""
    return [{"model": c["model"], "path": c["path"], "task": t, "kind": "probe" if t.startswith("probe") else "bench",
             "diag": t.startswith("diag"), "d": c[t]} for c in cfgs for t in TASKS if c.get(t)]


def _summ(v):
    v = sorted(v)
    return {"n": len(v), "median": statistics.median(v), "p5": analyze.pct(v, 5), "p95": analyze.pct(v, 95),
            "min": v[0], "max": v[-1], "mean": statistics.mean(v)} if v else None


def _fisher_greater(a, b, c, d):
    """One-sided Fisher exact p, P(X >= a) for the table [[a, b], [c, d]]."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    return sum(math.comb(r1, x) * math.comb(n - r1, c1 - x) for x in range(a, min(r1, c1) + 1)) / math.comb(n, c1)


def _ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def _spearman(x, y):
    rx, ry = _ranks(x), _ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


# ----------------------------------------------------------------- coverage
unreadable_reason = rc.unreadable_reason


def failure(lf):
    """Error string, class, failing buffer and server-log lines of a load failure."""
    lc = lf["load_check"]
    tail = _ANSI.sub("", lf.get("server_log_tail") or "")
    err = f"HTTP {lc['http_status']}: {lc['body']}" if "http_status" in lc else lc.get("error")
    lines = [ln for ln in tail.splitlines() if re.search(r"fail|error", ln, re.I)]
    m = _MAPPING.search(tail)
    if "fastrpc_mmap failed" in tail:
        cls = "fastrpc_mmap failed"
    elif "Timeout" in (err or "") and not lines:
        cls = "timeout, nothing logged"
    else:
        cls = err
    t = _LOG_TIME.match(lines[0]) if lines else None
    log_offset = None
    if t:
        start = datetime.fromisoformat(lf["started"].replace("Z", "+00:00"))
        logged = datetime.strptime(f"{start.year} {t.group(1)}", "%Y %b %d %H:%M:%S.%f").replace(tzinfo=start.tzinfo)
        log_offset = (logged - start).total_seconds()
    return {"error": err, "class": cls, "log_lines": [ln.rsplit("] ", 1)[-1] for ln in lines],
            "failed_buffer_bytes": int(m.group(2)) if m else None, "domain_id": int(m.group(1)) if m else None,
            "started": lf["started"], "log_time_minus_start_s": log_offset}


def coverage(cfgs, spec):
    by = {(c["model"], c["path"]): c for c in cfgs}
    cells, counts = [], collections.Counter()
    for m in spec["models"]:
        for p in rc.PATHS:
            c = by.get((m["name"], p))
            cell = {"model": m["name"], "path": p}
            if c is None:
                cell["status"] = "not offered" if p == "compiled" and not m.get("bundle") else "missing"
            elif c["load_failure"]:
                cell.update(status="load failure", reason=failure(c["load_failure"])["class"])
            else:
                why = unreadable_reason(c)
                if why:
                    cell.update(status="unreadable", reason=why, served_e2e=statistics.mean(rc.items(c).values()))
                else:
                    cell["status"] = "readable"
            counts[cell["status"]] += 1
            cells.append(cell)
    planned = counts["load failure"] + counts["unreadable"] + counts["readable"]
    return {"cells": cells, "counts": {"cells": len(cells), "planned": planned, "not_offered": counts["not offered"],
                                       "missing": counts["missing"], "load_failures": counts["load failure"],
                                       "measured": counts["unreadable"] + counts["readable"],
                                       "unreadable": counts["unreadable"], "readable": counts["readable"],
                                       "decided_by_stack": counts["load failure"] + counts["unreadable"]}}


# ------------------------------------------------------------ load boundary
def load_boundary(cfgs, arts):
    rows = []
    for c in cfgs:
        if c["path"] != "npu_llama":
            continue
        g = arts["models"][c["model"]]["gguf"]
        kv = g["kv_bytes_per_token_f16"] * _nctx(c["server"]) if g["kv_bytes_per_token_f16"] else None
        row = {"model": c["model"], "gguf": g["file"], "gguf_bytes": g["bytes"], "gguf_gib": g["bytes"] / GIB,
               "loaded": not c["load_failure"], "kv_cache_bytes": kv, "n_ctx": _nctx(c["server"])}
        if c["load_failure"]:
            f = failure(c["load_failure"])
            fb = f["failed_buffer_bytes"]
            row.update(f, failed_minus_kv=fb - kv if fb and kv else None)
        rows.append(row)
    rows.sort(key=lambda r: r["gguf_bytes"])
    ok = max((r for r in rows if r["loaded"]), key=lambda r: r["gguf_bytes"])
    bad = min((r for r in rows if not r["loaded"]), key=lambda r: r["gguf_bytes"])
    buffers = [r["failed_buffer_bytes"] for r in rows if r.get("failed_buffer_bytes")]
    offsets = collections.defaultdict(list)
    for r in rows:
        if r.get("log_time_minus_start_s") is not None:
            offsets[r["started"][:10]].append(r["log_time_minus_start_s"])
    return {"rows": rows, "largest_loaded": {"model": ok["model"], "bytes": ok["gguf_bytes"], "gib": ok["gguf_gib"]},
            "log_time_minus_start_s_by_day": {k: sorted(v) for k, v in sorted(offsets.items())},
            "smallest_failed": {"model": bad["model"], "bytes": bad["gguf_bytes"], "gib": bad["gguf_gib"]},
            "one_gib_between": ok["gguf_bytes"] < GIB < bad["gguf_bytes"],
            "failed_buffers": {"n": len(buffers), "min": min(buffers), "max": max(buffers),
                               "within_8_mib_of_1_gib": sum(abs(b - GIB) <= 8 * MIB for b in buffers)},
            "classes": dict(collections.Counter(r["class"] for r in rows if not r["loaded"]))}


def workaround_tests(cfgs, boundary):
    c = next((c for c in cfgs if c.get("workaround_tests")), None)
    if c is None:
        return None
    w = c["workaround_tests"]
    camp = next(r for r in boundary["rows"] if r["model"] == c["model"])
    attempts = [{**a, "failed_buffer_mib": a["failed_buffer_bytes"] / MIB} for a in w["attempts"]]
    return {"model": c["model"], "date": w["date"], "server": w["server"], "attempts": attempts,
            "campaign_attempt": {"failed_buffer_bytes": camp["failed_buffer_bytes"], "kv_cache_bytes": camp["kv_cache_bytes"],
                                 "failed_minus_kv": camp["failed_minus_kv"]},
            "any_loaded": any(a["loaded"] for a in attempts)}


# ----------------------------------------------------------- tools delivery
def _tools_finding(delivered, construction, ref, slot):
    if delivered:
        return ("native: the template renders the request's tools" if slot else
                "the template has no tools section, yet the tools reached the prompt, so the server added them")
    if construction.startswith("prefix/suffix table"):
        return "the prompt is built from the bundle's prefix/suffix table, which has no slot for tools"
    if ref == "system-message field only":
        return "the template reads tools only from a field on the system message, which the server leaves empty"
    if ref == "none":
        return "the template has no tools section and the server adds none"
    return "not explained by the template"


def tools_delivered(cfgs, arts, spec):
    thr = analyze.TOOLS_DELIVERED_MIN_PROMPT_TOKENS
    order = [m["name"] for m in spec["models"]]
    rows = []
    for c in sorted(cfgs, key=lambda c: (order.index(c["model"]), rc.PATHS.index(c["path"]))):
        if c["load_failure"]:
            continue
        a = arts["models"][c["model"]]
        if c["path"] == "compiled":
            b = a["bundle"]
            construction, ref, slot = b["prompt_construction"], b["jinja_template"]["tools_reference"], b["tool_slot"]
        else:
            t = a["gguf"]["chat_template"]
            construction, ref, slot = "GGUF chat template", t["tools_reference"], t["tools_reference"] == "request-level"
        meds = {s: _median_prompt(c, s) for s in (*SUITES, "diag_tools_in_prompt") if c.get(s)}
        delivered = all(meds[s] >= thr for s in SUITES)
        rows.append({"model": c["model"], "path": c["path"], "median_prompt_tokens": meds, "delivered": delivered,
                     "prompt_construction": construction, "template_tools_reference": ref, "tool_slot": slot,
                     "finding": _tools_finding(delivered, construction, ref, slot)})
    # one model served with and without its tools: the canonical difference is the tool block as rendered
    block = {}
    for m in order:
        mine = [r for r in rows if r["model"] == m]
        got = [r["median_prompt_tokens"]["canonical"] for r in mine if r["delivered"]]
        lost = [r["median_prompt_tokens"]["canonical"] for r in mine if not r["delivered"]]
        if got and lost:
            block[m] = {"delivered": got, "not_delivered": lost, "difference": sorted({g - x for g in got for x in lost})}
    return {"threshold_tokens": thr, "rows": rows, "tool_block_tokens": block,
            "not_delivered": [f"{r['model']}/{r['path']}" for r in rows if not r["delivered"]]}


def text_call_forms(cfg):
    """Reply forms and intent score of a configuration whose calls came back as text."""
    out, items = {}, {}
    for s in SUITES:
        reps = [r for c in cfg[s]["results"] for r in c["reps"]]
        texts = [r.get("content") or "" for r in reps]
        per_case = []
        for c in cfg[s]["results"]:
            oks = []
            for r in c["reps"]:
                name, args = analyze.parse_text_tool_call(r.get("content"))
                oks.append(analyze.e2e_ok(analyze.grade(c, name, args)) if name else False)
            per_case.append(sum(oks) / len(oks))
            items[f"{s}:{c['idx']}"] = per_case[-1]
        lo, hi = analyze.bootstrap_ci(per_case)
        out[s] = {"requests": len(reps), "cases": len(per_case), "structured": sum(bool(r.get("got_tool")) for r in reps),
                  "with_marker": sum("[TOOL_CALLS]" in t for t in texts),
                  "starting_with_marker": sum(t.lstrip().startswith("[TOOL_CALLS]") for t in texts),
                  "parsed_with_valid_args": sum("[TOOL_CALLS]" in t and analyze.parse_text_tool_call(t)[1] is not None
                                                for t in texts),
                  "more_than_one_call": sum(len(_TEXT_CALL.findall(t)) > 1 for t in texts),
                  "intent_e2e": statistics.mean(per_case), "intent_ci": [lo, hi]}
    v = list(items.values())
    lo, hi = analyze.bootstrap_ci(v)
    counts = ("requests", "structured", "with_marker", "starting_with_marker", "parsed_with_valid_args", "more_than_one_call")
    out["pooled"] = {"items": len(v), "intent_e2e": statistics.mean(v), "intent_ci": [lo, hi],
                     **{k: sum(out[s][k] for s in SUITES) for k in counts}}
    return out


# --------------------------------------------------------------- provenance
def _tally(values):
    """[[value, count], ...] in first-seen order (a list, so booleans and
    integers keep their type in JSON); dicts and lists as sorted JSON text."""
    c = collections.Counter(json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for v in values)
    return [[v, n] for v, n in c.items()]


def _diag_prompt_sha():
    """sha256 of the diagnostic system prompt as this checkout builds it."""
    tools = bench_toolcall.load_tools_file(os.path.join(rc.REPO, "suites", "tools_nested.json"))
    tools = [t for t in tools if t["function"]["name"] in rc.SUBSET]
    return hashlib.sha256(bench_tools_in_prompt.prompt_with_tools(tools).encode()).hexdigest()


def provenance(runs, cfgs, arts):
    head = {p: _sha(os.path.join(rc.REPO, p)) for p in HEAD_FILES}
    sys_prompt = hashlib.sha256(bench_toolcall.SYSTEM_PROMPT.encode()).hexdigest()
    native = [r["d"]["provenance"] for r in runs if r["kind"] == "bench" and not r["diag"]]
    diag = [r["d"]["provenance"] for r in runs if r["diag"]]
    probes = {t: [r["d"]["provenance"] for r in runs if r["task"] == t] for t in ("probe_agent", "probe_chat")}
    probe = probes["probe_agent"] + probes["probe_chat"]
    bench = native + diag
    items = []

    def add(item, group, values, expected=None):
        t = _tally(values)
        items.append({"item": item, "group": group, "files": len(values), "values": t,
                      "equals_checkout": None if expected is None else [v for v, _ in t] == [expected]})
    add("harness sha256", "tool-calling runs", [p["harness_sha256"] for p in bench], head["harness/bench_toolcall.py"])
    add("harness sha256", "probe runs", [p["harness_sha256"] for p in probe], head["harness/phase_probe.py"])
    add("diagnostic wrapper sha256", "diagnostic runs", [p["diagnostic_wrapper_sha256"] for p in diag],
        head["harness/bench_tools_in_prompt.py"])
    for f in ("suites/test_cases.yaml", "suites/test_cases_paraphrase.yaml"):
        add(f"suite sha256 ({os.path.basename(f)})", "tool-calling runs",
            [p["cases_sha256"] for p in bench if p["cases_file"] == f], head[f])
    add("tool schema sha256", "tool-calling runs", [p["tools_sha256"] for p in bench], head["suites/tools_nested.json"])
    add("tool subset", "tool-calling runs", [sorted(p["tool_names"]) for p in bench], json.dumps(sorted(rc.SUBSET)))
    add("system prompt sha256", "native runs", [p["system_prompt_sha256"] for p in native], sys_prompt)
    add("system prompt sha256", "diagnostic runs", [p["system_prompt_sha256"] for p in diag], _diag_prompt_sha())
    add("system prompt text re-hashes", "diagnostic runs",
        [hashlib.sha256(p["system_prompt"].encode()).hexdigest() == p["system_prompt_sha256"] for p in diag], True)
    add("schema_version", "all runs", [p.get("schema_version") for p in bench + probe])
    add("harness_version", "tool-calling runs", [p.get("harness_version") for p in bench])
    add("request config", "native runs", [p["request_config"] for p in native])
    add("request config", "diagnostic runs", [p["request_config"] for p in diag])
    for t, ps in probes.items():
        add("request config", f"{t} runs", [p["request_config"] for p in ps])
    add("server version", "all runs", [p["server_version"].replace("\n", " | ") for p in bench + probe])
    servers = [s for c in cfgs for s in _server_records(c)]
    add("server version", "server records", [s["server_version"].replace("\n", " | ") for s in servers])
    add("server options", "server records", [s["server_options"] for s in servers])
    add("power mode (runner label, not logged by the server)", "server records", [s["power_mode"] for s in servers])
    add("OS", "all runs", [p["board"]["os"]["PRETTY_NAME"] for p in bench + probe])
    add("kernel", "all runs", [p["board"]["kernel"] for p in bench + probe])
    add("client Python", "all runs", [p["python"] for p in bench + probe])
    by_art = collections.defaultdict(list)
    for s in servers + [c["load_failure"] for c in cfgs if c["load_failure"]]:
        by_art[s["artifact"]].append(s["artifact_sha256"])
    artifacts = []
    for name, a in arts["models"].items():
        for kind in ("gguf", "bundle"):
            if a[kind]:
                shas = by_art.get(a[kind]["file"], [])
                artifacts.append({"model": name, "kind": kind, "file": a[kind]["file"], "records": len(shas),
                                  "distinct_sha256": sorted(set(shas)),
                                  "matches_artifacts_json": set(shas) == {a[kind]["sha256_recorded"]},
                                  "matches_download_log": a[kind]["sha256_matches_download_log"]})
    return {"checkout_sha256": head, "items": items, "artifacts": artifacts,
            "files": {"tool_calling_native": len(native), "diagnostic": len(diag), "probe": len(probe),
                      "server_records": len(servers)},
            "all_single_valued": (all(len(i["values"]) == 1 for i in items)
                                  and all(len(a["distinct_sha256"]) == 1 for a in artifacts)),
            "all_equal_checkout": all(i["equals_checkout"] is not False for i in items)}


def _server_records(cfg):
    """server.json and, where a diagnostic ran, server.diag_tools_in_prompt.json."""
    recs = [cfg["server"], rc.load_json(os.path.join(cfg["dir"], "server.diag_tools_in_prompt.json"))]
    return [r for r in recs if r]


# ------------------------------------------------------------------ thermal
def _zones(tr, zone):
    return [x[zone] for x in tr if x.get(zone) is not None]


def _summary_agrees(r):
    """The run's stored thermal summary matches its trace."""
    tr = r["d"]["thermal_trace"]
    s = r["d"]["summary"]["thermal"] if r["kind"] == "bench" else r["d"]["thermal"]
    a = _zones(tr, "all")
    return s["max_all_c"] == max(a) and s["n_samples"] == len(tr) and s["mean_all_c"] == round(statistics.mean(a), 1)


def thermal(runs):
    traced = [r for r in runs if r["d"].get("thermal_trace")]
    main = [r for r in traced if not r["diag"]]
    out = {}
    for p in rc.PATHS:
        rr = [r for r in main if r["path"] == p]
        z = {k: [v for r in rr for v in _zones(r["d"]["thermal_trace"], k)] for k in ("all", "cpu", "nsp")}
        top = max(rr, key=lambda r: max(_zones(r["d"]["thermal_trace"], "all")))
        out[p] = {"runs_bench": sum(r["kind"] == "bench" for r in rr), "runs_probe": sum(r["kind"] == "probe" for r in rr),
                  "samples": len(z["all"]), "hottest_max_c": max(z["all"]), "hottest_mean_c": statistics.mean(z["all"]),
                  "hottest_run": f"{top['model']}/{top['task']}", "share_ge_70c": sum(x >= 70 for x in z["all"]) / len(z["all"]),
                  "cpu_max_c": max(z["cpu"]), "cpu_mean_c": statistics.mean(z["cpu"]),
                  "npu_max_c": max(z["nsp"]), "npu_mean_c": statistics.mean(z["nsp"]),
                  "margin_cpu_c": TRIP_C - max(z["cpu"]), "margin_npu_c": TRIP_C - max(z["nsp"])}
    top = max(traced, key=lambda r: max(_zones(r["d"]["thermal_trace"], "all")))
    return {"trip_c": TRIP_C, "by_path": out,
            "hottest": {"c": max(_zones(top["d"]["thermal_trace"], "all")), "model": top["model"], "path": top["path"],
                        "task": top["task"]},
            "diagnostic_max_c": max((max(_zones(r["d"]["thermal_trace"], "all")) for r in traced if r["diag"]), default=None),
            "summary_blocks_agree": sum(_summary_agrees(r) for r in traced), "runs": len(traced)}


def _snapshots(runs, paths, main_only=False):
    return [g["snapshot"] for r in runs if r["path"] in paths and not (main_only and r["diag"])
            for g in r["d"].get("readiness") or []]


def gates(runs, cfgs):
    """Gate counts and the container listing over every run; release
    temperature and CPU busy over the main matrix, as the thermal rows."""
    out = {}
    for p in rc.PATHS:
        snaps, main = _snapshots(runs, (p,)), _snapshots(runs, (p,), main_only=True)
        temps = [s["thermal_max_c"]["all"] for s in main]
        busy = [s["cpu_busy"] for s in main]
        out[p] = {"gates": len(snaps), "gates_main": len(main), "release_c": _summ(temps), "busy": _summ(busy),
                  "container_listed": sum(bool(s.get("containers")) for s in snaps),
                  "all_within_thresholds": all(s["thermal_max_c"]["all"] < GATE_MAX_C and s["cpu_busy"] < GATE_MAX_BUSY
                                               for s in snaps)}
    slack = {p: [] for p in rc.PATHS}
    for c in cfgs:
        if c.get("canonical") and c.get("paraphrase"):
            can = c["canonical"]
            lat = [r["latency_s"] for x in can["results"] for r in x["reps"]]
            run_s = _epoch(c["paraphrase"]["provenance"]["timestamp_utc"]) - _epoch(can["provenance"]["timestamp_utc"])
            slack[c["path"]].append({"model": c["model"], "slack_s": run_s - sum(lat) - statistics.median(lat),
                                     "run_s": run_s})
    for p, v in slack.items():
        out[p]["canonical_slack_runs"] = v
        out[p]["canonical_slack_s"] = _summ([x["slack_s"] for x in v])
        out[p]["canonical_slack_share"] = sum(x["slack_s"] for x in v) / sum(x["run_s"] for x in v)
    legs = {"cpu_legs": ("cpu",), "npu_legs": ("npu_llama", "compiled")}
    busy = {k: [s["cpu_busy"] for s in _snapshots(runs, v)] for k, v in legs.items()}
    return {"thresholds": {"max_c": GATE_MAX_C, "max_cpu_busy": GATE_MAX_BUSY}, "by_path": out,
            "busy_by_leg": {k: {**_summ(v), "p90": analyze.pct(v, 90)} for k, v in busy.items()}}


# ------------------------------------------------------- load-average model
def deconvolve(tr):
    """Mean active-task count per sampling interval from consecutive 1-minute
    load averages: n = (L_i - L_(i-1) e^(-dt/60)) / (1 - e^(-dt/60))."""
    out = []
    for a, b in zip(tr, tr[1:]):
        dt = b["t_s"] - a["t_s"]
        if a.get("load1") is None or b.get("load1") is None or dt <= 0:
            continue
        k = math.exp(-dt / TAU)
        out.append({"t0": a["t_s"], "t1": b["t_s"], "n": (b["load1"] - k * a["load1"]) / (1 - k)})
    return out


def _response(t, u, v):
    """Load-average response at t to one active task during [u, v]."""
    if t <= u:
        return 0.0
    if t <= v:
        return 1.0 - math.exp(-(t - u) / TAU)
    return math.exp(-(t - v) / TAU) - math.exp(-(t - u) / TAU)


class TimelineFit:
    """Block start times of one CPU bench run (warm-up, then each repetition)
    on the sampler clock, fitted to the recorded load average."""

    def __init__(self, d):
        cases, tr = d["results"], d["thermal_trace"]
        self.reps = d["summary"]["repeats"]
        self.D = [sum(c["reps"][r]["latency_s"] for c in cases) for r in range(self.reps)]
        self.warmup_guess = statistics.median(r["latency_s"] for c in cases for r in c["reps"])
        self.N = statistics.median(x["n"] for x in deconvolve(tr) if x["n"] > 4)
        self.gate_load1 = [float(g["snapshot"]["loadavg"].split()[0]) for g in d.get("readiness") or []]
        self.tr = [(x["t_s"], x["load1"]) for x in tr if x.get("load1") is not None]
        self.t0, self.t_last = self.tr[0][0], self.tr[-1][0]
        self.dt = (self.t_last - self.t0) / max(1, len(self.tr) - 1)

    def blocks(self, p):
        """p = [gate before warm-up, warm-up duration, gate before rep 1, ...]."""
        g, w = [p[0]] + p[2:], p[1]
        t = self.t0 - 0.5 + g[0]            # the first sample completes about 0.5 s after the sampler starts
        out = [(t, t + w)]
        t += w
        for r, dur in enumerate(self.D):
            t += g[r + 1]
            out.append((t, t + dur))
            t += dur
        return out

    def _design(self, p):
        bl = self.blocks(p)
        obs = list(self.tr[1:]) + [(u - 0.3, L) for (u, _), L in zip(bl, self.gate_load1)]
        rows, y = [], []
        for t, L in obs:
            e = math.exp(-(t - self.t0) / TAU)
            busy = sum(_response(t, u, v) for u, v in bl)
            rows.append((e, 1 - e - busy))
            y.append(L - self.N * busy)
        return rows, y, bl

    def sse(self, p):
        rows, y, bl = self._design(p)
        a11 = sum(m[0] * m[0] for m in rows)
        a12 = sum(m[0] * m[1] for m in rows)
        a22 = sum(m[1] * m[1] for m in rows)
        b1 = sum(m[0] * v for m, v in zip(rows, y))
        b2 = sum(m[1] * v for m, v in zip(rows, y))
        det = a11 * a22 - a12 * a12
        if abs(det) < 1e-12:
            return float("inf"), None
        L0, b = (b1 * a22 - b2 * a12) / det, (a11 * b2 - a12 * b1) / det
        res = sum((v - (L0 * m[0] + b * m[1])) ** 2 for m, v in zip(rows, y))
        end = bl[-1][1]
        # the last request ends after the last sample and before the next would have been taken
        res += 10 * max(0.0, self.t_last - end) ** 2 + 10 * max(0.0, end - self.t_last - self.dt - 1.0) ** 2
        return res, (L0, b)

    def _valid(self, p):
        return p[0] >= MIN_GATE_S and p[1] > 0.2 and all(g >= MIN_GATE_S for g in p[2:])

    def _optimise(self, p):
        best, _ = self.sse(p)
        step = 64.0
        while step >= 0.25:
            improved = True
            while improved:
                improved = False
                for i in range(len(p)):
                    for s in (step, -step):
                        q = p[:]
                        q[i] += s
                        if self._valid(q):
                            v, _ = self.sse(q)
                            if v < best - 1e-9:
                                best, p, improved = v, q, True
            step /= 2
        return best, p

    def run(self):
        R = self.reps
        slack = max(R + 1, self.t_last + self.dt / 2 - (self.t0 - 0.5) - sum(self.D) - self.warmup_guess)
        starts = []
        for share in range(R + 1):
            g = [MIN_GATE_S] * (R + 1)
            g[share] = max(MIN_GATE_S, slack - MIN_GATE_S * R)
            starts.append([g[0], self.warmup_guess] + g[1:])
        g = [max(MIN_GATE_S, slack / (R + 1))] * (R + 1)
        starts.append([g[0], self.warmup_guess] + g[1:])
        _, p = min((self._optimise(s) for s in starts), key=lambda x: x[0])
        _, (L0, b) = self.sse(p)
        self.p, self.L0, self.b, self.bl = p, L0, b, self.blocks(p)
        return self

    def residuals(self, iv):
        """Observed minus fitted active-task count per deconvolved interval."""
        A = self.N - self.b
        pred = []
        for t, _ in self.tr:
            e = math.exp(-(t - self.t0) / TAU)
            pred.append((t, self.L0 * e + self.b * (1 - e) + A * sum(_response(t, u, v) for u, v in self.bl)))
        out = []
        for x, ((ta, la), (tb, lb)) in zip(iv, zip(pred, pred[1:])):
            k = math.exp(-(tb - ta) / TAU)
            out.append({**x, "res": x["n"] - (lb - k * la) / (1 - k)})
        return out

    def requests(self, d):
        """Each request's start on the sampler clock, in run order."""
        out = []
        for r in range(self.reps):
            t = self.bl[r + 1][0]
            for c in d["results"]:
                out.append((t, r, c))
                t += c["reps"][r]["latency_s"]
        return out


def _excess(values, runs, own):
    """Share of intervals whose extra active-task count exceeds +0.5 and +1.0."""
    return {"runs": runs, "intervals": len(values), "own_load_tasks": own,
            "share_gt_0_5": sum(x > 0.5 for x in values) / len(values),
            "share_gt_1_0": sum(x > 1.0 for x in values) / len(values),
            "n_gt_1_0": sum(x > 1.0 for x in values), "max": max(values)}


def _median_deviation(runs, path):
    """Deviation of the active-task count from each run's median, over the
    runs of one path (resident container stopped)."""
    per_run = []
    for r in runs:
        if r["path"] == path and len(r["d"].get("thermal_trace") or []) >= 3:
            ns = [x["n"] for x in deconvolve(r["d"]["thermal_trace"])]
            med = statistics.median(ns)
            per_run.append((med, [n - med for n in ns]))
    meds = [m for m, _ in per_run]
    return {**_excess([x for _, v in per_run for x in v], len(per_run), statistics.median(meds)),
            "run_median_range": [min(meds), max(meds)]}


def outside_load(runs):
    """Residual load inside CPU repetitions (resident container running)
    against compiled runs (container stopped), and whether prefill slowdowns
    coincide with it. The npu_llama figures are returned apart, unused as a
    contrast (see the module docstring)."""
    fits, cpu_iv, slowed = {}, [], {"canonical": [], "paraphrase": []}
    for r in runs:
        if r["path"] != "cpu" or r["kind"] != "bench" or r["diag"]:
            continue
        d = r["d"]
        f = TimelineFit(d).run()
        fits[(r["model"], r["path"], r["task"])] = f
        resid = f.residuals(deconvolve(d["thermal_trace"]))
        cpu_iv += [x["res"] for x in resid if any(u + 1 <= x["t0"] and x["t1"] <= v - 1 for u, v in f.bl[1:])]
        reqs = []
        for t, rep, c in f.requests(d):
            tm = _timings(c["reps"][rep])
            end = t + tm["prompt_ms"] / 1000
            ov = [(min(end, x["t1"]) - max(t, x["t0"]), x["res"]) for x in resid if min(end, x["t1"]) > max(t, x["t0"])]
            w = sum(o for o, _ in ov)
            reqs.append({"tm": tm, "c": c, "excess": sum(o * v for o, v in ov) / w if w > 0 else None})
        if f.reps > 1:
            for q in reqs:
                q["slow"] = q["tm"]["prompt_ms"] / min(_timings(x)["prompt_ms"] for x in q["c"]["reps"]) - 1
        else:   # one repetition: against the run's median prefill rate
            mr = statistics.median(q["tm"]["prompt_per_second"] for q in reqs)
            for q in reqs:
                q["slow"] = mr / q["tm"]["prompt_per_second"] - 1
        slowed[r["task"]] += [q for q in reqs if q["excess"] is not None]
    coincidence = {}
    for suite, qs in slowed.items():
        a = sum(q["slow"] > SLOW and q["excess"] > EXCESS for q in qs)
        b = sum(q["slow"] > SLOW and q["excess"] <= EXCESS for q in qs)
        c = sum(q["slow"] <= SLOW and q["excess"] > EXCESS for q in qs)
        d_ = sum(q["slow"] <= SLOW and q["excess"] <= EXCESS for q in qs)
        coincidence[suite] = {"requests": len(qs), "table": [[a, b], [c, d_]],
                              "fisher_p_greater": _fisher_greater(a, b, c, d_),
                              "spearman": _spearman([q["excess"] for q in qs], [q["slow"] for q in qs])}
    return fits, {
        "cpu": _excess(cpu_iv, len(fits), statistics.median(f.N for f in fits.values())),
        "compiled": _median_deviation(runs, "compiled"),
        "npu_llama_not_used": _median_deviation(runs, "npu_llama"),
        "coincidence": coincidence,
        "fitted_gates_before_reps_2_3_s": _summ([g for (_, _, t), f in fits.items() if t == "canonical" for g in f.p[3:]]),
        "fit_runs_used": sorted("/".join(k) for k in fits)}


def prefill_counterfactual(runs):
    """Median and p95 of per-case mean latency (canonical, CPU) observed and
    with every request's prefill set to the fastest repetition of its case."""
    rows = []
    for r in runs:
        if r["path"] != "cpu" or r["task"] != "canonical":
            continue
        obs, cf, slowed = [], [], 0
        for c in r["d"]["results"]:
            base = min(_timings(x)["prompt_ms"] for x in c["reps"])
            obs.append(statistics.mean(x["latency_s"] for x in c["reps"]))
            cf.append(statistics.mean(x["latency_s"] - (_timings(x)["prompt_ms"] - base) / 1000 for x in c["reps"]))
            slowed += sum(_timings(x)["prompt_ms"] / base - 1 > SLOW for x in c["reps"])
        rows.append({"model": r["model"], "requests_slowed": slowed, "requests": 3 * len(obs),
                     "median_s": statistics.median(obs), "median_cf_s": statistics.median(cf),
                     "median_change": statistics.median(obs) / statistics.median(cf) - 1,
                     "p95_change": analyze.pct(obs, 95) / analyze.pct(cf, 95) - 1})
    ch = [x["median_change"] for x in rows]
    return {"rows": rows, "median_change": statistics.median(ch), "max_change": max(ch),
            "p95_change_median": statistics.median(x["p95_change"] for x in rows),
            "p95_change_max": max(x["p95_change"] for x in rows)}


def _distance(a, b, wins):
    return min(a - v if v < a else (u - b if u > b else 0.0) for u, v in wins)


def _sampler_rows(runs, fits, shift=0.0):
    """One row per canonical CPU request, its fitted start moved by shift s.
    Returns (rows, sampling period per run)."""
    rows, periods = [], []
    for r in runs:
        f = fits.get((r["model"], r["path"], r["task"]))
        if f is None or r["task"] != "canonical":
            continue
        ts = [x["t_s"] for x in r["d"]["thermal_trace"]]
        period = statistics.median(b - a for a, b in zip(ts, ts[1:]))
        periods.append(period)
        ssh = [(ts[i] - max(0.3, ts[i] - ts[i - 1] - SAMPLER_SLEEP_S), ts[i]) for i in range(1, len(ts))]
        placebo = [(u + period / 2, v + period / 2) for u, v in ssh]
        for t, rep, c in f.requests(r["d"]):
            t += shift
            tm = _timings(c["reps"][rep])
            pn = [_timings(x)["predicted_n"] for x in c["reps"]]
            ds = t + tm["prompt_ms"] / 1000
            de = ds + tm["predicted_ms"] / 1000
            dslow = None
            if len(set(pn)) == 1 and pn[0] > 0:
                dslow = tm["predicted_ms"] / min(_timings(x)["predicted_ms"] for x in c["reps"]) - 1
            rows.append({"model": r["model"], "dslow": dslow, "decode_s": tm["predicted_ms"] / 1000,
                         "dist": _distance(ds, de, ssh), "dist_placebo": _distance(ds, de, placebo),
                         "pexcess_ms": tm["prompt_ms"] - min(_timings(x)["prompt_ms"] for x in c["reps"]),
                         "sessions_in_prefill": sum(1 for u, v in ssh if v > t and u < ds)})
    return rows, periods


def _prefill_costs(rows):
    """Per model, the prefill excess (over the fastest repetition) of requests
    with k + 1 sampler sessions inside their prefill minus that of requests
    with k, k the fewest; both groups need MIN_GROUP requests."""
    costs = []
    for m in sorted({x["model"] for x in rows}):
        groups = collections.defaultdict(list)
        for x in rows:
            if x["model"] == m:
                groups[x["sessions_in_prefill"]].append(x["pexcess_ms"])
        ks = sorted(k for k, v in groups.items() if len(v) >= MIN_GROUP)
        if len(ks) >= 2 and ks[1] == ks[0] + 1:
            costs.append({"model": m, "median_diff_ms": statistics.median(groups[ks[1]]) - statistics.median(groups[ks[0]]),
                          "mean_diff_ms": statistics.mean(groups[ks[1]]) - statistics.mean(groups[ks[0]])})
    diffs = [x["median_diff_ms"] for x in costs]
    return {"models": len(costs), "median_of_median_diffs": statistics.median(diffs), "range": [min(diffs), max(diffs)],
            "per_model": costs, "median_of_mean_diffs": statistics.median(x["mean_diff_ms"] for x in costs),
            "means_above_medians": sum(x["mean_diff_ms"] > x["median_diff_ms"] for x in costs)}


def _overlap(rows):
    dd = [x for x in rows if x["dslow"] is not None]
    ov = [x for x in dd if x["dist"] == 0]
    far = [x for x in dd if x["dist"] >= 2]
    a, b = sum(x["dslow"] > SLOW for x in ov), sum(x["dslow"] <= SLOW for x in ov)
    c, d = sum(x["dslow"] > SLOW for x in far), sum(x["dslow"] <= SLOW for x in far)
    return dd, ov, far, {"table": [[a, b], [c, d]], "p_greater": _fisher_greater(a, b, c, d)}


def sampler_interference(runs, fits):
    """Decode and prefill time of canonical CPU requests against the thermal
    sampler's ssh sessions, which end at each sample time, on the fitted
    timeline and with that timeline shifted by SHIFTS_S."""
    rows, periods = _sampler_rows(runs, fits)
    dd, ov, far, fisher = _overlap(rows)
    bins = []
    for lo, hi, label in ((0, 1e-9, "overlapping"), (1e-9, 2, "0 to 2 s"), (2, 5, "2 to 5 s"), (5, math.inf, "5 s or more")):
        s = [x for x in dd if lo <= x["dist"] < hi]
        sp = [x for x in dd if lo <= x["dist_placebo"] < hi]
        bins.append({"distance": label, "n": len(s), "gt_3pct": sum(x["dslow"] > SLOW for x in s),
                     "gt_10pct": sum(x["dslow"] > 0.10 for x in s),
                     "median_slowdown": statistics.median(x["dslow"] for x in s),
                     "placebo_n": len(sp), "placebo_gt_3pct": sum(x["dslow"] > SLOW for x in sp)})
    costs = _prefill_costs(rows)
    period = statistics.median(periods)
    shifted = []
    for sh in SHIFTS_S:
        r2, _ = _sampler_rows(runs, fits, sh)
        _, ov2, _, f2 = _overlap(r2)
        c2 = _prefill_costs(r2)
        shifted.append({"shift_s": sh, "overlapping": len(ov2), "overlapping_gt_3pct": f2["table"][0][0],
                        "fisher_p_greater": f2["p_greater"], "prefill_cost_median_ms": c2["median_of_median_diffs"],
                        "prefill_cost_range_ms": c2["range"], "prefill_cost_models": c2["models"]})

    def extra_ms(xs):
        return 1000 * statistics.median(x["dslow"] * x["decode_s"] / (1 + x["dslow"]) for x in xs)
    return {"requests_same_length": len(dd), "bins": bins, "fisher": fisher,
            "extra_decode_ms_overlapping": extra_ms(ov), "extra_decode_ms_far": extra_ms(far),
            "prefill_cost_per_session_ms": costs, "timeline_shifted": shifted,
            "sampler_period_s": period, "share_of_compute_time": costs["median_of_median_diffs"] / 1000 / period}


# ------------------------------------------------------- timing and errors
def overhead(runs):
    out = {}
    for p in rc.PATHS:
        ms, share = [], []
        for r in runs:
            if r["path"] == p and r["kind"] == "bench" and not r["diag"]:
                for rep in (x for c in r["d"]["results"] for x in c["reps"]):
                    t = _timings(rep)
                    o = rep["latency_s"] - (t["prompt_ms"] + t["predicted_ms"]) / 1000
                    ms.append(1000 * o)
                    share.append(o / rep["latency_s"])
        out[p] = {"ms": _summ(ms), "share_median": statistics.median(share), "negative": sum(x < 0 for x in ms)}
    tok = collections.Counter()
    for r in runs:
        if r["kind"] == "bench":
            for rep in (x for c in r["d"]["results"] for x in c["reps"]):
                t, u = _timings(rep), rep.get("usage") or {}
                tok["requests"] += 1
                tok["prompt_counts_agree"] += t.get("prompt_n") == u.get("prompt_tokens")
                tok["completion_counts_agree"] += t.get("predicted_n") == u.get("completion_tokens")
                tok["cached_tokens"] += bool((u.get("prompt_tokens_details") or {}).get("cached_tokens"))
    return {"by_path": out, "token_counts": dict(tok)}


def _run_interval(r):
    """(start, end) of a run, epoch s: bench files stamp the start, probe files
    the end, and the thermal trace spans the run."""
    t, span = _epoch(r["d"]["provenance"]["timestamp_utc"]), r["d"]["thermal_trace"][-1]["t_s"]
    return (t - span, t) if r["kind"] == "probe" else (t, t + span)


def board_time(runs, cfgs):
    """Board hours by path: each server instance from its start (or, with no
    record of it, its first run) to the end of the last run it served."""
    hours = collections.Counter()
    for c in cfgs:
        if c["load_failure"]:
            continue
        starts = sorted(_epoch(s["started"]) for s in _server_records(c))
        mine = [(r, *_run_interval(r)) for r in runs if (r["model"], r["path"]) == (c["model"], c["path"])]
        unrecorded = min((a for _, a, _ in mine if all(s > a for s in starts)), default=None)
        segs = {}
        for r, a, b in mine:
            s0 = max((s for s in starts if s <= a), default=unrecorded)
            seg = segs.setdefault(s0, {"end": b, "diag": r["diag"]})
            seg["end"] = max(seg["end"], b)
        for s0, seg in segs.items():
            hours[f"{c['path']}{'_diag' if seg['diag'] else ''}"] += (seg["end"] - s0) / 3600
    total = sum(hours.values())
    main = {p: hours[p] for p in rc.PATHS}
    return {"hours": {k: hours[k] for k in sorted(hours)}, "total_h": total,
            "share_main": {p: v / sum(main.values()) for p, v in main.items()},
            "share_all": {p: (hours[p] + hours[p + "_diag"]) / total for p in rc.PATHS}}


def errors(runs):
    bench, probe = collections.Counter(), collections.Counter()
    for r in runs:
        d = r["d"]
        if r["kind"] == "bench":
            lim = d["provenance"]["request_config"]["max_tokens"]
            for rep in (x for c in d["results"] for x in c["reps"]):
                bench["requests"] += 1
                bench["errors"] += bool(rep.get("error"))
                bench["empty_reply_retries"] += bool(rep.get("retried_empty_response"))
                bench["empty_replies_graded"] += not rep.get("got_tool") and not (rep.get("content") or "").strip()
                bench["missing_server_timings"] += not _timings(rep)
                bench["reached_max_tokens"] += (rep.get("completion_tokens") or 0) >= lim
                bench["diagnostic_requests"] += r["diag"]
        else:
            for L in d["lengths"]:
                probe["lengths"] += 1
                probe["lengths_failed"] += "error" in L
                for rep in L.get("reps") or []:
                    probe["reps"] += 1
                    probe["rep_errors"] += "error" in rep
                    probe["max_tokens_not_honoured"] += rep.get("max_tokens_honored") is False
                    probe["chunks_differ_from_tokens"] += rep.get("n_chunks") != rep.get("n_out")
    return {"bench": dict(bench), "probe": dict(probe)}


# ------------------------------------------------------------------ compute
def compute(results_dir):
    cfgs = rc.configs(results_dir)
    spec = rc.spec()
    arts = rc.load_json(ARTIFACTS)
    runs = _runs(cfgs)
    cov = coverage(cfgs, spec)
    boundary = load_boundary(cfgs, arts)
    as_text = {(x["model"], x["path"]) for x in cov["cells"] if x.get("reason") == "calls returned as text"}
    fits, load = outside_load(runs)
    return {
        "scope": SCOPE,
        "coverage": cov,
        "load_boundary": boundary,
        "workaround_tests": workaround_tests(cfgs, boundary),
        "tools_delivered": tools_delivered(cfgs, arts, spec),
        "text_call_forms": {f"{c['model']}/{c['path']}": text_call_forms(c) for c in cfgs
                            if (c["model"], c["path"]) in as_text},
        "artifacts": arts,
        "provenance": provenance(runs, cfgs, arts),
        "measurement": {
            "thermal": thermal(runs), "gates": gates(runs, cfgs), "outside_load": load,
            "prefill_counterfactual": prefill_counterfactual(runs),
            "sampler_interference": sampler_interference(runs, fits),
            "overhead": overhead(runs), "board_time": board_time(runs, cfgs), "errors": errors(runs)},
        "method": {"seed": rc.SEED, "bootstrap_resamples": rc.B,
                   "tools_delivered_min_prompt_tokens": analyze.TOOLS_DELIVERED_MIN_PROMPT_TOKENS,
                   "slow_threshold": SLOW, "excess_load_threshold": EXCESS, "trip_c": TRIP_C,
                   "gate_thresholds": [GATE_MAX_C, GATE_MAX_BUSY], "load_check_timeout_s": LOAD_CHECK_TIMEOUT_S},
    }


# ------------------------------------------------------------------- tables
def _f(x, nd=1):
    return "-" if x is None else f"{x:,.{nd}f}"


def _pc(x, nd=1):
    return "-" if x is None else f"{100 * x:.{nd}f}"


def _ci(v, ci):
    return f"{_pc(v)} ({_pc(ci[0])}-{_pc(ci[1])})"


def _row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _head(cells):
    return [_row(cells), _row(["---"] * len(cells))]


def _m(model):
    return rc.MODEL_NAMES.get(model, model)


def _cell(c):
    if c["status"] in ("readable", "not offered", "missing"):
        return c["status"]
    served = f" ({_pc(c['served_e2e'])}% as served)" if "served_e2e" in c else ""
    return f"{c['status']}: {c['reason']}{served}"


def _value(v):
    s = str(v)
    return s[:12] if re.fullmatch(r"[0-9a-f]{64}", s) else s


def tables_md(data):
    L = []
    cov = data["coverage"]
    k = cov["counts"]
    L.append("## Coverage of the planned model and path steps\n")
    L.append(f"{data['scope']}. A step is planned when the campaign ran it; the compiled path is not offered for models "
             "without a catalog bundle. Unreadable means measured, but the score describes the serving stack: the "
             "median prompt of a suite was below "
             f"{data['tools_delivered']['threshold_tokens']:,} tokens (the tools never reached the model), or no reply was "
             "a structured call while most held a text call the server did not convert.\n")
    L += _head(["Model"] + [rc.PATH_NAMES[p] for p in rc.PATHS])
    by = {(c["model"], c["path"]): c for c in cov["cells"]}
    for m in dict.fromkeys(c["model"] for c in cov["cells"]):
        L.append(_row([_m(m)] + [_cell(by[(m, p)]) for p in rc.PATHS]))
    L.append("")
    L.append(f"Of the {k['cells']} cells, {k['planned']} are planned steps and {k['not_offered']} are not offered. Of the "
             f"planned steps, {k['measured']} were measured and {k['load_failures']} failed to load; {k['unreadable']} of the "
             f"measured are unreadable and {k['readable']} readable. The serving stack decided {k['decided_by_stack']} of "
             f"the {k['planned']} planned steps.\n")

    b = data["load_boundary"]
    L.append("## NPU llama.cpp path: load outcome by GGUF size\n")
    L.append(f"{data['scope']}. One load attempt per model (the runner's load check waits up to "
             f"{data['method']['load_check_timeout_s']} s). Failed buffer from the server log; f16 KV cache at the served "
             "context from the GGUF header (not modelled for Gemma 4).\n")
    L += _head(["Model", "GGUF bytes", "GiB", "Loaded", "Failed buffer (bytes)", "f16 KV cache (bytes)",
                "Failed minus KV (bytes)", "Error class"])
    for r in b["rows"]:
        L.append(_row([_m(r["model"]), f"{r['gguf_bytes']:,}", f"{r['gguf_gib']:.3f}", "yes" if r["loaded"] else "no",
                       _f(r.get("failed_buffer_bytes"), 0), _f(r["kv_cache_bytes"], 0) if r["kv_cache_bytes"] else "not modelled",
                       _f(r.get("failed_minus_kv"), 0), r.get("class", "-")]))
    fb = b["failed_buffers"]
    L.append("")
    between = "lies" if b["one_gib_between"] else "does not lie"
    L.append(f"The largest GGUF that loaded was {b['largest_loaded']['bytes']:,} B ({_m(b['largest_loaded']['model'])}) and "
             f"the smallest that failed {b['smallest_failed']['bytes']:,} B ({_m(b['smallest_failed']['model'])}); 1 GiB "
             f"{between} between them. The failing buffers range from {fb['min']:,} to {fb['max']:,} B, and "
             f"{fb['within_8_mib_of_1_gib']} of {fb['n']} are within 8 MiB of 1 GiB. These records do not settle the "
             "mechanism.\n")
    L += _head(["Model", "Load-check reply", "Server log lines", "Log time minus record start (s)"])
    for r in b["rows"]:
        if not r["loaded"]:
            L.append(_row([_m(r["model"]), f"`{r['error']}`", "<br>".join(f"`{x}`" for x in r["log_lines"]) or "none",
                           _f(r["log_time_minus_start_s"], 0)]))
    days = "; ".join(f"{d}: {', '.join(f'{x:.0f}' for x in v)} s" for d, v in b["log_time_minus_start_s_by_day"].items())
    L.append("")
    L.append(f"The log lines are stamped by the board's clock, the record start by the client's. Every failure's lines are "
             f"stamped earlier than its own record start, by about the same amount on a given day ({days}). A board clock "
             "behind the client's would give an offset that holds within a day; lines left from an earlier attempt would "
             "be unlikely to repeat the same offset across separate models. The records do not say how long a failed "
             "load took.\n")
    w = data["workaround_tests"]
    if w:
        L.append("")
        L.append(f"## NPU llama.cpp path: workaround tests, {_m(w['model'])} ({w['date']})\n")
        L.append(f"{w['server']}. Each setting is one load attempt.\n")
        L += _head(["Setting", "Loaded", "Devices used", "Failed buffer (bytes)", "MiB"])
        for a in w["attempts"]:
            L.append(_row([a["setting"], "yes" if a["loaded"] else "no", a["devices_used"], f"{a['failed_buffer_bytes']:,}",
                           f"{a['failed_buffer_mib']:.1f}"]))
        ca = w["campaign_attempt"]
        L.append("")
        L.append(f"The campaign's own attempt with default settings failed at {ca['failed_buffer_bytes']:,} B, which is the "
                 f"f16 KV cache ({ca['kv_cache_bytes']:,} B) plus {ca['failed_minus_kv']:,} B; the test's default attempt "
                 "failed at a different buffer. They are separate attempts.\n")

    t = data["tools_delivered"]
    L.append("## Tools delivered to the model\n")
    blocks = "; ".join(f"{_m(m)} {', '.join(f'{x:,.0f}' for x in v['difference'])}" for m, v in t["tool_block_tokens"].items())
    L.append(f"Median server-reported prompt tokens per suite; the tools count as delivered when both native suites reach "
             f"{t['threshold_tokens']:,} tokens. Where one model was served both with and without its tools, the canonical "
             f"medians differ by {blocks} tokens. The diagnostic column is the run with the tools written into the system "
             "prompt. Template facts from analysis/data/artifacts.json; the finding joins them with the token counts.\n")
    L += _head(["Model", "Path", "Canonical", "Paraphrase", "Diagnostic", "Delivered", "Prompt built from",
                "Template tools reference", "Finding"])
    for r in t["rows"]:
        mp = r["median_prompt_tokens"]
        L.append(_row([_m(r["model"]), rc.PATH_NAMES[r["path"]], _f(mp.get("canonical"), 0), _f(mp.get("paraphrase"), 0),
                       _f(mp.get("diag_tools_in_prompt"), 0), "yes" if r["delivered"] else "NO", r["prompt_construction"],
                       r["template_tools_reference"], r["finding"]]))
    for key, v in data["text_call_forms"].items():
        m, p = key.split("/")
        L.append("")
        L.append(f"## Replies returned as text: {_m(m)}, {rc.PATH_NAMES[p]}\n")
        L.append("A call is `[TOOL_CALLS]name[ARGS]` followed by its arguments, the form the GGUF template writes. "
                 "Intent is the harness grader applied to the call parsed from the text (a diagnostic, not the served "
                 "score); 95% percentile bootstrap CI over items.\n")
        L += _head(["Suite", "Requests", "Structured calls", "With [TOOL_CALLS]", "Starting with it",
                    "Parsed, valid JSON arguments", "More than one call", "Intent end to end (95% CI)"])
        for s in SUITES:
            x = v[s]
            L.append(_row([f"{s} ({x['cases']} items)", x["requests"], x["structured"], x["with_marker"],
                           x["starting_with_marker"], x["parsed_with_valid_args"], x["more_than_one_call"],
                           _ci(x["intent_e2e"], x["intent_ci"])]))
        x = v["pooled"]
        L.append(_row([f"pooled ({x['items']} items)", x["requests"], x["structured"], x["with_marker"],
                       x["starting_with_marker"], x["parsed_with_valid_args"], x["more_than_one_call"],
                       _ci(x["intent_e2e"], x["intent_ci"])]))

    L.append("")
    L.append("## Artifacts\n")
    L.append("From the GGUF headers and the bundle listings (analysis/inspect_artifacts.py); no tensor was read. The "
             "sha256 is the one recorded in the server records (first 12 hex digits); records counts every server, "
             "diagnostic-server and load-failure record naming the file.\n")
    L += _head(["Model", "Artifact", "Bytes", "Format", "Tools in template", "Context / parts / largest part (bytes)",
                "Prompt construction", "Tool slot", "sha256", "Records (distinct)", "Matches download log"])
    prov_art = {(a["model"], a["kind"]): a for a in data["provenance"]["artifacts"]}
    for m, a in data["artifacts"]["models"].items():
        g = a["gguf"]
        pa = prov_art[(m, "gguf")]
        fmt = f"{g['architecture']} {g['file_type']}, {g['origin']}{', imatrix' if g['imatrix'] else ''}"
        L.append(_row([_m(m), os.path.basename(g["file"]), f"{g['bytes']:,}", fmt, g["chat_template"]["tools_reference"],
                       "-", "GGUF chat template", "yes" if g["chat_template"]["tools_reference"] == "request-level" else "no",
                       _value(g["sha256_recorded"]), f"{pa['records']} ({len(pa['distinct_sha256'])})",
                       {True: "yes", False: "NO", None: "not listed"}[g["sha256_matches_download_log"]]]))
        bd = a["bundle"]
        if bd:
            pb = prov_art[(m, "bundle")]
            L.append(_row([_m(m), os.path.basename(bd["file"]), f"{bd['zip_bytes']:,}", f"W4A16 bundle, QAIRT {bd['qairt']}",
                           bd["jinja_template"]["tools_reference"],
                           f"{bd['context_size']:,} / {bd['n_parts']} / {bd['largest_part_bytes']:,}",
                           bd["prompt_construction"], "yes" if bd["tool_slot"] else "no", _value(bd["sha256_recorded"]),
                           f"{pb['records']} ({len(pb['distinct_sha256'])})",
                           {True: "yes", False: "NO", None: "not listed"}[bd["sha256_matches_download_log"]]]))
    L.append("")
    L += _head(["Model", "Artifact", "Template excerpt (tools evidence)"])
    for m, a in data["artifacts"]["models"].items():
        for kind in ("gguf", "bundle"):
            x = a[kind]
            if x:
                tm = x["chat_template"] if kind == "gguf" else x["jinja_template"]
                ex = "<br>".join("`" + ln.replace("|", "\\|").replace("`", "'") + "`" for ln in tm["excerpt"])
                if kind == "bundle":
                    ex += f"<br>{x['tool_slot_evidence']}"
                L.append(_row([_m(m), kind, ex]))

    p = data["provenance"]
    L.append("")
    L.append("## Provenance\n")
    L.append(f"The run files are {p['files']['tool_calling_native']} native tool-calling runs, {p['files']['diagnostic']} "
             f"diagnostic runs and {p['files']['probe']} probe runs, with {p['files']['server_records']} server records. "
             "Hashes are shortened to 12 hex digits, and the last column compares them with the files in this "
             "repository.\n")
    L += _head(["Item", "Files", "Distinct values", "Value (count)", "Equals this checkout"])
    for i in p["items"]:
        vals = "; ".join(f"`{_value(v)}` ({n})" for v, n in i["values"])
        eq = {None: "-", True: "yes", False: "NO"}[i["equals_checkout"]]
        L.append(_row([f"{i['item']}, {i['group']}", i["files"], len(i["values"]), vals.replace("|", "\\|"), eq]))
    L.append("")
    L.append("Every item is single-valued and every artifact carries one sha256 across its records.\n"
             if p["all_single_valued"] else
             "NOT every item is single-valued, or an artifact carries more than one sha256; see the JSON.\n")

    ms = data["measurement"]
    th, ga, ol, ov, bt, er = (ms["thermal"]["by_path"], ms["gates"]["by_path"], ms["outside_load"], ms["overhead"]["by_path"],
                              ms["board_time"], ms["errors"])
    L.append("## Measurement quality by serving path\n")
    L.append(f"{data['scope']}. The thermal rows cover main-matrix runs (diagnostics apart) sampled every 20 s: the "
             "hottest of all zones, the hottest CPU zone and the hottest NPU zone, and their margin to the "
             f"{ms['thermal']['trip_c']:.0f} C passive trip. Readiness gates release below "
             f"{ms['gates']['thresholds']['max_c']:.0f} C and CPU busy "
             f"{_pc(ms['gates']['thresholds']['max_cpu_busy'], 0)}%; release values are main matrix. The canonical run's time "
             "beyond requests runs from its start to the paraphrase run's start, minus the summed request latency and one "
             "warm-up (the median latency); it holds four readiness gates and a few seconds of fixed overhead. Overhead "
             "is client latency minus server prefill plus decode, main-matrix requests. Board hours run from each "
             "server's start to the end of the last run it served.\n")
    L += _head(["Measure"] + [rc.PATH_NAMES[q] for q in rc.PATHS])

    def row(label, fn):
        L.append(_row([label] + [fn(q) for q in rc.PATHS]))
    row("Runs (tool-calling / probe)", lambda q: f"{th[q]['runs_bench']} / {th[q]['runs_probe']}")
    row("Thermal samples", lambda q: f"{th[q]['samples']:,}")
    row("Hottest zone, max / mean C", lambda q: f"{th[q]['hottest_max_c']:.1f} / {th[q]['hottest_mean_c']:.1f}")
    row("CPU zones, max / mean C", lambda q: f"{th[q]['cpu_max_c']:.1f} / {th[q]['cpu_mean_c']:.1f}")
    row("NPU zones, max / mean C", lambda q: f"{th[q]['npu_max_c']:.1f} / {th[q]['npu_mean_c']:.1f}")
    row("Samples at or above 70 C", lambda q: f"{_pc(th[q]['share_ge_70c'])}%")
    row("Margin to trip, CPU / NPU zones C", lambda q: f"{th[q]['margin_cpu_c']:.1f} / {th[q]['margin_npu_c']:.1f}")
    row("Readiness gates, all runs (main matrix)", lambda q: f"{ga[q]['gates']} ({ga[q]['gates_main']})")
    row("Release temperature, main matrix, median (range) C",
        lambda q: f"{ga[q]['release_c']['median']:.1f} ({ga[q]['release_c']['min']:.1f}-{ga[q]['release_c']['max']:.1f})")
    row("CPU busy at release, main matrix, median / max",
        lambda q: f"{_pc(ga[q]['busy']['median'])}% / {_pc(ga[q]['busy']['max'])}%")
    row("Every gate below both thresholds", lambda q: "yes" if ga[q]["all_within_thresholds"] else "NO")
    row("Resident container listed at gates", lambda q: f"{ga[q]['container_listed']} of {ga[q]['gates']}")
    row("Canonical run time beyond requests, median (range) s",
        lambda q: f"{ga[q]['canonical_slack_s']['median']:.1f} ({ga[q]['canonical_slack_s']['min']:.1f}-"
                  f"{ga[q]['canonical_slack_s']['max']:.1f}), n={ga[q]['canonical_slack_s']['n']}")
    row("Same, share of canonical run time", lambda q: f"{_pc(ga[q]['canonical_slack_share'])}%")
    row("Overhead, median (p5-p95) ms, n",
        lambda q: f"{ov[q]['ms']['median']:.1f} ({ov[q]['ms']['p5']:.1f}-{ov[q]['ms']['p95']:.1f}), n={ov[q]['ms']['n']:,}")
    row("Overhead, median share of latency", lambda q: f"{_pc(ov[q]['share_median'])}%")
    row("Board hours, main / diagnostic",
        lambda q: f"{bt['hours'].get(q, 0):.2f} / {bt['hours'].get(q + '_diag', 0):.2f}")
    row("Share of board hours (all)", lambda q: f"{_pc(bt['share_all'][q])}%")
    L.append("")
    oc, op, on, bl = ol["cpu"], ol["compiled"], ol["npu_llama_not_used"], ms["gates"]["busy_by_leg"]
    co, cp = ol["coincidence"]["canonical"], ol["coincidence"]["paraphrase"]
    pc = ms["prefill_counterfactual"]
    fg = ol["fitted_gates_before_reps_2_3_s"]
    hot = ms["thermal"]["hottest"]
    L.append(f"The hottest reading was {hot['c']:.1f} C ({_m(hot['model'])}, {hot['path']}, {hot['task']}), and the "
             f"diagnostic runs reached {ms['thermal']['diagnostic_max_c']:.1f} C. The stored thermal summary agrees with "
             f"its trace in {ms['thermal']['summary_blocks_agree']} of {ms['thermal']['runs']} runs. On the fitted CPU "
             f"timelines the gates before canonical repetitions 2 and 3 lasted {fg['min']:.1f} to {fg['max']:.1f} s "
             f"(n={fg['n']}).\n")
    L.append(f"The CPU legs ran with the resident container up. Inside their repetitions the residual over the fitted "
             f"own load ({oc['own_load_tasks']:.2f} active tasks) exceeded +0.5 in {_pc(oc['share_gt_0_5'])}% and +1.0 in "
             f"{oc['n_gt_1_0']} ({_pc(oc['share_gt_1_0'])}%) of {oc['intervals']:,} intervals (largest {oc['max']:+.2f}, "
             f"{oc['runs']} runs). On the compiled runs, with the container stopped and an own load of "
             f"{op['own_load_tasks']:.2f}, the deviation from the run median exceeded +0.5 in {_pc(op['share_gt_0_5'])}% and "
             f"+1.0 in {op['n_gt_1_0']} ({_pc(op['share_gt_1_0'])}%) of {op['intervals']:,} intervals (largest "
             f"{op['max']:+.2f}, {op['runs']} runs). The llama.cpp NPU runs, also with the container stopped, are left out "
             f"of this contrast: their run medians range from {on['run_median_range'][0]:.1f} to "
             f"{on['run_median_range'][1]:.1f} active tasks and the deviation exceeded +1.0 in {_pc(on['share_gt_1_0'])}% "
             f"of {on['intervals']:,} intervals, so their own load varies too much within a run for the median to stand "
             f"for it. CPU busy at gate release averaged {_pc(bl['cpu_legs']['mean'])}% (nearest-rank p90 "
             f"{_pc(bl['cpu_legs']['p90'])}%, n={bl['cpu_legs']['n']}) on CPU legs and {_pc(bl['npu_legs']['mean'])}% (p90 "
             f"{_pc(bl['npu_legs']['p90'])}%, n={bl['npu_legs']['n']}) on both NPU legs, over all runs. Crossing canonical CPU requests whose prefill was more than {100 * SLOW:.0f}% slower than the "
             f"fastest repetition with residual load above +{EXCESS} gives {co['table']} (rows slowed and not slowed), "
             f"one-sided Fisher p = {co['fisher_p_greater']:.3f}, Spearman {co['spearman']:.3f}, n={co['requests']:,}; the "
             f"paraphrase suite, against each run's median prefill rate, gives p = {cp['fisher_p_greater']:.3f}, Spearman "
             f"{cp['spearman']:.3f}, n={cp['requests']:,}. Setting every prefill to the fastest repetition of its case moves "
             f"the CPU median turn time by a median {_pc(pc['median_change'])}% (max {_pc(pc['max_change'])}%) over "
             f"{len(pc['rows'])} runs.\n")
    eb, ep, tk = er["bench"], er["probe"], ms["overhead"]["token_counts"]
    timed_out = ", ".join(_m(r["model"]) for r in b["rows"] if r.get("class") == "timeout, nothing logged") or "none"
    L.append(f"Of {eb['requests']:,} tool-calling requests ({eb['diagnostic_requests']:,} of them diagnostic), "
             f"{eb['errors']} returned an error, {eb['empty_reply_retries']} were retried after an empty reply, "
             f"{eb['empty_replies_graded']} empty replies were graded, {eb['missing_server_timings']} lacked server timings "
             f"and {eb['reached_max_tokens']} reached max_tokens. Server and usage token counts agree in "
             f"{tk['prompt_counts_agree']:,} (prompt) and {tk['completion_counts_agree']:,} (completion) of "
             f"{tk['requests']:,} requests, and {tk['cached_tokens']} reported cached tokens. Of {ep['reps']} probe "
             f"repetitions over {ep['lengths']} lengths, {ep['rep_errors']} failed, {ep['lengths_failed']} lengths failed "
             f"calibration, {ep['max_tokens_not_honoured']} overran max_tokens and {ep['chunks_differ_from_tokens']} had a "
             f"streamed chunk count different from the token count. Board hours total {bt['total_h']:.2f}; artifact "
             f"staging, the {k['load_failures']} failed loads ({timed_out} waited out the "
             f"{data['method']['load_check_timeout_s']} s load check) and the runner's own steps are not in the results "
             "and not included.\n")

    si = ms["sampler_interference"]
    L.append("## Thermal sampler and CPU-path decode\n")
    L.append("Canonical CPU requests whose three repetitions produced the same number of tokens "
             f"(n={si['requests_same_length']:,}), by the distance between the decode phase and the nearest sampler ssh "
             "session on the fitted timeline. Slowdown is time per token over the fastest repetition of the case. The "
             "placebo shifts the sessions by half a sampling period, so its far bins lie near the real sessions and carry "
             "their higher rates.\n")
    L += _head(["Distance to a sampler session", "Requests", "Slowed > 3%", "Slowed > 10%", "Median slowdown",
                "Placebo: slowed > 3%"])
    for x in si["bins"]:
        L.append(_row([x["distance"], x["n"], f"{x['gt_3pct']} ({_pc(x['gt_3pct'] / x['n'])}%)", x["gt_10pct"],
                       f"{_pc(x['median_slowdown'])}%",
                       f"{x['placebo_gt_3pct']} of {x['placebo_n']} ({_pc(x['placebo_gt_3pct'] / x['placebo_n'])}%)"]))
    pcs = si["prefill_cost_per_session_ms"]
    L.append("")
    L.append(f"Overlapping decodes against those 2 s or more away give {si['fisher']['table']}, one-sided Fisher p = "
             f"{si['fisher']['p_greater']:.2g}. An overlapping decode takes a median {si['extra_decode_ms_overlapping']:.0f} ms "
             f"longer than the fastest repetition ({si['extra_decode_ms_far']:.0f} ms when 2 s or more away). Each "
             f"session inside a prefill adds a median {pcs['median_of_median_diffs']:.0f} ms ({pcs['models']} models, range "
             f"{pcs['range'][0]:.0f} to {pcs['range'][1]:.0f} ms; {pcs['median_of_mean_diffs']:.0f} ms from the means, "
             f"which exceed the medians in {pcs['means_above_medians']} of {pcs['models']} models), about "
             f"{_pc(si['share_of_compute_time'])}% of CPU compute time at one session per {si['sampler_period_s']:.1f} s.\n")
    a, z = si["timeline_shifted"]
    L.append(f"These figures rest on the fitted timeline. With every fitted request moved {abs(a['shift_s']):.1f} s earlier "
             f"or later, the overlapping decodes number {a['overlapping']} and {z['overlapping']} "
             f"({a['overlapping_gt_3pct']} and {z['overlapping_gt_3pct']} slowed by more than 3%; Fisher p = "
             f"{a['fisher_p_greater']:.2g} and {z['fisher_p_greater']:.2g}), and a session inside a prefill adds a median "
             f"{a['prefill_cost_median_ms']:.0f} and {z['prefill_cost_median_ms']:.0f} ms (ranges "
             f"{a['prefill_cost_range_ms'][0]:.0f} to {a['prefill_cost_range_ms'][1]:.0f} and "
             f"{z['prefill_cost_range_ms'][0]:.0f} to {z['prefill_cost_range_ms'][1]:.0f} ms). The counts by distance and "
             "the low end of the per-model range move most with the fit.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    print(tables_md(compute(sys.argv[1] if len(sys.argv) > 1 else os.path.join(rc.REPO, "results", "v2"))), end="")
