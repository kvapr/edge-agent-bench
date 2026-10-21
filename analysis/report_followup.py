#!/usr/bin/env python3
"""GenieX v0.9.3 follow-up, for the addendum to the version 2 report. Standard
library, plus PyYAML through report_common. Reads a results directory laid out
as results/v2-followup (<phase>/<model>/<path>/<file>.json), the version 2
results in results/v2 for the comparison, campaign/v2_followup.json for the
step lists and analysis/data/artifacts.json for GGUF sizes.

    python3 analysis/report_followup.py results/v2-followup \\
        --json analysis/out/followup.json --md analysis/out/followup_tables.md

- Triage. Each step of the triage phase sent the first six canonical cases
  once (all level 1 queries). Its outcome under v0.9.3 uses the report's
  readable rule (report_common.unreadable_reason, with the six replies standing
  in for both suites) and the load-failure parse of report_stack; under v0.8.0
  it is the version 2 outcome of the same step. Passes use analyze.e2e_ok.
  For steps that ran under both releases, the first repetition of the same six
  cases in the version 2 canonical run gives the comparison.
- Runtime. The llama.cpp micro-batch (n_ubatch) each server logged.
- Length sweep. harness/sweep_prompt_length.py results: per model, series and
  case, the server-reported prompt length of each request and whether it
  returned a structured call and passed; summarised by length band.
"""
import argparse
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze  # noqa: E402
import report_common as rc  # noqa: E402
import report_stack  # noqa: E402

V2 = os.path.join(rc.REPO, "results", "v2")
PATH_NAMES = {**rc.PATH_NAMES, "gpu": "GPU (llama.cpp)"}
_OPENCL = re.compile(r"failed to allocate OpenCL buffer of size (\d+)")
_UBATCH = re.compile(r"n_ubatch=(\d+)")
BANDS = ((0, 999, "below 1,000"), (1000, 1499, "1,000 to 1,499"), (1500, 1999, "1,500 to 1,999"),
         (2000, 10 ** 9, "2,000 and above"))


def _phase_steps(name):
    spec = rc.load_json(os.path.join(rc.REPO, "campaign", "v2_followup.json"))
    return next(p["steps"] for p in spec["phases"] if p["name"] == name)


def _failure(lf):
    tail = report_stack._ANSI.sub("", lf.get("server_log_tail") or "")
    f = report_stack.failure(lf)
    m = _OPENCL.search(tail)
    if m:
        return {"class": "OpenCL buffer allocation failed", "failed_buffer_bytes": int(m.group(1))}
    return {"class": f["class"], "failed_buffer_bytes": f["failed_buffer_bytes"]}


def _ubatch(server):
    m = _UBATCH.search(report_stack._ANSI.sub("", (server or {}).get("server_log_runtime_lines") or ""))
    return int(m.group(1)) if m else None


def _status(cfg):
    """(status, reason) of one step from its parsed files."""
    if cfg.get("load_failure") and not cfg.get("canonical"):
        return "load failure", _failure(cfg["load_failure"])["class"]
    if not cfg.get("canonical"):
        return "not measured", None
    why = rc.unreadable_reason(cfg)
    return ("unreadable", why) if why else ("readable", None)


def _first_reps(run, n=6):
    return [c["reps"][0] for c in run["results"][:n] if c.get("reps")]


def triage(root):
    arts = rc.load_json(os.path.join(HERE, "data", "artifacts.json"))["models"]
    rows = []
    for step in _phase_steps("triage"):
        model, path = step.split("/")
        d = os.path.join(root, "triage", model, path)
        new = {"load_failure": rc.load_json(os.path.join(d, "load_failure.json")),
               "server": rc.load_json(os.path.join(d, "server.json"))}
        t = rc.load_json(os.path.join(d, "triage.json"))
        if t:
            new["canonical"] = new["paraphrase"] = t
        v2d = os.path.join(V2, model, path)
        old = {k: rc.load_json(os.path.join(v2d, k + ".json")) for k in ("canonical", "paraphrase", "load_failure", "server")}
        row = {"model": model, "path": path, "v093": _status(new), "v080": _status(old) if os.path.isdir(v2d) else ("not measured", None),
               "gguf_gib": (arts[model]["gguf"]["bytes"] / 2 ** 30) if path != "compiled" else None,
               "n_ubatch_v093": _ubatch(new["server"]), "n_ubatch_v080": _ubatch(old["server"]),
               "server_version": (t or new["load_failure"] or {}).get("provenance", {}).get("server_version")
               or (new["server"] or {}).get("server_version")}
        if new["load_failure"]:
            row["failure"] = _failure(new["load_failure"])
        if t:
            reps = _first_reps(t)
            toks = [r["prompt_tokens"] for r in reps if r.get("prompt_tokens")]
            row.update(n=len(reps), prompt_median=statistics.median(toks) if toks else None,
                       structured=sum(bool(r.get("got_tool")) for r in reps),
                       text_calls=sum(bool(r.get("text_tool_call") or (not r.get("got_tool") and
                                                                       analyze.parse_text_tool_call(r.get("content"))[0]))
                                      for r in reps),
                       passed=sum(analyze.e2e_ok(r) for r in reps),
                       latency_median=statistics.median(r["latency_s"] for r in reps))
            if old["canonical"]:
                oreps = _first_reps(old["canonical"])
                row.update(v080_passed=sum(analyze.e2e_ok(r) for r in oreps),
                           v080_latency_median=statistics.median(r["latency_s"] for r in oreps))
        rows.append(row)
    decided = [r for r in rows if r["path"] != "gpu" and r["v080"][0] in ("load failure", "unreadable")]
    outcome = {}
    for r in decided:
        key = r["v093"][0] if r["v093"][0] == "readable" else f"{r['v093'][0]}: {r['v093'][1]}"
        outcome.setdefault(key, []).append(f"{r['model']}/{r['path']}")
    stamps = []
    for r in rows:
        t = rc.load_json(os.path.join(root, "triage", r["model"], r["path"], "triage.json"))
        if t:
            stamps.append(t["provenance"]["timestamp_utc"])
    return {"rows": rows, "decided_v080": len(decided), "outcome_of_decided": outcome,
            "still_decided": sum(len(v) for k, v in outcome.items() if k != "readable"),
            "first_request_utc": min(stamps), "last_run_utc": max(stamps)}


def _band(tokens):
    return next(name for lo, hi, name in BANDS if lo <= tokens <= hi)


def sweep(root):
    out = {}
    for step in _phase_steps("lengthsweep"):
        model, path = step.split("/")
        d = rc.load_json(os.path.join(root, "lengthsweep", model, path, "sweep_length.json"))
        cases, bands = [], {}
        for r in d["results"]:
            pts = [{"prompt_tokens": x["prompt_tokens"], "structured": x["structured_call"], "passed": x["passed"]}
                   for x in r["sweep"]]
            with_call = [p["prompt_tokens"] for p in pts if p["structured"]]
            without = [p["prompt_tokens"] for p in pts if not p["structured"]]
            cases.append({"series": r["series"], "level": r["level"], "points": pts,
                          "longest_with_call": max(with_call) if with_call else None,
                          "shortest_without_call": min(without) if without else None,
                          "reference": ({"prompt_tokens": r["reference"]["prompt_tokens"],
                                         "structured": r["reference"]["structured_call"],
                                         "passed": r["reference"]["passed"]} if "reference" in r else None)})
            for p in pts:
                b = bands.setdefault(r["series"], {}).setdefault(_band(p["prompt_tokens"]), {"n": 0, "structured": 0, "passed": 0})
                b["n"] += 1
                b["structured"] += p["structured"]
                b["passed"] += p["passed"]
        refs = [c["reference"] for c in cases if c["reference"]]
        out[model] = {"path": path, "cases": cases, "bands": bands,
                      "reference": {"n": len(refs), "structured": sum(r["structured"] for r in refs),
                                    "passed": sum(r["passed"] for r in refs),
                                    "prompt_tokens": [r["prompt_tokens"] for r in refs]},
                      "range": [min(p["prompt_tokens"] for c in cases for p in c["points"]),
                                max(p["prompt_tokens"] for c in cases for p in c["points"])],
                      "timestamp_utc": d["provenance"]["timestamp_utc"],
                      "server_version": d["provenance"].get("server_version")}
    return out


def compute(results_dir):
    tri = triage(results_dir)
    return {"scope": "QCS9075, GenieX v0.9.3 (QAIRT 2.50, llama.cpp e467e5b), 2026-10-21", "triage": tri,
            "sweep": sweep(results_dir)}


# ------------------------------------------------------------------- tables
def _row(cells):
    return "| " + " | ".join("-" if c is None else str(c) for c in cells) + " |"


def _head(cells):
    return [_row(cells), _row(["---"] * len(cells))]


def _st(s):
    return s[0] if not s[1] else f"{s[0]}: {s[1]}"


def tables_md(data):
    L = ["# Follow-up tables (generated by analysis/report_followup.py; do not edit)\n",
         f"Scope: {data['scope']}.\n", "## Triage of the stack-decided steps, anchors and the GPU path\n",
         "Each step sent the first six canonical cases once (all level 1 queries). Outcomes use the report's readable "
         "rule; v0.8.0 is the version 2 outcome of the same step. Passes are end to end. Latency is the median of the "
         "six requests; the v0.8.0 latency is the first repetition of the same six cases in the version 2 run.\n"]
    L += _head(["Model", "Path", "GGUF, GiB", "v0.8.0", "v0.9.3", "Failing buffer, bytes", "Prompt tokens (median)",
                "Structured calls", "Text calls", "Passed", "Latency v0.9.3, s", "Passed v0.8.0", "Latency v0.8.0, s",
                "n_ubatch v0.8.0 / v0.9.3"])
    for r in data["triage"]["rows"]:
        L.append(_row([rc.MODEL_NAMES.get(r["model"], r["model"]), PATH_NAMES[r["path"]],
                       f"{r['gguf_gib']:.3f}" if r["gguf_gib"] else None, _st(r["v080"]), _st(r["v093"]),
                       f"{r['failure']['failed_buffer_bytes']:,}" if r.get("failure", {}).get("failed_buffer_bytes") else None,
                       f"{r['prompt_median']:,.0f}" if r.get("prompt_median") else None,
                       f"{r['structured']} of {r['n']}" if "n" in r else None,
                       f"{r['text_calls']} of {r['n']}" if "n" in r else None,
                       f"{r['passed']} of {r['n']}" if "n" in r else None,
                       f"{r['latency_median']:.2f}" if "n" in r else None,
                       f"{r['v080_passed']} of 6" if "v080_passed" in r else None,
                       f"{r['v080_latency_median']:.2f}" if "v080_latency_median" in r else None,
                       f"{r['n_ubatch_v080'] or '-'} / {r['n_ubatch_v093'] or '-'}"]))
    t = data["triage"]
    L.append("")
    L.append(f"Of the {t['decided_v080']} steps the stack decided under v0.8.0, {t['still_decided']} are still decided by "
             f"it under v0.9.3. Outcomes under v0.9.3:\n")
    for k, v in sorted(t["outcome_of_decided"].items()):
        L.append(f"- {k}: {len(v)} ({', '.join(v)})")
    L.append(f"\nTriage requests ran from {t['first_request_utc'][:16]} to {t['last_run_utc'][:16]} UTC (file timestamps).\n")
    L.append("## Tool calling against prompt length (length sweep)\n")
    L.append("One tool per request (the one the case expects); the compact series uses a one-sentence system prompt and "
             "the tool without descriptions, the benchmark series the benchmark prompt and the full tool. Padding is "
             "neutral numbered notes. Prompt lengths are server-reported. In the compact series the tool descriptions "
             "that name the commands are removed, so a structured call, not a pass, is the measure there.\n")
    L += _head(["Model", "Series", "Prompt length, tokens", "Requests", "Structured calls", "Passed"])
    for model, s in data["sweep"].items():
        for series, bands in s["bands"].items():
            for _, _, name in BANDS:
                if name in bands:
                    b = bands[name]
                    L.append(_row([rc.MODEL_NAMES[model] + " (compiled)", series, name, b["n"], b["structured"], b["passed"]]))
        ref = s["reference"]
        L.append(_row([rc.MODEL_NAMES[model] + " (compiled)", "full four-tool benchmark request",
                       f"{min(ref['prompt_tokens']):,} to {max(ref['prompt_tokens']):,}", ref["n"], ref["structured"], ref["passed"]]))
    L.append("")
    L += _head(["Model", "Series", "Case level", "Longest prompt with a structured call", "Shortest prompt without one",
                "Prompt lengths (outcome: P passed, c call, - no call)"])
    for model, s in data["sweep"].items():
        for c in s["cases"]:
            pts = " ".join(f"{p['prompt_tokens']}{'P' if p['passed'] else ('c' if p['structured'] else '-')}" for p in c["points"])
            L.append(_row([rc.MODEL_NAMES[model], c["series"], c["level"].split("_")[0],
                           f"{c['longest_with_call']:,}" if c["longest_with_call"] else None,
                           f"{c['shortest_without_call']:,}" if c["shortest_without_call"] else None, pts]))
    L.append("")
    for model, s in data["sweep"].items():
        L.append(f"- {rc.MODEL_NAMES[model]} (compiled): prompts from {s['range'][0]:,} to {s['range'][1]:,} tokens, "
                 f"run {s['timestamp_utc'][:16]} UTC.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results_dir")
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    data = compute(a.results_dir)
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
        f.write("\n")
    with open(a.md, "w") as f:
        f.write(tables_md(data))
    print(f"wrote {a.json} and {a.md}")


if __name__ == "__main__":
    main()
