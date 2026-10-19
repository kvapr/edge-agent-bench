#!/usr/bin/env python3
"""Statistical analysis of edge-agent-bench result files. Standard library plus
PyYAML, which the harness grader it imports also needs.

Reads bench_toolcall.py results (files with "summary" and "results") and
phase_probe.py results (files with "lengths") under one or more directories,
and writes markdown tables plus a machine-readable JSON summary.

Statistics:
- Accuracy: each case's pass fraction across repetitions, averaged over cases.
  95% confidence intervals come from a percentile bootstrap over cases
  (10,000 resamples, fixed seed), so they reflect how many cases the suite
  has, which dominates the uncertainty at temperature 0.
- Paired comparisons (same suite, two configurations): exact two-sided
  McNemar test on the per-case majority outcome, with Holm correction within
  each family of comparisons.
- Latency: per-case mean latency across repetitions; median with a bootstrap
  CI, plus 5th and 95th percentiles. Server-side prefill and decode rates come
  from the timings the server reports with each response, where available.
- "Intent" accuracy is a diagnostic only: when a reply carries its tool call
  as text instead of a structured tool_calls entry, the text is parsed and
  graded with the same grader. The headline metric is always the structured,
  deployed-path result.

Usage:
  python3 analysis/analyze.py results/v2 --md docs/results_v2_tables.md --json analysis/summary_v2.json
"""
import argparse
import glob
import json
import math
import os
import random
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "harness"))
from bench_toolcall import grade  # noqa: E402  (the harness's own grader)

B_RESAMPLES = 10000
# The agent request carries about 2,230 tokens of tool definitions on top of a
# roughly 470-token system prompt. A median server-reported prompt below this
# threshold means the serving stack did not render the tools into the prompt,
# so the model never saw them; such scores describe the stack, not the model.
TOOLS_DELIVERED_MIN_PROMPT_TOKENS = 1500
SEED = 20261014


# ----------------------------------------------------------------- statistics
def bootstrap_ci(values, stat=statistics.mean, b=B_RESAMPLES, seed=SEED, alpha=0.05):
    """Percentile bootstrap CI of stat(values), resampling the values."""
    if not values:
        return None, None
    if len(set(values)) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(b))
    lo = stats[int(math.floor(alpha / 2 * b))]
    hi = stats[min(b - 1, int(math.ceil((1 - alpha / 2) * b)) - 1)]
    return lo, hi


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def holm(pvals):
    """Holm step-down adjusted p-values, returned in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [None] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    k = max(1, min(len(s), math.ceil(p / 100.0 * len(s))))
    return s[k - 1]


# ------------------------------------------------------------ intent parsing
_HERMES_JSON = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_XML_FUNC = re.compile(r"<function=([A-Za-z0-9_]+)>(.*?)</function>", re.S)
_XML_PARAM = re.compile(r"<parameter=([A-Za-z0-9_]+)>\s*(.*?)\s*</parameter>", re.S)
# Mistral-family native forms: "[TOOL_CALLS]name[ARGS]{...}" (newer tokenizers,
# e.g. Ministral 3) and "[TOOL_CALLS][{"name": ..., "arguments": {...}}]".
_MISTRAL_ARGS = re.compile(r"\[TOOL_CALLS\]\s*([A-Za-z0-9_\-]+)\s*\[ARGS\]\s*")
_MISTRAL_LIST = re.compile(r"\[TOOL_CALLS\]\s*(?=\[)")
_DECODER = json.JSONDecoder()


def parse_text_tool_call(content):
    """Best-effort parse of a tool call written as text. Returns (name, args)
    or (None, None). Handles the Hermes JSON form, the XML-parameter form and
    the two Mistral-family forms."""
    if not content:
        return None, None
    m = _MISTRAL_ARGS.search(content)
    if m:
        try:
            args = _DECODER.raw_decode(content, m.end())[0]
            return m.group(1), args if isinstance(args, dict) else None
        except json.JSONDecodeError:
            return m.group(1), None
    m = _MISTRAL_LIST.search(content)
    if m:
        try:
            calls = _DECODER.raw_decode(content, m.end())[0]
            if isinstance(calls, list) and calls and isinstance(calls[0], dict):
                args = calls[0].get("arguments", calls[0].get("parameters"))
                if isinstance(args, str):
                    args = json.loads(args)
                return calls[0].get("name"), args if isinstance(args, dict) else None
        except (json.JSONDecodeError, AttributeError):
            pass
    m = _HERMES_JSON.search(content)
    if m:
        try:
            obj = json.loads(m.group(1))
            args = obj.get("arguments", obj.get("parameters"))
            if isinstance(args, str):
                args = json.loads(args)
            return obj.get("name"), args if isinstance(args, dict) else None
        except (json.JSONDecodeError, AttributeError):
            pass
    m = _XML_FUNC.search(content)
    if m:
        args = {}
        for pm in _XML_PARAM.finditer(m.group(2)):
            v = pm.group(2)
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
            args[pm.group(1)] = v
        return m.group(1), args
    return None, None


_STYLES = None


def _paraphrase_styles():
    """input text -> paraphrase style (A, B or C), from the paraphrase suite."""
    global _STYLES
    if _STYLES is None:
        _STYLES = {}
        p = os.path.join(HERE, "..", "suites", "test_cases_paraphrase.yaml")
        try:
            import yaml
            for items in (yaml.safe_load(open(p)) or {}).values():
                for it in items or []:
                    _STYLES[it["input"]] = it.get("style")
        except Exception:  # noqa: BLE001
            pass
    return _STYLES


# --------------------------------------------------------------- per-file
def parse_label(label):
    """'model=X;path=Y;suite=Z' -> dict; anything else -> {'label': label}."""
    out = {}
    for part in (label or "").split(";"):
        if "=" in part:
            k, _, v = part.partition("=")
            out[k.strip()] = v.strip()
    return out or {"label": label or ""}


def e2e_ok(r):
    return bool(r.get("ok_tool") and r.get("ok_command") and r.get("ok_params_lenient") and r.get("ok_device"))


def bench_stats(path):
    d = json.load(open(path))
    prov = d.get("provenance", {})
    label = prov.get("label") or d.get("summary", {}).get("label") or os.path.basename(path)
    cfg = parse_label(label)
    cases = d["results"]
    p_e2e, p_tool, p_intent, majority, lat_case = [], [], [], {}, []
    text_calls = retried = errors = n_req = 0
    srv_prefill, srv_decode, prompt_toks = [], [], []
    for c in cases:
        reps = c.get("reps") or []
        if not reps:
            continue
        oks = [e2e_ok(r) for r in reps]
        p_e2e.append(sum(oks) / len(oks))
        p_tool.append(sum(bool(r.get("ok_tool")) for r in reps) / len(reps))
        majority[c["idx"]] = sum(oks) * 2 > len(oks)
        intent = []
        for r in reps:
            n_req += 1
            # The harness flags text calls by its own pattern at run time; the
            # analysis also recognises forms that pattern does not know.
            name, args = (None, None) if r.get("got_tool") else parse_text_tool_call(r.get("content"))
            is_text_call = bool(r.get("text_tool_call") or name)
            text_calls += is_text_call
            retried += bool(r.get("retried_empty_response"))
            errors += bool(r.get("error"))
            if is_text_call:
                g = grade(c, name, args) if name else {}
                intent.append(e2e_ok(g))
            else:
                intent.append(e2e_ok(r))
            t = (r.get("server_extra") or {}).get("timings") or {}
            if isinstance(t.get("prompt_per_second"), (int, float)):
                srv_prefill.append(t["prompt_per_second"])
            if isinstance(t.get("predicted_per_second"), (int, float)):
                srv_decode.append(t["predicted_per_second"])
            if isinstance(r.get("prompt_tokens"), (int, float)) and r["prompt_tokens"] > 0:
                prompt_toks.append(r["prompt_tokens"])
        p_intent.append(sum(intent) / len(intent))
        lats = [r["latency_s"] for r in reps if isinstance(r.get("latency_s"), (int, float))]
        if lats:
            lat_case.append(statistics.mean(lats))
    acc = statistics.mean(p_e2e) if p_e2e else None
    acc_lo, acc_hi = bootstrap_ci(p_e2e)
    tool = statistics.mean(p_tool) if p_tool else None
    tool_lo, tool_hi = bootstrap_ci(p_tool)
    lat_med = statistics.median(lat_case) if lat_case else None
    lat_lo, lat_hi = bootstrap_ci(lat_case, stat=statistics.median) if lat_case else (None, None)
    s = d.get("summary", {})
    suite_tag = cfg.get("suite") or os.path.basename(path)
    style_of = _paraphrase_styles()
    by_style = {}
    for c in cases:
        st = style_of.get(c.get("input"))
        if st and c.get("reps"):
            by_style.setdefault(st, []).append(sum(e2e_ok(r) for r in c["reps"]) / len(c["reps"]))
    return {
        "file": path, "label": label, **cfg,
        "by_style": {k: statistics.mean(v) for k, v in sorted(by_style.items())},
        "_items": {f"{suite_tag}:{c['idx']}": sum(e2e_ok(r) for r in c["reps"]) / len(c["reps"])
                   for c in cases if c.get("reps")},
        "n_cases": len(p_e2e), "repeats": s.get("repeats"),
        "e2e": acc, "e2e_ci": [acc_lo, acc_hi],
        "tool_sel": tool, "tool_sel_ci": [tool_lo, tool_hi],
        "intent_e2e": statistics.mean(p_intent) if p_intent else None,
        "unstable_cases": [c["idx"] for c in cases if 0 < sum(e2e_ok(r) for r in c.get("reps", [])) < len(c.get("reps", []))],
        "text_tool_call_rate": text_calls / n_req if n_req else None,
        "retried_empty": retried, "errors": errors, "n_requests": n_req,
        "latency_p50_s": lat_med, "latency_p50_ci": [lat_lo, lat_hi],
        "latency_p5_s": pct(lat_case, 5), "latency_p95_s": pct(lat_case, 95),
        "prompt_tokens_median": statistics.median(prompt_toks) if prompt_toks else None,
        "tools_delivered": (statistics.median(prompt_toks) >= TOOLS_DELIVERED_MIN_PROMPT_TOKENS) if prompt_toks else None,
        "server_prefill_tok_s_median": statistics.median(srv_prefill) if srv_prefill else None,
        "server_decode_tok_s_median": statistics.median(srv_decode) if srv_decode else None,
        "thermal": s.get("thermal"),
        "server_version": (prov.get("server_version") or "").replace("\n", " | ")[:160],
        "_majority": majority,
    }


def probe_stats(path):
    d = json.load(open(path))
    prov = d.get("provenance", {})
    label = prov.get("label") or d.get("label") or os.path.basename(path)
    rows = []
    for L in d.get("lengths", []):
        reps = [r for r in L.get("reps") or [] if "error" not in r]
        if not reps:
            rows.append({"target": L.get("target"), "error": L.get("error", "no valid reps")})
            continue

        def ms(key, src=None):
            v = [((r.get("server_extra") or {}).get("timings") or {}).get(src) if src else r.get(key)
                 for r in reps]
            v = [x for x in v if isinstance(x, (int, float))]
            return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0) if v else (None, None)
        rows.append({
            "target": L.get("target"),
            "prompt_tokens": statistics.median([r["prompt_tokens"] for r in reps if r.get("prompt_tokens")] or [0]) or L.get("prompt_tokens"),
            "n": len(reps),
            "ttft_s": ms("ttft_s"), "prefill_client": ms("prefill_tok_s"),
            "prefill_server": ms(None, "prompt_per_second"),
            "decode_client": ms("decode_tok_s"), "decode_server": ms(None, "predicted_per_second"),
            "e2e_s": ms("e2e_s"),
            "max_tokens_honored": all(r.get("max_tokens_honored", True) for r in reps),
            "n_out": statistics.median([r.get("n_out") or 0 for r in reps]),
        })
    return {"file": path, "label": label, **parse_label(label), "gen_tokens": d.get("gen_tokens"),
            "rows": rows, "thermal": d.get("thermal"),
            "server_version": (prov.get("server_version") or "").replace("\n", " | ")[:160]}


# ------------------------------------------------------------ pooling
def pooled(bench):
    """One entry per (model, path) pooling every item from all its suites:
    each canonical case's pass fraction across repetitions and each
    paraphrase's single outcome, weighted equally."""
    groups = {}
    for s in bench:
        if s.get("model") and s.get("path"):
            groups.setdefault((s["model"], s["path"]), []).append(s)
    out = []
    for (m, p), ss in sorted(groups.items()):
        items = {}
        for s in ss:
            items.update(s["_items"])
        vals = list(items.values())
        lo, hi = bootstrap_ci(vals)
        lat = [s["latency_p50_s"] for s in ss if s.get("suite") == "canonical" and s.get("latency_p50_s")]
        out.append({"model": m, "path": p, "label": f"model={m};path={p};suite=all",
                    "n_items": len(vals), "e2e": statistics.mean(vals) if vals else None, "e2e_ci": [lo, hi],
                    "suites": sorted(s.get("suite") for s in ss),
                    "latency_p50_s": lat[0] if lat else None,
                    "tools_delivered": None if any(s.get("tools_delivered") is None for s in ss)
                    else all(s["tools_delivered"] for s in ss),
                    "_majority": {k: v > 0.5 for k, v in items.items()}})
    return out


# ------------------------------------------------------------ comparisons
def paired(a, b):
    """McNemar on the cases both configurations share."""
    common = sorted(set(a["_majority"]) & set(b["_majority"]))
    b_cnt = sum(1 for i in common if a["_majority"][i] and not b["_majority"][i])
    c_cnt = sum(1 for i in common if b["_majority"][i] and not a["_majority"][i])
    return {"a": a["label"], "b": b["label"], "n_common": len(common),
            "a_only_pass": b_cnt, "b_only_pass": c_cnt, "p_exact": mcnemar_exact(b_cnt, c_cnt)}


def comparisons(bench):
    """Family 1: serving paths within a model and suite. Family 2: models
    within a path and suite. Holm correction within each family."""
    fams = {"paths_within_model": [], "models_within_path": []}
    by = {}
    for s in bench:
        by.setdefault((s.get("model"), s.get("suite")), []).append(s)
    for (_m, _su), group in by.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                fams["paths_within_model"].append(paired(group[i], group[j]))
    by = {}
    for s in bench:
        by.setdefault((s.get("path"), s.get("suite")), []).append(s)
    for (_p, _su), group in by.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                fams["models_within_path"].append(paired(group[i], group[j]))
    for fam in fams.values():
        for row, adj in zip(fam, holm([r["p_exact"] for r in fam])):
            row["p_holm"] = adj
    return fams


def diagnostics(diag, bench):
    """Each diagnostic run (label carries mode=...) beside the native run of the
    same model, path and suite, with an exact McNemar test on shared cases."""
    native = {(s.get("model"), s.get("path"), s.get("suite")): s for s in bench}
    out = []
    for d in sorted(diag, key=lambda x: x["label"]):
        n = native.get((d.get("model"), d.get("path"), d.get("suite")))
        row = {"label": d["label"], "mode": d.get("mode"), "model": d.get("model"), "path": d.get("path"),
               "suite": d.get("suite"), "e2e": d["e2e"], "e2e_ci": d["e2e_ci"], "tool_sel": d["tool_sel"],
               "prompt_tokens_median": d["prompt_tokens_median"], "latency_p50_s": d["latency_p50_s"],
               "native_label": n["label"] if n else None, "native_e2e": n["e2e"] if n else None,
               "native_tools_delivered": n["tools_delivered"] if n else None}
        if n:
            row["paired_vs_native"] = paired(n, d)
        out.append(row)
    return out


# ------------------------------------------------------------------ output
def f(x, nd=1, pctg=False):
    if x is None:
        return "-"
    return f"{100 * x:.{nd}f}" if pctg else f"{x:.{nd}f}"


def write_md(bench, probes, comps, out, pools=(), diags=()):
    L = []
    if pools:
        L.append("## Headline: end-to-end accuracy over all items (canonical + paraphrase), 95% bootstrap CI\n")
        L.append("| Model | Path | Items | End to end | Median s/turn | Tools seen |")
        L.append("|---|---|---|---|---|---|")
        for p in sorted(pools, key=lambda x: -(x["e2e"] or 0)):
            seen = {True: "yes", False: "NO"}.get(p.get("tools_delivered"), "-")
            L.append(f"| {p['model']} | {p['path']} | {p['n_items']} | {f(p['e2e'], 1, True)} "
                     f"({f(p['e2e_ci'][0], 1, True)}-{f(p['e2e_ci'][1], 1, True)}) | {f(p['latency_p50_s'], 2)} | {seen} |")
        L.append("")
    if diags:
        L.append("## Diagnostic runs beside the native run of the same configuration\n")
        L.append("| Diagnostic | Native end to end | Diagnostic end to end | Prompt tok | Only native passes | Only diagnostic passes | p |")
        L.append("|---|---|---|---|---|---|---|")
        for d in diags:
            pr = d.get("paired_vs_native") or {}
            L.append(f"| {d['label']} | {f(d['native_e2e'], 1, True)} | {f(d['e2e'], 1, True)} "
                     f"({f(d['e2e_ci'][0], 1, True)}-{f(d['e2e_ci'][1], 1, True)}) | {f(d['prompt_tokens_median'], 0)} | "
                     f"{pr.get('a_only_pass', '-')} | {pr.get('b_only_pass', '-')} | "
                     f"{('%.3g' % pr['p_exact']) if pr else '-'} |")
        L.append("")
    L.append("## Tool-calling accuracy (95% bootstrap CI over cases)\n")
    L.append("| Configuration | Cases | Reps | End to end | Tool selection | Intent (diag.) | Text calls | Prompt tok | Tools seen | p50 s/turn | p5-p95 s | Server prefill / decode tok/s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in sorted(bench, key=lambda x: x["label"]):
        L.append(
            f"| {s['label']} | {s['n_cases']} | {s['repeats']} | "
            f"{f(s['e2e'], 1, True)} ({f(s['e2e_ci'][0], 1, True)}-{f(s['e2e_ci'][1], 1, True)}) | "
            f"{f(s['tool_sel'], 1, True)} | {f(s['intent_e2e'], 1, True)} | {f(s['text_tool_call_rate'], 1, True)}% | "
            f"{f(s['prompt_tokens_median'], 0)} | {'yes' if s['tools_delivered'] else ('NO' if s['tools_delivered'] is False else '-')} | "
            f"{f(s['latency_p50_s'], 2)} ({f(s['latency_p50_ci'][0], 2)}-{f(s['latency_p50_ci'][1], 2)}) | "
            f"{f(s['latency_p5_s'], 2)}-{f(s['latency_p95_s'], 2)} | "
            f"{f(s['server_prefill_tok_s_median'], 0)} / {f(s['server_decode_tok_s_median'], 1)} |")
    for fam, rows in comps.items():
        if not rows:
            continue
        L.append(f"\n## Paired comparisons: {fam.replace('_', ' ')} (exact McNemar, Holm-adjusted)\n")
        L.append("| A | B | Common cases | Only A passes | Only B passes | p | p (Holm) |")
        L.append("|---|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda x: x["p_holm"]):
            L.append(f"| {r['a']} | {r['b']} | {r['n_common']} | {r['a_only_pass']} | {r['b_only_pass']} | "
                     f"{r['p_exact']:.3g} | {r['p_holm']:.3g} |")
    L.append("\n## Phase throughput (mean, sd across repetitions)\n")
    L.append("| Configuration | Prompt tok | TTFT s | Prefill tok/s client / server | Decode tok/s client / server | max_tokens honored |")
    L.append("|---|---|---|---|---|---|")
    for p in sorted(probes, key=lambda x: x["label"]):
        for r in p["rows"]:
            if "error" in r:
                L.append(f"| {p['label']} | {r['target']} | error: {str(r['error'])[:60]} | | | |")
                continue
            L.append(f"| {p['label']} | {r['prompt_tokens']} | {f(r['ttft_s'][0], 2)} | "
                     f"{f(r['prefill_client'][0], 0)} / {f(r['prefill_server'][0], 0)} | "
                     f"{f(r['decode_client'][0], 1)} / {f(r['decode_server'][0], 1)} | {r['max_tokens_honored']} |")
    open(out, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--md", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    files = sorted({p for d in a.dirs for p in glob.glob(os.path.join(d, "**", "*.json"), recursive=True)})
    bench, probes, diag = [], [], []
    for p in files:
        try:
            d = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            continue
        if "results" in d and "summary" in d:
            st = bench_stats(p)
            # diagnostic runs (label carries mode=...) are reported separately
            (diag if st.get("mode") else bench).append(st)
        elif "lengths" in d:
            probes.append(probe_stats(p))
    pools = pooled(bench)
    comps = comparisons(bench)
    comps_all = comparisons([{**p, "suite": "all"} for p in pools])
    comps["paths_within_model_all_items"] = comps_all["paths_within_model"]
    comps["models_within_path_all_items"] = comps_all["models_within_path"]
    diags = diagnostics(diag, bench)
    write_md(bench, probes, comps, a.md, pools, diags)
    out = {"bench": [{k: v for k, v in s.items() if not k.startswith("_")} for s in bench],
           "pooled": [{k: v for k, v in p.items() if not k.startswith("_")} for p in pools],
           "probes": probes, "comparisons": comps, "diagnostics": diags,
           "method": {"bootstrap_resamples": B_RESAMPLES, "seed": SEED, "test": "exact McNemar, Holm"}}
    json.dump(out, open(a.json, "w"), indent=1)
    print(f"{len(bench)} bench files, {len(probes)} probe files, {len(diag)} diagnostic files -> {a.md}, {a.json}")


if __name__ == "__main__":
    main()
