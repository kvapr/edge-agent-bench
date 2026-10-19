#!/usr/bin/env python3
"""Speed: throughput, agent turn time, speed-up over the CPU and the trade-off
between accuracy and turn time, for the version 2 report. Standard library
plus PyYAML. Everything is computed from a results directory laid out as results/v2,
except the size fits, which read analysis/data/model_size.json (written from
the GGUF headers by analysis/model_size.py).

- Throughput probes. probe_agent.json holds five prompt lengths with 128
  tokens generated, probe_chat.json one prompt of about 100 tokens with 300
  generated, each with 5 repetitions. For each length, analyze.probe_stats
  gives the mean and sample standard deviation of time to first token, client
  and server prefill rate, client and server decode rate and end-to-end time.
  Client prefill is prompt tokens over client time to first token, so it
  carries the request overhead; server rates are the timings the server
  returns with each response.
- Agent turn time. Per canonical case, the mean client latency over its 3
  repetitions; then the median over the 65 cases with a percentile bootstrap
  CI over cases and nearest-rank p5 and p95, as analyze.bench_stats computes
  it. The paraphrase figure is the median over its 195 single-repetition
  cases.
- Prefill share. Server prompt_ms over client latency, per request and summed
  over the canonical suite; decode share likewise from predicted_ms.
- Prefix reuse. llama.cpp-style servers report the prompt tokens they actually
  processed as timings.prompt_n, and OpenAI-style usage reports reused tokens
  as prompt_tokens_details.cached_tokens. A request reused cached tokens if
  either shows it (prompt_n below usage.prompt_tokens, or cached_tokens above
  0).
- Speed-up over the CPU, same model. Per canonical case, CPU turn time over
  NPU turn time; the median over cases with a percentile bootstrap CI, and the
  range. The ratio of the two medians, with a paired bootstrap CI that
  resamples cases jointly, is given beside it. A pair is comparable only when
  both paths' prompts carried the tools (analyze's prompt-length rule).
- Workload shape. Chat speed-up is CPU over NPU end-to-end time at the chat
  probe; the agent-to-chat ratio divides the agent-turn ratio of medians by it.
- Seconds per correct action. Median canonical turn time over pooled
  end-to-end accuracy (canonical and paraphrase items weighted equally), for
  readable configurations: the prompts carried the tools, and the replies are
  not entirely in a native tool-call format the server returned as text.
- Pareto frontier. Readable configurations not dominated in (pooled accuracy,
  median turn time); each step along it is tested with exact McNemar on the
  pooled items' majority outcomes, Holm-adjusted over the steps.
- Size scaling. Least-squares slope of log rate against log non-lookup
  parameter count, by path. These are descriptive fits on a handful of models
  of mixed architecture, not a model of the hardware.
- The 'model' block is a modelled estimate, not a measurement, of turn time if
  the server reused the cached static prefix; its assumptions are stated in
  the block.

Every bootstrap uses report_common.SEED and report_common.B.

Usage:
  python3 analysis/report_speed.py [results_dir]
"""
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import report_common as rc  # noqa: E402
from report_common import analyze  # noqa: E402

SIZE_FILE = os.path.join(HERE, "data", "model_size.json")
PROBE_FIELDS = ("ttft_s", "prefill_client", "prefill_server", "decode_client", "decode_server", "e2e_s")
# Modelled prefix reuse: uncached suffix tokens (user turn plus template) and
# which probe prefill rate they are read at.
SUFFIX_FAST = (50, "shortest")
SUFFIX_SLOW = (100, "longest")


def _timings(r):
    return (r.get("server_extra") or {}).get("timings") or {}


def _key(cfg):
    return f"{cfg['model']}/{cfg['path']}"


def _reps(cfg, suites=("canonical", "paraphrase")):
    return [r for s in suites if cfg.get(s) for c in cfg[s]["results"] for r in c.get("reps") or []]


def _median_prompt(cfg, suite):
    toks = [r["prompt_tokens"] for r in _reps(cfg, (suite,)) if (r.get("prompt_tokens") or 0) > 0]
    return statistics.median(toks) if toks else None


tools_seen = rc.tools_seen
readable = rc.readable


# ------------------------------------------------------------------ probes
def probe_rows(cfg, kind):
    """analyze.probe_stats rows for probe_<kind>.json, as {mean, sd} dicts."""
    out = []
    for row in analyze.probe_stats(os.path.join(cfg["dir"], f"probe_{kind}.json"))["rows"]:
        if "error" in row:
            out.append({"target": row["target"], "error": row["error"]})
            continue
        out.append({"target": row["target"], "prompt_tokens": row["prompt_tokens"], "n": row["n"],
                    **{k: {"mean": row[k][0], "sd": row[k][1]} for k in PROBE_FIELDS},
                    "max_tokens_honored": row["max_tokens_honored"], "n_out": row["n_out"]})
    return out


def probe_integrity(cfg, rows):
    """Client against server decode rate per repetition, max_tokens honoured,
    client over server prefill rate at the shortest and longest agent length,
    and time to first token beyond the server's prompt time at the longest."""
    rel, n_reps, honoured, shapes = [], 0, 0, 0
    for kind in ("agent", "chat"):
        for L in cfg[f"probe_{kind}"]["lengths"]:
            shapes += 1
            for r in L.get("reps") or []:
                if "error" in r:
                    continue
                n_reps += 1
                honoured += bool(r.get("max_tokens_honored"))
                srv = _timings(r).get("predicted_per_second")
                if r.get("decode_tok_s") and srv:
                    rel.append(r["decode_tok_s"] / srv - 1)
    longest = max(cfg["probe_agent"]["lengths"], key=lambda L: L["target"])
    over = [r["ttft_s"] - _timings(r)["prompt_ms"] / 1000 for r in longest["reps"] if "prompt_ms" in _timings(r)]
    agent = _by_target(rows["agent"])
    ratio = {t: agent[t]["prefill_client"]["mean"] / agent[t]["prefill_server"]["mean"] for t in (min(agent), max(agent))}
    return {"shapes": shapes, "reps": n_reps, "max_tokens_honoured_reps": honoured,
            "decode_rel_diff_mean": statistics.mean(rel), "decode_rel_diff_max_abs": max(abs(x) for x in rel),
            "decode_rel_n": len(rel),
            "prefill_client_over_server_shortest": ratio[min(agent)],
            "prefill_client_over_server_longest": ratio[max(agent)],
            "ttft_minus_server_prompt_s_longest": statistics.mean(over)}


def chat_prefill_share(cfg):
    """Mean server prompt time over mean end-to-end time at the chat probe."""
    reps = [r for r in cfg["probe_chat"]["lengths"][0]["reps"] if "error" not in r]
    return (statistics.mean(_timings(r)["prompt_ms"] for r in reps) / 1000
            / statistics.mean(r["e2e_s"] for r in reps))


def _by_target(rows):
    return {r["target"]: r for r in rows if "error" not in r}


# ------------------------------------------------------------------- bench
def case_latency(cfg, suite="canonical"):
    """{case idx: mean latency over its repetitions}."""
    out = {}
    for c in cfg[suite]["results"]:
        lats = [r["latency_s"] for r in c.get("reps") or [] if isinstance(r.get("latency_s"), (int, float))]
        if lats:
            out[c["idx"]] = statistics.mean(lats)
    return out


def turn_stats(cfg):
    lat = list(case_latency(cfg).values())
    lo, hi = analyze.bootstrap_ci(lat, stat=statistics.median)
    reps = [r for r in _reps(cfg, ("canonical",)) if isinstance(r.get("latency_s"), (int, float)) and "prompt_ms" in _timings(r)]
    pre = [_timings(r)["prompt_ms"] / 1000 for r in reps]
    dec = [_timings(r)["predicted_ms"] / 1000 for r in reps]
    tot = [r["latency_s"] for r in reps]
    share = [p / t for p, t in zip(pre, tot)]
    cap = cfg["canonical"]["provenance"]["request_config"]["max_tokens"]
    out_toks = [r.get("completion_tokens") or 0 for r in _reps(cfg)]
    return {
        "n_cases": len(lat), "n_requests": len(reps), "prompt_tokens_median": _median_prompt(cfg, "canonical"),
        "median_s": statistics.median(lat), "median_ci": [lo, hi], "ci": "95% percentile bootstrap over cases",
        "p5_s": analyze.pct(lat, 5), "p95_s": analyze.pct(lat, 95),
        "paraphrase_median_s": statistics.median(case_latency(cfg, "paraphrase").values()),
        "paraphrase_n_cases": len(case_latency(cfg, "paraphrase")),
        "prefill_share_aggregate": sum(pre) / sum(tot), "decode_share_aggregate": sum(dec) / sum(tot),
        "other_share_aggregate": 1 - (sum(pre) + sum(dec)) / sum(tot),
        "prefill_share_request_median": statistics.median(share),
        "prefill_share_request_p5": analyze.pct(share, 5), "prefill_share_request_p95": analyze.pct(share, 95),
        "output_tokens_median": statistics.median(r.get("completion_tokens") or 0 for r in reps),
        "max_tokens": cap, "requests_at_max_tokens": sum(t >= cap for t in out_toks), "requests_both_suites": len(out_toks),
    }


def prefix_reuse(cfgs):
    """Requests that reused cached prompt tokens, by kind of run."""
    out = {}
    for kind, suites in (("bench", ("canonical", "paraphrase")), ("diagnostic", ("diag_tools_in_prompt",)),
                         ("probe", ("probe_agent", "probe_chat"))):
        n = with_n = fewer = cached = 0
        fields = set()
        for cfg in cfgs:
            for s in suites:
                d = cfg.get(s)
                if not d:
                    continue
                reps = [r for c in d.get("results") or d.get("lengths") or [] for r in c.get("reps") or []]
                for r in (r for r in reps if "error" not in r):
                    n += 1
                    t = _timings(r)
                    fields.update(t)
                    total = (r.get("usage") or {}).get("prompt_tokens") or r.get("prompt_tokens")
                    if "prompt_n" in t:
                        with_n += 1
                        fewer += t["prompt_n"] < total
                    cached += (((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0) > 0
        out[kind] = {"requests": n, "with_prompt_n": with_n, "prompt_n_below_prompt_tokens": fewer,
                     "cached_tokens_above_zero": cached, "timings_fields": sorted(fields)}
    return out


# ---------------------------------------------------------------- speed-up
def _ratio_of_medians(pairs):
    return statistics.median(c for c, _ in pairs) / statistics.median(n for _, n in pairs)


def speedup(cpu, npu, probes):
    """Paired per-case speed-up of npu over cpu (same model), plus probe ratios."""
    a, b = case_latency(cpu), case_latency(npu)
    ids = sorted(set(a) & set(b))
    ratios = [a[i] / b[i] for i in ids]
    pairs = [(a[i], b[i]) for i in ids]
    seen = (tools_seen(cpu), tools_seen(npu))
    reason = None if all(seen) else ("tools missing on both paths" if not any(seen) else
                                     f"tools missing on {rc.PATH_NAMES['cpu' if not seen[0] else npu['path']]}")
    pc, pn = probes[_key(cpu)], probes[_key(npu)]
    ac, an = _by_target(pc["agent"]), _by_target(pn["agent"])
    short, long_ = min(ac), max(ac)
    cc, cn = pc["chat"][0], pn["chat"][0]
    out = {
        "model": npu["model"], "path": npu["path"], "comparable": reason is None, "not_comparable_reason": reason,
        "n_cases": len(ids), "per_case_median": statistics.median(ratios),
        "per_case_median_ci": list(analyze.bootstrap_ci(ratios, stat=statistics.median)),
        "per_case_min": min(ratios), "per_case_max": max(ratios),
        "ratio_of_medians": _ratio_of_medians(pairs),
        "ratio_of_medians_ci": list(analyze.bootstrap_ci(pairs, stat=_ratio_of_medians)),
        "ci": "95% percentile bootstrap over cases, resampled jointly",
        "ttft_longest_speedup": ac[long_]["ttft_s"]["mean"] / an[long_]["ttft_s"]["mean"],
        "agent_probe_longest_speedup": ac[long_]["e2e_s"]["mean"] / an[long_]["e2e_s"]["mean"],
        "chat_speedup": cc["e2e_s"]["mean"] / cn["e2e_s"]["mean"],
        "prefill_ratio_shortest": an[short]["prefill_server"]["mean"] / ac[short]["prefill_server"]["mean"],
        "prefill_ratio_longest": an[long_]["prefill_server"]["mean"] / ac[long_]["prefill_server"]["mean"],
        "decode_ratio_longest": an[long_]["decode_server"]["mean"] / ac[long_]["decode_server"]["mean"],
        "decode_ratio_chat": cn["decode_server"]["mean"] / cc["decode_server"]["mean"],
        "probe_lengths": [short, long_],
    }
    out["agent_over_chat"] = out["ratio_of_medians"] / out["chat_speedup"] if reason is None else None
    return out


# ------------------------------------------------------- accuracy and time
def _dominated_by(q, pts):
    return [o["config"] for o in pts if o is not q and o["e2e"] >= q["e2e"] and o["t"] <= q["t"]
            and (o["e2e"] > q["e2e"] or o["t"] < q["t"])]


def frontier(pts):
    """Members of the Pareto frontier in (accuracy up, time down), fastest first."""
    return sorted((q for q in pts if not _dominated_by(q, pts)), key=lambda q: q["t"])


def tradeoff(pts, items):
    """Seconds per correct action, the frontier, and McNemar tests along it.
    pts: [{config, e2e, t, ...}], items: {config: {item id: pass fraction}}."""
    for q in pts:
        q["dominated_by"] = _dominated_by(q, pts)
        q["s_per_correct"] = q["t"] / q["e2e"] if q["e2e"] else None
    front = frontier(pts)
    steps = []
    for x, y in zip(front, front[1:]):
        a, b = items[x["config"]], items[y["config"]]
        only_x, only_y, p = rc.mcnemar_items(a, b)
        steps.append({"faster": x["config"], "slower": y["config"], "n_items": len(set(a) & set(b)),
                      "e2e_diff": y["e2e"] - x["e2e"], "time_ratio": y["t"] / x["t"],
                      "faster_only_passes": only_x, "slower_only_passes": only_y, "p": p})
    for st, adj in zip(steps, rc.holm([st["p"] for st in steps])):
        st["p_holm"] = adj
        st["significant"] = adj < 0.05
    return {"points": sorted(pts, key=lambda q: q["t"]), "frontier": [q["config"] for q in front], "steps": steps,
            "ci": "95% percentile bootstrap over pooled items",
            "test": "exact McNemar on pooled items' majority outcomes, Holm over the steps, alpha 0.05"}


# ----------------------------------------------------------------- scaling
def loglog(xs, ys):
    lx, ly = [math.log(x) for x in xs], [math.log(y) for y in ys]
    mx, my = statistics.mean(lx), statistics.mean(ly)
    slope = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sum((x - mx) ** 2 for x in lx)
    res = sum((y - my - slope * (x - mx)) ** 2 for x, y in zip(lx, ly))
    tot = sum((y - my) ** 2 for y in ly)
    return {"exponent": slope, "r2": 1 - res / tot if tot else None, "n": len(xs)}


def scaling(probes, sizes):
    """Exponent b in rate = a * N^b, N = non-lookup parameters, per path."""
    if not sizes:
        return None
    quantities = {
        "prefill_shortest": lambda p: _by_target(p["agent"])[min(_by_target(p["agent"]))]["prefill_server"]["mean"],
        "prefill_longest": lambda p: _by_target(p["agent"])[max(_by_target(p["agent"]))]["prefill_server"]["mean"],
        "decode_chat": lambda p: p["chat"][0]["decode_server"]["mean"],
        "decode_longest": lambda p: _by_target(p["agent"])[max(_by_target(p["agent"]))]["decode_server"]["mean"],
    }
    fits = []
    for path in rc.PATHS:
        keys = [k for k in probes if k.endswith("/" + path) and k.split("/")[0] in sizes]
        for subset, sel in (("all", keys), ("Qwen3 dense", [k for k in keys if k.startswith("qwen3-")])):
            if len(sel) < 3:
                continue
            for q, get in quantities.items():
                fit = loglog([sizes[k.split("/")[0]]["params_non_lookup"] for k in sel], [get(probes[k]) for k in sel])
                fits.append({"path": path, "quantity": q, "models": subset, **fit})
    return {"descriptive": True, "x": "non-lookup parameters from the GGUF header (analysis/data/model_size.json)",
            "y": "server rate, mean of 5 repetitions", "fits": fits}


# ------------------------------------------------------------------- model
def prefix_model(cfgs, probes):
    """Modelled turn time with the static prefix cached (not measured)."""
    rows = {}
    for cfg in cfgs:
        k = _key(cfg)
        if not tools_seen(cfg):
            continue
        rest = [statistics.mean(r["latency_s"] - _timings(r)["prompt_ms"] / 1000 for r in c["reps"])
                for c in cfg["canonical"]["results"] if c.get("reps")]
        agent = _by_target(probes[k]["agent"])
        rate = {"shortest": agent[min(agent)]["prefill_server"]["mean"], "longest": agent[max(agent)]["prefill_server"]["mean"]}
        base = statistics.median(rest)
        rows[k] = {"measured_median_s": statistics.median(case_latency(cfg).values()),
                   "without_prefill_median_s": base,
                   "modelled_fast_s": base + SUFFIX_FAST[0] / rate[SUFFIX_FAST[1]],
                   "modelled_slow_s": base + SUFFIX_SLOW[0] / rate[SUFFIX_SLOW[1]]}
    for k, v in rows.items():
        cpu = rows.get(k.split("/")[0] + "/cpu")
        if cpu and not k.endswith("/cpu"):
            v["measured_speedup"] = cpu["measured_median_s"] / v["measured_median_s"]
            v["modelled_speedup"] = sorted([cpu["modelled_fast_s"] / v["modelled_fast_s"],
                                            cpu["modelled_slow_s"] / v["modelled_slow_s"]])
    return {
        "measured": False,
        "assumptions": [
            "This block is a model, not a measurement. It keeps each turn's decode and non-server time as measured "
            "(the median over canonical cases of the per-case mean of client latency minus server prompt time).",
            f"It assumes that only the user turn and template suffix is read anew, either {SUFFIX_FAST[0]} tokens at "
            f"the server prefill rate of the shortest agent probe (optimistic) or {SUFFIX_SLOW[0]} tokens at the rate "
            "of the longest (pessimistic).",
            "Reading a short suffix that attends over a cached 2,700-token prefix was not measured, and whether either "
            "path can reuse a prefix under GenieX v0.8.0 is not known.",
            "Only configurations whose prompts carried the tools are modelled, and requests are single-turn.",
        ],
        "rows": rows,
    }


# ----------------------------------------------------------------- compute
def compute(results_dir):
    every = rc.configs(results_dir)
    order = {m: i for i, m in enumerate(rc.MODEL_NAMES)}
    cfgs = sorted((c for c in every if c.get("canonical") and c.get("probe_agent") and c.get("probe_chat")),
                  key=lambda c: (order.get(c["model"], len(order)), rc.PATHS.index(c["path"])))
    by_key = {_key(c): c for c in cfgs}
    probes = {k: {"agent": probe_rows(c, "agent"), "chat": probe_rows(c, "chat"),
                  "chat_prefill_share": chat_prefill_share(c)} for k, c in by_key.items()}
    integrity = {k: probe_integrity(c, probes[k]) for k, c in by_key.items()}
    turns = {k: {**turn_stats(c), "tools_seen": tools_seen(c), "readable": readable(c)} for k, c in by_key.items()}
    speedups = {k: speedup(by_key[c["model"] + "/cpu"], c, probes) for k, c in by_key.items()
                if c["path"] != "cpu" and c["model"] + "/cpu" in by_key}
    ttft = {}
    for k, p in probes.items():
        agent = _by_target(p["agent"])
        top = agent[max(agent)]
        ttft[k] = {"target": top["target"], "prompt_tokens": top["prompt_tokens"], "n": top["n"], **top["ttft_s"],
                   "speedup_vs_cpu": speedups[k]["ttft_longest_speedup"] if k in speedups else None}

    items = {k: rc.items(by_key[k]) for k, t in turns.items() if t["readable"]}
    pts = []
    for k, it in items.items():
        lo, hi = analyze.bootstrap_ci(list(it.values()))
        pts.append({"config": k, "e2e": statistics.mean(it.values()), "e2e_ci": [lo, hi], "n_items": len(it),
                    "t": turns[k]["median_s"]})
    trade = tradeoff(pts, items)
    canon = [{"config": k, "e2e": statistics.mean(rc.items(by_key[k], suites_=("canonical",)).values()),
              "t": turns[k]["median_s"]} for k in items]
    tail = [{"config": q["config"], "e2e": q["e2e"], "t": turns[q["config"]]["p95_s"]} for q in pts]
    trade["sensitivity"] = {"canonical_accuracy": [q["config"] for q in frontier(canon)],
                            "p95_turn_time": [q["config"] for q in frontier(tail)]}

    by_path = {}
    for path in rc.PATHS:
        ks = [k for k in integrity if k.endswith("/" + path)]
        if not ks:
            continue
        rng = {f: [min(integrity[k][f] for k in ks), max(integrity[k][f] for k in ks)]
               for f in ("decode_rel_diff_mean", "prefill_client_over_server_shortest",
                         "prefill_client_over_server_longest", "ttft_minus_server_prompt_s_longest")}
        by_path[path] = {"configs": len(ks), "range": rng,
                         "decode_rel_diff_max_abs": max(integrity[k]["decode_rel_diff_max_abs"] for k in ks),
                         "within_0_1pct": sum(abs(integrity[k]["decode_rel_diff_mean"]) <= 0.001 for k in ks)}
    integrity_total = {
        "configs": len(integrity), "shapes": sum(v["shapes"] for v in integrity.values()),
        "reps": sum(v["reps"] for v in integrity.values()),
        "max_tokens_honoured_reps": sum(v["max_tokens_honoured_reps"] for v in integrity.values()),
        "bench_requests": sum(t["requests_both_suites"] for t in turns.values()),
        "bench_requests_at_max_tokens": sum(t["requests_at_max_tokens"] for t in turns.values()),
        "by_path": by_path,
    }
    return {
        "scope": "QCS9075 (Dragonwing IQ-9075 EVK), GenieX v0.8.0 (QAIRT 2.45, llama.cpp 9425611), "
                 "campaign 2026-10-14 to 2026-10-17",
        "configs": list(by_key), "not_loaded": sorted(_key(c) for c in every if c.get("load_failure")),
        "probes": probes, "ttft_longest": ttft, "turn": turns, "prefix_reuse": prefix_reuse(every),
        "speedup": speedups, "probe_integrity": integrity, "integrity_total": integrity_total,
        "tradeoff": trade, "scaling": scaling(probes, rc.load_json(SIZE_FILE)),
        "model": prefix_model(cfgs, probes),
        "method": {"bootstrap_resamples": rc.B, "seed": rc.SEED, "sd": "sample standard deviation (n-1)"},
    }


# ------------------------------------------------------------------ tables
def _f(x, nd=1):
    return "-" if x is None else f"{x:,.{nd}f}"


def _pc(x, nd=1):
    return "-" if x is None else f"{100 * x:.{nd}f}"


def _msd(d, nd):
    return f"{_f(d['mean'], nd)} ± {_f(d['sd'], nd)}"


def _name(k):
    m, p = k.split("/")
    return f"{rc.MODEL_NAMES.get(m, m)} | {rc.PATH_NAMES.get(p, p)}"


def _throughput_cells(k, row):
    if "error" in row:
        return [_name(k), "error: " + str(row["error"])[:60]] + [""] * 7
    return [_name(k), f"{row['prompt_tokens']:,.0f}", _msd(row["ttft_s"], 2), _msd(row["prefill_client"], 0),
            _msd(row["prefill_server"], 0), _msd(row["decode_client"], 1), _msd(row["decode_server"], 1),
            _msd(row["e2e_s"], 2)]


def _row(cells):
    return "| " + " | ".join(cells) + " |"


def _plain(k):
    return _name(k).replace(" | ", " ")


def tables_md(data):
    L = []
    L.append("## Throughput at agent-like prompt lengths\n")
    L.append(f"{data['scope']}. Synthetic filler prompts at five lengths with 128 tokens generated, 5 repetitions per "
             "length, mean ± sample standard deviation. Client prefill is prompt tokens over client time to first "
             "token, so it includes the request overhead; server rates are the timings the server returns.\n")
    L.append("| Model | Path | Target | Prompt tok | TTFT s | Prefill tok/s, client | Prefill tok/s, server | "
             "Decode tok/s, client | Decode tok/s, server | End to end s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, p in data["probes"].items():
        for row in p["agent"]:
            cells = _throughput_cells(k, row)
            L.append(_row(cells[:1] + [f"{row['target']:,}"] + cells[1:]))

    L.append("\n## Throughput at a chat-like shape\n")
    L.append("About 100 prompt tokens and 300 generated, 5 repetitions, mean ± sample standard deviation. The prefill "
             "share is the mean server prompt time over the mean end-to-end time.\n")
    L.append("| Model | Path | Prompt tok | TTFT s | Prefill tok/s, client | Prefill tok/s, server | "
             "Decode tok/s, client | Decode tok/s, server | End to end s | Prefill share % |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, p in data["probes"].items():
        L.append(_row(_throughput_cells(k, p["chat"][0]) + [_pc(p["chat_prefill_share"])]))

    L.append("\n## Time to first token at the longest agent probe\n")
    L.append("Client time to first token, mean ± sample standard deviation over 5 repetitions. The speed-up is the "
             "CPU mean over the path mean for the same model.\n")
    L.append("| Model | Path | Prompt tok | TTFT s | Speed-up over CPU |")
    L.append("|---|---|---|---|---|")
    for k, t in data["ttft_longest"].items():
        sp = f"{t['speedup_vs_cpu']:.1f}x" if t["speedup_vs_cpu"] else "-"
        L.append(f"| {_name(k)} | {t['prompt_tokens']:,.0f} | {t['mean']:.2f} ± {t['sd']:.2f} | {sp} |")

    L.append("\n## Agent turn time\n")
    L.append("Canonical suite, 65 cases with 3 repetitions each. Each case's mean client latency is taken, then the "
             "median over cases with a 95% percentile bootstrap CI over cases, and nearest-rank p5 and p95. The "
             "paraphrase column is the median over 195 single-repetition cases. Shares are summed server prompt and "
             "decode time over summed client latency for the 195 canonical requests; the per-request column gives the "
             "median and p5-p95 of each request's prefill share. Rows whose prompts lacked the tools ran on short "
             "prompts and are not agent turns.\n")
    L.append("| Model | Path | Tools seen | Prompt tok | Median s/turn (95% CI) | p5-p95 s | Paraphrase median s | "
             "Prefill share % | Decode share % | Prefill share per request % (p5-p95) | Output tok | At max_tokens |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, t in data["turn"].items():
        L.append(f"| {_name(k)} | {'yes' if t['tools_seen'] else 'no'} | {t['prompt_tokens_median']:,.0f} | "
                 f"{t['median_s']:.2f} ({t['median_ci'][0]:.2f}-{t['median_ci'][1]:.2f}) | "
                 f"{t['p5_s']:.2f}-{t['p95_s']:.2f} | {t['paraphrase_median_s']:.2f} | "
                 f"{_pc(t['prefill_share_aggregate'])} | {_pc(t['decode_share_aggregate'])} | "
                 f"{_pc(t['prefill_share_request_median'])} ({_pc(t['prefill_share_request_p5'])}-"
                 f"{_pc(t['prefill_share_request_p95'])}) | {t['output_tokens_median']:.0f} | "
                 f"{t['requests_at_max_tokens']} of {t['requests_both_suites']} |")
    pr = data["prefix_reuse"]
    L.append("\nThe server reports the prompt tokens it processed as timings.prompt_n, which falls below "
             "usage.prompt_tokens when a cached prefix is reused; usage also carries cached_tokens. "
             + " ".join(f"Of {v['requests']:,} {kind} requests, {v['prompt_n_below_prompt_tokens']} processed fewer "
                        f"prompt tokens than the prompt held and {v['cached_tokens_above_zero']} reported cached tokens."
                        for kind, v in pr.items()))
    if data["not_loaded"]:
        L.append(f"\nThe configurations that did not load are absent: {', '.join(map(_plain, data['not_loaded']))}.")

    L.append("\n## Speed-up over the CPU path, same model\n")
    L.append("The per-case speed-up is CPU turn time over NPU turn time for each of the 65 canonical cases, given as "
             "the median with a 95% percentile bootstrap CI over cases and the range. The ratio of medians uses a "
             "paired bootstrap that resamples cases jointly. Turn ratios are comparable only where both paths' "
             "prompts carried the tools. Probe columns use the means of 5 repetitions at the longest agent probe "
             "(128 tokens out) and at the chat shape; decode ratios are NPU over CPU server rates.\n")
    L.append("| Model | Path | Cases | Per-case speed-up, median (95% CI) | Range | Ratio of medians (95% CI) | "
             "TTFT x | Agent probe x | Chat x | Agent / chat | Decode NPU/CPU, chat | Decode NPU/CPU, longest |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, s in data["speedup"].items():
        if s["comparable"]:
            turn = (f"{s['per_case_median']:.2f} ({s['per_case_median_ci'][0]:.2f}-{s['per_case_median_ci'][1]:.2f}) | "
                    f"{s['per_case_min']:.2f}-{s['per_case_max']:.2f} | "
                    f"{s['ratio_of_medians']:.2f} ({s['ratio_of_medians_ci'][0]:.2f}-{s['ratio_of_medians_ci'][1]:.2f})")
        else:
            turn = f"not comparable, {s['not_comparable_reason']} | - | -"
        L.append(f"| {_name(k)} | {s['n_cases']} | {turn} | {s['ttft_longest_speedup']:.1f} | "
                 f"{s['agent_probe_longest_speedup']:.2f} | {s['chat_speedup']:.2f} | {_f(s['agent_over_chat'], 2)} | "
                 f"{s['decode_ratio_chat']:.2f} | {s['decode_ratio_longest']:.2f} |")

    tr = data["tradeoff"]
    L.append("\n## Accuracy against turn time, and seconds per correct action\n")
    L.append("Only readable configurations are shown, those whose prompts carried the tools and whose replies were not "
             "entirely in a native tool-call format returned unparsed. End to end is pooled over the canonical and paraphrase items, "
             "weighted equally, with a 95% percentile bootstrap CI over items. Seconds per correct action is the "
             "median canonical turn time over that accuracy, and assumes single-turn requests with no prefix reuse.\n")
    L.append("| Model | Path | Items | End to end % (95% CI) | Median s/turn | Seconds per correct action | "
             "On frontier | Dominated by |")
    L.append("|---|---|---|---|---|---|---|---|")
    for q in tr["points"]:
        dom = q["dominated_by"]
        dom_s = ", ".join(_plain(d) for d in dom[:2]) + (f" and {len(dom) - 2} more" if len(dom) > 2 else "")
        L.append(f"| {_name(q['config'])} | {q['n_items']} | {_pc(q['e2e'])} ({_pc(q['e2e_ci'][0])}-{_pc(q['e2e_ci'][1])}) | "
                 f"{q['t']:.2f} | {_f(q['s_per_correct'], 2)} | {'yes' if not dom else 'no'} | {dom_s or '-'} |")
    L.append("\nSteps along the frontier, fastest first, tested with exact McNemar on the majority outcome of each "
             f"shared item and Holm-adjusted over the {len(tr['steps'])} steps.\n")
    L.append("| Faster | Slower | Items | Accuracy gain, points | Time ratio | Only faster passes | Only slower passes | "
             "p | p (Holm) | Significant at 0.05 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in tr["steps"]:
        L.append(f"| {_plain(s['faster'])} | {_plain(s['slower'])} | {s['n_items']} | {_pc(s['e2e_diff'])} | "
                 f"{s['time_ratio']:.2f} | {s['faster_only_passes']} | {s['slower_only_passes']} | {s['p']:.2g} | "
                 f"{s['p_holm']:.2g} | {'yes' if s['significant'] else 'no'} |")
    sens = tr["sensitivity"]
    L.append("\nWith canonical-only accuracy the frontier is " + ", ".join(map(_plain, sens["canonical_accuracy"]))
             + ". With p95 instead of median turn time it is " + ", ".join(map(_plain, sens["p95_turn_time"])) + ".")

    it = data["integrity_total"]
    L.append("\n## Speed instrument agreement\n")
    L.append("Client decode rate (streamed chunks over their arrival time) against the server's predicted_per_second, "
             "per repetition over all six probe shapes, 30 repetitions per configuration. Prefill columns give the "
             "client rate over the server rate (ratio of 5-repetition means) at the shortest and longest agent probe, "
             "and the last column the mean time to first token beyond the server's prompt time at the longest probe. "
             "Each cell is the range over the path's configurations.\n")
    L.append("| Path | Configurations | Decode, mean relative difference % | Within ±0.1% | Largest single "
             "repetition % | Prefill client/server, shortest | Prefill client/server, longest | TTFT overhead ms, longest |")
    L.append("|---|---|---|---|---|---|---|---|")
    for path, v in it["by_path"].items():
        r = v["range"]
        L.append(f"| {rc.PATH_NAMES[path]} | {v['configs']} | {100 * r['decode_rel_diff_mean'][0]:+.2f} to "
                 f"{100 * r['decode_rel_diff_mean'][1]:+.2f} | {v['within_0_1pct']} | "
                 f"{100 * v['decode_rel_diff_max_abs']:.2f} | {r['prefill_client_over_server_shortest'][0]:.3f}-"
                 f"{r['prefill_client_over_server_shortest'][1]:.3f} | {r['prefill_client_over_server_longest'][0]:.3f}-"
                 f"{r['prefill_client_over_server_longest'][1]:.3f} | "
                 f"{1000 * r['ttft_minus_server_prompt_s_longest'][0]:.0f}-{1000 * r['ttft_minus_server_prompt_s_longest'][1]:.0f} |")
    L.append(f"\nmax_tokens was honoured in {it['max_tokens_honoured_reps']} of {it['reps']} probe repetitions "
             f"over {it['shapes']} shapes, and {it['bench_requests_at_max_tokens']} of {it['bench_requests']:,} "
             "tool-calling requests reached the max_tokens cap.")

    sc = data["scaling"]
    if sc:
        L.append("\n## Throughput against model size (descriptive)\n")
        L.append("Least-squares exponent b in rate = a N^b, where N is the non-lookup parameter count from the GGUF "
                 "header and the rate is the server mean over 5 repetitions. Shortest and longest are the agent probe "
                 "lengths, and chat is the chat probe. The fits describe these models only; architectures differ "
                 "within 'all', and n is small.\n")
        L.append("| Path | Quantity | Models | n | Exponent b | R^2 |")
        L.append("|---|---|---|---|---|---|")
        for f_ in sc["fits"]:
            L.append(f"| {rc.PATH_NAMES[f_['path']]} | {f_['quantity'].replace('_', ' ')} | {f_['models']} | "
                     f"{f_['n']} | {f_['exponent']:.2f} | {f_['r2']:.3f} |")

    mo = data["model"]
    L.append("\n## Modelled turn time with the static prefix cached (a model, not a measurement)\n")
    L.append(" ".join(mo["assumptions"]) + "\n")
    L.append("| Model | Path | Measured median s/turn | Modelled s/turn | Measured speed-up over CPU | "
             "Modelled speed-up over CPU |")
    L.append("|---|---|---|---|---|---|")
    for k, v in mo["rows"].items():
        lo, hi = sorted([v["modelled_fast_s"], v["modelled_slow_s"]])
        ms = f"{v['measured_speedup']:.2f}" if "measured_speedup" in v else "-"
        mm = f"{v['modelled_speedup'][0]:.2f}-{v['modelled_speedup'][1]:.2f}" if "modelled_speedup" in v else "-"
        L.append(f"| {_name(k)} | {v['measured_median_s']:.2f} | {lo:.2f}-{hi:.2f} | {ms} | {mm} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    print(tables_md(compute(sys.argv[1] if len(sys.argv) > 1 else os.path.join(rc.REPO, "results", "v2"))), end="")
