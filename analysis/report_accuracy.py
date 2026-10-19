#!/usr/bin/env python3
"""Accuracy of the measured configurations, with the uncertainty of their ranks
and of the serving-path effects.

Reads only <model>/<path>/{canonical,paraphrase}.json under the results
directory, through analysis/report_common.py. An item is a canonical case (its
pass fraction over 3 repetitions) or a paraphrase (1 repetition); pooled scores
weight the 260 items equally, as analysis/analyze.py does.

Per measured configuration (24):
- end to end, pooled and per suite, with the 95% percentile item bootstrap CI
  of analysis/out/v2.json, and for pooled and paraphrase items a cluster
  bootstrap CI that resamples the 65 base requests (a canonical case with its
  three paraphrases);
- tool selection, and lenient and strict parameter rates, cumulative as in the
  harness summary (right tool, then right command, then parameters);
- the format funnel: a structured call, a tool name among the four offered,
  the right tool, end to end (item-weighted like the headline);
- tools_seen, analyze.py's rule (median server-reported prompt tokens at least
  TOOLS_DELIVERED_MIN_PROMPT_TOKENS in every suite), and readability. A score
  is not read as model capability under report_common.unreadable_reason:
  the prompts lacked the tools, or no reply was a structured call while most
  replies held a text tool call (the model's own format, which the server did
  not parse).

Over readable configurations only:
- rank uncertainty: a joint cluster bootstrap (the same base requests drawn for
  every configuration in a resample, in report_common.cluster_bootstrap's
  order) gives P(rank 1) and the 95% percentile rank interval, among all
  readable configurations and among the NPU ones; tied scores share the best
  rank;
- serving-path effects within a model (CPU against each NPU path; canonical,
  paraphrase and pooled items): the difference, paired item and cluster
  bootstrap CIs, exact McNemar on majority outcomes, and Holm within
  analyze.py's families, which hold every pair of measured configurations, so
  the p-values equal analysis/out/v2.json;
- CPU neighbours in pooled order, and Qwen3-8B against the two 4B models, with
  the same statistics;
- minimum detectable paired differences (80% power) for the pairs whose
  difference is read as small or absent, at alpha 0.05 and at 0.05/m, the
  smallest Holm threshold in the pair's family of m tests;
- suite weighting: ranks under the published pooled mean, 50/50
  canonical/paraphrase, canonical only and paraphrase only, with Kendall tau-b
  of each against the published scores.

Accuracies and differences are fractions of 1 in the JSON and percentages or
percentage points in the tables; a difference is the second configuration
minus the first.

Usage: python3 analysis/report_accuracy.py [results_dir]   (default results/v2)
"""
import itertools
import math
import os
import random
import statistics
import sys

import report_common as rc
from report_common import analyze

SUITES = ("canonical", "paraphrase")
SCOPES = ("canonical", "paraphrase", "pooled")
PATH_PAIRS = (("cpu", "npu_llama"), ("cpu", "compiled"))
WEIGHTINGS = ("published", "half", "canonical", "paraphrase")
WEIGHTING_NAMES = {"published": "Published (pooled items)", "half": "50/50 canonical/paraphrase",
                   "canonical": "Canonical only", "paraphrase": "Paraphrase only"}
# Pairs whose difference is read as small or absent; their minimum detectable
# differences bound what could have gone unseen.
MDE_PAIRS = (
    (("gemma-4-e4b", "cpu"), ("qwen3-4b-2507", "cpu")),
    (("qwen3-4b-2507", "cpu"), ("qwen3-4b", "cpu")),
    (("qwen3-4b", "cpu"), ("granite-4.0-micro", "cpu")),
    (("qwen3-4b-2507", "compiled"), ("qwen3-8b", "compiled")),
    (("qwen3-4b-2507", "cpu"), ("qwen3-4b-2507", "compiled")),
    (("qwen3-8b", "cpu"), ("qwen3-8b", "compiled")),
    (("qwen3-1.7b", "cpu"), ("qwen3-1.7b", "npu_llama")),
    (("qwen3.5-0.8b", "cpu"), ("qwen3.5-0.8b", "npu_llama")),
)
EIGHT_B, FOUR_BS = ("qwen3-8b", "cpu"), (("qwen3-4b", "cpu"), ("qwen3-4b-2507", "cpu"))


def _params(kind, device=False):
    def ok(r):
        return bool(r.get("ok_tool") and r.get("ok_command") and r.get(kind) and (not device or r.get("ok_device")))
    return ok


# Per-repetition flags beside end to end (analyze.e2e_ok, report_common.items' default).
FLAGS = {
    "tool": lambda r: bool(r.get("ok_tool")),
    "structured": lambda r: bool(r.get("got_tool")),
    "offered": lambda r: r.get("got_tool") in rc.SUBSET,
    "params_lenient": _params("ok_params_lenient"),
    "params_strict": _params("ok_params_strict"),
    "e2e_strict": _params("ok_params_strict", device=True),
}


# ------------------------------------------------------------ per configuration
def scoped(vals, scope):
    """The items of one suite ('canonical', 'paraphrase') or all ('pooled')."""
    return vals if scope == "pooled" else {k: v for k, v in vals.items() if k.startswith(scope + ":")}


def mean(vals):
    return statistics.mean(vals.values()) if vals else None


def ident(key):
    return {"model": key[0], "path": key[1]}


def replies(cfg):
    return [r for s in SUITES for c in cfg[s]["results"] for r in c.get("reps") or []]


def prompt_medians(cfg):
    """Median server-reported prompt tokens per suite, taken as analyze.bench_stats does."""
    out = {}
    for s in SUITES:
        toks = [r["prompt_tokens"] for c in cfg[s]["results"] for r in c.get("reps") or []
                if isinstance(r.get("prompt_tokens"), (int, float)) and r["prompt_tokens"] > 0]
        out[s] = statistics.median(toks) if toks else None
    return out


def parsed_text_call(r):
    if r.get("got_tool"):
        return False
    name, args = analyze.parse_text_tool_call(r.get("content"))
    return name is not None and isinstance(args, dict)


def turn_p50(cfg):
    """analyze.py's turn time: median over canonical cases of each case's mean latency."""
    per_case = [[r["latency_s"] for r in c.get("reps") or [] if isinstance(r.get("latency_s"), (int, float))]
                for c in cfg["canonical"]["results"]]
    lat = [statistics.mean(x) for x in per_case if x]
    return statistics.median(lat) if lat else None


def item_ci(vals):
    return list(analyze.bootstrap_ci(list(vals.values())))


def config_record(cfg, e2e):
    flags = {name: rc.items(cfg, fn) for name, fn in FLAGS.items()}
    tool = flags["tool"]
    can, par = scoped(e2e, "canonical"), scoped(e2e, "paraphrase")
    toks = prompt_medians(cfg)
    seen = None if None in toks.values() else all(t >= analyze.TOOLS_DELIVERED_MIN_PROMPT_TOKENS
                                                  for t in toks.values())
    reps = replies(cfg)
    structured = sum(bool(r.get("got_tool")) for r in reps)
    parsed = sum(parsed_text_call(r) for r in reps)
    reason = rc.unreadable_reason(cfg)
    m, p = cfg["model"], cfg["path"]
    return {
        "model": m, "path": p, "model_name": rc.MODEL_NAMES.get(m, m), "path_name": rc.PATH_NAMES[p],
        "n_items": len(e2e), "e2e": mean(e2e), "e2e_ci_item": item_ci(e2e),
        "e2e_ci_cluster": list(rc.cluster_bootstrap(e2e)),
        "canonical": {"n": len(can), "e2e": mean(can), "e2e_ci_item": item_ci(can),
                      "tool": mean(scoped(tool, "canonical")), "tool_ci_item": item_ci(scoped(tool, "canonical"))},
        "paraphrase": {"n": len(par), "e2e": mean(par), "e2e_ci_item": item_ci(par),
                       "e2e_ci_cluster": list(rc.cluster_bootstrap(par))},
        "tool": mean(tool), "tool_ci_item": item_ci(tool),
        "params_lenient": mean(flags["params_lenient"]), "params_strict": mean(flags["params_strict"]),
        "e2e_strict": mean(flags["e2e_strict"]),
        "funnel": {"structured_call": mean(flags["structured"]), "offered_tool": mean(flags["offered"]),
                   "right_tool": mean(tool), "e2e": mean(e2e)},
        "tools_seen": seen, "prompt_tokens_median": toks,
        "requests": len(reps), "structured_requests": structured, "parsed_text_call_requests": parsed,
        "readable": reason is None, "unreadable_reason": reason,
        "turn_p50_s": turn_p50(cfg),
    }


# ------------------------------------------------------------ comparisons
def holm_families(e2e):
    """analyze.py's comparison families over every measured configuration:
    paths within a model and models within a path, per suite (one family for
    both suites) and pooled. {(scope, a, b): test} for both orders of a pair."""
    out = {}
    for fam, group in (("paths_within_model", 0), ("models_within_path", 1)):
        for scopes, suffix in ((("canonical", "paraphrase"), ""), (("pooled",), "_all_items")):
            groups = {}
            for key in sorted(e2e):
                groups.setdefault(key[group], []).append(key)
            tests = [(s, a, b, rc.mcnemar_items(scoped(e2e[a], s), scoped(e2e[b], s))[2])
                     for s in scopes for keys in groups.values() for a, b in itertools.combinations(keys, 2)]
            for (s, a, b, p), q in zip(tests, rc.holm([t[3] for t in tests])):
                out[(s, a, b)] = out[(s, b, a)] = {"family": fam + suffix, "m": len(tests), "p": p, "p_holm": q}
    return out


def compare(e2e, a, b, scope, fams):
    """b minus a over the items both configurations share."""
    ea, eb = scoped(e2e[a], scope), scoped(e2e[b], scope)
    diff, ci = rc.paired_diff_ci(ea, eb)
    only_a, only_b, p = rc.mcnemar_items(ea, eb)
    fam = fams[(scope, a, b)]
    return {"a": ident(a), "b": ident(b), "scope": scope, "n": len(ea.keys() & eb.keys()),
            "acc_a": mean(ea), "acc_b": mean(eb), "diff": diff, "ci_item": list(ci),
            # one canonical item per base request, so clustering would only redraw the item bootstrap
            "ci_cluster": None if scope == "canonical" else list(rc.paired_diff_ci(ea, eb, cluster=True)[1]),
            "only_a": only_a, "only_b": only_b, "p": p,
            "p_holm": fam["p_holm"], "holm_family": fam["family"], "holm_m": fam["m"]}


def mde_row(e2e, a, b, fams):
    only_a, only_b, _ = rc.mcnemar_items(e2e[a], e2e[b])
    n = len(e2e[a].keys() & e2e[b].keys())
    rate = (only_a + only_b) / n
    m = fams[("pooled", a, b)]["m"]
    return {"a": ident(a), "b": ident(b), "n": n, "discordant_rate": rate, "diff": mean(e2e[b]) - mean(e2e[a]),
            "mde_alpha_0_05": rc.mde_paired(rate, n), "holm_m": m, "holm_worst_alpha": 0.05 / m,
            "mde_holm_worst": rc.mde_paired(rate, n, alpha=0.05 / m), "power": 0.8}


# ------------------------------------------------------------ ranks
def competition_ranks(scores):
    """{key: 1 + number of keys with a strictly higher score}."""
    return {k: 1 + sum(o > s for o in scores.values()) for k, s in scores.items()}


def rank_uncertainty(e2e, keys):
    """P(rank 1) and the 95% percentile rank interval of each configuration in
    keys under a joint cluster bootstrap of the pooled score."""
    bases = rc.item_bases()
    clusters = sorted({bases[i] for k in keys for i in e2e[k]})
    where = {b: j for j, b in enumerate(clusters)}
    sums, counts = {}, {}
    for k in keys:
        sums[k], counts[k] = [0.0] * len(clusters), [0] * len(clusters)
        for i, v in e2e[k].items():
            sums[k][where[bases[i]]] += v
            counts[k][where[bases[i]]] += 1
    rng = random.Random(rc.SEED)
    n = len(clusters)
    ranks = {k: [] for k in keys}
    for _ in range(rc.B):
        draw = [rng.randrange(n) for _ in range(n)]
        # rounded so that equal scores tie exactly despite summation order
        score = {k: round(sum(map(sums[k].__getitem__, draw)) / sum(map(counts[k].__getitem__, draw)), 9)
                 for k in keys}
        for k, r in competition_ranks(score).items():
            ranks[k].append(r)
    full = competition_ranks({k: round(mean(e2e[k]), 9) for k in keys})
    out = []
    for k in sorted(keys, key=lambda k: (full[k], k)):
        rs = sorted(ranks[k])
        out.append({**ident(k), "e2e": mean(e2e[k]), "rank": full[k], "p_rank1": rs.count(1) / rc.B,
                    "rank_ci": [rs[int(math.floor(0.025 * rc.B))], rs[int(math.ceil(0.975 * rc.B)) - 1]]})
    return out


def kendall_tau_b(x, y):
    conc = disc = tied_x = tied_y = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        dx, dy = x[i] - x[j], y[i] - y[j]
        tied_x += dx == 0
        tied_y += dy == 0
        conc += dx * dy > 0
        disc += dx * dy < 0
    n0 = len(x) * (len(x) - 1) // 2
    return (conc - disc) / math.sqrt((n0 - tied_x) * (n0 - tied_y))


def weighting(records, keys):
    fns = {"published": lambda r: r["e2e"], "half": lambda r: (r["canonical"]["e2e"] + r["paraphrase"]["e2e"]) / 2,
           "canonical": lambda r: r["canonical"]["e2e"], "paraphrase": lambda r: r["paraphrase"]["e2e"]}
    scores = {w: {k: round(fns[w](records[k]), 9) for k in keys} for w in WEIGHTINGS}
    ranks = {w: competition_ranks(scores[w]) for w in WEIGHTINGS}
    order = sorted(keys, key=lambda k: (ranks["published"][k], k))
    rows = [{**ident(k), **{w: {"score": scores[w][k], "rank": ranks[w][k]} for w in WEIGHTINGS}} for k in order]
    tau = {w: kendall_tau_b([scores["published"][k] for k in order], [scores[w][k] for k in order])
           for w in WEIGHTINGS[1:]}
    return {"rows": rows, "kendall_tau_b": tau}


# ------------------------------------------------------------ compute
def compute(results_dir):
    cfgs = {(c["model"], c["path"]): c for c in rc.configs(results_dir) if c["canonical"] and c["paraphrase"]}
    e2e = {k: rc.items(c) for k, c in cfgs.items()}
    records = {k: config_record(c, e2e[k]) for k, c in cfgs.items()}
    readable = sorted(k for k in records if records[k]["readable"])
    fams = holm_families(e2e)

    order = sorted(records, key=lambda k: (not records[k]["readable"], -records[k]["e2e"], k))
    path_effects = [compare(e2e, (m, pa), (m, pb), s, fams)
                    for m in sorted({k[0] for k in readable}) for pa, pb in PATH_PAIRS
                    if (m, pa) in readable and (m, pb) in readable for s in SCOPES]
    cpu = sorted((k for k in readable if k[1] == "cpu"), key=lambda k: (-records[k]["e2e"], k))
    neighbours = [{"rank": i + 1, **compare(e2e, lower, higher, "pooled", fams)}
                  for i, (higher, lower) in enumerate(zip(cpu, cpu[1:]))]
    return {
        "method": {
            "items": "65 canonical cases (pass fraction over 3 repetitions) and 195 paraphrases (1 repetition), "
                     "equally weighted when pooled",
            "intervals": f"95% percentile bootstrap, {rc.B} resamples, seed {rc.SEED}; item CIs resample items, "
                         "cluster CIs resample the 65 base requests",
            "paired_test": "exact McNemar on majority outcomes; Holm within analyze.py's families",
            "readable_rule": f"tools seen (median prompt tokens >= {analyze.TOOLS_DELIVERED_MIN_PROMPT_TOKENS} in "
                             "every suite), and not (no structured call while most replies hold a text tool call); "
                             "report_common.unreadable_reason",
            "rank_ties": "tied scores share the best rank",
            "units": "fractions of 1; differences are second minus first",
        },
        "configs": [records[k] for k in order],
        "ranks": {"overall": rank_uncertainty(e2e, readable),
                  "npu": rank_uncertainty(e2e, [k for k in readable if k[1] != "cpu"])},
        "path_effects": path_effects,
        "cpu_neighbours": neighbours,
        "qwen3_8b_vs_4b": [compare(e2e, four, EIGHT_B, s, fams) for four in FOUR_BS for s in SCOPES],
        "mde": [mde_row(e2e, a, b, fams) for a, b in MDE_PAIRS if a in readable and b in readable],
        "weighting": weighting(records, readable),
    }


# ------------------------------------------------------------ tables
def pc(x):
    return "-" if x is None else f"{100 * x:.1f}"


def pp(x):
    return "-" if x is None else f"{round(100 * x, 1) + 0.0:+.1f}"


def ci(c):
    return "-" if not c or c[0] is None else f"{pc(c[0])}-{pc(c[1])}"


def dci(c):
    return "-" if not c or c[0] is None else f"{pp(c[0])} to {pp(c[1])}"


def fp(p):
    if p >= 0.995:
        return "1.00"
    return f"{p:#.2g}" if p >= 1e-4 else f"{p:.1e}"


def name(d):
    return f"{rc.MODEL_NAMES.get(d['model'], d['model'])}, {rc.PATH_NAMES[d['path']]}"


def tables_md(data):
    L = []
    cfgs = data["configs"]
    n = cfgs[0]["n_items"]
    L.append("## Accuracy by configuration\n")
    L.append(f"End to end over {n} items (65 canonical cases, each the pass fraction over 3 repetitions, and 195 "
             "paraphrases at 1 repetition, equally weighted). Item CI: 95% percentile bootstrap over items, as in "
             "analysis/out/v2.json. Base-request CI: 95% cluster bootstrap over the 65 base requests (a canonical "
             "case with its three paraphrases). Suite CIs are item CIs (n = 65 and 195). Tool selection is pooled "
             "over all items and over the canonical suite. Median turn: median over the canonical cases of each "
             "case's mean latency. Readable configurations are ranked first; the rest cannot be read as model "
             "capability for the reason given.\n")
    L.append("| Model | Path | Items | End to end, % | Item CI | Base-request CI | Canonical, % (CI) | "
             "Paraphrase, % (CI) | Tool selection, % pooled / canonical | Median turn, s | Tools seen | "
             "Not readable because |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in cfgs:
        L.append(f"| {r['model_name']} | {r['path_name']} | {r['n_items']} | {pc(r['e2e'])} | {ci(r['e2e_ci_item'])} | "
                 f"{ci(r['e2e_ci_cluster'])} | {pc(r['canonical']['e2e'])} ({ci(r['canonical']['e2e_ci_item'])}) | "
                 f"{pc(r['paraphrase']['e2e'])} ({ci(r['paraphrase']['e2e_ci_item'])}) | "
                 f"{pc(r['tool'])} / {pc(r['canonical']['tool'])} | "
                 f"{'-' if r['turn_p50_s'] is None else format(r['turn_p50_s'], '.2f')} | "
                 f"{ {True: 'yes', False: 'no'}.get(r['tools_seen'], '-') } | {r['unreadable_reason'] or ''} |")
    L.append("")
    for r in cfgs:
        if r["unreadable_reason"] == "tools not delivered":
            t = r["prompt_tokens_median"]
            L.append(f"- {name(r)}: median prompt {t['canonical']:.0f} (canonical) and {t['paraphrase']:.0f} "
                     f"(paraphrase) tokens, below the {analyze.TOOLS_DELIVERED_MIN_PROMPT_TOKENS:,}-token threshold.")
        elif r["unreadable_reason"]:
            L.append(f"- {name(r)}: {r['structured_requests']} of {r['requests']} replies are structured calls and "
                     f"{r['parsed_text_call_requests']} are text calls that parse in full.")
    L.append("")

    L.append("## Format funnel and parameter grading\n")
    L.append(f"Item-weighted over {n} items, as the headline. Structured call: the reply carries a tool_calls entry. "
             "Offered tool: its name is one of the four tools in the request. Right tool: the expected tool. "
             "Lenient and strict parameters are cumulative (right tool, right command, then parameters) as in the "
             "harness summary; end to end also requires the device and uses lenient parameters, and strict end to "
             "end uses strict ones.\n")
    L.append("| Model | Path | Structured call, % | Offered tool, % | Right tool, % | End to end, % | "
             "Lenient parameters, % | Strict parameters, % | Strict end to end, % |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in cfgs:
        f = r["funnel"]
        L.append(f"| {r['model_name']} | {r['path_name']} | {pc(f['structured_call'])} | {pc(f['offered_tool'])} | "
                 f"{pc(f['right_tool'])} | {pc(f['e2e'])} | {pc(r['params_lenient'])} | {pc(r['params_strict'])} | "
                 f"{pc(r['e2e_strict'])} |")
    L.append("")

    ov, npu = data["ranks"]["overall"], {(x["model"], x["path"]): x for x in data["ranks"]["npu"]}
    L.append("## Rank uncertainty\n")
    L.append(f"Readable configurations only ({len(ov)} overall, {len(npu)} on the NPU paths). Joint cluster "
             f"bootstrap of the pooled score: each of {rc.B:,} resamples draws 65 base requests with replacement "
             f"(seed {rc.SEED}) for all configurations at once. P(rank 1) is the share of resamples in which the "
             "configuration scores highest; a tie for the highest score counts for each tied configuration, so the "
             "column can sum to more than 100%. The rank interval holds the central 95% of the resampled ranks, "
             "with tied scores sharing the best rank.\n")
    L.append("| Rank | Model | Path | End to end, % | P(rank 1), % | 95% rank interval | NPU-only rank | "
             "NPU-only P(rank 1), % | NPU-only 95% rank interval |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for x in ov:
        u = npu.get((x["model"], x["path"]))
        tail = (f"{u['rank']} | {pc(u['p_rank1'])} | [{u['rank_ci'][0]}, {u['rank_ci'][1]}]" if u else "| |")
        L.append(f"| {x['rank']} | {rc.MODEL_NAMES.get(x['model'], x['model'])} | {rc.PATH_NAMES[x['path']]} | "
                 f"{pc(x['e2e'])} | {pc(x['p_rank1'])} | [{x['rank_ci'][0]}, {x['rank_ci'][1]}] | {tail} |")
    L.append("")

    def cmp_header(first, second):
        L.append(f"| {first} | {second} | Items | Scope | First, % | Second, % | Difference, pp | Paired item CI | "
                 "Paired base-request CI | Only first / only second passes | McNemar p | Holm p (family size) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

    def cmp_row(c, first, second):
        L.append(f"| {first} | {second} | {c['n']} | {c['scope']} | {pc(c['acc_a'])} | {pc(c['acc_b'])} | "
                 f"{pp(c['diff'])} | {dci(c['ci_item'])} | {dci(c['ci_cluster']) if c['ci_cluster'] else 'as item'} | "
                 f"{c['only_a']} / {c['only_b']} | {fp(c['p'])} | {fp(c['p_holm'])} ({c['holm_m']}) |")

    L.append("## Serving-path effects within a model\n")
    L.append("Readable pairs only. Difference: NPU path minus CPU, in percentage points, over the items both share. "
             "Paired CIs are 95% percentile bootstraps resampling items, or base requests (canonical items are one "
             "per base request, so there the cluster CI is the item CI). McNemar: exact, on majority outcomes. Holm "
             "within analyze.py's families: the 28 suite-level or the 14 pooled serving-path comparisons, as in "
             "analysis/out/v2.json.\n")
    cmp_header("Model", "Comparison")
    for c in data["path_effects"]:
        cmp_row(c, rc.MODEL_NAMES.get(c["a"]["model"], c["a"]["model"]),
                f"{rc.PATH_NAMES[c['a']['path']]} to {rc.PATH_NAMES[c['b']['path']]}")
    L.append("")

    L.append("## CPU path: neighbouring ranks\n")
    L.append("Readable CPU configurations in order of pooled end to end; each row compares a configuration with the "
             f"next one down over {n} pooled items. Difference: higher minus lower. CIs and McNemar as above; Holm "
             f"within analyze.py's {data['cpu_neighbours'][0]['holm_m']} pooled model comparisons.\n")
    L.append("| Rank | Model | End to end, % | Next model | Difference, pp | Paired item CI | Paired base-request CI | "
             "Only higher / only lower passes | McNemar p | Holm p |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in data["cpu_neighbours"]:
        L.append(f"| {c['rank']} | {rc.MODEL_NAMES[c['b']['model']]} | {pc(c['acc_b'])} | "
                 f"{rc.MODEL_NAMES[c['a']['model']]} ({pc(c['acc_a'])}) | {pp(c['diff'])} | {dci(c['ci_item'])} | "
                 f"{dci(c['ci_cluster'])} | {c['only_b']} / {c['only_a']} | {fp(c['p'])} | {fp(c['p_holm'])} |")
    L.append("")

    L.append("## Qwen3-8B against the 4B models on the CPU\n")
    L.append("Difference: Qwen3-8B minus the 4B model. Holm within analyze.py's families: 210 suite-level or 105 "
             "pooled model comparisons.\n")
    cmp_header("4B model", "Against")
    for c in data["qwen3_8b_vs_4b"]:
        cmp_row(c, rc.MODEL_NAMES[c["a"]["model"]], rc.MODEL_NAMES[c["b"]["model"]])
    L.append("")

    L.append("## Minimum detectable differences\n")
    L.append("Paired difference detectable with 80% power by a two-sided test (normal approximation to McNemar), "
             f"from the observed rate of items whose majority outcome differs, over {n} pooled items. The Holm-worst "
             "alpha is 0.05/m for the pair's family of m tests (105 model comparisons, 14 serving-path "
             "comparisons).\n")
    L.append("| First | Second | Observed difference, pp | Discordant items, % | MDE at alpha 0.05, pp | Holm-worst alpha | "
             "MDE at Holm-worst alpha, pp |")
    L.append("|---|---|---|---|---|---|---|")
    for x in data["mde"]:
        L.append(f"| {name(x['a'])} | {name(x['b'])} | {pp(x['diff'])} | {pc(x['discordant_rate'])} | "
                 f"{100 * x['mde_alpha_0_05']:.1f} | 0.05/{x['holm_m']} | {100 * x['mde_holm_worst']:.1f} |")
    L.append("")

    w = data["weighting"]
    L.append("## Sensitivity of the ranking to suite weighting\n")
    L.append(f"Readable configurations ({len(w['rows'])}). Published: the pooled mean over all items, which weights "
             "paraphrases 3:1. 50/50: the mean of the two suite scores. Each cell gives the score in % and the rank "
             "(tied scores share the best rank). Kendall tau-b is computed on the scores against the published "
             "ones.\n")
    L.append("| Model | Path | " + " | ".join(WEIGHTING_NAMES[k] for k in WEIGHTINGS) + " |")
    L.append("|---|---|" + "---|" * len(WEIGHTINGS))
    for r in w["rows"]:
        L.append(f"| {rc.MODEL_NAMES.get(r['model'], r['model'])} | {rc.PATH_NAMES[r['path']]} | "
                 + " | ".join(f"{pc(r[k]['score'])} ({r[k]['rank']})" for k in WEIGHTINGS) + " |")
    L.append("| Kendall tau-b against published | | 1.000 | "
             + " | ".join(f"{w['kendall_tau_b'][k]:.3f}" for k in WEIGHTINGS[1:]) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    print(tables_md(compute(sys.argv[1] if len(sys.argv) > 1 else os.path.join(rc.REPO, "results", "v2"))))
