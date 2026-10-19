#!/usr/bin/env python3
"""Robustness of the version 2 tool-calling results to rephrasing, to the
choice of expected labels and to repetition, for docs/REPORT.md.

Everything comes from a results directory laid out as results/v2, the suites
and analysis/data/label_review.json. Accuracy is end to end as analyze.e2e_ok
grades it; a canonical case counts its pass fraction over three repetitions,
a paraphrase its single outcome.

- Pools. A configuration is readable when the prompts of both suites carried
  the tool definitions (median prompt of at least
  analyze.TOOLS_DELIVERED_MIN_PROMPT_TOKENS tokens) and the server returned at
  least one structured call. The functional pool is the readable
  configurations with canonical end to end of at least 50%, a threshold fixed
  before item results were examined.
- Paraphrase gap. Each paraphrase belongs to one base request (its canonical
  case). Per base request, d = mean of its three paraphrase outcomes minus
  the canonical pass fraction; the mean of d is paraphrase minus canonical
  accuracy. The interval is a 95% percentile bootstrap over the 65 base
  requests (analyze.bootstrap_ci in canonical order), the test an exact
  two-sided sign test on the signs of d. Holm is applied across the
  functional pool and, for comparison, across every configuration whose
  prompts carried the tools; a configuration is borderline when the interval
  and the two adjusted tests do not all agree.
- Styles A (casual), B (polite) and C (situational): accuracy over each
  style's 65 paraphrases with an item bootstrap, change against canonical,
  and an exact McNemar test of the style outcome against the canonical
  majority outcome, with Holm over the pool and the three styles and, as
  for the gap, over the tools-seen configurations and the three styles.
  The share of a base request's content words that a paraphrase keeps
  first drops every whitespace-separated token that contains a digit (the
  device name @poesw1, port numbers, values such as 25W), then counts the
  remaining lowercase alphabetic words minus a short stoplist that also
  drops port and interface.
- Gap decomposition. Each failing reply gets its first failing check in
  grading order, with a wrong tool split into a real but wrong tool, the
  expected command used as the function name, and any other unoffered name.
  End to end is one minus the sum of the failure rates, so the per-category
  changes add up to the gap exactly.
- Levels: canonical accuracy per suite level with a case bootstrap, and
  paraphrase accuracy per level.
- Label review: the paraphrases listed in analysis/data/label_review.json are
  regraded accepting their alternatives, and the gap is recomputed.
- Stability, over the canonical suite's three repetitions: cases whose three
  replies are identical (tool name, arguments with keys sorted, and message
  text), cases whose grade changes, and accuracy per repetition; pooled by
  path, the reply pairs that differ only in argument key order and the first
  differing character of differing text replies.
- Same GGUF: for each model run from one file on cpu and npu_llama, the
  share of requests (same item and repetition) with identical replies and
  with identical grades, beside the identity of repetitions within each path.
  The results keep the parsed call, not the reply bytes, so identical here
  means the same tool name, the same arguments in the order the server sent
  them, and the same message text; the count with keys sorted is given too.
- Gaps and interval bounds are rounded to 12 decimals, so a mean that is
  zero up to floating-point error reads as zero and not as a negative bound.
- Hardest items: how many readable configurations fail each item (a
  canonical case fails when it passes in fewer than half its repetitions),
  the functional-pool pass rate and the commonest failing answers.

    python3 analysis/report_robustness.py [results/v2]
"""
import collections
import itertools
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import report_common as rc  # noqa: E402
from report_common import analyze  # noqa: E402
from bench_toolcall import norm  # noqa: E402  (the grader's own normalisation)

FUNCTIONAL_MIN = 0.5
STYLES = ("A", "B", "C")
STYLE_NAMES = {"A": "casual", "B": "polite", "C": "situational"}
LEVEL_NAMES = {"level1_queries": "L1 queries", "level2_enable_disable": "L2 enable/disable",
               "level3_single_param": "L3 one parameter", "level4_multi_param": "L4 several parameters"}
CATEGORIES = ("wrong_tool_real", "cmd_as_tool", "other_unoffered", "wrong_command", "wrong_params",
              "wrong_device", "text_call", "no_call", "other")
CATEGORY_NAMES = {"wrong_tool_real": "Wrong offered tool", "cmd_as_tool": "Command as function name",
                  "other_unoffered": "Other unoffered name", "wrong_command": "Wrong command",
                  "wrong_params": "Wrong parameters", "wrong_device": "Wrong device",
                  "text_call": "Call as text", "no_call": "No call", "other": "Other"}
STOPWORDS = {"the", "to", "on", "in", "of", "a", "an", "for", "and", "is", "it", "its", "port", "interface"}
# Port formats the diagnose_port_issues schema lists: '5', '1/5', 'Gi1/5', 'GigabitEthernet1/5'.
DIAG_PORT = re.compile(r"^(?:gi(?:gabitethernet)?)?(?:1/)?(\d{1,2})$", re.I)
LABEL_REVIEW = os.path.join(HERE, "data", "label_review.json")
ITEM_98 = 98


def mean(xs):
    xs = list(xs)
    return statistics.mean(xs) if xs else None


def clean(x):
    # -0.0 + 0.0 is 0.0, so a rounded negative zero becomes a plain zero.
    return None if x is None else round(x, 12) + 0.0


def boot(values):
    return [clean(x) for x in analyze.bootstrap_ci(values)]


def ok(r):
    return analyze.e2e_ok(r)


def name(key):
    m, p = key.split("/")
    return f"{rc.MODEL_NAMES.get(m, m)}, {rc.PATH_NAMES.get(p, p)}"


def sig(r, sort_keys=True):
    """A reply as (tool name, arguments, message text)."""
    return r.get("got_tool"), json.dumps(r.get("got_args"), sort_keys=sort_keys), r.get("content")


# ------------------------------------------------------------------ loading
def load(results_dir):
    """Suites, the base canonical index of each paraphrase, and every
    configuration that has both tool-calling suites, checked against the
    suites case by case."""
    can, par = rc.suites()
    base = {c["input"]: c["idx"] for c in can}
    par_base = [base[p["base_input"]] for p in par]
    cfgs = {}
    for c in rc.configs(results_dir):
        if not (c["canonical"] and c["paraphrase"]):
            continue
        for suite, cases in (("canonical", can), ("paraphrase", par)):
            if [x["input"] for x in c[suite]["results"]] != [x["input"] for x in cases]:
                raise ValueError(f"{c['dir']}: {suite} cases differ from the suite")
        c["pf"] = [mean(ok(r) for r in x["reps"]) for x in c["canonical"]["results"]]
        c["py"] = [mean(ok(r) for r in x["reps"]) for x in c["paraphrase"]["results"]]
        cfgs[f"{c['model']}/{c['path']}"] = c
    return can, par, par_base, cfgs


tools_seen = rc.tools_seen
readable = rc.readable


# ------------------------------------------------------------- paraphrase gap
def by_base(values, par_base, n_base):
    groups = [[] for _ in range(n_base)]
    for j, v in enumerate(values):
        groups[par_base[j]].append(v)
    return groups


def paired_gap(pf, py, par_base):
    """Paraphrase minus canonical, paired by base request."""
    d = [mean(g) - pf[b] for b, g in enumerate(by_base(py, par_base, len(pf)))]
    worse, better = sum(x < 0 for x in d), sum(x > 0 for x in d)
    return {"canonical": mean(pf), "paraphrase": mean(py), "gap": clean(mean(d)), "ci": boot(d),
            "bases_worse": worse, "bases_better": better, "bases_same": len(d) - worse - better,
            "p_sign": analyze.mcnemar_exact(worse, better)}


def gap_section(cfgs, seen, pool, par_base):
    rows = {k: paired_gap(cfgs[k]["pf"], cfgs[k]["py"], par_base) for k in seen}
    for fam, keys in (("p_holm_pool", pool), ("p_holm_seen", seen)):
        for k, adj in zip(keys, rc.holm([rows[k]["p_sign"] for k in keys])):
            rows[k][fam] = adj
    for k in pool:
        r = rows[k]
        r["verdicts"] = {"ci": r["ci"][1] < 0 or r["ci"][0] > 0, "holm_pool": r["p_holm_pool"] < 0.05,
                         "holm_seen": r["p_holm_seen"] < 0.05}
        r["borderline"] = len(set(r["verdicts"].values())) > 1
    gaps = [rows[k]["gap"] for k in pool]
    return {"configs": rows, "pool_mean_gap": clean(mean(gaps)), "pool_range": [min(gaps), max(gaps)],
            "pool_significant": {"ci_below_zero": sum(rows[k]["ci"][1] < 0 for k in pool),
                                 "sign_holm_pool": sum(rows[k]["p_holm_pool"] < 0.05 for k in pool),
                                 "sign_holm_seen": sum(rows[k]["p_holm_seen"] < 0.05 for k in pool),
                                 "n": len(pool)},
            "borderline": [k for k in pool if rows[k]["borderline"]]}


# -------------------------------------------------------------------- styles
def style_values(py, par, par_base, style):
    """A style's outcomes in base-request order."""
    out = [None] * (max(par_base) + 1)
    for j, p in enumerate(par):
        if p["style"] == style:
            out[par_base[j]] = py[j]
    return out


def content_words(text):
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", re.sub(r"\S*\d\S*", " ", text.lower()))
    return set(words) - STOPWORDS


def style_section(cfgs, seen, pool, can, par, par_base):
    rows = {}
    for k in seen:
        pf, py = cfgs[k]["pf"], cfgs[k]["py"]
        maj = [v > 0.5 for v in pf]
        rows[k] = {}
        for st in STYLES:
            ys = style_values(py, par, par_base, st)
            only_can = sum(1 for b, y in enumerate(ys) if maj[b] and not y > 0.5)
            only_style = sum(1 for b, y in enumerate(ys) if y > 0.5 and not maj[b])
            rows[k][st] = {"acc": mean(ys), "ci": boot(ys), "n": len(ys),
                           "change": clean(mean(ys) - mean(pf)), "canonical_only": only_can, "style_only": only_style,
                           "p_mcnemar": analyze.mcnemar_exact(only_can, only_style)}
    for field, keys in (("p_holm_pool", pool), ("p_holm_seen", seen)):
        fam = [(k, st) for k in keys for st in STYLES]
        for (k, st), adj in zip(fam, rc.holm([rows[k][st]["p_mcnemar"] for k, st in fam])):
            rows[k][st][field] = adj
    can_items = [mean(cfgs[k]["pf"][i] for k in pool) for i in range(len(can))]
    pool_rows = {"canonical": {"acc": mean(can_items), "ci": boot(can_items), "n": len(can_items)}}
    for st in STYLES:
        items = [mean(cfgs[k]["py"][j] for k in pool) for j, p in enumerate(par) if p["style"] == st]
        changes = [rows[k][st]["change"] for k in pool]
        pool_rows[st] = {"acc": mean(items), "ci": boot(items), "n": len(items),
                         "mean_change": clean(mean(changes)), "change_range": [min(changes), max(changes)],
                         "significant_holm_pool": sum(rows[k][st]["p_holm_pool"] < 0.05 for k in pool),
                         "significant_holm_seen": sum(rows[k][st]["p_holm_seen"] < 0.05 for k in pool)}
    words = {"canonical": {"mean_words": mean(len(c["input"].split()) for c in can)}}
    for st in STYLES:
        kept = [len(content_words(can[par_base[j]]["input"]) & content_words(p["input"]))
                / len(content_words(can[par_base[j]]["input"])) for j, p in enumerate(par) if p["style"] == st]
        words[st] = {"content_words_kept": mean(kept), "n": len(kept),
                     "mean_words": mean(len(p["input"].split()) for p in par if p["style"] == st)}
    return {"configs": rows, "pool": pool_rows, "wording": words}


# ------------------------------------------------------------ decomposition
def category(case, r):
    """First failing check of one reply in grading order, or None for a pass."""
    if ok(r):
        return None
    if r.get("error"):
        return "other"
    tool = r.get("got_tool")
    if not tool:
        text_name, _ = analyze.parse_text_tool_call(r.get("content"))
        return "text_call" if r.get("text_tool_call") or text_name else "no_call"
    if not r.get("ok_tool"):
        if tool in rc.SUBSET:
            return "wrong_tool_real"
        return "cmd_as_tool" if tool == case["expected_command"] else "other_unoffered"
    if r.get("got_args") is None:
        return "other"
    if not r.get("ok_command"):
        return "wrong_command"
    if not r.get("ok_params_lenient"):
        return "wrong_params"
    return "wrong_device"


def failure_rates(results):
    rates = collections.Counter()
    for c in results:
        for r in c["reps"]:
            k = category(c, r)
            if k:
                rates[k] += 1 / (len(c["reps"]) * len(results))
    return rates


def renamed_passes(case, r):
    """A command-as-function call graded with the function name moved into
    command_name and any top-level parameters moved into parameters."""
    a = dict(r.get("got_args") or {})
    top = {k: v for k, v in a.items() if k not in ("device_name", "command_name", "parameters")}
    params = a.get("parameters") if isinstance(a.get("parameters"), dict) else {}
    args = {"device_name": a.get("device_name"), "command_name": r["got_tool"], "parameters": {**top, **params}}
    return ok(analyze.grade(case, case["expected_tool"], args))


def decomposition_section(cfgs, readable_keys, pool):
    rows = {}
    for k in readable_keys:
        cr, pr = cfgs[k]["canonical"]["results"], cfgs[k]["paraphrase"]["results"]
        rc_, rp = failure_rates(cr), failure_rates(pr)
        contrib = {c: clean(rc_.get(c, 0.0) - rp.get(c, 0.0)) for c in CATEGORIES}
        gap = clean(mean(cfgs[k]["py"]) - mean(cfgs[k]["pf"]))
        assert abs(sum(contrib.values()) - gap) < 1e-9, k
        renamed = sum(1 for c in pr for r in c["reps"]
                      if category(c, r) == "cmd_as_tool" and renamed_passes(c, r))
        rows[k] = {"gap": gap, "contribution": contrib, "paraphrase_cmd_as_tool_renamed_pass": renamed,
                   "n_paraphrase": len(pr)}
    pool_mean = {c: clean(mean(rows[k]["contribution"][c] for k in pool)) for c in CATEGORIES}
    return {"configs": rows, "pool_mean": pool_mean, "pool_mean_gap": clean(mean(rows[k]["gap"] for k in pool))}


# -------------------------------------------------------------------- levels
def level_section(cfgs, readable_keys, pool, can, par):
    levels = list(dict.fromkeys(c["level"] for c in can))
    rows = {}
    for k in readable_keys:
        rows[k] = {}
        for lv in levels:
            vc = [v for v, c in zip(cfgs[k]["pf"], can) if c["level"] == lv]
            vp = [v for v, p in zip(cfgs[k]["py"], par) if p["level"] == lv]
            rows[k][lv] = {"canonical": mean(vc), "canonical_ci": boot(vc), "n_canonical": len(vc),
                           "paraphrase": mean(vp), "n_paraphrase": len(vp)}
    pool_rows = {}
    for lv in levels:
        vc = [mean(cfgs[k]["pf"][i] for k in pool) for i, c in enumerate(can) if c["level"] == lv]
        vp = [mean(cfgs[k]["py"][j] for k in pool) for j, p in enumerate(par) if p["level"] == lv]
        pool_rows[lv] = {"canonical": mean(vc), "canonical_ci": boot(vc), "n_canonical": len(vc),
                         "paraphrase": mean(vp), "paraphrase_ci": boot(vp), "n_paraphrase": len(vp)}
    return {"levels": levels, "configs": rows, "pool": pool_rows}


# -------------------------------------------------------------- label review
def load_label_review(par):
    review = rc.load_json(LABEL_REVIEW)
    for it in review["items"]:
        p = par[it["idx"]]
        exp = {"tool": p["expected_tool"], "command": p["expected_command"], "params": p["expected_params"]}
        if it["input"] != p["input"] or it["expected"] != exp:
            raise ValueError(f"label_review.json item {it['idx']} does not match the paraphrase suite")
    return review["items"]


def alternative_passes(alt, r):
    a = r.get("got_args")
    if r.get("got_tool") != alt["tool"] or not isinstance(a, dict) or norm(a.get("device_name", "")) != "poesw1":
        return False
    if "command" in alt and a.get("command_name") != alt["command"]:
        return False
    if alt["tool"] == "diagnose_port_issues":
        m = DIAG_PORT.match(str(a.get("port", "")).strip())
        return bool(m) and alt["params"]["port"] == f"1/{int(m.group(1))}"
    return True


def label_sets(items):
    return {"robustness": {it["idx"] for it in items if it["robustness"]},
            "grader": {it["idx"] for it in items if it["grader"] == "primary"},
            "merged": {it["idx"] for it in items if it["robustness"] or it["grader"] == "primary"},
            "merged_and_wider": {it["idx"] for it in items}}


def regraded(cfg, alts, flagged):
    """Paraphrase outcomes with the alternatives accepted on the flagged items."""
    out = []
    for j, (c, v) in enumerate(zip(cfg["paraphrase"]["results"], cfg["py"])):
        if j in flagged:
            v = mean(ok(r) or any(alternative_passes(a, r) for a in alts[j]) for r in c["reps"])
        out.append(v)
    return out


def label_section(cfgs, readable_keys, pool, par, par_base):
    items = load_label_review(par)
    alts = {it["idx"]: it["alternatives"] for it in items}
    sets = label_sets(items)
    merged = sets["merged"]
    rows = {}
    for k in pool:
        y_alt = regraded(cfgs[k], alts, merged)
        g = paired_gap(cfgs[k]["pf"], y_alt, par_base)
        rows[k] = {"paraphrase": mean(cfgs[k]["py"]), "paraphrase_alt": g["paraphrase"],
                   "gap": clean(mean(cfgs[k]["py"]) - g["canonical"]), "gap_alt": g["gap"], "gap_alt_ci": g["ci"],
                   "flagged_pass": sum(cfgs[k]["py"][j] for j in merged),
                   "flagged_pass_alt": sum(y_alt[j] for j in merged)}
    by_set = {}
    for s, flagged in sets.items():
        gaps = [mean(regraded(cfgs[k], alts, flagged)) - mean(cfgs[k]["pf"]) for k in pool]
        by_set[s] = {"n_items": len(flagged), "pool_mean_gap_alt": clean(mean(gaps))}
    per_item = []
    for it in items:
        j = it["idx"]
        reps = {k: cfgs[k]["paraphrase"]["results"][j]["reps"] for k in readable_keys}
        per_item.append({**it, "base_idx": par_base[j],
                         "pool_pass": sum(ok(r) for k in pool for r in reps[k]),
                         "pool_alt": sum(any(alternative_passes(a, r) for a in alts[j]) for k in pool for r in reps[k]),
                         "readable_pass": sum(ok(r) for r in itertools.chain(*reps.values())),
                         "readable_alt": sum(any(alternative_passes(a, r) for a in alts[j])
                                             for r in itertools.chain(*reps.values())),
                         "n_pool": len(pool), "n_readable": len(readable_keys)})
    return {"items": per_item, "flagged": sorted(merged), "configs": rows, "by_set": by_set,
            "pool_mean_gap": clean(mean(rows[k]["gap"] for k in pool)),
            "pool_mean_gap_alt": clean(mean(rows[k]["gap_alt"] for k in pool)),
            "pool_mean_paraphrase": mean(rows[k]["paraphrase"] for k in pool),
            "pool_mean_paraphrase_alt": mean(rows[k]["paraphrase_alt"] for k in pool)}


# ----------------------------------------------------------------- stability
def first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def stability(results):
    reps_n = len(results[0]["reps"])
    out = {"n_cases": len(results), "reps": reps_n, "identical_cases": 0, "grade_changes": 0,
           "pairs": 0, "identical_pairs": 0, "key_order_only_pairs": 0, "text_pairs": 0, "text_pairs_first_char": 0}
    per_rep = [0] * reps_n
    for c in results:
        reps = c["reps"]
        sigs = [sig(r) for r in reps]
        grades = [ok(r) for r in reps]
        out["identical_cases"] += len(set(sigs)) == 1
        out["grade_changes"] += 0 < sum(grades) < len(grades)
        for i, g in enumerate(grades):
            per_rep[i] += g
        for i, j in itertools.combinations(range(len(reps)), 2):
            out["pairs"] += 1
            if sigs[i] == sigs[j]:
                out["identical_pairs"] += 1
                out["key_order_only_pairs"] += sig(reps[i], False) != sig(reps[j], False)
            elif not reps[i].get("got_tool") and not reps[j].get("got_tool"):
                out["text_pairs"] += 1
                out["text_pairs_first_char"] += first_difference(reps[i].get("content") or "",
                                                                 reps[j].get("content") or "") == 0
    out["per_rep_acc"] = [v / len(results) for v in per_rep]
    return out


def stability_section(cfgs, seen):
    rows = {}
    for k, c in cfgs.items():
        rows[k] = {**stability(c["canonical"]["results"]), "tools_seen": k in seen, "acc": mean(c["pf"])}
    by_path = {}
    for p in rc.PATHS:
        ks = [k for k in rows if k.endswith("/" + p)]
        agg = {f: sum(rows[k][f] for k in ks) for f in ("n_cases", "identical_cases", "grade_changes", "pairs",
                                                       "identical_pairs", "key_order_only_pairs", "text_pairs",
                                                       "text_pairs_first_char")}
        agg["configs"] = len(ks)
        agg["configs_per_rep_varies"] = sum(len(set(rows[k]["per_rep_acc"])) > 1 for k in ks)
        by_path[p] = agg
    return {"configs": rows, "by_path": by_path}


# ------------------------------------------------------------------ same GGUF
def same_gguf_section(cfgs, seen):
    rows = {}
    for m in sorted({k.split("/")[0] for k in cfgs}):
        a, b = cfgs.get(f"{m}/cpu"), cfgs.get(f"{m}/npu_llama")
        if not (a and b):
            continue
        row = {"same_file": (a["server"] or {}).get("artifact_sha256") == (b["server"] or {}).get("artifact_sha256"),
               "tools_seen": f"{m}/cpu" in seen and f"{m}/npu_llama" in seen}
        for s in ("canonical", "paraphrase"):
            pairs = [(x, y) for ca, cb in zip(a[s]["results"], b[s]["results"]) for x, y in zip(ca["reps"], cb["reps"])]
            row[s] = {"requests": len(pairs), "identical": sum(sig(x, False) == sig(y, False) for x, y in pairs),
                      "identical_keys_sorted": sum(sig(x) == sig(y) for x, y in pairs),
                      "same_grade": sum(ok(x) == ok(y) for x, y in pairs)}
        within_a, within_b, cross, det_both, det_same = [], [], [], 0, 0
        for ca, cb in zip(a["canonical"]["results"], b["canonical"]["results"]):
            sa, sb = [sig(r, False) for r in ca["reps"]], [sig(r, False) for r in cb["reps"]]
            within_a.append(mean(x == y for x, y in itertools.combinations(sa, 2)))
            within_b.append(mean(x == y for x, y in itertools.combinations(sb, 2)))
            cross.append(mean(x == y for x in sa for y in sb))
            if len(set(sa)) == 1 and len(set(sb)) == 1:
                det_both += 1
                det_same += sa[0] == sb[0]
        delta = [c - (x + y) / 2 for c, x, y in zip(cross, within_a, within_b)]
        row.update({"within_cpu": mean(within_a), "within_npu": mean(within_b), "cross": mean(cross),
                    "cross_minus_within": clean(mean(delta)), "cross_minus_within_ci": boot(delta),
                    "deterministic_both": det_both, "deterministic_both_same": det_same,
                    "acc_cpu": mean(a["pf"] + a["py"]), "acc_npu": mean(b["pf"] + b["py"])})
        for f in ("requests", "identical", "identical_keys_sorted", "same_grade"):
            row[f] = row["canonical"][f] + row["paraphrase"][f]
        rows[m] = row
    return rows


# ------------------------------------------------------------- hardest items
def answer(case, r):
    """What a failing reply did, compactly."""
    k = category(case, r)
    if k in ("text_call", "no_call", "other"):
        return k.replace("_", " ")
    a = r.get("got_args") if isinstance(r.get("got_args"), dict) else {}
    if k in ("cmd_as_tool", "other_unoffered"):
        return f"function {r['got_tool']}"
    if k == "wrong_device":
        return f"device {a.get('device_name')!r}"
    if k == "wrong_params":
        got = a.get("parameters") if isinstance(a.get("parameters"), dict) else {}
        bad = {p: got.get(p, "<missing>") for p, v in case["expected_params"].items()
               if norm(got.get(p, "")) != norm(v)}
        return f"{a.get('command_name')} with {json.dumps(bad)}"
    return f"{r['got_tool']}:{a['command_name']}" if a.get("command_name") else r["got_tool"]


def hardest_section(cfgs, readable_keys, pool, can, par, flagged, top=10):
    items = []
    for suite, cases, field in (("canonical", can, "pf"), ("paraphrase", par, "py")):
        for i, case in enumerate(cases):
            fails = [k for k in readable_keys if cfgs[k][field][i] < 0.5]
            alts = collections.Counter(answer(case, r) for k in pool
                                       for r in cfgs[k][suite]["results"][i]["reps"] if not ok(r))
            items.append({"suite": suite, "idx": i, "input": case["input"], "style": case.get("style"),
                          "expected": f"{case['expected_tool']}:{case['expected_command']}",
                          "readable_fail": len(fails), "pool_fail": sum(k in pool for k in fails),
                          "pool_pass": mean(cfgs[k][field][i] for k in pool),
                          "top_failing_answers": alts.most_common(3)})
    order = sorted(items, key=lambda x: (-x["readable_fail"], x["pool_pass"], x["suite"], x["idx"]))
    hardest = {s: [x for x in order if x["suite"] == s][:top] for s in ("canonical", "paraphrase")}
    p98 = next(x for x in items if x["suite"] == "paraphrase" and x["idx"] == ITEM_98)
    readable_98 = collections.Counter(answer(par[ITEM_98], r) for k in readable_keys
                                      for r in cfgs[k]["paraphrase"]["results"][ITEM_98]["reps"] if not ok(r))
    base_98 = next(c["idx"] for c in can if c["input"] == par[ITEM_98]["base_input"])
    fails = collections.Counter()
    diag = collections.Counter()
    diag_flagged = 0
    for k in pool:
        for j, c in enumerate(cfgs[k]["paraphrase"]["results"]):
            for r in c["reps"]:
                if ok(r):
                    continue
                fails[par[j]["style"]] += 1
                if r.get("got_tool") == "diagnose_port_issues":
                    diag[par[j]["style"]] += 1
                    diag_flagged += j in flagged
    return {"hardest": hardest, "n_readable": len(readable_keys), "n_pool": len(pool),
            "item_98": {**p98, "base_input": par[ITEM_98]["base_input"], "base_idx": base_98,
                        "base_pool_pass": mean(cfgs[k]["pf"][base_98] for k in pool),
                        "readable_pass": len(readable_keys) - p98["readable_fail"],
                        "readable_failing_answers": readable_98.most_common()},
            "pool_paraphrase_failures": {"by_style": dict(fails), "diagnose_by_style": dict(diag),
                                         "total": sum(fails.values()), "diagnose": sum(diag.values()),
                                         "diagnose_on_flagged": diag_flagged},
            "items": [{k: x[k] for k in ("suite", "idx", "readable_fail", "pool_pass")} for x in items]}


# ------------------------------------------------------------------ compute
def compute(results_dir):
    can, par, par_base, cfgs = load(results_dir)
    seen = [k for k in cfgs if tools_seen(cfgs[k])]
    readable_keys = [k for k in seen if readable(cfgs[k])]
    order = sorted(readable_keys, key=lambda k: (-mean(cfgs[k]["pf"]), k))
    pool = [k for k in order if mean(cfgs[k]["pf"]) >= FUNCTIONAL_MIN]
    labels = label_section(cfgs, order, pool, par, par_base)
    return {
        "method": {"bootstrap_resamples": rc.B, "seed": rc.SEED, "functional_min_canonical": FUNCTIONAL_MIN,
                   "interval": "95% percentile bootstrap", "gap_test": "exact two-sided sign test over base requests",
                   "style_test": "exact McNemar, style outcome vs canonical majority"},
        "pools": {"tools_seen": seen, "readable": order, "functional": pool},
        "gap": gap_section(cfgs, seen, pool, par_base),
        "styles": style_section(cfgs, seen, pool, can, par, par_base),
        "decomposition": decomposition_section(cfgs, order, pool),
        "levels": level_section(cfgs, order, pool, can, par),
        "labels": labels,
        "stability": stability_section(cfgs, seen),
        "same_gguf": same_gguf_section(cfgs, seen),
        "hardest": hardest_section(cfgs, order, pool, can, par, set(labels["flagged"])),
    }


# ------------------------------------------------------------------- tables
def p1(x):
    return "-" if x is None else f"{100 * x:.1f}"


def pp(x):
    return "-" if x is None else f"{100 * x:+.1f}"


def ci1(ci):
    return f"{p1(ci[0])}-{p1(ci[1])}"


def cipp(ci):
    return f"{pp(ci[0])} to {pp(ci[1])}"


def pv(x):
    return "-" if x is None else f"{x:.2g}"


def borderline_note(verdicts):
    names = {"ci": "the interval", "holm_pool": "pool Holm", "holm_seen": "tools-seen Holm"}
    yes = " and ".join(names[v] for v in names if verdicts[v])
    no = " or ".join(names[v] for v in names if not verdicts[v])
    return f"borderline: a loss by {yes}, not by {no}"


def gap_table(d):
    g, pool = d["gap"], d["pools"]["functional"]
    L = ["## Paraphrase minus canonical, paired by base request\n",
         f"Canonical: 65 cases x 3 repetitions; paraphrase: 195 items x 1, three per base request. "
         f"Interval: 95% percentile bootstrap over the 65 base requests ({d['method']['bootstrap_resamples']:,} "
         f"resamples). Test: exact sign test over base requests, Holm across the functional pool "
         f"({len(pool)} configurations) and across the {len(d['pools']['tools_seen'])} configurations whose "
         f"prompts carried the tools. Borderline: the interval and the two adjusted tests do not all agree; the "
         f"note says which of them show a loss.\n",
         "| Configuration | Canonical | Paraphrase | Gap (pp) | 95% CI (pp) | Bases worse / better / same | "
         "Sign p | Holm, pool | Holm, tools seen | Note |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k in d["pools"]["readable"]:
        r = g["configs"][k]
        note = borderline_note(r["verdicts"]) if r.get("borderline") else ("" if k in pool else "outside pool")
        L.append(f"| {name(k)} | {p1(r['canonical'])} | {p1(r['paraphrase'])} | {pp(r['gap'])} | {cipp(r['ci'])} | "
                 f"{r['bases_worse']} / {r['bases_better']} / {r['bases_same']} | {pv(r['p_sign'])} | "
                 f"{pv(r.get('p_holm_pool'))} | {pv(r['p_holm_seen'])} | {note} |")
    s = g["pool_significant"]
    hidden = [k for k in d["pools"]["tools_seen"] if k not in d["pools"]["readable"]]
    if hidden:
        L.append(f"\nNot shown, and counted only in the tools-seen family: {', '.join(name(k) for k in hidden)}, "
                 "which carried the tools but returned no structured call.")
    L.append(f"\nFunctional pool mean gap {pp(g['pool_mean_gap'])} pp (range {pp(g['pool_range'][0])} to "
             f"{pp(g['pool_range'][1])}). Loss established in {s['ci_below_zero']} of {s['n']} by the interval, "
             f"{s['sign_holm_pool']} of {s['n']} by the sign test with Holm across the pool, and "
             f"{s['sign_holm_seen']} of {s['n']} with Holm across the tools-seen configurations.\n")
    dec = d["decomposition"]
    L += ["## Paraphrase gap by failure category\n",
          "Change in each failure category's rate from canonical to paraphrase, in pp; negative values are "
          "losses. Rows add up to the gap. The last column counts paraphrases (of 195) whose call names the "
          "expected command as the function and would pass with the name moved into command_name "
          "(diagnostic only).\n",
          "| Configuration | Gap | " + " | ".join(CATEGORY_NAMES[c] for c in CATEGORIES) + " | Renamed would pass |",
          "|---|---|" + "---|" * len(CATEGORIES) + "---|"]
    for k in pool:
        r = dec["configs"][k]
        L.append(f"| {name(k)} | {pp(r['gap'])} | " + " | ".join(pp(r["contribution"][c]) for c in CATEGORIES)
                 + f" | {r['paraphrase_cmd_as_tool_renamed_pass']} |")
    L.append(f"| Functional pool mean | {pp(dec['pool_mean_gap'])} | "
             + " | ".join(pp(dec["pool_mean"][c]) for c in CATEGORIES) + " | |\n")
    return L


def style_table(d):
    s, pool, n_seen = d["styles"], d["pools"]["functional"], len(d["pools"]["tools_seen"])
    L = ["## Accuracy by paraphrase style\n",
         "Each style has 65 paraphrases, one per base request. Interval: 95% percentile bootstrap over items. "
         "Change: style minus canonical accuracy, pp. Holm: exact McNemar of style outcome against canonical "
         f"majority outcome, adjusted across the functional pool and the three styles ({3 * len(pool)} tests) and "
         f"across the {n_seen} configurations whose prompts carried the tools and the three styles "
         f"({3 * n_seen} tests).\n",
         "| Configuration | Canonical | " + " | ".join(f"{st} {STYLE_NAMES[st]}" for st in STYLES) + " | "
         + " | ".join(f"{st} change" for st in STYLES) + " | Holm p, pool, " + " / ".join(STYLES)
         + " | Holm p, tools seen, " + " / ".join(STYLES) + " |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k in pool:
        r = s["configs"][k]
        L.append(f"| {name(k)} | {p1(d['gap']['configs'][k]['canonical'])} | "
                 + " | ".join(f"{p1(r[st]['acc'])} ({ci1(r[st]['ci'])})" for st in STYLES) + " | "
                 + " | ".join(pp(r[st]["change"]) for st in STYLES) + " | "
                 + " / ".join(pv(r[st]["p_holm_pool"]) for st in STYLES) + " | "
                 + " / ".join(pv(r[st]["p_holm_seen"]) for st in STYLES) + " |")
    pr = s["pool"]
    L.append(f"| Functional pool, item means | {p1(pr['canonical']['acc'])} ({ci1(pr['canonical']['ci'])}) | "
             + " | ".join(f"{p1(pr[st]['acc'])} ({ci1(pr[st]['ci'])})" for st in STYLES) + " | "
             + " | ".join(f"{pp(pr[st]['mean_change'])} (range {pp(pr[st]['change_range'][0])} to "
                          f"{pp(pr[st]['change_range'][1])})" for st in STYLES) + " | "
             + " / ".join(f"{pr[st]['significant_holm_pool']}" for st in STYLES) + f" of {len(pool)} significant | "
             + " / ".join(f"{pr[st]['significant_holm_seen']}" for st in STYLES) + f" of {len(pool)} significant |")
    w = s["wording"]
    L += ["", "| Wording | Canonical | " + " | ".join(f"{st} {STYLE_NAMES[st]}" for st in STYLES) + " |",
          "|---|---|---|---|---|",
          "| Share of the base request's content words kept (%) | 100.0 | "
          + " | ".join(p1(w[st]["content_words_kept"]) for st in STYLES) + " |",
          f"| Mean words per request | {w['canonical']['mean_words']:.1f} | "
          + " | ".join(f"{w[st]['mean_words']:.1f}" for st in STYLES) + " |", ""]
    return L


def level_table(d):
    lv, pool = d["levels"], d["pools"]["functional"]
    levels = lv["levels"]
    head = " | ".join(f"{LEVEL_NAMES.get(x, x)} (n={lv['pool'][x]['n_canonical']})" for x in levels)
    phead = " | ".join(f"Para. {LEVEL_NAMES.get(x, x).split()[0]} (n={lv['pool'][x]['n_paraphrase']})" for x in levels)
    L = ["## Accuracy by canonical level\n",
         "Canonical accuracy per level (cases x 3 repetitions) with a 95% percentile bootstrap over cases, and "
         "paraphrase accuracy per level (items x 1). The pool row averages each item over the functional pool "
         "and bootstraps over items.\n",
         f"| Configuration | {head} | {phead} |", "|---|" + "---|" * (2 * len(levels))]
    for k in pool:
        r = lv["configs"][k]
        L.append(f"| {name(k)} | " + " | ".join(f"{p1(r[x]['canonical'])} ({ci1(r[x]['canonical_ci'])})" for x in levels)
                 + " | " + " | ".join(p1(r[x]["paraphrase"]) for x in levels) + " |")
    L.append("| Functional pool | " + " | ".join(f"{p1(lv['pool'][x]['canonical'])} ({ci1(lv['pool'][x]['canonical_ci'])})"
                                                for x in levels)
             + " | " + " | ".join(f"{p1(lv['pool'][x]['paraphrase'])} ({ci1(lv['pool'][x]['paraphrase_ci'])})"
                                  for x in levels) + " |\n")
    return L


def alt_text(alt):
    if alt["tool"] == "diagnose_port_issues":
        return f"diagnose_port_issues, port {alt['params']['port']}"
    return f"{alt['tool']}:{alt['command']}"


def label_table(d):
    lb, pool = d["labels"], d["pools"]["functional"]
    n_pool, n_read = len(pool), len(d["pools"]["readable"])
    L = ["## Debatable paraphrase labels\n",
         "From analysis/data/label_review.json. Flagged by: R = robustness review, G = grader audit (G* its wider list). "
         f"Pass and alternative counts are replies, over the functional pool ({n_pool}) and all readable "
         f"configurations ({n_read}).\n",
         "| Item | Style | Request | Label | Alternative | Flagged by | Pool pass / alternative | "
         "Readable pass / alternative |",
         "|---|---|---|---|---|---|---|---|"]
    for it in lb["items"]:
        flags = ", ".join(f for f in ("R" if it["robustness"] else "",
                                      {"primary": "G", "wider": "G*"}.get(it["grader"], "")) if f)
        L.append(f"| {it['idx']} | {it['style']} | {it['input']} | {it['expected']['command']} | "
                 f"{'; '.join(alt_text(a) for a in it['alternatives'])} | {flags} | "
                 f"{it['pool_pass']} / {it['pool_alt']} | {it['readable_pass']} / {it['readable_alt']} |")
    n_merged = lb["by_set"]["merged"]["n_items"]
    L += ["", f"Gap with the alternatives accepted on the {n_merged} items flagged by R or G (wider-list items "
          "excluded). Interval: 95% percentile bootstrap over the 65 base requests. The flag sets are compared "
          "in the table after this one.\n",
          "| Configuration | Paraphrase | Paraphrase, alternatives accepted | Gap (pp) | Gap, alternatives accepted (pp) | "
          "95% CI (pp) | Flagged items passed, as labelled / with alternatives |",
          "|---|---|---|---|---|---|---|"]
    for k in pool:
        r = lb["configs"][k]
        L.append(f"| {name(k)} | {p1(r['paraphrase'])} | {p1(r['paraphrase_alt'])} | {pp(r['gap'])} | "
                 f"{pp(r['gap_alt'])} | {cipp(r['gap_alt_ci'])} | {r['flagged_pass']:.0f} / {r['flagged_pass_alt']:.0f} |")
    L.append(f"| Functional pool mean | {p1(lb['pool_mean_paraphrase'])} | {p1(lb['pool_mean_paraphrase_alt'])} | "
             f"{pp(lb['pool_mean_gap'])} | {pp(lb['pool_mean_gap_alt'])} | | |")
    sets = {"robustness": "R (robustness review)", "grader": "G (grader audit)", "merged": "R or G (used above)",
            "merged_and_wider": "R, G or G*"}
    L += ["", f"Functional pool mean gap with the alternatives accepted, by flag set (as labelled "
          f"{pp(lb['pool_mean_gap'])} pp).\n",
          "| Flag set | Items | Pool mean gap, alternatives accepted (pp) |", "|---|---|---|"]
    L += [f"| {sets[s]} | {v['n_items']} | {pp(v['pool_mean_gap_alt'])} |" for s, v in lb["by_set"].items()]
    L.append("")
    return L


def stability_table(d):
    st = d["stability"]
    L = ["## Repetition stability, canonical suite\n",
         "65 cases x 3 repetitions sent in the same order, temperature 0 requested. Identical: the three replies "
         "match in tool name, arguments (keys sorted) and message text. Grade changes: cases whose end-to-end "
         "grade differs across repetitions. Pairwise: share of the 195 repetition pairs with identical replies.\n",
         "| Configuration | Tools seen | Accuracy | Identical (of 65) | Grade changes (of 65) | Accuracy by repetition | "
         "Pairwise identical (%) |",
         "|---|---|---|---|---|---|---|"]
    for p in rc.PATHS:
        for k in sorted((k for k in st["configs"] if k.endswith("/" + p)), key=lambda k: -st["configs"][k]["acc"]):
            r = st["configs"][k]
            L.append(f"| {name(k)} | {'yes' if r['tools_seen'] else 'no'} | {p1(r['acc'])} | {r['identical_cases']} | "
                     f"{r['grade_changes']} | {' / '.join(p1(x) for x in r['per_rep_acc'])} | "
                     f"{p1(r['identical_pairs'] / r['pairs'])} |")
    L += ["", "| Path | Configurations | Cases | Identical in all 3 | Grade changes | Configurations whose per-repetition "
          "accuracy varies | Identical pairs differing only in key order | Differing text pairs, differing at "
          "the first character |",
          "|---|---|---|---|---|---|---|---|"]
    for p, r in st["by_path"].items():
        L.append(f"| {rc.PATH_NAMES[p]} | {r['configs']} | {r['n_cases']} | {r['identical_cases']} "
                 f"({p1(r['identical_cases'] / r['n_cases'])}%) | {r['grade_changes']} | {r['configs_per_rep_varies']} | "
                 f"{r['key_order_only_pairs']} | {r['text_pairs_first_char']} of {r['text_pairs']} |")
    L.append("")
    return L


def same_gguf_table(d):
    L = ["## Same GGUF on the CPU and on the llama.cpp NPU backend\n",
         "Requests are matched by item and repetition (195 canonical repetitions and 195 paraphrases). The "
         "results keep the parsed call, not the reply bytes, so identical means the same tool name, the same "
         "arguments in the order the server sent them and the same message text; the next column sorts the "
         "argument keys first, as the stability table does. Within and across: mean share of identical reply "
         "pairs (server key order) per canonical case, over the 3 pairs within a path and the 9 pairs across "
         "paths; the interval is a 95% percentile bootstrap over the 65 cases.\n",
         "| Model | Same file | Tools seen | Identical replies | Identical, keys sorted | Identical grades | "
         "Canonical identity within CPU / "
         "within NPU / across | Across minus within (pp, 95% CI) | Deterministic on both paths: same reply | "
         "Pooled accuracy CPU / NPU |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in d["same_gguf"].items():
        L.append(f"| {rc.MODEL_NAMES.get(m, m)} | {'yes' if r['same_file'] else 'no'} | {'yes' if r['tools_seen'] else 'no'} | "
                 f"{r['identical']} of {r['requests']} ({p1(r['identical'] / r['requests'])}%) | "
                 f"{r['identical_keys_sorted']} ({p1(r['identical_keys_sorted'] / r['requests'])}%) | "
                 f"{r['same_grade']} of {r['requests']} ({p1(r['same_grade'] / r['requests'])}%) | "
                 f"{p1(r['within_cpu'])} / {p1(r['within_npu'])} / {p1(r['cross'])} | "
                 f"{pp(r['cross_minus_within'])} ({cipp(r['cross_minus_within_ci'])}) | "
                 f"{r['deterministic_both_same']} of {r['deterministic_both']} | {p1(r['acc_cpu'])} / {p1(r['acc_npu'])} |")
    L.append("")
    return L


def hardest_table(d):
    h = d["hardest"]
    L = ["## Hardest items\n",
         f"Ranked by the number of readable configurations ({h['n_readable']}) that fail the item, then by the "
         f"functional-pool pass rate ({h['n_pool']} configurations; canonical cases count their 3 repetitions). "
         "Failing answers: the commonest replies among failing pool repetitions.\n",
         "| Suite | Item | Style | Request | Label | Readable failing | Pool pass | Commonest failing answers in the pool |",
         "|---|---|---|---|---|---|---|---|"]
    for s in ("paraphrase", "canonical"):
        for x in h["hardest"][s]:
            alts = "; ".join(f"{a} ({n})" for a, n in x["top_failing_answers"])
            L.append(f"| {s} | {x['idx']} | {x['style'] or '-'} | {x['input']} | {x['expected']} | "
                     f"{x['readable_fail']} of {h['n_readable']} | {p1(x['pool_pass'])} | {alts} |")
    i98, f = h["item_98"], h["pool_paraphrase_failures"]
    L.append(f"\nParaphrase {i98['idx']} ({i98['input']}) is labelled {i98['expected']} and passes in "
             f"{i98['readable_pass']} of {h['n_readable']} readable configurations; their failing replies are "
             + ", ".join(f"{a} ({n})" for a, n in i98["readable_failing_answers"])
             + f". Its base request, canonical case {i98['base_idx']} ({i98['base_input']}), passes "
             f"{p1(i98['base_pool_pass'])}% in the functional pool. Of {f['total']} failing pool paraphrase replies, "
             f"{f['diagnose']} ({p1(f['diagnose'] / f['total'])}%) call diagnose_port_issues "
             f"(A {f['diagnose_by_style'].get('A', 0)}, B {f['diagnose_by_style'].get('B', 0)}, "
             f"C {f['diagnose_by_style'].get('C', 0)}), {f['diagnose_on_flagged']} of them on the flagged items.\n")
    return L


def tables_md(data):
    L = []
    for part in (gap_table, style_table, level_table, label_table, stability_table, same_gguf_table, hardest_table):
        L += part(data)
    return "\n".join(L)


if __name__ == "__main__":
    print(tables_md(compute(sys.argv[1] if len(sys.argv) > 1 else "results/v2")))
