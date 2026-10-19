"""Shared loading and statistics for the report modules (analysis/report_*.py).

Everything here reads a results directory laid out as results/v2 is
(<model>/<path>/<file>.json) and the suites in suites/. The statistics reuse
analysis/analyze.py where it already has the function, so the report's numbers
agree with analysis/out/v2.json.
"""
import glob
import json
import math
import os
import random
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "harness"))
import analyze  # noqa: E402
import yaml  # noqa: E402  (PyYAML, which the harness also needs)

SEED = analyze.SEED
B = analyze.B_RESAMPLES
PATHS = ("cpu", "npu_llama", "compiled")
PATH_NAMES = {"cpu": "CPU (llama.cpp)", "npu_llama": "NPU (llama.cpp)", "compiled": "NPU (compiled)"}
SUBSET = ("query_device_info", "configure_device_setting", "diagnose_port_issues", "search_documentation")

# Display names, in the order models are listed in campaign/v2_models.json.
MODEL_NAMES = {
    "qwen3-0.6b": "Qwen3-0.6B", "qwen3-1.7b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B",
    "qwen3-4b-2507": "Qwen3-4B-Instruct-2507", "qwen3-8b": "Qwen3-8B",
    "gemma-4-e4b": "Gemma-4-E4B-it", "smollm2-1.7b": "SmolLM2-1.7B-Instruct",
    "phi-4-mini": "Phi-4-mini-instruct", "granite-4.0-micro": "Granite-4.0-H-Micro",
    "ministral-3-3b": "Ministral-3-3B-Instruct-2512", "gemma-4-e2b": "Gemma-4-E2B-it",
    "qwen3.5-0.8b": "Qwen3.5-0.8B", "qwen3.5-2b": "Qwen3.5-2B",
}


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def spec():
    """The campaign's model list (campaign/v2_models.json)."""
    return load_json(os.path.join(REPO, "campaign", "v2_models.json"))


def suites():
    """(canonical, paraphrase) case lists in harness order, each case a dict
    with 'idx' and 'level' added; paraphrases keep base_input and style."""
    out = []
    for name in ("test_cases.yaml", "test_cases_paraphrase.yaml"):
        data = yaml.safe_load(open(os.path.join(REPO, "suites", name)))
        cases = []
        for level, items in data.items():
            for it in items:
                cases.append({**it, "level": level})
        for i, c in enumerate(cases):
            c["idx"] = i
        out.append(cases)
    return out[0], out[1]


_CMD = re.compile(r"\b((?:show|set|enable|disable|configure|clear|reset|restart|reboot|save|run|add|remove|"
                  r"delete|create|get|test|ping|apply|update|change)_[a-z0-9_]+)\b")


def tool_schema():
    """{tool name: set of command names}. The schema names commands in each
    tool's description (suites/tools_nested.json), so they are read from there
    and from the expected commands of both suites."""
    tools = load_json(os.path.join(REPO, "suites", "tools_nested.json"))
    out = {}
    for t in tools:
        fn = t.get("function", t)
        props = fn.get("parameters", {}).get("properties", {})
        text = fn.get("description", "") + " " + json.dumps(props)
        out[fn["name"]] = set(_CMD.findall(text)) - set(out) - {fn["name"]}
    for cases in suites():
        for c in cases:
            if c.get("expected_tool") and c.get("expected_command"):
                out.setdefault(c["expected_tool"], set()).add(c["expected_command"])
    return out


def command_names():
    """Every command name the schema or the suites mention."""
    return set().union(*tool_schema().values())


def configs(results_dir):
    """Every <model>/<path> directory under results_dir, as dicts holding the
    parsed files that exist (canonical, paraphrase, probe_agent, probe_chat,
    diag_tools_in_prompt, server, load_failure)."""
    out = []
    for d in sorted(glob.glob(os.path.join(results_dir, "*", "*"))):
        if not os.path.isdir(d):
            continue
        model, path = d.split(os.sep)[-2:]
        c = {"model": model, "path": path, "dir": d}
        for name in ("canonical", "paraphrase", "probe_agent", "probe_chat", "diag_tools_in_prompt",
                     "server", "load_failure", "workaround_tests"):
            c[name] = load_json(os.path.join(d, name + ".json"))
        out.append(c)
    return out


def items(cfg, grade_fn=analyze.e2e_ok, suites_=("canonical", "paraphrase")):
    """{item id: pass fraction} over the chosen suites, with item ids
    'canonical:<idx>' and 'paraphrase:<idx>' as analyze.py uses them.
    grade_fn(rep, case) -> bool lets a caller re-grade under another rule;
    it is called with (rep) when it takes one argument."""
    out = {}
    for s in suites_:
        d = cfg.get(s)
        if not d:
            continue
        for c in d["results"]:
            reps = c.get("reps") or []
            if not reps:
                continue
            try:
                oks = [grade_fn(r, c) for r in reps]
            except TypeError:
                oks = [grade_fn(r) for r in reps]
            out[f"{s}:{c['idx']}"] = sum(bool(x) for x in oks) / len(oks)
    return out


_BASE = None


def item_bases():
    """{item id: base canonical input}, so the four items built on one base
    request (the canonical case and its three paraphrases) form one cluster."""
    global _BASE
    if _BASE is None:
        can, par = suites()
        _BASE = {f"canonical:{c['idx']}": c["input"] for c in can}
        _BASE.update({f"paraphrase:{c['idx']}": c["base_input"] for c in par})
    return _BASE


def cluster_bootstrap(item_vals, b=B, seed=SEED, alpha=0.05, stat=None):
    """Percentile CI of the item mean, resampling base requests (clusters)
    instead of items. item_vals: {item id: value}."""
    bases = item_bases()
    groups = {}
    for k, v in item_vals.items():
        groups.setdefault(bases.get(k, k), []).append(v)
    keys = sorted(groups)
    if not keys:
        return None, None
    rng = random.Random(seed)
    n = len(keys)
    stats = []
    for _ in range(b):
        vals = []
        for _ in range(n):
            vals.extend(groups[keys[rng.randrange(n)]])
        stats.append(statistics.mean(vals) if stat is None else stat(vals))
    stats.sort()
    return stats[int(math.floor(alpha / 2 * b))], stats[min(b - 1, int(math.ceil((1 - alpha / 2) * b)) - 1)]


def paired_diff_ci(a, b_, b=B, seed=SEED, alpha=0.05, cluster=False):
    """Mean of (b - a) over shared items, with a percentile bootstrap CI that
    resamples items jointly (or base-request clusters when cluster=True)."""
    common = sorted(set(a) & set(b_))
    diffs = {k: b_[k] - a[k] for k in common}
    if not diffs:
        return None, (None, None)
    mean = statistics.mean(diffs.values())
    if cluster:
        return mean, cluster_bootstrap(diffs, b=b, seed=seed, alpha=alpha)
    return mean, analyze.bootstrap_ci(list(diffs.values()), b=b, seed=seed, alpha=alpha)


def mcnemar_items(a, b_):
    """Exact McNemar on majority outcomes of shared items: (only a, only b, p)."""
    common = sorted(set(a) & set(b_))
    only_a = sum(1 for k in common if a[k] > 0.5 and not b_[k] > 0.5)
    only_b = sum(1 for k in common if b_[k] > 0.5 and not a[k] > 0.5)
    return only_a, only_b, analyze.mcnemar_exact(only_a, only_b)


def mde_paired(discordant_rate, n, alpha=0.05, power=0.8):
    """Minimum detectable paired difference (normal approximation to McNemar)
    for a given rate of discordant items and n items."""
    z_a = statistics.NormalDist().inv_cdf(1 - alpha / 2)
    z_b = statistics.NormalDist().inv_cdf(power)
    return (z_a + z_b) * math.sqrt(discordant_rate / n)


def tools_seen(cfg):
    """True when the median server-reported prompt of both suites was long
    enough to hold the tool definitions (analyze.py's threshold); None when a
    suite is missing."""
    meds = []
    for s in ("canonical", "paraphrase"):
        d = cfg.get(s)
        if not d:
            return None
        toks = [r["prompt_tokens"] for c in d["results"] for r in c.get("reps") or []
                if isinstance(r.get("prompt_tokens"), (int, float)) and r["prompt_tokens"] > 0]
        if not toks:
            return None
        meds.append(statistics.median(toks))
    return all(m >= analyze.TOOLS_DELIVERED_MIN_PROMPT_TOKENS for m in meds)


def unreadable_reason(cfg):
    """Why a measured configuration's score says nothing about the model, or
    None. 'tools not delivered': the prompts never contained the tool
    definitions. 'calls returned as text': no reply was a structured call while
    most replies held a text tool call, the model's own call format that the
    server did not parse."""
    if tools_seen(cfg) is False:
        return "tools not delivered"
    reps = [r for s in ("canonical", "paraphrase") if cfg.get(s)
            for c in cfg[s]["results"] for r in c.get("reps") or []]
    structured = sum(bool(r.get("got_tool")) for r in reps)
    text = sum(bool(r.get("text_tool_call") or (not r.get("got_tool")
                                                and analyze.parse_text_tool_call(r.get("content"))[0]))
               for r in reps)
    return "calls returned as text" if structured == 0 and text * 2 > len(reps) else None


def readable(cfg):
    """A measured configuration whose score can be read as model capability."""
    return bool(cfg.get("canonical")) and unreadable_reason(cfg) is None


def holm(ps):
    return analyze.holm(ps)
