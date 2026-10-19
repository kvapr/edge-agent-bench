#!/usr/bin/env python3
"""Failures, format defects and the grading sensitivity of accuracy (version 2).

Reads a results directory laid out as results/v2, the suites and
analysis/data/grader_audit.json, and computes:

Error taxonomy. Every failing repetition (analyze.e2e_ok false) of the two
native suites gets one category, the first that applies in this order:
  request_error     the request failed, or the reply has neither a call nor text
  text_parseable    no structured call; analyze.parse_text_tool_call recovers one
                    (Hermes JSON, XML-parameter or Mistral form)
  text_malformed    no structured call; the text holds a call whose JSON is not
                    valid as written (short of closing braces, or unrecoverable)
  text_other_shape  no structured call; valid JSON or call syntax that the server
                    did not convert: untagged or fenced JSON, tool_name keys or
                    wrapper objects, a Python-literal dict, Gemma call syntax, or
                    call arguments without a tool name
  text_pseudo       no structured call; a tool or command name followed by
                    arguments that are not JSON
  prose             no call of any kind
  command_as_tool   structured call whose function name is a documented command
  other_tool        structured call to another name outside the offered tools
  wrong_tool        one of the four offered tools, but not the expected one
  wrong_command     right tool; command_name wrong or missing, or no JSON arguments
  wrong_device      right tool and command; device_name is not poesw1
  wrong_params      right tool, command and device; the lenient check fails
Counts are repetitions (a canonical case counts three times) and shares are of
failing repetitions, per configuration and suite, per configuration, and
pooled over all configurations and over the 20 whose prompts carried the tools
(median prompt of at least analyze.TOOLS_DELIVERED_MIN_PROMPT_TOKENS in each
suite).

Closing braces. In every reply without a structured call, the first object
that opens with {"name" (tagged or bare) is cut at the next </tool_call> or at
the end of the text, and read as valid JSON as written, valid after appending
exactly one '}', or other. Nothing else is repaired. Variants reported beside
it: a final ')' read as the missing '}', and a reply that repeats the call,
read up to the start of the second call. The balance is closing minus opening
braces in the cut text. In the diagnostic files the balance is counted over
the whole reply.

Command names as tools. The share of repetitions whose structured call names
a command (report_common.command_names()) as the function, item-weighted with
a percentile bootstrap CI over items, compared between the compiled bundle and
each GGUF path of the same model by exact McNemar on per-item majority (Holm
over the pairs), and pooled over the five Qwen3 models on model-qualified
items. Arguments are flat when they carry neither command_name nor a
parameters object.

Sensitivity of accuracy (diagnostic only; the as-served score is the result).
Per readable configuration (report_common.readable),
pooled end to end under four readings: as served; with conservative brace
repair (a text call that is valid as written or after one appended '}');
brace repair plus renaming (a function named after a documented command
becomes that command's tool, with command_name set to the name unless already
present); and the grader audit's semantic rules (port numbers, '@', misspelt
device keys, flat parameters, command synonyms and command-named functions;
text calls still fail). Compiled minus CPU for the five Qwen3 models under
each reading, with a paired item bootstrap CI and exact McNemar (Holm over the
five models). Ministral-3-3B intent: its text calls parsed by
analyze.parse_text_tool_call and graded, with item and base-request cluster CIs.

Tools-in-prompt diagnostics. Each diag_tools_in_prompt.json as recorded, and
with tolerant decoding: where the recorded parser found no call, the first
JSON object in the reply is read with json raw_decode and accepted when
nothing or exactly one surplus '}' follows it. Both are compared with the
native canonical run by exact McNemar on per-case majority.

Grader audit. analysis/data/grader_audit.json holds a seeded stratified sample of
repetitions with blind verdicts (A exact, B right action off convention,
C wrong, D defensible other reading). False-negative rates among failures and
false-positive rates among passes come with Wilson 95% intervals, unweighted
and weighted by inverse sampling probability (Kish effective n). Agreement
between the semantic rules and the blind A/B verdicts is Cohen's kappa.

Every bootstrap uses report_common.SEED and report_common.B.

    python3 analysis/report_errors.py [results_dir]
"""
import ast
import copy
import json
import math
import os
import re
import statistics
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import report_common as rc  # noqa: E402
from report_common import analyze  # noqa: E402

grade = analyze.grade
e2e_ok = analyze.e2e_ok
AUDIT_FILE = os.path.join(HERE, "data", "grader_audit.json")

OFFERED = set(rc.SUBSET)
SCHEMA = rc.tool_schema()
CMD_TOOL = {cmd: tool for tool, cmds in SCHEMA.items() for cmd in cmds}
COMMANDS = set(CMD_TOOL)
QWEN3 = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-4b-2507", "qwen3-8b")
NATIVE = ("canonical", "paraphrase")
RULES = ("served", "brace", "brace_rename", "semantic")
RULE_NAMES = {"served": "As served", "brace": "Brace repair", "brace_rename": "Brace repair and rename",
              "semantic": "Semantic rules"}
CATEGORIES = ("request_error", "text_parseable", "text_malformed", "text_other_shape", "text_pseudo", "prose",
              "command_as_tool", "other_tool", "wrong_tool", "wrong_command", "wrong_device", "wrong_params")
CAT_NAMES = {"request_error": "Request error or empty", "text_parseable": "Text call, parseable",
             "text_malformed": "Text call, malformed JSON", "text_other_shape": "Text call, other shape",
             "text_pseudo": "Text call, pseudo-code", "prose": "Prose, no call", "command_as_tool": "Command as tool",
             "other_tool": "Other tool name", "wrong_tool": "Wrong tool (of 4)", "wrong_command": "Wrong command",
             "wrong_device": "Wrong device", "wrong_params": "Wrong params"}
DEC = json.JSONDecoder()


# ------------------------------------------------------------ statistics
_CI = {}


def ci(values):
    """analyze.bootstrap_ci, memoised on the value sequence (the seed is fixed,
    so equal inputs give equal intervals)."""
    key = tuple(values)
    if key not in _CI:
        _CI[key] = list(analyze.bootstrap_ci(list(values), b=rc.B, seed=rc.SEED))
    return _CI[key]


def rate(item_vals):
    vals = list(item_vals.values())
    return {"rate": statistics.mean(vals), "ci": ci(vals), "n_items": len(vals)}


def wilson(p, n, z=1.959963984540054):
    """Wilson score interval for a proportion p observed on n (possibly an
    effective, non-integer n)."""
    if not n:
        return [None, None]
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    # at p = 0 or 1 the bound is exact; avoid a rounding residue
    return [0.0 if p == 0 else max(0.0, c - h), 1.0 if p == 1 else min(1.0, c + h)]


def kappa(pairs):
    """Cohen's kappa for a list of (rater 1, rater 2) booleans."""
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    p1, p2 = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return (po - pe) / (1 - pe) if pe < 1 else None


def mcnemar(a, b):
    """Exact McNemar on per-item majority: {'a_only', 'b_only', 'p'}."""
    a_only, b_only, p = rc.mcnemar_items(a, b)
    return {"a_only": a_only, "b_only": b_only, "p": p}


def add_holm(rows, key="p"):
    for row, adj in zip(rows, rc.holm([r[key] for r in rows])):
        row["p_holm"] = adj
    return rows


# ------------------------------------------------------------ configurations
def native_configs(results_dir):
    """Configurations with both native suites, in results-directory order, each
    with a tools_seen flag decided per suite as analyze.pooled does."""
    out = []
    for c in rc.configs(results_dir):
        if not all(c.get(s) for s in NATIVE):
            continue
        c["tools_seen"] = bool(rc.tools_seen(c))
        out.append(c)
    return out


def reps(cfg, suites=NATIVE):
    """(suite, case, rep) for every repetition of the chosen suites."""
    for s in suites:
        for case in (cfg.get(s) or {}).get("results", []):
            for r in case.get("reps") or []:
                yield s, case, r


def label(cfg):
    return f"{rc.MODEL_NAMES.get(cfg['model'], cfg['model'])}, {rc.PATH_NAMES.get(cfg['path'], cfg['path'])}"


# ------------------------------------------------------------ text replies
_FENCE = re.compile(r"```(?:json|JSON|python|tool_call)?\s*\n?(.*?)(?:```|$)", re.S)
_GEMMA = re.compile(r"(?:call:)?([A-Za-z_][A-Za-z0-9_]*)\s*\{")
_NAMED = re.compile(r'"(?:name|tool_name)"\s*:\s*"([A-Za-z0-9_]+)"')
_END_MARKERS = ("</tool_call>", "</think>", "</s>", "<|im_end|>", "<tool_call>", "```")
_ARG_KEYS = {"device_name", "command_name", "command", "commands", "parameters", "params", "arguments", "port",
             "vlan_id", "power_limit", "speed", "duplex", "mode", "description", "action", "vlan_list",
             "time_range", "query", "doc_type", "device_model", "max_results"}
_KNOWN = sorted(OFFERED | set(SCHEMA) | COMMANDS, key=lambda n: (-len(n), n))  # longest name first
_PSEUDO = re.compile(r"^(?:@\S+[\s,:-]*)?[`\"']?(" + "|".join(map(re.escape, _KNOWN)) + r")\b")


def _decode_region(region):
    """(object, defect) for the first JSON value in region. The defect is None
    for valid JSON, 'short' when the text cut at the next end marker parses
    after appending up to three braces, 'literal' for a Python-literal dict,
    and 'unrecoverable' otherwise."""
    i, j = region.find("{"), region.find("[")
    if i < 0 or (0 <= j < i and region[j + 1:].lstrip().startswith("{")):
        i = j if j >= 0 else i
    try:
        return DEC.raw_decode(region, i)[0], None
    except json.JSONDecodeError:
        pass
    body = region[i:]
    for mk in _END_MARKERS:
        p = body.find(mk, 1)
        if p > 0:
            body = body[:p]
    body = body.rstrip()
    last = body.rfind("}")
    if last < 0:
        return None, "unrecoverable"
    for k in (1, 2, 3):
        try:
            return json.loads(body[:last + 1] + "}" * k), "short"
        except json.JSONDecodeError:
            continue
    # small models sometimes write a Python dict; only when it opens the reply
    if re.fullmatch(r"\s*(@\S+[\s,:-]*)?", region[:i]):
        try:
            obj = ast.literal_eval(body[:last + 1])
            if isinstance(obj, (dict, list)):
                return obj, "literal"
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            pass
    return None, "unrecoverable"


def _call_name(obj):
    """Tool name of a decoded call object, through a list or wrapper object."""
    if isinstance(obj, list):
        obj = obj[0] if obj and isinstance(obj[0], dict) else None
    if not isinstance(obj, dict):
        return None
    for wrap in ("tool_call", "function_call", "function"):
        if isinstance(obj.get(wrap), dict):
            obj = obj[wrap]
            break
    return next((obj[k] for k in ("name", "tool_name", "tool", "function") if isinstance(obj.get(k), str)), None)


def _call_arguments(obj):
    """A decoded object without a tool name that still reads as call arguments."""
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        obj = obj[0]
    if not isinstance(obj, dict) or not obj:
        return False
    keys = set(obj)
    return bool(keys & {"command_name", "command", "commands"} or keys & OFFERED or keys & COMMANDS
                or keys <= _ARG_KEYS)


def _gemma_args_ok(content):
    """None if the reply has no Gemma-style call, else whether its arguments decode."""
    m = _GEMMA.search(content)
    if not m:
        return None
    body = content[m.end() - 1:].replace('<|"|>', '"')
    body = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', body)
    try:
        return isinstance(DEC.raw_decode(body)[0], dict)
    except json.JSONDecodeError:
        return False


def text_category(content):
    """Category of a non-empty reply that carries no structured call."""
    if analyze.parse_text_tool_call(content)[0]:
        return "text_parseable"
    text = content
    m = _FENCE.search(text)
    if m and "{" in m.group(1):
        text = m.group(1)
    if '<|"|>' in content:
        ok = _gemma_args_ok(content)
        if ok is not None:
            return "text_other_shape" if ok else "text_malformed"
    tagged = "<tool_call>" in text
    region = text.split("<tool_call>", 1)[1] if tagged else text
    if "{" in region or "[" in region:
        obj, defect = _decode_region(region)
        if obj is not None:
            if _call_name(obj) or _call_arguments(obj):
                return "text_malformed" if defect == "short" else "text_other_shape"
            return "prose"  # a JSON object that is data, not a call
        if tagged or _NAMED.search(region):
            return "text_malformed"
    if _PSEUDO.search(content.strip()):
        return "text_pseudo"
    return "prose"


def category(r):
    """Taxonomy category of a failing repetition (None for a pass)."""
    if e2e_ok(r):
        return None
    content = r.get("content") or ""
    if r.get("error") or (not r.get("got_tool") and not content.strip()):
        return "request_error"
    name = r.get("got_tool")
    if not name:
        return text_category(content)
    if name not in OFFERED:
        return "command_as_tool" if name in COMMANDS else "other_tool"
    if not r.get("ok_tool"):
        return "wrong_tool"
    if r.get("got_args") is None or not r.get("ok_command"):
        return "wrong_command"
    if not r.get("ok_device"):
        return "wrong_device"
    return "wrong_params"


# ------------------------------------------------------------ brace reading
_HERMES = re.compile(r'\{\s*"name"\s*:')
_TRAIL = re.compile(r"(\s|</?tool_call>|</think>)+$")  # tags left between a call and its repeat


def hermes_call(content):
    """One extraction rule for every reply: (class, object, cut text) for the
    first {"name" object, cut at the next </tool_call> or the end. Class is
    'valid', 'short_one' (valid after one appended brace) or 'other'; (None,
    None, None) when the reply has no such object."""
    m = _HERMES.search(content or "")
    if not m:
        return None, None, None
    end = content.find("</tool_call>", m.start())
    seg = content[m.start():end if end >= 0 else len(content)].strip()
    for cls, text in (("valid", seg), ("short_one", seg + "}")):
        try:
            return cls, json.loads(text), seg
        except json.JSONDecodeError:
            pass
    return "other", None, seg


def _loads_ok(text):
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def call_of(obj):
    """(name, arguments) of a decoded {"name", "arguments"} object."""
    if not isinstance(obj, dict):
        return None, None
    args = obj.get("arguments", obj.get("parameters"))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = None
    return obj.get("name"), args if isinstance(args, dict) else None


def brace_table(cfg):
    """Hermes-shaped text replies of one configuration, by reading."""
    out = {"hermes_shaped": 0, "untagged": 0, "valid": 0, "short_one": 0, "other": 0,
           "paren_for_brace": 0, "repeated_call_short_one": 0, "balance": Counter()}
    for _s, _c, r in reps(cfg):
        if r.get("got_tool"):
            continue
        cls, _obj, seg = hermes_call(r.get("content"))
        if cls is None:
            continue
        out["hermes_shaped"] += 1
        out["untagged"] += "<tool_call>" not in r["content"]
        out[cls] += 1
        out["balance"][seg.count("}") - seg.count("{")] += 1
        if cls == "other":
            second = _HERMES.search(seg, 1)
            if seg.endswith(")") and _loads_ok(seg[:-1] + "}"):
                out["paren_for_brace"] += 1
            elif second and _loads_ok(_TRAIL.sub("", seg[:second.start()]) + "}"):
                out["repeated_call_short_one"] += 1
    out["balance"] = {str(k): v for k, v in sorted(out["balance"].items())}
    return out


# ------------------------------------------------------------ readings
def served_ok(r, case):
    return e2e_ok(r)


def recovered_call(r):
    """The structured call, else a text call valid as written or after one brace."""
    if r.get("got_tool"):
        return r["got_tool"], r.get("got_args")
    cls, obj, _ = hermes_call(r.get("content"))
    return call_of(obj) if cls in ("valid", "short_one") else (None, None)


def rename(name, args):
    """A function named after a documented command becomes that command's
    tool, with command_name set to the name unless already present."""
    if not isinstance(name, str) or name not in CMD_TOOL or not isinstance(args, dict):
        return name, args
    return CMD_TOOL[name], {"command_name": name, **args}


def brace_ok(r, case):
    if r.get("got_tool"):
        return e2e_ok(r)
    name, args = recovered_call(r)
    return bool(name) and e2e_ok(grade(case, name, args))


def brace_rename_ok(r, case):
    name, args = recovered_call(r)
    if not name:
        return False
    return e2e_ok(grade(case, *rename(name, args)))


# The grader audit's synonym map: self-explaining names seen in the data,
# mapped to the documented command (and the PoE mode they imply).
SYNONYMS = {
    "show_all_interfaces": "show_all_ports_status", "show_interface_status": "show_port_status",
    "get_port_status": "show_port_status", "query_system_info": "show_system_info",
    "show_poe_budget": "show_poe_power_budget", "show_all_poe_power_budget": "show_poe_power_budget",
    "query_vlan_config": "show_vlan_config", "vlan config": "show_vlan_config",
    "configure_port_description": "set_port_description", "set_interface_duplex": "configure_port_duplex",
    "set_port_duplex": "configure_port_duplex", "set_port_speed": "configure_port_speed",
    "configure_interface_speed": "configure_port_speed", "configure_vlan": "configure_access_vlan",
    "assign_vlan": "configure_access_vlan", "assign_port_vlan": "configure_access_vlan",
    "disable_port (port)": "disable_port",
    "enable_poe": ("configure_poe_mode", "enable"), "enable_poe_on_port": ("configure_poe_mode", "enable"),
    "enable_poe_port": ("configure_poe_mode", "enable"), "disable_poe": ("configure_poe_mode", "disable"),
    "disable_poe_mode": ("configure_poe_mode", "disable"), "disable_poe_port": ("configure_poe_mode", "disable"),
    "disable_poe_on_port": ("configure_poe_mode", "disable"),
}
_PORT = re.compile(r"^\s*(?:(?:port|interface|gi|gigabitethernet|ge)\s*)?(\d{1,2})\s*$", re.I)


def canon_cmd(cmd):
    """(documented command, implied PoE mode or None) under the synonym map."""
    v = SYNONYMS.get(cmd.strip()) if isinstance(cmd, str) else None
    if v is None:
        return cmd, None
    return (v, None) if isinstance(v, str) else v


def norm_port(v):
    if isinstance(v, int) and not isinstance(v, bool):
        return f"1/{v}"
    m = _PORT.match(v) if isinstance(v, str) else None
    return f"1/{int(m.group(1))}" if m else v


def semantic_ok(r, case):
    """The grader audit's semantic rule set over a structured call: a bare
    port N reads as '1/N', a leading '@' is dropped, 'devices_name' and
    'devices' count as device_name, parameters placed at the top level count,
    command synonyms map to the documented command, and a function named after
    a command is graded as that command's tool. Text calls fail."""
    tool, args = r.get("got_tool"), r.get("got_args")
    if r.get("error") or not tool:
        return False
    if not isinstance(args, dict):
        return e2e_ok(grade(case, tool, args))
    args = copy.deepcopy(args)
    dev = next((args[k] for k in ("device_name", "devices_name", "devices") if k in args), "")
    if isinstance(dev, list) and dev:
        dev = dev[0]
    args["device_name"] = dev.strip().lstrip("@") if isinstance(dev, str) else dev
    mode = None
    if tool not in OFFERED:
        own, own_mode = canon_cmd(args.get("command_name"))
        cmd, mode = (own, own_mode) if isinstance(own, str) and own in CMD_TOOL else canon_cmd(tool)
        tool = CMD_TOOL.get(cmd, tool) if isinstance(cmd, str) else tool
        args["command_name"] = cmd
    cmd, syn_mode = canon_cmd(args.get("command_name"))
    args["command_name"] = cmd
    mode = syn_mode or mode
    params = args.get("parameters")
    top = {k: v for k, v in args.items() if k not in ("device_name", "command_name", "parameters", "devices_name",
                                                        "devices")}
    params = {**top, **(params if isinstance(params, dict) else {})}
    if mode and "mode" not in params:
        params["mode"] = mode
    if "port" in params:
        params["port"] = norm_port(params["port"])
    args["parameters"] = params
    return e2e_ok(grade(case, tool, args))


READINGS = {"served": served_ok, "brace": brace_ok, "brace_rename": brace_rename_ok, "semantic": semantic_ok}


def intent_ok(r, case):
    """analyze.py's intent reading: a text call that its parser recovers is graded."""
    if r.get("got_tool"):
        return e2e_ok(r)
    name, args = analyze.parse_text_tool_call(r.get("content"))
    return bool(name) and e2e_ok(grade(case, name, args))


# ------------------------------------------------------------ sections
def taxonomy(cfgs):
    def block(rs):
        fails = [category(r) for r in rs]
        fails = [c for c in fails if c]
        cnt = Counter(fails)
        return {"n_reps": len(rs), "n_fail": len(fails), "counts": {c: cnt.get(c, 0) for c in CATEGORIES},
                "shares": {c: (cnt.get(c, 0) / len(fails) if fails else None) for c in CATEGORIES}}

    rows, all_reps, seen_reps = [], [], []
    for cfg in cfgs:
        by_suite = {s: [r for _s, _c, r in reps(cfg, (s,))] for s in NATIVE}
        pooled = by_suite["canonical"] + by_suite["paraphrase"]
        rows.append({"model": cfg["model"], "path": cfg["path"], "tools_seen": cfg["tools_seen"],
                     "suites": {s: block(v) for s, v in by_suite.items()}, "pooled": block(pooled)})
        all_reps += pooled
        if cfg["tools_seen"]:
            seen_reps += pooled
    return {"categories": list(CATEGORIES), "configs": rows,
            "all": {**block(all_reps), "n_configs": len(cfgs)},
            "tools_seen": {**block(seen_reps), "n_configs": sum(c["tools_seen"] for c in cfgs)}}


def braces(cfgs):
    rows = []
    for cfg in cfgs:
        t = brace_table(cfg)
        if t["hermes_shaped"]:
            rows.append({"model": cfg["model"], "path": cfg["path"], **t})
    q = [r for r in rows if r["model"] in QWEN3]
    keys = ("hermes_shaped", "untagged", "valid", "short_one", "other", "paren_for_brace", "repeated_call_short_one")
    qwen = {k: sum(r[k] for r in q) for k in keys}
    bal = Counter()
    for r in q:
        bal.update({int(k): v for k, v in r["balance"].items()})
    qwen["balance"] = {str(k): v for k, v in sorted(bal.items())}
    return {"configs": rows, "qwen3": qwen}


def diag_braces(results_dir):
    rows = []
    for cfg in rc.configs(results_dir):
        d = cfg.get("diag_tools_in_prompt")
        if not d:
            continue
        bal, surplus_fail = Counter(), 0
        for case in d["results"]:
            for r in case["reps"]:
                text = r.get("content") or ""
                b = text.count("}") - text.count("{")
                bal[b] += 1
                surplus_fail += b == 1 and not e2e_ok(r)
        rows.append({"model": cfg["model"], "path": cfg["path"], "n_replies": sum(bal.values()),
                     "surplus_one": bal.get(1, 0), "surplus_one_failing": surplus_fail,
                     "balance": {str(k): v for k, v in sorted(bal.items())}})
    return rows


def _is_cmd(r, case):
    return r.get("got_tool") in COMMANDS


def _flat(args):
    return not ("command_name" in (args or {}) or isinstance((args or {}).get("parameters"), dict))


def command_as_tool(cfgs):
    rows, items = [], {}
    for cfg in cfgs:
        calls = [r for _s, _c, r in reps(cfg) if r.get("got_tool")]
        cmd = [r for r in calls if r["got_tool"] in COMMANDS]
        it = rc.items(cfg, _is_cmd)
        items[(cfg["model"], cfg["path"])] = it
        rows.append({"model": cfg["model"], "path": cfg["path"], "tools_seen": cfg["tools_seen"],
                     "n_reps": sum(1 for _ in reps(cfg)), "structured_calls": len(calls), "cmd_as_tool": len(cmd),
                     "flat": sum(_flat(r.get("got_args")) for r in cmd), **rate(it)})
    tests = []
    for m in sorted({c["model"] for c in cfgs if c["path"] == "compiled"}):
        for p in ("cpu", "npu_llama"):
            if (m, "compiled") in items and (m, p) in items:
                tests.append({"model": m, "gguf_path": p, **mcnemar(items[(m, "compiled")], items[(m, p)])})
    add_holm(tests)
    pooled = {}
    for p in ("compiled", "cpu"):
        q = {f"{c['model']}|{k}": v for c in cfgs if c["model"] in QWEN3 and c["path"] == p
             for k, v in items[(c["model"], p)].items()}
        sel = [r for r in rows if r["model"] in QWEN3 and r["path"] == p]
        pooled[p] = {"cmd_as_tool": sum(r["cmd_as_tool"] for r in sel), "n_reps": sum(r["n_reps"] for r in sel),
                     "flat": sum(r["flat"] for r in sel), "_items": q, **rate(q)}
    pooled["test"] = mcnemar(pooled["compiled"].pop("_items"), pooled["cpu"].pop("_items"))
    flat = {}
    for name, sel in (("compiled", lambda r: r["path"] == "compiled"), ("gguf", lambda r: r["path"] != "compiled"),
                      ("qwen3_compiled", lambda r: r["model"] in QWEN3 and r["path"] == "compiled"),
                      ("qwen3_cpu", lambda r: r["model"] in QWEN3 and r["path"] == "cpu")):
        s = [r for r in rows if sel(r)]
        flat[name] = {"cmd_as_tool": sum(r["cmd_as_tool"] for r in s), "flat": sum(r["flat"] for r in s)}
    return {"configs": rows, "compiled_vs_gguf": tests, "qwen3_pooled": pooled, "flat_arguments": flat}


def sensitivity(cfgs):
    readable = [c for c in cfgs if rc.readable(c)]
    items, rows = {}, []
    for cfg in readable:
        row = {"model": cfg["model"], "path": cfg["path"]}
        for rule in RULES:
            it = rc.items(cfg, READINGS[rule])
            items[(cfg["model"], cfg["path"], rule)] = it
            passing = sum(READINGS[rule](r, case) for _s, case, r in reps(cfg))
            row[rule] = {**rate(it), "passing_reps": passing}
        rows.append(row)
    diffs = {}
    for rule in RULES:
        fam = []
        for m in QWEN3:
            cpu, comp = items.get((m, "cpu", rule)), items.get((m, "compiled", rule))
            if cpu is None or comp is None:
                continue
            mean, (lo, hi) = rc.paired_diff_ci(cpu, comp)
            t = mcnemar(comp, cpu)
            fam.append({"model": m, "compiled": statistics.mean(comp.values()), "cpu": statistics.mean(cpu.values()),
                        "diff": mean, "ci": [lo, hi], "compiled_only": t["a_only"], "cpu_only": t["b_only"],
                        "p": t["p"]})
        diffs[rule] = add_holm(fam)
    return {"readable": [f"{c['model']}/{c['path']}" for c in readable], "configs": rows,
            "compiled_minus_cpu": diffs}


def ministral(cfgs):
    cfg = next((c for c in cfgs if c["model"] == "ministral-3-3b" and c["path"] == "cpu"), None)
    if cfg is None:
        return None
    out = {}
    for name, suites in (("pooled", NATIVE), ("canonical", ("canonical",)), ("paraphrase", ("paraphrase",))):
        it = rc.items(cfg, intent_ok, suites)
        out[name] = {**rate(it), "cluster_ci": list(rc.cluster_bootstrap(it)),
                     "passing_reps": sum(intent_ok(r, case) for _s, case, r in reps(cfg, suites)),
                     "n_reps": sum(1 for _ in reps(cfg, suites)),
                     "served": statistics.mean(rc.items(cfg, served_ok, suites).values())}
    return out


def tolerant_call(content):
    """The first JSON object of the reply (inside <tool_call> when tagged), read
    with raw_decode and accepted when nothing or one surplus '}' follows it."""
    text = content or ""
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    if m:
        text = m.group(1)
    i = text.find("{")
    if i < 0:
        return None, None
    try:
        obj, end = DEC.raw_decode(text, i)
    except json.JSONDecodeError:
        return None, None
    if text[end:].strip() not in ("", "}") or not isinstance(obj, dict):
        return None, None
    fn = obj.get("function") if isinstance(obj.get("function"), dict) else {}
    name = fn.get("name") or obj.get("name") or obj.get("tool_name")
    args = fn.get("arguments") or obj.get("arguments") or obj.get("parameters") or obj.get("args")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = None
    return name, args if isinstance(args, dict) else None


def tolerant_ok(r, case):
    if r.get("got_tool"):
        return e2e_ok(r)
    name, args = tolerant_call(r.get("content"))
    return bool(name) and e2e_ok(grade(case, name, args))


def diagnostics(results_dir):
    rows = []
    for cfg in rc.configs(results_dir):
        if not cfg.get("diag_tools_in_prompt") or not cfg.get("canonical"):
            continue
        # the diagnostic re-runs the canonical suite; its items share the native ids
        dcfg = {"canonical": cfg["diag_tools_in_prompt"]}
        native = rc.items(cfg, served_ok, ("canonical",))
        rec, tol = rc.items(dcfg, served_ok, ("canonical",)), rc.items(dcfg, tolerant_ok, ("canonical",))
        rescued = sum(tolerant_ok(r, c) and not e2e_ok(r) for _s, c, r in reps(dcfg, ("canonical",)))
        rows.append({"model": cfg["model"], "path": cfg["path"], "native": rate(native),
                     "recorded": rate(rec), "tolerant": rate(tol), "rescued_reps": rescued,
                     "vs_native_recorded": mcnemar(native, rec), "vs_native_tolerant": mcnemar(native, tol)})
    return rows


def audit_category(r):
    """The audit's mechanical category: the first failing check in grading order."""
    if e2e_ok(r):
        return "pass"
    if r.get("error"):
        return "error"
    if not r.get("got_tool"):
        name, _ = analyze.parse_text_tool_call(r.get("content"))
        return "text_call" if name or r.get("text_tool_call") else "no_call"
    if not r.get("ok_tool"):
        return "wrong_tool_offered" if r["got_tool"] in OFFERED else "wrong_tool_unoffered"
    if r.get("got_args") is None:
        return "bad_args"
    if not r.get("ok_command"):
        return "wrong_command"
    if not r.get("ok_params_lenient"):
        return "params"
    return "device"


def grader_audit(cfgs, audit_file=AUDIT_FILE):
    audit = rc.load_json(audit_file)
    if not audit:
        return None
    lookup, pop = {}, Counter()
    semantic_rescued = n_fail = 0
    for cfg in (c for c in cfgs if c["tools_seen"]):
        for s in NATIVE:
            for case in cfg[s]["results"]:
                for i, r in enumerate(case["reps"]):
                    cat = audit_category(r)
                    lookup[(cfg["model"], cfg["path"], s, case["idx"], i + 1)] = (case, r, cat)
                    pop[(cat, cfg["model"], cfg["path"])] += 1
                    if cat != "pass":
                        n_fail += 1
                        semantic_rescued += semantic_ok(r, case)
    items = []
    for it in audit["items"]:
        case, r, cat = lookup[(it["model"], it["path"], it["suite"], it["idx"], it["rep"])]
        items.append({**it, "recomputed_category": cat, "semantic": semantic_ok(r, case)})
    # each item stands for N/n repetitions of its (category, configuration) cell
    drawn = Counter((it["mechanical_category"], it["model"], it["path"]) for it in items)
    for it in items:
        cell = (it["mechanical_category"], it["model"], it["path"])
        it["weight"] = pop[cell] / drawn[cell]
    fails = [it for it in items if it["mechanical_category"] != "pass"]
    passes = [it for it in items if it["mechanical_category"] == "pass"]

    def share(pool, codes):
        k = sum(it["verdict"] in codes for it in pool)
        sw = sum(it["weight"] for it in pool)
        est = sum(it["weight"] for it in pool if it["verdict"] in codes) / sw
        n_eff = sw ** 2 / sum(it["weight"] ** 2 for it in pool)
        return {"k": k, "n": len(pool), "share": k / len(pool), "wilson": wilson(k / len(pool), len(pool)),
                "weighted": est, "weighted_wilson": wilson(est, n_eff), "n_eff": n_eff}

    agree = [(it["verdict"] in "AB", it["semantic"]) for it in items]
    by_cat, population = {}, Counter()
    for it in items:
        by_cat.setdefault(it["mechanical_category"], Counter())[it["verdict"]] += 1
    for (cat, _m, _p), n in pop.items():
        population[cat] += n
    return {
        "seed": audit["seed"], "n_items": len(items), "n_failures": len(fails), "n_passes": len(passes),
        "n_configs": len({(it["model"], it["path"]) for it in items}),
        "category_mismatches": sum(it["mechanical_category"] != it["recomputed_category"] for it in items),
        "population": dict(sorted(population.items())),
        "verdicts_by_category": {k: dict(sorted(v.items())) for k, v in sorted(by_cat.items())},
        "fn_exact": share(fails, "A"), "fn_semantic": share(fails, "AB"), "fn_broad": share(fails, "ABD"),
        "fp_wrong": share(passes, "C"), "fp_off_convention": share(passes, "B"),
        "fp_not_exact": share(passes, "BCD"),
        "agreement": {"n": len(agree), "agree": sum(a == b for a, b in agree), "kappa": kappa(agree),
                      "disagreements": [{"id": it["id"], "verdict": it["verdict"], "semantic": it["semantic"],
                                         "reason": it["reason"]}
                                        for it, (a, b) in zip(items, agree) if a != b]},
        "exhaustive_semantic": {"rescued": semantic_rescued, "n_fail": n_fail, "share": semantic_rescued / n_fail},
    }


def compute(results_dir):
    cfgs = native_configs(results_dir)
    return {
        "taxonomy": taxonomy(cfgs),
        "braces": braces(cfgs),
        "diag_braces": diag_braces(results_dir),
        "command_as_tool": command_as_tool(cfgs),
        "sensitivity": sensitivity(cfgs),
        "ministral_intent": ministral(cfgs),
        "diagnostics": diagnostics(results_dir),
        "grader_audit": grader_audit(cfgs),
        "method": {"bootstrap_resamples": rc.B, "seed": rc.SEED, "interval": "percentile bootstrap over items",
                   "test": "exact McNemar on per-item majority, Holm within each family"},
    }


# ------------------------------------------------------------------ output
def pc(x, nd=1):
    return "-" if x is None else f"{100 * x:.{nd}f}"


def pci(d):
    return f"{pc(d['rate'])} ({pc(d['ci'][0])}-{pc(d['ci'][1])})"


def pct_ci(d):
    return f"{pc(d['rate'])}% (95% CI {pc(d['ci'][0])}-{pc(d['ci'][1])})"


def pv(p):
    return "-" if p is None else f"{p:.3g}"


def ordered(rows):
    """Rows in the campaign's model order, then cpu, npu_llama, compiled."""
    models = list(rc.MODEL_NAMES)
    return sorted(rows, key=lambda r: (models.index(r["model"]) if r["model"] in models else len(models),
                                       rc.PATHS.index(r["path"]) if r["path"] in rc.PATHS else len(rc.PATHS)))


def _taxonomy_md(t):
    cats = [c for c in CATEGORIES if t["all"]["counts"][c]]
    L = ["## Error taxonomy\n",
         "Failing repetitions by category, both suites pooled (65 canonical cases x 3 repetitions plus 195 "
         "paraphrases x 1, so 390 repetitions per configuration). Each cell is a count and its share of the "
         "configuration's failing repetitions in percent. Categories are mutually exclusive and assigned in "
         "column order; the module docstring defines them."
         + ("" if len(cats) == len(CATEGORIES) else " No request errors or empty replies occurred, so that "
            "column is omitted.") + "\n",
         "| Configuration | Tools seen | Failing reps | " + " | ".join(CAT_NAMES[c] for c in cats) + " |",
         "|---|---|---|" + "---|" * len(cats)]

    def cells(b):
        return " | ".join(f"{b['counts'][c]} ({pc(b['shares'][c])})" if b["counts"][c] else "0" for c in cats)

    for r in ordered(t["configs"]):
        b = r["pooled"]
        L.append(f"| {label(r)} | {'yes' if r['tools_seen'] else 'no'} | {b['n_fail']} of {b['n_reps']} | "
                 f"{cells(b)} |")
    for key, lab in (("all", "All configurations"), ("tools_seen", "Configurations with tools seen")):
        b = t[key]
        L.append(f"| {lab} ({b['n_configs']}) | | {b['n_fail']} of {b['n_reps']} | {cells(b)} |")
    L += ["", "## Error taxonomy by suite\n",
          "Counts of failing repetitions by category, per configuration and suite.\n",
          "| Configuration | Suite | Reps | Failing | " + " | ".join(CAT_NAMES[c] for c in cats) + " |",
          "|---|---|---|---|" + "---|" * len(cats)]
    for r in ordered(t["configs"]):
        for s, b in r["suites"].items():
            L.append(f"| {label(r)} | {s} | {b['n_reps']} | {b['n_fail']} | "
                     + " | ".join(str(b["counts"][c]) for c in cats) + " |")
    return L


def _balance(d):
    return ", ".join(f"{int(k):+d}: {v}" if int(k) else f"0: {v}" for k, v in d.items())


def _braces_md(b, diag):
    q = b["qwen3"]
    L = ["", "## Closing braces in text tool calls\n",
         "Replies without a structured call whose first object opens with {\"name\" (tagged or bare), cut at the "
         "next </tool_call> or the end of the reply. A reply counts as valid after one brace only if appending "
         "exactly one '}' makes it valid JSON; nothing else is repaired. The two right-hand counts are variants "
         "within 'Other': a final ')' read as the missing '}', and a reply that repeats the call, read up to the "
         "second call. Balance is closing minus opening braces in the cut text (count of replies).\n",
         "| Configuration | Hermes-shaped replies | Untagged | Valid as written | Valid after one '}' | Other | "
         "Other: ')' for '}' | Other: repeated call, first one '}' short | Brace balance |",
         "|---|---|---|---|---|---|---|---|---|"]
    rows = ordered(b["configs"]) + [{"label": "Qwen3, five models, all paths", **q}]
    for r in rows:
        L.append(f"| {r.get('label') or label(r)} | {r['hermes_shaped']} | {r['untagged']} | {r['valid']} | "
                 f"{r['short_one']} | {r['other']} | {r['paren_for_brace']} | {r['repeated_call_short_one']} | "
                 f"{_balance(r['balance'])} |")
    n = q["hermes_shaped"]
    if n:
        k = q["short_one"] + q["repeated_call_short_one"]
        L += ["", f"Of the {n} Hermes-shaped Qwen3 replies, {q['valid']} are valid as written and {q['short_one']} "
              f"({pc(q['short_one'] / n)}%) are valid after one appended brace. With the first of a repeated call "
              f"isolated, {k} ({pc(k / n)}%) are one closing brace short. Reading a final ')' as the brace adds "
              f"{q['paren_for_brace']}."]
    L += ["", "## Closing braces in the tools-in-prompt diagnostics\n",
          "Closing minus opening braces over the whole reply, every repetition of each diagnostic file.\n",
          "| Configuration | Replies | One surplus '}' | Of which failing as recorded | Brace balance |",
          "|---|---|---|---|---|"]
    for r in ordered(diag):
        L.append(f"| {label(r)} | {r['n_replies']} | {r['surplus_one']} | {r['surplus_one_failing']} | "
                 f"{_balance(r['balance'])} |")
    return L


def _command_md(c):
    L = ["", "## Command names called as tools\n",
         "Structured calls whose function name is a documented command instead of one of the four offered tools. "
         "The rate is item-weighted over 260 items (canonical cases averaged over their 3 repetitions), with a 95% "
         f"percentile bootstrap CI over items ({rc.B:,} resamples, seed {rc.SEED}). Flat arguments carry neither "
         "command_name nor a parameters object.\n",
         "| Configuration | Tools seen | Structured calls | Command as tool, reps | Rate % (95% CI) | Flat arguments |",
         "|---|---|---|---|---|---|"]
    for r in ordered(c["configs"]):
        L.append(f"| {label(r)} | {'yes' if r['tools_seen'] else 'no'} | {r['structured_calls']} | "
                 f"{r['cmd_as_tool']} of {r['n_reps']} | {pci(r)} | {r['flat']} |")
    L += ["", "Compiled bundle against each GGUF path of the same model: items (of 260) where the command-as-tool "
          "call is the majority outcome on one path only, exact McNemar, Holm over the "
          f"{len(c['compiled_vs_gguf'])} pairs. For Gemma-4-E4B the compiled prompts lacked the tools, and for "
          "SmolLM2-1.7B the GGUF prompts did, so those two rows are confounded.\n",
          "| Model | GGUF path | Compiled only | GGUF only | p | p (Holm) |", "|---|---|---|---|---|---|"]
    for t in ordered([{**t, "path": t["gguf_path"]} for t in c["compiled_vs_gguf"]]):
        L.append(f"| {rc.MODEL_NAMES.get(t['model'], t['model'])} | {rc.PATH_NAMES[t['gguf_path']]} | "
                 f"{t['a_only']} | {t['b_only']} | {pv(t['p'])} | {pv(t['p_holm'])} |")
    q, f = c["qwen3_pooled"], c["flat_arguments"]
    L += ["", "Pooled over the five Qwen3 models (1,300 model-items per path), the item-weighted rate is "
          f"{pct_ci(q['compiled'])} compiled against {pct_ci(q['cpu'])} on the CPU. The repetition counts are "
          f"{q['compiled']['cmd_as_tool']} and {q['cpu']['cmd_as_tool']} command-named calls, each in "
          f"{q['compiled']['n_reps']:,} repetitions. Compiled-only {q['test']['a_only']}, CPU-only "
          f"{q['test']['b_only']} model-items, exact McNemar p {pv(q['test']['p'])} (model-items treated as "
          "independent).\n",
          "| Calls naming a command as the tool | Calls | Flat arguments | Share flat % |", "|---|---|---|---|"]
    for key, lab in (("compiled", "All compiled configurations"), ("gguf", "All GGUF configurations"),
                     ("qwen3_compiled", "Five Qwen3 models, compiled"), ("qwen3_cpu", "Five Qwen3 models, CPU")):
        n, k = f[key]["cmd_as_tool"], f[key]["flat"]
        L.append(f"| {lab} | {n} | {k} | {pc(k / n) if n else '-'} |")
    return L


def _sensitivity_md(s, m):
    L = ["", "## Accuracy under alternative readings (diagnostic)\n",
         "Pooled end to end over 260 items, % with 95% item bootstrap CI, for each readable configuration (tools "
         "seen and at least one structured call). The as-served score is the result; the other columns show how "
         "much of it rests on the form of the reply. Brace repair reads a text call that is valid JSON as written "
         "or after one appended '}'. Rename maps a function named after a documented command to that command's "
         "tool. Semantic rules are the grader audit's rule set over structured calls only.\n",
         "| Configuration | " + " | ".join(RULE_NAMES[r] for r in RULES) + " |", "|---|" + "---|" * len(RULES)]
    for r in ordered(s["configs"]):
        L.append(f"| {label(r)} | " + " | ".join(pci(r[k]) for k in RULES) + " |")
    L += ["", "## Compiled against CPU under each reading\n",
          "Compiled minus CPU in percentage points, paired item bootstrap CI over 260 items; items passing on one "
          "path only (per-item majority), exact McNemar, Holm over the five models within each reading.\n",
          "| Model | Reading | Compiled % | CPU % | Difference pp (95% CI) | Compiled only | CPU only | p | p (Holm) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for rule in RULES:
        for t in ordered([{**t, "path": "compiled"} for t in s["compiled_minus_cpu"][rule]]):
            L.append(f"| {rc.MODEL_NAMES.get(t['model'], t['model'])} | {RULE_NAMES[rule]} | {pc(t['compiled'])} | "
                     f"{pc(t['cpu'])} | {100 * t['diff']:+.1f} ({100 * t['ci'][0]:+.1f} to {100 * t['ci'][1]:+.1f}) | "
                     f"{t['compiled_only']} | {t['cpu_only']} | {pv(t['p'])} | {pv(t['p_holm'])} |")
    if m:
        L += ["", "## Ministral-3-3B intent (diagnostic)\n",
              "The server returned Ministral-3-3B's native [TOOL_CALLS] calls as text, so it scores 0.0% as served. "
              "Intent parses those calls with analyze.parse_text_tool_call and grades them with the harness "
              "grader. Item CIs resample items; cluster CIs resample the 65 base requests.\n",
              "| Suite | Items | Repetitions passing | As served % | Intent % | Item 95% CI | Cluster 95% CI |",
              "|---|---|---|---|---|---|---|"]
        for k in ("pooled", "canonical", "paraphrase"):
            d = m[k]
            L.append(f"| {k} | {d['n_items']} | {d['passing_reps']} of {d['n_reps']} | {pc(d['served'])} | "
                     f"{pc(d['rate'])} | {pc(d['ci'][0])}-{pc(d['ci'][1])} | "
                     f"{pc(d['cluster_ci'][0])}-{pc(d['cluster_ci'][1])} |")
    return L


def _diagnostics_md(rows):
    L = ["", "## Tools-in-prompt diagnostics\n",
         "Canonical suite (65 cases x 3 repetitions) with the tool definitions written into the system prompt, "
         "beside the native canonical run of the same configuration; % with 95% bootstrap CI over cases. Tolerant "
         "decoding reads a reply the recorded parser rejected with json raw_decode, accepting one surplus '}'. "
         "Paired counts are cases passing (per-case majority) in the native run only and in the diagnostic only, "
         "exact McNemar. A diagnostic score shows whether the model can make the calls when it sees the tools; "
         "it is neither a lower nor an upper bound on the native score.\n",
         "| Configuration | Native % | Recorded % | Tolerant % | Reps rescued | Native only / diagnostic only, "
         "recorded | p | Native only / diagnostic only, tolerant | p |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in ordered(rows):
        a, b = r["vs_native_recorded"], r["vs_native_tolerant"]
        L.append(f"| {label(r)} | {pci(r['native'])} | {pci(r['recorded'])} | {pci(r['tolerant'])} | "
                 f"{r['rescued_reps']} | {a['a_only']} / {a['b_only']} | {pv(a['p'])} | {b['a_only']} / "
                 f"{b['b_only']} | {pv(b['p'])} |")
    return L


def _audit_md(g):
    if not g:
        return []

    def row(lab, d):
        return (f"| {lab} | {d['k']} of {d['n']} | {pc(d['share'])} | {pc(d['wilson'][0])}-{pc(d['wilson'][1])} | "
                f"{pc(d['weighted'])} ({pc(d['weighted_wilson'][0])}-{pc(d['weighted_wilson'][1])}, "
                f"n_eff {d['n_eff']:.0f}) |")

    a, e = g["agreement"], g["exhaustive_semantic"]
    L = ["", "## Grader audit\n",
         f"A stratified sample (seed {g['seed']}) of {g['n_failures']} failing and {g['n_passes']} passing "
         f"repetitions from the {g['n_configs']} configurations whose prompts carried the tools, judged blind by "
         "one rater from the request and the reply (A exact; B the requested action, off convention or off "
         "schema; C not the requested action or no structured call; D a defensible other reading). Intervals are "
         "Wilson 95%; the weighted column reweights each item by the inverse of its sampling probability within "
         f"its mechanical category and configuration (Kish effective n). Sampled categories match the "
         f"recomputed ones for {g['n_items'] - g['category_mismatches']} of {g['n_items']} items.\n",
         "| Quantity | Count | Share % | Wilson 95% | Weighted % (Wilson 95%) |", "|---|---|---|---|---|",
         row("False negatives: failures judged exact (A)", g["fn_exact"]),
         row("Failures judged A or B (right action, wrong form)", g["fn_semantic"]),
         row("Failures judged A, B or D", g["fn_broad"]),
         row("False positives: passes judged wrong (C)", g["fp_wrong"]),
         row("Passes judged off convention (B)", g["fp_off_convention"]),
         row("Passes judged not exact (B, C or D)", g["fp_not_exact"]),
         "", f"The semantic rules agree with the blind A/B verdicts on {a['agree']} of {a['n']} items "
         f"(Cohen's kappa {a['kappa']:.3f}). Over every failure in those configurations the semantic rules "
         f"rescue {e['rescued']} of {e['n_fail']} ({pc(e['share'])}%).\n",
         "| Mechanical category | Population | Sampled | A | B | C | D |", "|---|---|---|---|---|---|---|"]
    for cat, v in g["verdicts_by_category"].items():
        L.append(f"| {cat} | {g['population'].get(cat, 0)} | {sum(v.values())} | "
                 + " | ".join(str(v.get(x, 0)) for x in "ABCD") + " |")
    return L


def tables_md(data):
    L = _taxonomy_md(data["taxonomy"]) + _braces_md(data["braces"], data["diag_braces"])
    L += _command_md(data["command_as_tool"]) + _sensitivity_md(data["sensitivity"], data["ministral_intent"])
    L += _diagnostics_md(data["diagnostics"]) + _audit_md(data["grader_audit"])
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    print(tables_md(compute(sys.argv[1] if len(sys.argv) > 1 else os.path.join(rc.REPO, "results", "v2"))))
