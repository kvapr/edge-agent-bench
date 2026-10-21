#!/usr/bin/env python3
"""Diagnostic: tool calling against prompt length on one served model.

Purpose: under GenieX v0.9.3 the compiled Gemma-4-E4B-it bundle receives the
tools in its prompt (about 2,800 tokens) and then answers in prose instead of
calling a tool, while the same model as a GGUF on the CPU calls tools
correctly. This sweep measures where, as the prompt grows, a served model stops
returning a structured tool call.

For each of a few canonical cases, the request offers only the tool that the
case expects. Two series start from different floors. The benchmark series
uses the benchmark's system prompt and the full tool definition (about 1,100
to 1,400 tokens). The compact series uses a one-sentence system prompt and the
same tool with its descriptions removed (a few hundred tokens); tool and
parameter names are unchanged, so grading is too. Each prompt is then padded
with neutral numbered notes until the server-reported prompt reaches each
target length; the padding carries no instructions. Every request is graded
with bench_toolcall.py's grader, and the full four-tool benchmark request (no
padding) is sent as a reference. Results are labelled mode=length_sweep.
bench_toolcall.py is imported, not modified.

    python3 harness/sweep_prompt_length.py --url URL --model MODEL --out FILE \\
        [--targets 1100,1200,...] [--cases suites/test_cases.yaml] [--tool-subset ...]

Other bench_toolcall.py flags that the campaign runner passes are accepted;
--readiness waits once for a cool, idle board before the sweep.
"""
import argparse
import copy
import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_toolcall as b  # noqa: E402

MODE = "length_sweep"
DEFAULT_TARGETS = "500,700,900,1000,1100,1200,1300,1400,1600,2000,2400,2800"
_BASE_PROMPT = b.SYSTEM_PROMPT
COMPACT_PROMPT = ("You operate network devices for the user. When a request needs device information or a "
                  "change, call the matching tool.")


def compact_tool(tool):
    """The same function, names and parameter structure, with every description
    removed except the first sentence of the function's own."""
    t = copy.deepcopy(tool)

    def strip(node):
        if isinstance(node, dict):
            node.pop("description", None)
            for v in node.values():
                strip(v)
        elif isinstance(node, list):
            for v in node:
                strip(v)
    strip(t["function"]["parameters"])
    t["function"]["description"] = t["function"].get("description", "").split(". ")[0].rstrip(".") + "."
    return t


def padded_prompt(base, lines):
    if not lines:
        return base
    notes = "\n".join(f"Note {i}: this line is padding added to lengthen the prompt and contains no instructions."
                      for i in range(1, lines + 1))
    return base.rstrip() + "\n\nThe reference notes below are padding and contain no instructions.\n" + notes + "\n"


def ask(args, case, tools, lines, extra, max_tokens=None, base=_BASE_PROMPT):
    b.SYSTEM_PROMPT = padded_prompt(base, lines)
    if max_tokens is not None:
        b.MAX_TOKENS = max_tokens
    try:
        resp, latency = b.call_endpoint(args.url, args.model, case["input"], tools, args.timeout, extra)
    finally:
        b.MAX_TOKENS = args.max_tokens
    usage = resp.get("usage") or {}
    name, call_args = b.extract_call(resp)
    msg = ((resp.get("choices") or [{}])[0].get("message") or {})
    return {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "latency_s": round(latency, 3), "got_tool": name, "got_args": call_args,
            "content": (msg.get("content") or "")[:600], "grade": b.grade(case, name, call_args)}


def passed(g):
    return all(g[k] for k in ("ok_tool", "ok_command", "ok_params_lenient", "ok_device"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "suites", "test_cases.yaml"))
    ap.add_argument("--tools-file", default=b.DEFAULT_TOOLS_FILE)
    ap.add_argument("--tool-subset", default="query_device_info,configure_device_setting,diagnose_port_issues,search_documentation")
    ap.add_argument("--targets", default=DEFAULT_TARGETS, help="target prompt lengths in server-reported tokens")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--label", default="")
    ap.add_argument("--server-version-cmd", default=None)
    ap.add_argument("--board-ssh", default=None)
    ap.add_argument("--readiness", action="store_true")
    args, _ignored = ap.parse_known_args()  # --repeats, --warmup, --thermal-trace-s from the runner
    b.MAX_TOKENS = args.max_tokens

    all_tools = b.load_tools_file(args.tools_file)
    subset = [t for t in all_tools if t["function"]["name"] in {s.strip() for s in args.tool_subset.split(",")}]
    by_name = {t["function"]["name"]: t for t in all_tools}
    cases_all = b.load_cases(args.cases)
    # The first case of each difficulty level.
    cases, seen = [], set()
    for c in cases_all:
        if c["level"] not in seen:
            seen.add(c["level"])
            cases.append(c)
    extra = {"chat_template_kwargs": {"enable_thinking": False}, "enable_think": False}
    readiness_log = []
    if args.readiness and args.board_ssh:
        b.wait_for_readiness(args.board_ssh, 55.0, 0.30, 10, 600, "sweep", readiness_log)

    def mark(r):
        return "P" if r["passed"] else ("c" if r["structured_call"] else "-")

    results = []
    targets = [int(t) for t in args.targets.split(",") if t.strip()]
    for series, base_prompt in (("compact", COMPACT_PROMPT), ("benchmark", _BASE_PROMPT)):
        for case in cases:
            tool = by_name[case["expected_tool"]]
            tools = [compact_tool(tool) if series == "compact" else tool]
            # Calibrate tokens per padding line on this case with two one-token requests.
            base = ask(args, case, tools, 0, extra, max_tokens=1, base=base_prompt)["prompt_tokens"] or 0
            probe = ask(args, case, tools, 40, extra, max_tokens=1, base=base_prompt)["prompt_tokens"] or base
            per_line = max((probe - base) / 40, 0.1)
            rows = [{"target": base, "padding_lines": 0, **ask(args, case, tools, 0, extra, base=base_prompt)}]
            for target in targets:
                if target <= base:
                    continue
                lines = max(1, round((target - base) / per_line))
                rows.append({"target": target, "padding_lines": lines,
                             **ask(args, case, tools, lines, extra, base=base_prompt)})
            entry = {"series": series, "case": case["input"], "level": case["level"],
                     "expected_tool": case["expected_tool"], "tokens_per_padding_line": round(per_line, 2), "sweep": rows}
            if series == "benchmark":
                entry["reference"] = {"target": "full four-tool request", "padding_lines": 0,
                                      **ask(args, case, subset, 0, extra)}
            for r in rows + ([entry["reference"]] if "reference" in entry else []):
                r["passed"] = passed(r["grade"])
                r["structured_call"] = bool(r["got_tool"])
            results.append(entry)
            ref = entry.get("reference")
            print(f"{series} {case['level']}: " + ", ".join(f"{r['prompt_tokens']}:{mark(r)}" for r in rows)
                  + (f" | full {ref['prompt_tokens']}:{mark(ref)}" if ref else ""), flush=True)

    here = os.path.abspath(__file__)
    out = {
        "provenance": {
            "label": args.label, "diagnostic_mode": MODE,
            "diagnostic_wrapper_sha256": hashlib.sha256(open(here, "rb").read()).hexdigest(),
            "harness_sha256": b.sha256_file(os.path.abspath(b.__file__)),
            "system_prompt_sha256": hashlib.sha256(_BASE_PROMPT.encode()).hexdigest(),
            "cases_file": os.path.relpath(os.path.abspath(args.cases), os.path.join(os.path.dirname(here), "..")),
            "cases_sha256": b.sha256_file(args.cases), "tools_sha256": b.sha256_file(args.tools_file),
            "server_version": b.server_version(args.board_ssh, args.server_version_cmd),
            "endpoint_models": b.get_endpoint_models(args.url), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(), "request_config": {"temperature": 0, "max_tokens": args.max_tokens, "extra_body": extra},
            "readiness": readiness_log,
        },
        "summary": {
            "cases": len(cases),
            "by_series_and_target": {},
            "reference_passed": sum(r["reference"]["passed"] for r in results if "reference" in r),
        },
        "results": results,
    }
    for r in results:
        for row in r["sweep"]:
            k = f"{r['series']} " + (str(row["target"]) if row["padding_lines"] else "unpadded")
            s = out["summary"]["by_series_and_target"].setdefault(k, {"structured": 0, "passed": 0, "n": 0, "prompt_tokens": []})
            s["n"] += 1
            s["structured"] += row["structured_call"]
            s["passed"] += row["passed"]
            s["prompt_tokens"].append(row["prompt_tokens"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
