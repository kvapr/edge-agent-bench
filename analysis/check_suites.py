#!/usr/bin/env python3
"""Mechanical validity checks for the test suites. Exits non-zero on any issue.

Checks that every expected command is one the system prompt names, that
parameters follow the prompt's stated formats (ports as '1/N', VLAN IDs and PoE
power limits as integers), that every paraphrase points to a canonical base
case and carries exactly that case's labels, and that no input repeats.
"""
import collections
import importlib.util
import os
import re
import sys

import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
spec = importlib.util.spec_from_file_location("b", os.path.join(ROOT, "harness", "bench_toolcall.py"))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def main():
    prompt = b.SYSTEM_PROMPT
    cmds = set(re.findall(r"\b([a-z_]+)\s*\(", prompt)) | set(re.findall(r"\b(show_[a-z_]+)\b", prompt))
    can = yaml.safe_load(open(os.path.join(ROOT, "suites", "test_cases.yaml")))
    par = yaml.safe_load(open(os.path.join(ROOT, "suites", "test_cases_paraphrase.yaml")))
    issues = []

    def check(item, where):
        c = item.get("expected_command")
        if c and c not in cmds:
            issues.append(f"{where}: command {c!r} is not named in the system prompt")
        for k, v in (item.get("expected_params") or {}).items():
            if k == "port" and not (isinstance(v, str) and re.fullmatch(r"1/\d+", v)):
                issues.append(f"{where}: port {v!r} is not in 1/N form")
            if k in ("vlan_id", "power_limit") and not isinstance(v, int):
                issues.append(f"{where}: {k} {v!r} is not an integer")

    for lv, items in can.items():
        for i, it in enumerate(items):
            check(it, f"canonical {lv}[{i}]")
    base = {it["input"]: it for items in can.values() for it in items}
    styles = collections.Counter()
    for lv, items in par.items():
        for i, it in enumerate(items):
            check(it, f"paraphrase {lv}[{i}]")
            styles[it.get("style")] += 1
            bi = base.get(it.get("base_input"))
            if not bi:
                issues.append(f"paraphrase {lv}[{i}]: base case not found")
                continue
            for k in ("expected_tool", "expected_command", "expected_params"):
                if (bi.get(k) or None) != (it.get(k) or None):
                    issues.append(f"paraphrase {lv}[{i}]: {k} differs from its base case")
    for name, suite in (("canonical", can), ("paraphrase", par)):
        inputs = [it["input"] for items in suite.values() for it in items]
        if len(inputs) != len(set(inputs)):
            issues.append(f"{name}: {len(inputs) - len(set(inputs))} repeated inputs")
    n_can = sum(len(v) for v in can.values())
    n_par = sum(len(v) for v in par.values())
    print(f"canonical {n_can} cases, paraphrase {n_par} cases, styles {dict(styles)}, "
          f"commands named in prompt {len(cmds)}")
    for x in issues:
        print("ISSUE", x)
    print("PASS" if not issues else f"FAIL ({len(issues)} issues)")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
