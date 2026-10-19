#!/usr/bin/env python3
"""Diagnostic variant of bench_toolcall.py: the tools are written into the
system prompt instead of being passed as the request's "tools" field.

Purpose: some serving stacks silently drop the "tools" field for some models,
so the model never sees the tool definitions and cannot call them (visible as a
server-reported prompt of a few hundred tokens instead of about 2,700). This
variant measures what such a model can do when it does see the tools. It is a
diagnostic, not the benchmark's headline protocol, and its results are labelled
mode=tools_in_prompt. It should be calibrated on a configuration where native
tool calling works, to show how closely the two protocols agree.

Everything except the delivery of the tools is unchanged: same cases, same
grader, same provenance. bench_toolcall.py itself is imported, not modified.
Usage and flags are identical to bench_toolcall.py.
"""
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_toolcall as b  # noqa: E402

MODE = "tools_in_prompt"
_ORIG_PROMPT = b.SYSTEM_PROMPT


def prompt_with_tools(tools):
    lines = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    return (_ORIG_PROMPT.rstrip() + "\n\n"
            "You can call the following tools. Each line is one tool definition in JSON:\n"
            "<tools>\n" + lines + "\n</tools>\n\n"
            "To call a tool, reply with only a tool call in this form:\n"
            '<tool_call>{"name": "<tool name>", "arguments": {<arguments as a JSON object>}}</tool_call>\n')


_JSON_OBJ = re.compile(r"\{.*\}|\[.*\]", re.S)


def parse_text_call(content):
    """Read a tool call from free text, accepting the common shapes models use:
    a <tool_call> block, a bare JSON object, or a one-element JSON list, with the
    name under name/tool_name/function and arguments under arguments/parameters."""
    if not content:
        return None, None
    fx = re.search(r"<function=([A-Za-z0-9_]+)>(.*?)</function>", content, re.S)
    if fx:  # XML-parameter form: <function=NAME><parameter=KEY>VALUE</parameter>...
        args = {}
        for pm in re.finditer(r"<parameter=([A-Za-z0-9_]+)>\s*(.*?)\s*</parameter>", fx.group(2), re.S):
            v = pm.group(2)
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
            args[pm.group(1)] = v
        return fx.group(1), args
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.S)
    text = m.group(1) if m else content
    m2 = _JSON_OBJ.search(text)
    if not m2:
        return None, None
    try:
        obj = json.loads(m2.group(0))
    except json.JSONDecodeError:
        return None, None
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        return None, None
    fn = obj.get("function") if isinstance(obj.get("function"), dict) else None
    name = (fn or {}).get("name") or obj.get("name") or obj.get("tool_name") or \
        (obj.get("function") if isinstance(obj.get("function"), str) else None)
    args = (fn or {}).get("arguments") or obj.get("arguments") or obj.get("parameters") or obj.get("args")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = None
    return name, args if isinstance(args, dict) else None


_orig_call = b.call_endpoint
_orig_extract = b.extract_call


def call_endpoint(url, model, user_msg, tools, timeout, extra_body):
    # Same request as the harness, but the tools travel inside the system
    # prompt and the "tools" field is omitted.
    b.SYSTEM_PROMPT = prompt_with_tools(tools)
    return _orig_call(url, model, user_msg, None, timeout, {**extra_body})


def extract_call(resp):
    name, args = _orig_extract(resp)          # a structured call still counts
    if name:
        return name, args
    try:
        content = resp["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return None, None
    return parse_text_call(content)


def main():
    b.call_endpoint = call_endpoint
    b.extract_call = extract_call
    # The harness omits null fields poorly for some servers; drop "tools": None.
    b.OMIT_PARAMS = set(b.OMIT_PARAMS) | {"tools"}
    here = os.path.abspath(__file__)
    wrapper_sha = hashlib.sha256(open(here, "rb").read()).hexdigest()
    orig_dump = b.json.dump

    def dump_with_mode(obj, fp, *a, **k):
        if isinstance(obj, dict) and "provenance" in obj:
            # record the prompt actually sent, tools included
            obj["provenance"]["system_prompt"] = b.SYSTEM_PROMPT
            obj["provenance"]["system_prompt_sha256"] = hashlib.sha256(b.SYSTEM_PROMPT.encode()).hexdigest()
            obj["provenance"]["diagnostic_mode"] = MODE
            obj["provenance"]["diagnostic_wrapper_sha256"] = wrapper_sha
        return orig_dump(obj, fp, *a, **k)
    b.json.dump = dump_with_mode
    b.main()


if __name__ == "__main__":
    main()
