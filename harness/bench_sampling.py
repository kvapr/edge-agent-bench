#!/usr/bin/env python3
"""Variant of bench_toolcall.py that adds sampling fields to every request.

Purpose: at temperature 0 the llama.cpp paths of GenieX v0.8.0 did not give
repeatable replies, and the server does not record the sampler it applied, so
the cause is not established (report Section 11). This variant sends extra
sampling fields with every request, for example {"top_k": 1} to ask for greedy
decoding or {"seed": 7} to change the seed of a sampled default, so a rerun can
show whether the variation comes from sampling. Its results are labelled
mode=sampling and record the fields sent.

A server may ignore fields it does not support; the variant cannot detect that,
so a result whose replies still vary says only that the fields did not remove
the variation. Everything except the added fields is unchanged: same cases,
same grader, same provenance. bench_toolcall.py itself is imported, not
modified. Flags are those of bench_toolcall.py plus --sampling:

    python3 harness/bench_sampling.py --sampling '{"top_k": 1}' <bench_toolcall.py flags>
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_toolcall as b  # noqa: E402

MODE = "sampling"


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--sampling", required=True,
                     help='JSON object merged into every request body, e.g. \'{"top_k": 1}\'')
    ours, rest = pre.parse_known_args()
    sampling = json.loads(ours.sampling)
    if not isinstance(sampling, dict) or not sampling:
        sys.exit("--sampling must be a non-empty JSON object")
    sys.argv = [sys.argv[0], *rest]

    orig_call = b.call_endpoint

    def call_endpoint(url, model, user_msg, tools, timeout, extra_body):
        # Same request as the harness, with the sampling fields added last so
        # they win over any default of the same name.
        return orig_call(url, model, user_msg, tools, timeout, {**extra_body, **sampling})
    b.call_endpoint = call_endpoint

    wrapper_sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    orig_dump = b.json.dump

    def dump_with_mode(obj, fp, *a, **k):
        if isinstance(obj, dict) and "provenance" in obj:
            prov = obj["provenance"]
            prov["diagnostic_mode"] = MODE
            prov["diagnostic_wrapper_sha256"] = wrapper_sha
            prov["sampling"] = sampling
            rc = prov.get("request_config") or {}
            rc["extra_body"] = {**(rc.get("extra_body") or {}), **sampling}
        return orig_dump(obj, fp, *a, **k)
    b.json.dump = dump_with_mode
    b.main()


if __name__ == "__main__":
    main()
