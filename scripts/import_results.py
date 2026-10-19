#!/usr/bin/env python3
"""Copy raw campaign results into the repository with private details removed.

The raw outputs of campaign/run_campaign.py live outside the repository because
they record where the run happened: the board's address in each request URL
and the client's hostname in each provenance block. This script copies every
result JSON from a raw campaign directory into a destination inside the
repository, replacing the board address with "<board>", the client hostname
with "client", and the home directory with "~". Nothing else is altered, so
the analysis gives identical numbers on the raw and the imported copy (the
script checks this when --verify is given).

The values to remove are taken from the environment at run time, never from
this file: EAB_BOARD_IP (required) and the current hostname. Files with a
".runner-bug" infix are skipped; they record failures caused by a runner bug
that was fixed before the affected runs were repeated. Harness logs (*.log) are
not copied; every number in them is also in the JSON.

Usage: EAB_BOARD_IP=<addr> python3 scripts/import_results.py <raw dir> <dest dir> [--verify]
"""
import json
import os
import re
import socket
import subprocess
import sys


def main():
    if len(sys.argv) < 3 or not os.environ.get("EAB_BOARD_IP"):
        print(__doc__)
        return 2
    src, dst = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    board = os.environ["EAB_BOARD_IP"]
    host = socket.gethostname()
    home = os.path.expanduser("~")
    subs = [(re.compile(re.escape(board)), "<board>"),
            (re.compile(r"(?<![\w-])" + re.escape(host) + r"(?![\w-])"), "client"),
            (re.compile(re.escape(home) + r"(?=/|\b)"), "~")]

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, str):
            for rx, rep in subs:
                o = rx.sub(rep, o)
        return o

    n = skipped = 0
    for dirpath, _, files in os.walk(src):
        for fn in sorted(files):
            if not fn.endswith(".json") or ".runner-bug" in fn:
                skipped += fn.endswith(".json")
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), src)
            obj = clean(json.load(open(os.path.join(src, rel))))
            out = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
                f.write("\n")
            n += 1
    leftover = []
    for dirpath, _, files in os.walk(dst):
        for fn in files:
            text = open(os.path.join(dirpath, fn), encoding="utf-8").read()
            if board in text or re.search(r"(?<![\w-])" + re.escape(host) + r"(?![\w-])", text) or home in text:
                leftover.append(os.path.relpath(os.path.join(dirpath, fn), dst))
    print(f"imported {n} files into {dst} (skipped {skipped} runner-bug records)")
    if leftover:
        print("PRIVATE VALUES REMAIN in:", ", ".join(leftover))
        return 1
    if "--verify" in sys.argv:
        here = os.path.dirname(os.path.abspath(__file__))
        an = os.path.join(here, "..", "analysis", "analyze.py")
        tmp = os.path.join(dst, ".verify")
        os.makedirs(tmp, exist_ok=True)
        res = []
        for d, tag in ((src, "raw"), (dst, "imported")):
            subprocess.run([sys.executable, an, d, "--json", os.path.join(tmp, tag + ".json"),
                            "--md", os.path.join(tmp, tag + ".md")], check=True, capture_output=True)
            j = json.load(open(os.path.join(tmp, tag + ".json")))
            res.append(json.dumps({k: j[k] for k in ("pooled", "comparisons") if k in j}, sort_keys=True))
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)
        print("analysis identical on raw and imported copies:", res[0] == res[1])
        return 0 if res[0] == res[1] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
