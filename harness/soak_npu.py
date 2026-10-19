#!/usr/bin/env python3
"""Soak test for the Genie NPU endpoint: N consecutive requests with a mix of
shapes, including deliberately oversized prompts. Verifies no degradation
(latency creep, empty outputs, engine death) and that oversize fails clean.

Usage:
  python3 soak_npu.py --url http://<host>:<port> --n 300 \
      --out results/soak_npu.json
"""
import argparse
import hashlib
import json
import os
import socket
import statistics
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

TOOLS = [{"type": "function", "function": {
    "name": "get_interface_details",
    "description": "Get status of a switch interface",
    "parameters": {"type": "object", "properties": {
        "device": {"type": "string"}, "port": {"type": "integer"}},
        "required": ["device", "port"]}}}]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_endpoint_models(url):
    """GET {url}/v1/models with a 10s timeout. Must never raise."""
    try:
        req = urllib.request.Request(url.rstrip("/") + "/v1/models")
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    s = json.dumps(data)
    if len(s) > 2000:
        return s[:2000]
    return data


def req(url, body, timeout=180):
    r = urllib.request.Request(url, json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            d = json.loads(resp.read())
        msg = d["choices"][0]["message"]
        text = msg.get("content") or json.dumps(msg.get("tool_calls") or "")
        return time.time() - t0, len(text), None
    except urllib.error.HTTPError as e:
        return time.time() - t0, 0, f"HTTP{e.code}"
    except Exception as e:  # noqa: BLE001
        return time.time() - t0, 0, type(e).__name__

def shape(i):
    kind = i % 4
    if i % 25 == 24:   # oversized prompt: ~6k tokens > 4096 ctx
        return "oversize", {"model": "q", "max_tokens": 32, "messages": [
            {"role": "user", "content": "network telemetry: " + "port up, " * 3000}]}
    if kind == 0:
        return "chat", {"model": "q", "max_tokens": 48, "messages": [
            {"role": "user", "content": f"In one sentence, what is VLAN {100+i}?"}]}
    if kind == 1:
        return "tool", {"model": "q", "max_tokens": 128, "tools": TOOLS, "messages": [
            {"role": "system", "content": "You are a network assistant."},
            {"role": "user", "content": f"Is port {1+i%8} on switch sw-{i%5} up?"}]}
    if kind == 2:
        return "multiturn", {"model": "q", "max_tokens": 96, "tools": TOOLS, "messages": [
            {"role": "user", "content": f"Is port {1+i%8} on sw-1 up?"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "get_interface_details",
                             "arguments": json.dumps({"device": "sw-1", "port": 1+i%8})}}]},
            {"role": "tool", "tool_call_id": "c1",
             "content": "Link UP, 1Gbps full duplex, PoE 8.2W"}]}
    return "long", {"model": "q", "max_tokens": 160, "messages": [
        {"role": "user", "content": "List three common causes of PoE faults, briefly."}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="base URL, e.g. http://<host>:<port>")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "soak_npu.json"))
    args = ap.parse_args()

    url = args.url.rstrip("/") + "/v1/chat/completions"
    n = args.n

    fails, lat_by_kind, empties = [], {}, 0
    t_start = time.time()
    for i in range(n):
        kind, body = shape(i)
        dt, nchars, err = req(url, body)
        lat_by_kind.setdefault(kind, []).append(dt)
        if kind == "oversize":
            ok = err is not None            # MUST fail cleanly
            if not ok:
                fails.append((i, kind, "oversize request unexpectedly succeeded"))
        else:
            if err:
                fails.append((i, kind, err))
            elif nchars == 0:
                empties += 1
                fails.append((i, kind, "EMPTY OUTPUT"))
        if (i + 1) % 25 == 0:
            med = sorted(lat_by_kind.get("chat", [0]))[len(lat_by_kind.get("chat", [0])) // 2]
            print(f"[{i+1}/{n}] fails={len(fails)} empties={empties} "
                  f"chat_med={med:.1f}s elapsed={(time.time()-t_start)/60:.1f}m", flush=True)

    print("\n=== SOAK SUMMARY ===")
    print(f"requests: {n}, failures: {len(fails)}, empty outputs: {empties}")
    latency_stats = {}
    for kind, ls in sorted(lat_by_kind.items()):
        ls2 = sorted(ls)
        mean_v = statistics.mean(ls2)
        std_v = statistics.stdev(ls2) if len(ls2) >= 2 else None
        p95_v = ls2[int(len(ls2) * 0.95) - 1]
        latency_stats[kind] = {
            "n": len(ls2),
            "median": ls2[len(ls2) // 2],
            "mean": round(mean_v, 3),
            "std": round(std_v, 3) if std_v is not None else None,
            "p95": p95_v,
        }
        std_disp = f"{std_v:5.1f}" if std_v is not None else "  n/a"
        print(f"  {kind:10s} n={len(ls):3d} med={ls2[len(ls2)//2]:5.1f}s "
              f"mean={mean_v:5.1f}s std={std_disp}s "
              f"p95={p95_v:5.1f}s")
    for f in fails[:12]:
        print("  FAIL:", f)

    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "harness_sha256": sha256_file(os.path.abspath(__file__)),
        "endpoint_models": get_endpoint_models(args.url),
    }

    with open(args.out, "w") as fh:
        json.dump({"n": n, "failures": fails, "empties": empties,
                   "latencies": {k: v for k, v in lat_by_kind.items()},
                   "latency_stats": latency_stats,
                   "provenance": provenance}, fh)


if __name__ == "__main__":
    main()
