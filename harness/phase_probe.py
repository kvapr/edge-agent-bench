#!/usr/bin/env python3
"""Prefill/decode throughput probe for any OpenAI-compatible chat endpoint.

Measures prefill (prompt processing) and decode (generation) throughput as a
function of prompt length. Supports two measurement methods (--method):

  diff   - the original streaming-free method: two identical non-streaming
           requests per rep that differ only in max_tokens (1 vs gen_tokens).
  stream - one streaming (SSE) request per rep with max_tokens=gen_tokens;
           per-token-chunk arrival times give ttft/decode directly. Needed
           for servers that ignore max_tokens (so "diff" can't measure
           anything) but do stream token-by-token.
  auto   - (default) probe the first length with one streaming request; use
           "stream" for the whole run if it yields >=4 content chunks, else
           fall back to "diff".

See METHOD_NOTE below for the exact per-field math in both methods.

A length whose calibration fails (HTTP error, timeout, etc.) after the first
length does NOT abort the run: it is recorded as {"target","error","reps":
[]} in "lengths" and the run continues with the next length, so a bad length
near the end never discards completed work. Exit code is 4 (no --out file
written) only if the FIRST length's calibration fails; 5 if the file was
written but at least one later length failed; 0 if every length produced
data.

Python stdlib only. Work only in this directory. Does not contact any host
other than 127.0.0.1 on its own -- --url/--board-ssh point wherever the
caller tells it to.

Usage:
  python3 phase_probe.py --url http://<host>:<port> --model qwen3-8b \
      --lengths 256,512,1024,2048,2800 --repeats 5 --gen-tokens 128 \
      --out results/phase_probe_qwen3-8b.json --label npu-geniex-qairt

  # force the streaming method (e.g. server is known to ignore max_tokens)
  python3 phase_probe.py --url http://<host>:<port> --model qwen3-8b \
      --out results/phase_probe.json --label npu-geniex-qairt --method stream

  # gate each length on board thermal/CPU idleness first
  python3 phase_probe.py --url http://<host>:<port> --model qwen3-8b \
      --out results/phase_probe.json --label npu-geniex-qairt \
      --board-ssh user@<board> --readiness
"""
import argparse
import json
import os
import socket
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

# Import shared measurement-hygiene helpers from bench_toolcall.py (same
# directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_toolcall import (  # noqa: E402
    sha256_file,
    get_endpoint_models,
    board_snapshot,
    wait_for_readiness,
    server_version,
    ThermalSampler,
    STANDARD_RESP_KEYS,
)

_REQ_COUNTER = [0]


def probe_user_msg():
    """USER_MSG plus a request number, so that no two probe requests are
    byte-identical. Some servers (seen on GenieX v0.8.0) return an empty reply
    to a request that repeats the previous one verbatim, which would wipe out
    every repetition after the first at each prompt length. The suffix adds a
    few tokens; prefill uses each request's own reported prompt token count."""
    _REQ_COUNTER[0] += 1
    return f"{USER_MSG} (probe request {_REQ_COUNTER[0]})"


USER_MSG = (
    "Explain in detail how Ethernet PoE negotiation works between a switch "
    "and a powered device, step by step, covering classification, power "
    "budgeting and fault handling."
)

# Neutral ~40-word paragraph about network operations, used to build the
# filler system prompt when --system-file is not given. No lab names, no
# IPs. Tiled (repeated word-by-word) to reach the requested word count.
BUILTIN_FILLER = (
    "Network operations teams monitor switch ports, cabling, and power "
    "delivery throughout the day, logging status changes, tracking "
    "utilization trends, and verifying that configuration changes take "
    "effect without disrupting active sessions or degrading link quality "
    "across the deployment."
)

METHOD_NOTE = (
    "Two measurement methods, chosen via --method (default 'auto'). "
    "'diff': two identical non-streaming requests per rep that differ only "
    "in max_tokens (1 vs gen_tokens). tA_s/tB_s are their wall-clock "
    "latencies; ctA/ctB are usage.completion_tokens from each response; pt "
    "is usage.prompt_tokens from request A. ttft_s = tA_s -- NOT a pure "
    "prefill-only measurement: it includes prefill, one decode step, and "
    "request/response overhead. prefill_tok_s = pt / tA_s. decode_tok_s = "
    "(ctB - ctA) / (tB_s - tA_s), or null with decode_skipped='early_stop' "
    "if ctB - ctA < 16, or decode_skipped='server_ignores_max_tokens' "
    "instead if EVERY rep at that length had ctA == ctB (the server did "
    "not honor max_tokens=1, so decode speed cannot be inferred this way). "
    "e2e_s = tB_s. "
    "'stream': one streaming (SSE) request per rep with "
    "max_tokens=gen_tokens. ttft_s = arrival time of the first chunk whose "
    "delta.content is non-empty. n_out = usage.completion_tokens if the "
    "server sends a final usage chunk (requested via "
    "stream_options.include_usage, retried once without it on HTTP 400), "
    "else the number of content chunks received (token_source records "
    "which). prefill_tok_s = pt / ttft_s, where pt is the PER-LENGTH "
    "prompt token count from that length's ordinary (non-streaming) "
    "calibration request -- not a per-rep value. decode_tok_s = (n_out - "
    "1) / (t_last_chunk - t_first_chunk) when n_out >= 16 and that "
    "duration is positive, else null with a decode_skipped reason. e2e_s "
    "= wall-clock time to [DONE]/EOF. "
    "'auto': calibrates the first length, then sends one streaming probe "
    "request (max_tokens=gen_tokens) at that length; uses 'stream' for "
    "the whole run if it yields >=4 content chunks, else 'diff'. The "
    "outcome is recorded as top-level 'method' and 'method_reason'. "
    "Calibration requests (finding a prompt whose token count is close to "
    "a length's target) are always non-streaming with max_tokens=1 and are "
    "excluded from all timing, in both methods -- if a server ignores "
    "max_tokens this only costs calibration time, not correctness. "
    "Per-length summary keys (ttft_s, prefill_tok_s, decode_tok_s, e2e_s "
    "as mean/std) have the same meaning and shape in both methods. "
    "If calibration fails for a length after the first, that length is "
    "recorded as {'target','error','reps':[]} and the run continues; the "
    "process exits 4 (no output file written) only if the FIRST length's "
    "calibration fails (server looks unusable); exits 5 (output file "
    "still written) if the file was written but at least one later length "
    "failed calibration; exits 0 if every length produced data."
)


def build_filler(word_count, source_words):
    """Tile source_words (a non-empty list of words) to exactly word_count
    words and join with spaces."""
    if word_count <= 0:
        return ""
    n = len(source_words)
    return " ".join(source_words[i % n] for i in range(word_count))


def load_source_words(system_file):
    if system_file:
        with open(system_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = BUILTIN_FILLER
    words = text.split()
    if not words:
        raise ValueError(f"filler source text is empty (system_file={system_file!r})")
    return words


def _request_body(model, system_text, user_text, max_tokens, temperature, extra_body):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        body.update(extra_body)
    return body


def call_chat(url, model, system_text, user_text, max_tokens, temperature, extra_body, timeout):
    """POST one non-streaming /v1/chat/completions request. Returns
    (resp_dict, latency_s). Raises on network/HTTP/JSON errors -- caller
    decides how to record them."""
    body = _request_body(model, system_text, user_text, max_tokens, temperature, extra_body)
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    latency = time.time() - t0
    return resp, latency


def _read_sse_stream(resp, t_start):
    """Read an SSE-formatted streaming chat completion response line by
    line. Returns a dict with timing/content info. Never raises for
    malformed lines -- they are skipped."""
    first_chunk_t = None
    last_chunk_t = None
    n_chunks = 0
    usage = None
    chunk_ts = []
    server_extra = {}
    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        now = time.time()
        choices = obj.get("choices") or []
        content = None
        if choices:
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
        if isinstance(content, str) and content != "":
            n_chunks += 1
            chunk_ts.append(now - t_start)
            if first_chunk_t is None:
                first_chunk_t = now
            last_chunk_t = now
        if obj.get("usage"):
            usage = obj["usage"]
        for k, v in obj.items():
            if k not in STANDARD_RESP_KEYS:
                server_extra[k] = v
    t_end = time.time()
    return {
        "first_chunk_t": (first_chunk_t - t_start) if first_chunk_t is not None else None,
        "last_chunk_t": (last_chunk_t - t_start) if last_chunk_t is not None else None,
        "n_chunks": n_chunks,
        "usage": usage,
        "e2e_s": t_end - t_start,
        "chunk_ts": chunk_ts,
        "server_extra": server_extra,
    }


def call_chat_stream(url, model, system_text, user_text, max_tokens, temperature, extra_body, timeout):
    """POST one streaming /v1/chat/completions request (stream=True) and
    read the SSE body. Requests stream_options.include_usage=True; on HTTP
    400 retries once without it (some servers reject the field). Returns
    the dict from _read_sse_stream. Raises on other network/HTTP errors."""

    def _do(include_usage):
        body = _request_body(model, system_text, user_text, max_tokens, temperature, extra_body)
        body["stream"] = True
        if include_usage:
            body["stream_options"] = {"include_usage": True}
        req = urllib.request.Request(
            url.rstrip("/") + "/v1/chat/completions",
            json.dumps(body).encode(),
            {"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        t_start = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return _read_sse_stream(r, t_start)

    try:
        return _do(True)
    except urllib.error.HTTPError as e:
        if e.code == 400:
            return _do(False)
        raise


def calibrate_prompt(url, model, target_tokens, source_words, extra_body, timeout):
    """Find a filler word count whose usage.prompt_tokens is close to
    target_tokens. Start at words = round(target*0.75), measure, then scale
    by target/actual and re-measure, up to 3 adjustment rounds (4 requests
    total). Returns (words, system_text, achieved_prompt_tokens). Always
    non-streaming with max_tokens=1; these calibration requests are
    excluded from timing. Raises on network/HTTP/JSON errors or if
    usage.prompt_tokens is missing -- caller decides how to handle that."""
    words = max(1, round(target_tokens * 0.75))
    system_text = build_filler(words, source_words)
    actual = None
    max_adjust_rounds = 3
    for round_i in range(max_adjust_rounds + 1):
        system_text = build_filler(words, source_words)
        resp, _lat = call_chat(url, model, system_text, probe_user_msg(), 1, 0, extra_body, timeout)
        usage = resp.get("usage") or {}
        actual = usage.get("prompt_tokens")
        if actual is None:
            raise RuntimeError(
                f"calibration response missing usage.prompt_tokens (target={target_tokens})"
            )
        if target_tokens <= 0:
            break
        err = abs(actual - target_tokens) / target_tokens
        if err <= 0.03:
            break
        if round_i == max_adjust_rounds:
            break
        ratio_base = actual if actual > 0 else 1
        words = max(1, round(words * target_tokens / ratio_base))
    return words, system_text, actual


def detect_method(url, model, system_text, gen_tokens, extra_body, timeout):
    """Send one streaming probe request (max_tokens=gen_tokens) and decide
    between 'stream' and 'diff'. Never raises: a failed/unsupported probe
    just falls back to 'diff'."""
    try:
        result = call_chat_stream(url, model, system_text, probe_user_msg(), gen_tokens, 0, extra_body, timeout)
    except Exception as e:  # noqa: BLE001
        return "diff", f"stream probe failed ({type(e).__name__}: {e}), falling back to diff"
    n = result["n_chunks"]
    if n >= 4:
        return "stream", f"stream probe returned {n} content chunks (>=4)"
    return "diff", f"stream probe returned {n} content chunks (<4)"


def run_diff_rep(url, model, system_text, gen_tokens, extra_body, timeout):
    """One 'diff'-method rep: two non-streaming requests differing only in
    max_tokens (1 vs gen_tokens). Returns a rep dict; never raises."""
    try:
        respA, tA = call_chat(url, model, system_text, probe_user_msg(), 1, 0, extra_body, timeout)
        respB, tB = call_chat(url, model, system_text, probe_user_msg(), gen_tokens, 0, extra_body, timeout)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}

    usageA = respA.get("usage") or {}
    usageB = respB.get("usage") or {}
    if (
        "completion_tokens" not in usageA
        or "completion_tokens" not in usageB
        or "prompt_tokens" not in usageA
    ):
        return {
            "tA_s": round(tA, 3),
            "tB_s": round(tB, 3),
            "error": "usage missing from response",
        }

    ctA = usageA["completion_tokens"]
    ctB = usageB["completion_tokens"]
    pt = usageA["prompt_tokens"]

    ttft_s = tA
    prefill_tok_s = (pt / tA) if tA > 0 else None

    rep = {
        "tA_s": round(tA, 3),
        "tB_s": round(tB, 3),
        "ctA": ctA,
        "ctB": ctB,
        "ttft_s": round(ttft_s, 3),
        "prefill_tok_s": round(prefill_tok_s, 3) if prefill_tok_s is not None else None,
        "e2e_s": round(tB, 3),
    }

    if ctB - ctA < 16:
        rep["decode_tok_s"] = None
        rep["decode_skipped"] = "early_stop"
    elif (tB - tA) <= 0:
        rep["decode_tok_s"] = None
        rep["decode_skipped"] = "non_positive_duration"
    else:
        rep["decode_tok_s"] = round((ctB - ctA) / (tB - tA), 3)
    return rep


def run_stream_rep(url, model, system_text, gen_tokens, extra_body, timeout, prompt_tokens):
    """One 'stream'-method rep: a single streaming request with
    max_tokens=gen_tokens. Returns a rep dict; never raises."""
    try:
        result = call_chat_stream(url, model, system_text, probe_user_msg(), gen_tokens, 0, extra_body, timeout)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}

    first_t = result["first_chunk_t"]
    last_t = result["last_chunk_t"]
    n_chunks = result["n_chunks"]
    usage = result["usage"] or {}
    e2e_s = result["e2e_s"]

    if first_t is None:
        return {
            "e2e_s": round(e2e_s, 3),
            "n_chunks": n_chunks,
            "error": "no content chunks received",
        }

    if isinstance(usage.get("completion_tokens"), (int, float)):
        n_out = usage["completion_tokens"]
        token_source = "usage"
    else:
        n_out = n_chunks
        token_source = "chunks"

    ttft_s = first_t
    # Prefer this request's own prompt token count (sent in the final usage
    # chunk); fall back to the per-length calibration count.
    rep_pt = usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), (int, float)) else None
    pt_used = rep_pt if rep_pt else prompt_tokens
    prefill_tok_s = (
        (pt_used / ttft_s) if (pt_used is not None and ttft_s > 0) else None
    )

    rep = {
        "ttft_s": round(ttft_s, 3),
        "prefill_tok_s": round(prefill_tok_s, 3) if prefill_tok_s is not None else None,
        "n_chunks": n_chunks,
        "n_out": n_out,
        "prompt_tokens": pt_used,
        "token_source": token_source,
        "e2e_s": round(e2e_s, 3),
    }

    if result.get("server_extra"):
        rep["server_extra"] = result["server_extra"]
    chunk_ts = result.get("chunk_ts") or []
    # A server that ignores max_tokens generates past gen_tokens, and decode
    # slows as the context grows, so an average over the whole generation is
    # not comparable across servers. Decode is therefore always measured over
    # the first gen_tokens content chunks (one chunk per token on the servers
    # used here; n_chunks and n_out are both recorded so this can be checked).
    rep["max_tokens_honored"] = (n_out is not None and n_out <= gen_tokens * 1.05)
    k = min(gen_tokens, len(chunk_ts))
    if n_out is None or n_out < 16 or k < 16:
        rep["decode_tok_s"] = None
        rep["decode_skipped"] = "early_stop"
    elif chunk_ts[k - 1] - chunk_ts[0] <= 0:
        rep["decode_tok_s"] = None
        rep["decode_skipped"] = "non_positive_duration"
    else:
        rep["decode_tok_s"] = round((k - 1) / (chunk_ts[k - 1] - chunk_ts[0]), 3)
        rep["decode_tokens_measured"] = k
        dur_all = (last_t - first_t) if last_t is not None else None
        if not rep["max_tokens_honored"] and dur_all and dur_all > 0:
            rep["decode_tok_s_full_generation"] = round((n_out - 1) / dur_all, 3)
    return rep


def print_rep_status(target, rep_i, repeats, rep):
    if "error" in rep:
        print(f"[len={target} rep={rep_i}/{repeats}] ERROR {rep['error']}", flush=True)
        return
    ttft = rep.get("ttft_s")
    prefill = rep.get("prefill_tok_s")
    decode = rep.get("decode_tok_s")
    skip = rep.get("decode_skipped")
    e2e = rep.get("e2e_s")
    ttft_disp = f"{ttft:.3f}s" if ttft is not None else "n/a"
    prefill_disp = f"{prefill:.1f}tok/s" if prefill is not None else "n/a"
    decode_disp = f"{decode:.1f}tok/s" if decode is not None else f"skipped({skip})"
    e2e_disp = f"{e2e:.3f}s" if e2e is not None else "n/a"
    if "n_out" in rep:
        extra = f" n_chunks={rep.get('n_chunks')} n_out={rep.get('n_out')} src={rep.get('token_source')}"
    else:
        extra = f" ctA={rep.get('ctA')} ctB={rep.get('ctB')}"
    print(
        f"[len={target} rep={rep_i}/{repeats}] ttft={ttft_disp} "
        f"prefill={prefill_disp} decode={decode_disp} e2e={e2e_disp}{extra}",
        flush=True,
    )


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return {"mean": None, "std": None}
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if n >= 2 else None
    return {
        "mean": round(mean, 3),
        "std": round(std, 3) if std is not None else None,
    }


def fmt_stat(stat):
    if stat["mean"] is None:
        return "n/a"
    if stat["std"] is None:
        return f"{stat['mean']:.3f}±n/a"
    return f"{stat['mean']:.3f}±{stat['std']:.3f}"


def main():
    ap = argparse.ArgumentParser(
        description="Measure prefill/decode throughput of an OpenAI-compatible "
                     "chat endpoint as a function of prompt length."
    )
    ap.add_argument("--url", required=True, help="base URL, e.g. http://<host>:<port>")
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="256,512,1024,2048,2800",
                     help="comma-separated target prompt token counts")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--gen-tokens", type=int, default=128,
                     help="max_tokens for the 'real generation' request(s)")
    ap.add_argument("--warmup", type=int, default=1,
                     help="throwaway requests at the first length before timing")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="", help="free text stored in the output JSON")
    ap.add_argument("--extra-json", default=None,
                     help='JSON object merged into every request body, e.g. '
                          '\'{"enable_think": false}\'')
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--board-ssh", default=None,
                     help="USER@HOST to collect board provenance (and, with "
                          "--readiness, to gate the run) over ssh")
    ap.add_argument("--readiness", action="store_true",
                     help="poll --board-ssh before warmup and before each length; "
                          "requires --board-ssh")
    ap.add_argument("--system-file", default=None,
                     help="use this file's text as the filler corpus instead of "
                          "the built-in paragraph")
    ap.add_argument("--server-version-cmd", default=None,
                     help="command run on --board-ssh whose output identifies the "
                          "server binary that handles the requests (stored in provenance)")
    ap.add_argument("--thermal-trace-s", type=float, default=20.0,
                     help="with --board-ssh, sample board thermal zones every N seconds "
                          "for the whole run (0 disables)")
    ap.add_argument("--method", choices=("auto", "stream", "diff"), default="auto",
                     help="measurement method: 'diff' (two non-streaming requests "
                          "per rep), 'stream' (one SSE request per rep), or 'auto' "
                          "(probe the first length and pick automatically)")
    args = ap.parse_args()

    if args.readiness and not args.board_ssh:
        ap.error("--readiness requires --board-ssh")

    try:
        lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    except ValueError as e:
        ap.error(f"--lengths must be a comma-separated list of ints: {e}")
    if not lengths:
        ap.error("--lengths produced an empty list")

    extra_body = {}
    if args.extra_json:
        try:
            extra_body = json.loads(args.extra_json)
        except json.JSONDecodeError as e:
            ap.error(f"--extra-json is not valid JSON: {e}")
        if not isinstance(extra_body, dict):
            ap.error("--extra-json must decode to a JSON object")

    try:
        source_words = load_source_words(args.system_file)
    except (OSError, ValueError) as e:
        ap.error(str(e))

    readiness_log = []

    def check_ready(label):
        ok = wait_for_readiness(
            args.board_ssh, 55.0, 0.30, 10, 600, label, readiness_log
        )
        if not ok:
            print(
                f"readiness: board not ready within 600s; aborting without "
                f"writing {args.out}",
                flush=True,
            )
            sys.exit(3)

    board_before = None
    board_after = None
    if args.board_ssh and not args.readiness:
        board_before = board_snapshot(args.board_ssh)
    srv_version = server_version(args.board_ssh, args.server_version_cmd)
    sampler = ThermalSampler(args.board_ssh, args.thermal_trace_s).start()

    if args.readiness and args.warmup:
        check_ready("warmup")

    warmup_remaining = args.warmup if args.warmup and args.warmup > 0 else 0
    lengths_out = []
    failed_targets = []
    chosen_method = None
    method_reason = None

    for idx, target in enumerate(lengths):
        if args.readiness:
            check_ready(f"length_{target}")

        try:
            words, system_text, achieved = calibrate_prompt(
                args.url, args.model, target, source_words, extra_body, args.timeout
            )
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            if idx == 0:
                print(
                    f"error: calibration failed for length target={target} "
                    f"(first length -- server looks unusable): {msg}; "
                    f"aborting without writing {args.out}",
                    flush=True,
                )
                sys.exit(4)
            print(
                f"error: calibration failed for length target={target}: {msg}; "
                f"skipping this length and continuing",
                flush=True,
            )
            lengths_out.append({"target": target, "error": msg, "reps": []})
            failed_targets.append(target)
            continue

        print(
            f"calibrated length target={target}: words={words} "
            f"achieved_prompt_tokens={achieved}",
            flush=True,
        )

        if chosen_method is None:
            if args.method == "auto":
                chosen_method, method_reason = detect_method(
                    args.url, args.model, system_text, args.gen_tokens, extra_body, args.timeout
                )
                print(f"auto method detection: {chosen_method} ({method_reason})", flush=True)
            else:
                chosen_method = args.method
                method_reason = f"explicit --method={args.method}"

        if warmup_remaining:
            for _ in range(warmup_remaining):
                try:
                    if chosen_method == "stream":
                        call_chat_stream(
                            args.url, args.model, system_text, probe_user_msg(),
                            args.gen_tokens, 0, extra_body, args.timeout,
                        )
                    else:
                        call_chat(
                            args.url, args.model, system_text, probe_user_msg(),
                            args.gen_tokens, 0, extra_body, args.timeout,
                        )
                except Exception:  # noqa: BLE001
                    pass
            warmup_remaining = 0

        reps = []
        for rep_i in range(1, args.repeats + 1):
            if chosen_method == "stream":
                rep = run_stream_rep(
                    args.url, args.model, system_text, args.gen_tokens,
                    extra_body, args.timeout, achieved,
                )
            else:
                rep = run_diff_rep(
                    args.url, args.model, system_text, args.gen_tokens,
                    extra_body, args.timeout,
                )
            reps.append(rep)
            print_rep_status(target, rep_i, args.repeats, rep)

        if chosen_method == "diff":
            non_error = [r for r in reps if "error" not in r]
            if non_error and all(r.get("ctA") == r.get("ctB") for r in non_error):
                for r in non_error:
                    if r.get("decode_skipped") == "early_stop":
                        r["decode_skipped"] = "server_ignores_max_tokens"

        valid = [r for r in reps if "error" not in r]
        ttft_vals = [r["ttft_s"] for r in valid if r.get("ttft_s") is not None]
        prefill_vals = [r["prefill_tok_s"] for r in valid if r.get("prefill_tok_s") is not None]
        e2e_vals = [r["e2e_s"] for r in valid if r.get("e2e_s") is not None]
        decode_vals = [r["decode_tok_s"] for r in valid if r.get("decode_tok_s") is not None]

        decode_stat = mean_std(decode_vals)
        decode_stat["n"] = len(decode_vals)

        lengths_out.append({
            "target": target,
            "prompt_tokens": achieved,
            "reps": reps,
            "ttft_s": mean_std(ttft_vals),
            "prefill_tok_s": mean_std(prefill_vals),
            "decode_tok_s": decode_stat,
            "e2e_s": mean_std(e2e_vals),
        })

    sampler.stop()
    if args.board_ssh:
        if args.readiness:
            board_before = readiness_log[0]["snapshot"] if readiness_log else None
        board_after = board_snapshot(args.board_ssh)

    provenance = {
        "schema_version": 3,
        "label": args.label,
        "server_version": srv_version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "harness_sha256": sha256_file(os.path.abspath(__file__)),
        "request_config": {
            "temperature": 0,
            "gen_tokens": args.gen_tokens,
            "extra_body": extra_body,
        },
        "endpoint_models": get_endpoint_models(args.url),
        "board": board_before,
        "board_after": board_after,
    }

    output = {
        "label": args.label,
        "url": args.url,
        "model": args.model,
        "gen_tokens": args.gen_tokens,
        "repeats": args.repeats,
        "method": chosen_method,
        "method_reason": method_reason,
        "method_note": METHOD_NOTE,
        "provenance": provenance,
        "lengths": lengths_out,
        "thermal": sampler.summary(),
        "thermal_trace": sampler.samples,
    }
    if args.readiness:
        output["readiness"] = readiness_log

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    if failed_targets:
        print(
            f"WARNING: {len(failed_targets)} length(s) failed calibration and "
            f"were skipped: {failed_targets}",
            flush=True,
        )

    print()
    print("length(tok)  n  ttft_s mean±std  prefill_tok/s mean±std  "
          "decode_tok/s mean±std  e2e_s(B) mean±std")
    for entry in lengths_out:
        if "error" in entry:
            continue
        n_valid = sum(1 for r in entry["reps"] if "error" not in r)
        print(
            f"{entry['target']:<11d}  {n_valid:<1d}  "
            f"{fmt_stat(entry['ttft_s']):<16}  "
            f"{fmt_stat(entry['prefill_tok_s']):<22}  "
            f"{fmt_stat(entry['decode_tok_s']):<20}  "
            f"{fmt_stat(entry['e2e_s'])}"
        )

    if failed_targets:
        sys.exit(5)


if __name__ == "__main__":
    main()
