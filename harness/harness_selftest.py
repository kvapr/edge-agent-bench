#!/usr/bin/env python3
"""Self-test for the bench_toolcall.py / soak_npu.py / phase_probe.py
measurement-hygiene upgrade. Runs entirely against a fake, local
(127.0.0.1) OpenAI-compatible HTTP server -- no board, no real model, no
network host other than 127.0.0.1.

Usage:
  python3 harness_selftest.py
"""
import http.server
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_PATH = os.path.join(HERE, "bench_toolcall.py")
SAMPLING_PATH = os.path.join(HERE, "bench_sampling.py")
SOAK_PATH = os.path.join(HERE, "soak_npu.py")
PHASE_PROBE_PATH = os.path.join(HERE, "phase_probe.py")

# Must exactly match test_cases.yaml's first case (level1_queries[0]).
FIRST_CASE_INPUT = "@poesw1 show all ports status"
FIRST_CASE_TOOL_CALL = {
    "name": "query_device_info",
    "arguments": {
        "device_name": "poesw1",
        "command_name": "show_all_ports_status",
        "parameters": {},
    },
}

TOOL_NAMES = [
    "query_device_info",
    "configure_device_setting",
    "diagnose_port_issues",
    "search_documentation",
]

# Long dummy reply used for non-tool-call responses (see FakeHandler.do_POST
# below) -- gives every response a substantial, non-empty content string
# regardless of requested max_tokens. The generation length that
# phase_probe.py's differential math actually measures comes from
# usage.completion_tokens, not from this string's length.
DUMMY_LONG_CONTENT = (
    "This is a long dummy response used only to exercise differential "
    "prefill/decode timing math in the harness self-test suite. " * 20
)

# A special model name that puts FakeHandler into "fake-400-long" mode: any
# request (calibration, streaming probe, or timed rep -- streaming or not)
# whose prompt exceeds FAKE_400_LONG_MAX_PROMPT_TOKENS gets HTTP 400. Used
# to exercise phase_probe.py's per-length calibration-failure handling
# (partial results / exit 5) without touching a real server.
#
# The threshold is deliberately NOT the round number 150 suggested as a
# first cut: with the built-in filler + fixed USER_MSG in phase_probe.py,
# calibrate_prompt()'s *first* attempt for target=100 (words =
# round(100*0.75) = 75) measures ~185 prompt tokens before self-correcting
# down to ~103; a 150 cutoff would reject that first attempt too and make
# even the "healthy" length=100 case fail outright. target=200's
# calibration never measures below ~205 tokens even fully converged. 195
# sits in the gap between those two, so length=100 always succeeds and
# length=200 always fails, for every request kind, reliably.
FAKE_400_LONG_MODEL = "fake-400-long"
FAKE_400_LONG_MAX_PROMPT_TOKENS = 195

# "fake-textcall": answers the first case with its tool call written as text
# in message.content (Hermes style) and no structured tool_calls entry, the
# way some model/server pairs do. The harness must grade it a failure and
# flag it as a text tool call.
FAKE_TEXTCALL_MODEL = "fake-textcall"
# "fake-ignore-max": streams FAKE_IGNORE_MAX_TOKENS tokens whatever max_tokens
# says, and adds a server-specific "timings" field, like a server that does
# not honor the output limit. phase_probe.py must then measure decode over
# exactly --gen-tokens tokens and flag max_tokens_honored = false.
FAKE_IGNORE_MAX_MODEL = "fake-ignore-max"
FAKE_IGNORE_MAX_TOKENS = 64
# "fake-identical-empty": otherwise like "fake", but a request whose model and
# messages are identical to the previous request gets an empty reply with zero
# token counts, reproducing a quirk seen on GenieX v0.8.0. The harness must
# never let this reach grading.
FAKE_IDENTICAL_EMPTY_MODEL = "fake-identical-empty"
_LAST_REQUEST = {"key": None}
_LAST_LOCK = threading.Lock()
# Sampling fields of every request the fake server receives, so the test of
# bench_sampling.py can check that they reach the server.
_SEEN_SAMPLING = []


def make_tools_file(path):
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Fake schema for {name} (harness_selftest.py)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "device_name": {"type": "string"},
                        "command_name": {"type": "string"},
                        "parameters": {"type": "object"},
                    },
                    "required": ["device_name"],
                },
            },
        }
        for name in TOOL_NAMES
    ]
    with open(path, "w") as f:
        json.dump(tools, f)


class FakeHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # silence
        pass

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._send_json({"data": [{"id": "fake"}]})
        else:
            self._send_json({"error": "not found"}, code=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw)
        except Exception:  # noqa: BLE001
            body = {}

        messages = body.get("messages", []) or []
        with _LAST_LOCK:
            _SEEN_SAMPLING.append({k: body[k] for k in ("top_k", "seed", "temperature") if k in body})

        # usage.completion_tokens tracks the requested max_tokens (capped at
        # 64) and usage.prompt_tokens is a rough len(content)//4 count over
        # all message content. This makes phase_probe.py's differential and
        # streaming prefill/decode math (which reads only these usage
        # fields / chunk timings, never anything server-specific) produce
        # finite, non-trivial numbers for any prompt length or --gen-tokens
        # value the self-test exercises.
        try:
            requested_max_tokens = int(body.get("max_tokens", 16))
        except (TypeError, ValueError):
            requested_max_tokens = 16
        completion_tokens = min(requested_max_tokens, 64)
        if body.get("model") == FAKE_IGNORE_MAX_MODEL:
            completion_tokens = FAKE_IGNORE_MAX_TOKENS

        if body.get("model") == FAKE_IDENTICAL_EMPTY_MODEL:
            key = json.dumps([body.get("model"), messages], sort_keys=True)
            with _LAST_LOCK:
                repeat = (key == _LAST_REQUEST["key"])
                _LAST_REQUEST["key"] = key
            if repeat:
                empty = {"choices": [{"message": {"role": "assistant", "content": ""}}],
                         "usage": {"completion_tokens": 0, "prompt_tokens": 0}}
                if body.get("stream"):
                    self._handle_stream(0, 0)
                else:
                    self._send_json(empty)
                return

        all_content = "".join(
            m.get("content", "") for m in messages if isinstance(m.get("content"), str)
        )
        prompt_tokens = len(all_content) // 4

        # fake-400-long mode: reject any request whose prompt exceeds the
        # threshold, regardless of streaming or not. See the module-level
        # comment on FAKE_400_LONG_MODEL for why the threshold is 195.
        if (
            body.get("model") == FAKE_400_LONG_MODEL
            and prompt_tokens > FAKE_400_LONG_MAX_PROMPT_TOKENS
        ):
            self._send_json(
                {"error": {"message": "prompt too long (fake-400-long test mode)",
                           "type": "invalid_request_error"}},
                code=400,
            )
            return

        if body.get("stream"):
            self._handle_stream(completion_tokens, prompt_tokens,
                                extra_timings=(body.get("model") == FAKE_IGNORE_MAX_MODEL))
            return

        # Sleep scales with completion_tokens (base + jitter + per-token
        # term) so that a max_tokens=1 request is reliably faster than a
        # larger max_tokens request -- phase_probe.py's diff-method
        # decode_tok_s depends on tB_s > tA_s holding in practice, not
        # just on average.
        time.sleep(0.005 + completion_tokens * 0.001 + random.uniform(0, 0.01))

        user_msg = ""
        for m in messages:
            if m.get("role") == "user":
                user_msg = m.get("content", "")

        if user_msg.strip() == FIRST_CASE_INPUT and body.get("model") == FAKE_TEXTCALL_MODEL:
            text_call = ("<tool_call>\n<function=" + FIRST_CASE_TOOL_CALL["name"] + ">\n"
                         "<parameter=device_name>\npoesw1\n</parameter>\n</function>\n</tool_call>")
            resp = {
                "choices": [{"message": {"role": "assistant", "content": text_call}}],
                "usage": {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens},
            }
        elif user_msg.strip() == FIRST_CASE_INPUT:
            resp = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": FIRST_CASE_TOOL_CALL["name"],
                                "arguments": json.dumps(FIRST_CASE_TOOL_CALL["arguments"]),
                            },
                        }],
                    },
                }],
                "usage": {
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                },
            }
        else:
            resp = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": DUMMY_LONG_CONTENT,
                    },
                }],
                "usage": {
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                },
            }
        self._send_json(resp)

    def _handle_stream(self, completion_tokens, prompt_tokens, extra_timings=False):
        """Fake SSE stream: one 'data: {...}' chunk per token (delta.content
        = 'tok ') for completion_tokens tokens, ~2ms apart, then a final
        chunk carrying 'usage', then 'data: [DONE]'. No Content-Length or
        chunked framing is sent -- the response is close-delimited
        (Connection: close), which urllib/http.client read correctly via
        EOF, same as many real streaming servers over HTTP/1.1."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            for _ in range(completion_tokens):
                chunk = {
                    "choices": [{"index": 0, "delta": {"content": "tok "}, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.002)
            final_chunk = {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                "usage": {
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                },
            }
            if extra_timings:
                final_chunk["timings"] = {"prompt_per_second": 100.0, "predicted_per_second": 50.0}
            self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def start_fake_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


def run(cmd, timeout=120):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def main():
    server, port = start_fake_server()
    url = f"http://127.0.0.1:{port}"
    tmpdir = tempfile.mkdtemp(prefix="harness_selftest_")
    try:
        tools_file = os.path.join(tmpdir, "tools.json")
        make_tools_file(tools_file)

        # --- run 1: repeats + warmup ---
        out1 = os.path.join(tmpdir, "out1.json")
        rc, stdout, stderr = run([
            sys.executable, BENCH_PATH,
            "--url", url,
            "--model", "fake",
            "--tools-file", tools_file,
            "--tool-subset", ",".join(TOOL_NAMES),
            "--limit", "3",
            "--repeats", "3",
            "--warmup", "1",
            "--out", out1,
        ])
        assert rc == 0, f"bench_toolcall.py (repeats run) exited {rc}\nstdout:\n{stdout}\nstderr:\n{stderr}"

        with open(out1) as f:
            data = json.load(f)

        prov = data["provenance"]
        assert prov["schema_version"] == 3, f"schema_version != 3: {prov.get('schema_version')}"
        assert prov.get("harness_version") == "2.0", f"harness_version: {prov.get('harness_version')}"
        sp = prov.get("system_prompt") or ""
        assert sp and prov.get("system_prompt_sha256"), "system prompt or its hash missing from provenance"
        assert sp.splitlines()[0] == (
            "You are a network device management assistant for managed PoE switches."), \
            f"system prompt first line is not the neutral wording: {sp.splitlines()[0]!r}"
        assert "thermal_trace" in data, "output missing thermal_trace"
        assert isinstance(data["results"][0]["reps"][0].get("prompt_tokens"), int), \
            "rep prompt_tokens not recorded"
        assert "fake" in json.dumps(prov["endpoint_models"]), \
            f"endpoint_models does not contain 'fake': {prov['endpoint_models']}"

        summary = data["summary"]
        assert summary["repeats"] == 3, f"summary.repeats != 3: {summary.get('repeats')}"
        assert len(summary["e2e_pct_by_rep"]) == 3, \
            f"len(e2e_pct_by_rep) != 3: {summary.get('e2e_pct_by_rep')}"
        assert summary["overall_e2e_pct"] == 33.3, \
            f"overall_e2e_pct != 33.3: {summary.get('overall_e2e_pct')}"
        assert summary["unstable_cases"] == [], \
            f"unstable_cases not empty: {summary.get('unstable_cases')}"
        assert isinstance(summary["latency_p95_s"], (int, float)), \
            f"latency_p95_s is not a number: {summary.get('latency_p95_s')}"
        for key in ("latency_p5_s", "latency_p50_s", "prompt_tokens_median"):
            assert isinstance(summary.get(key), (int, float)), f"summary.{key} missing: {summary.get(key)}"
        assert summary.get("text_tool_call_reps") == 0, \
            f"structured-call run should have no text tool calls: {summary.get('text_tool_call_reps')}"

        results = data["results"]
        assert len(results) == 3, f"expected 3 results, got {len(results)}"

        c0 = results[0]
        assert c0["pass_count"] == 3, f"case0 pass_count != 3: {c0.get('pass_count')}"
        assert c0["stable"] is True, f"case0 not stable: {c0}"
        assert len(c0["latencies_s"]) == 3, f"case0 latencies_s len != 3: {c0.get('latencies_s')}"
        assert isinstance(c0["latency_std_s"], (int, float)), \
            f"case0 latency_std_s is not a number: {c0.get('latency_std_s')}"

        for idx in (1, 2):
            c = results[idx]
            assert c["pass_count"] == 0, f"case{idx} pass_count != 0: {c.get('pass_count')}"
            assert c["stable"] is True, f"case{idx} not stable: {c}"

        by_level = summary["by_level"]
        for lv, stats in by_level.items():
            for key in ("tool", "cmd", "params_l", "params_s", "e2e"):
                assert isinstance(stats[key], float), \
                    f"by_level[{lv!r}][{key!r}] is not a float: {stats[key]!r}"
        e2e_sum = sum(stats["e2e"] for stats in by_level.values())
        assert abs(e2e_sum - 1.0) < 1e-9, \
            f"sum of by_level[*]['e2e'] != 1.0: {e2e_sum} (by_level={by_level})"

        # --- run 2: default flags (no --repeats/--warmup) ---
        out2 = os.path.join(tmpdir, "out2.json")
        rc, stdout, stderr = run([
            sys.executable, BENCH_PATH,
            "--url", url,
            "--model", "fake",
            "--tools-file", tools_file,
            "--tool-subset", ",".join(TOOL_NAMES),
            "--limit", "3",
            "--out", out2,
        ])
        assert rc == 0, f"bench_toolcall.py (default run) exited {rc}\nstdout:\n{stdout}\nstderr:\n{stderr}"

        with open(out2) as f:
            data2 = json.load(f)
        summary2 = data2["summary"]
        for key in ("model", "url", "shuffle_seed", "n_cases", "by_level",
                    "overall_e2e_pct", "overall_lenient_pct", "overall_strict_pct",
                    "tool_sel_pct", "median_latency_s", "mean_latency_s"):
            assert key in summary2, f"classic summary key missing: {key}"
        r0 = data2["results"][0]
        assert isinstance(r0["latency_s"], float), f"results[0].latency_s not a float: {r0.get('latency_s')}"
        assert len(r0["reps"]) == 1, f"results[0].reps length != 1: {len(r0['reps'])}"

        # --- run 3: soak_npu.py against the fake server ---
        soak_out = os.path.join(tmpdir, "soak_out.json")
        rc, stdout, stderr = run([
            sys.executable, SOAK_PATH,
            "--url", url,
            "--n", "8",
            "--out", soak_out,
        ])
        assert rc == 0, f"soak_npu.py exited {rc}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        with open(soak_out) as f:
            soak_data = json.load(f)
        assert "provenance" in soak_data, "soak_npu.py output missing 'provenance'"
        assert "latency_stats" in soak_data, "soak_npu.py output missing 'latency_stats'"

        # --- run 4: phase_probe.py against the fake server, default --method
        # (auto). The fake server now supports SSE ("stream": true), and a
        # gen-tokens=32 probe yields 32 content chunks (>=4), so auto
        # detection is expected to pick "stream" for this run. ---
        phase_out = os.path.join(tmpdir, "phase_out.json")
        rc, stdout, stderr = run([
            sys.executable, PHASE_PROBE_PATH,
            "--url", url,
            "--model", "fake",
            "--lengths", "100,200",
            "--repeats", "2",
            "--gen-tokens", "32",
            "--warmup", "1",
            "--out", phase_out,
            "--label", "selftest",
        ])
        assert rc == 0, f"phase_probe.py exited {rc}\nstdout:\n{stdout}\nstderr:\n{stderr}"

        with open(phase_out) as f:
            phase_data = json.load(f)

        assert phase_data["provenance"]["schema_version"] == 3, \
            f"phase_probe provenance.schema_version != 3: {phase_data['provenance'].get('schema_version')}"
        r0 = phase_data["lengths"][0]["reps"][0]
        assert r0.get("max_tokens_honored") is True, f"honoring server flagged wrong: {r0}"
        assert r0.get("decode_tokens_measured") == 32, f"decode not measured over 32 tokens: {r0}"

        assert phase_data.get("method") == "stream", \
            f"phase_probe (auto method run) did not pick 'stream': {phase_data.get('method')!r}"
        assert isinstance(phase_data.get("method_reason"), str) and phase_data["method_reason"], \
            f"phase_probe method_reason is not a non-empty string: {phase_data.get('method_reason')!r}"

        phase_lengths = phase_data["lengths"]
        assert len(phase_lengths) == 2, \
            f"expected 2 lengths in phase_probe output, got {len(phase_lengths)}"

        for entry in phase_lengths:
            assert len(entry["reps"]) == 2, \
                f"length {entry.get('target')}: expected 2 reps, got {len(entry['reps'])}"

            prefill_mean = entry["prefill_tok_s"]["mean"]
            assert isinstance(prefill_mean, (int, float)) and prefill_mean > 0, \
                f"length {entry.get('target')}: prefill_tok_s.mean not finite positive: {prefill_mean!r}"

            decode_mean = entry["decode_tok_s"]["mean"]
            assert isinstance(decode_mean, (int, float)) and decode_mean > 0, \
                f"length {entry.get('target')}: decode_tok_s.mean not finite positive: {decode_mean!r}"

        # --- run 5: phase_probe.py with --method stream (explicit) ---
        phase_stream_out = os.path.join(tmpdir, "phase_stream_out.json")
        rc, stdout, stderr = run([
            sys.executable, PHASE_PROBE_PATH,
            "--url", url,
            "--model", "fake",
            "--lengths", "100,200",
            "--repeats", "2",
            "--gen-tokens", "32",
            "--warmup", "1",
            "--method", "stream",
            "--out", phase_stream_out,
            "--label", "selftest-stream",
        ])
        assert rc == 0, \
            f"phase_probe.py --method stream exited {rc}\nstdout:\n{stdout}\nstderr:\n{stderr}"

        with open(phase_stream_out) as f:
            phase_stream_data = json.load(f)

        assert phase_stream_data.get("method") == "stream", \
            f"phase_probe --method stream: method != 'stream': {phase_stream_data.get('method')!r}"

        for entry in phase_stream_data["lengths"]:
            prefill_mean = entry["prefill_tok_s"]["mean"]
            assert isinstance(prefill_mean, (int, float)) and prefill_mean > 0, \
                f"stream length {entry.get('target')}: prefill_tok_s.mean not finite positive: {prefill_mean!r}"
            decode_mean = entry["decode_tok_s"]["mean"]
            assert isinstance(decode_mean, (int, float)) and decode_mean > 0, \
                f"stream length {entry.get('target')}: decode_tok_s.mean not finite positive: {decode_mean!r}"

        # --- run 6: phase_probe.py against fake-400-long -- length=100
        # should calibrate fine, length=200 should fail calibration (HTTP
        # 400) and be skipped without discarding length=100's data. ---
        phase_400_out = os.path.join(tmpdir, "phase_400_out.json")
        rc, stdout, stderr = run([
            sys.executable, PHASE_PROBE_PATH,
            "--url", url,
            "--model", FAKE_400_LONG_MODEL,
            "--lengths", "100,200",
            "--repeats", "2",
            "--gen-tokens", "32",
            "--warmup", "1",
            "--out", phase_400_out,
            "--label", "selftest-400long",
        ])
        assert rc == 5, \
            f"phase_probe.py (fake-400-long run) exited {rc}, expected 5\nstdout:\n{stdout}\nstderr:\n{stderr}"
        assert os.path.exists(phase_400_out), \
            "phase_probe.py (fake-400-long run) did not write --out despite exit 5"

        with open(phase_400_out) as f:
            phase_400_data = json.load(f)

        p400_lengths = phase_400_data["lengths"]
        assert len(p400_lengths) == 2, f"expected 2 length entries, got {len(p400_lengths)}"
        assert "error" not in p400_lengths[0], \
            f"lengths[0] (target=100) unexpectedly has 'error': {p400_lengths[0]}"
        assert len(p400_lengths[0]["reps"]) == 2, \
            f"lengths[0]: expected 2 reps, got {len(p400_lengths[0]['reps'])}"
        assert "error" in p400_lengths[1], \
            f"lengths[1] (target=200) expected to have 'error': {p400_lengths[1]}"
        assert p400_lengths[1]["reps"] == [], \
            f"lengths[1].reps expected empty: {p400_lengths[1]['reps']}"

        # --- run 7: a reply whose tool call arrives as text must fail grading
        # and be flagged as a text tool call ---
        tc_out = os.path.join(tmpdir, "textcall.json")
        rc, stdout, stderr = run([
            sys.executable, BENCH_PATH, "--url", url, "--model", FAKE_TEXTCALL_MODEL,
            "--tools-file", tools_file, "--tool-subset", ",".join(TOOL_NAMES),
            "--limit", "2", "--out", tc_out,
        ])
        assert rc == 0, f"bench_toolcall.py (textcall run) exited {rc}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        with open(tc_out) as f:
            tc = json.load(f)
        tc0 = tc["results"][0]
        assert tc0["reps"][0]["text_tool_call"] is True, f"text tool call not flagged: {tc0['reps'][0]}"
        assert tc0["pass_count"] == 0, "a text tool call must not pass grading"
        assert tc["summary"]["text_tool_call_reps"] == 1, \
            f"text_tool_call_reps != 1: {tc['summary'].get('text_tool_call_reps')}"

        # --- run 8: a server that ignores max_tokens; decode must still be
        # measured over exactly --gen-tokens tokens, and the server's own
        # timing fields must be kept ---
        ig_out = os.path.join(tmpdir, "ignore_max.json")
        rc, stdout, stderr = run([
            sys.executable, PHASE_PROBE_PATH, "--url", url, "--model", FAKE_IGNORE_MAX_MODEL,
            "--lengths", "100", "--repeats", "2", "--gen-tokens", "32", "--warmup", "0",
            "--method", "stream", "--out", ig_out, "--label", "selftest-ignore-max",
        ])
        assert rc == 0, f"phase_probe.py (ignore-max run) exited {rc}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        with open(ig_out) as f:
            ig = json.load(f)
        for r in ig["lengths"][0]["reps"]:
            assert r.get("max_tokens_honored") is False, f"ignoring server not flagged: {r}"
            assert r.get("n_out") == FAKE_IGNORE_MAX_TOKENS, f"n_out != {FAKE_IGNORE_MAX_TOKENS}: {r}"
            assert r.get("decode_tokens_measured") == 32, f"decode window != 32 tokens: {r}"
            assert isinstance(r.get("decode_tok_s_full_generation"), (int, float)), \
                f"full-generation decode not kept for comparison: {r}"
            assert "timings" in (r.get("server_extra") or {}), f"server timing fields not kept: {r}"

        # --- run 9: identical-request quirk. Warm-up must not repeat case 0, so
        # case 0 still passes; a forced back-to-back repeat (one case, two
        # reps, no warm-up) must be retried, not graded as a failure ---
        qe_out = os.path.join(tmpdir, "quirk_bench.json")
        rc, stdout, stderr = run([
            sys.executable, BENCH_PATH, "--url", url, "--model", FAKE_IDENTICAL_EMPTY_MODEL,
            "--tools-file", tools_file, "--tool-subset", ",".join(TOOL_NAMES),
            "--limit", "3", "--warmup", "2", "--out", qe_out,
        ])
        assert rc == 0, f"bench (quirk warm-up run) exited {rc}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        with open(qe_out) as f:
            qe = json.load(f)
        assert qe["results"][0]["pass_count"] == 1, \
            f"case 0 lost to the identical-request quirk after warm-up: {qe['results'][0]['reps'][0]}"
        qr_out = os.path.join(tmpdir, "quirk_retry.json")
        rc, stdout, stderr = run([
            sys.executable, BENCH_PATH, "--url", url, "--model", FAKE_IDENTICAL_EMPTY_MODEL,
            "--tools-file", tools_file, "--tool-subset", ",".join(TOOL_NAMES),
            "--limit", "1", "--repeats", "2", "--warmup", "0", "--out", qr_out,
        ])
        assert rc == 0, f"bench (quirk retry run) exited {rc}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        with open(qr_out) as f:
            qr = json.load(f)
        reps_q = qr["results"][0]["reps"]
        assert reps_q[1].get("retried_empty_response") is True, f"empty reply not retried: {reps_q[1]}"
        assert qr["results"][0]["pass_count"] == 2, f"retried rep not graded on the real answer: {reps_q}"
        qp_out = os.path.join(tmpdir, "quirk_probe.json")
        rc, stdout, stderr = run([
            sys.executable, PHASE_PROBE_PATH, "--url", url, "--model", FAKE_IDENTICAL_EMPTY_MODEL,
            "--lengths", "100", "--repeats", "3", "--gen-tokens", "32", "--warmup", "1",
            "--method", "stream", "--out", qp_out, "--label", "selftest-quirk",
        ])
        assert rc == 0, f"phase_probe (quirk run) exited {rc}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        with open(qp_out) as f:
            qp = json.load(f)
        for r in qp["lengths"][0]["reps"]:
            assert r.get("decode_tok_s"), f"probe repetition hit the identical-request quirk: {r}"

        # --- run 10: bench_sampling.py must send its fields with every request,
        # warm-up included, record them, and leave grading unchanged ---
        sm_out = os.path.join(tmpdir, "sampling.json")
        with _LAST_LOCK:
            del _SEEN_SAMPLING[:]
        rc, stdout, stderr = run([
            sys.executable, SAMPLING_PATH, "--sampling", '{"top_k": 1, "seed": 7}',
            "--url", url, "--model", "fake", "--tools-file", tools_file,
            "--tool-subset", ",".join(TOOL_NAMES), "--limit", "2", "--repeats", "2",
            "--warmup", "1", "--out", sm_out, "--label", "selftest-sampling",
        ])
        assert rc == 0, f"bench_sampling.py exited {rc}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        with open(sm_out) as f:
            sm = json.load(f)
        smp = sm["provenance"]
        assert smp.get("diagnostic_mode") == "sampling", f"mode not recorded: {smp.get('diagnostic_mode')}"
        assert smp.get("sampling") == {"top_k": 1, "seed": 7}, f"sampling not recorded: {smp.get('sampling')}"
        assert smp["request_config"]["extra_body"].get("top_k") == 1, \
            f"request_config misses the sampling fields: {smp['request_config']}"
        assert smp.get("diagnostic_wrapper_sha256") and smp.get("harness_sha256"), "hashes missing"
        with _LAST_LOCK:
            seen = list(_SEEN_SAMPLING)
        assert len(seen) == 5, f"expected 1 warm-up + 4 timed requests, saw {len(seen)}"
        assert all(s == {"top_k": 1, "seed": 7, "temperature": 0} for s in seen), \
            f"a request lacked the sampling fields: {seen}"
        assert sm["results"][0]["pass_count"] == 2, \
            f"grading changed under bench_sampling.py: {sm['results'][0]['reps']}"

    except AssertionError as e:
        print(f"SELFTEST FAIL: {e}")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"SELFTEST FAIL (unexpected {type(e).__name__}): {e}")
        sys.exit(1)
    finally:
        server.shutdown()
        server.server_close()

    print("SELFTEST PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
