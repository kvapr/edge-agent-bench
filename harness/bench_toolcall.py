#!/usr/bin/env python3
"""Tool-calling accuracy benchmark for on-device LLM agents.

Runs the canonical 65-case suite (suites/test_cases.yaml, levels 1-4) or the
195-case paraphrase suite against any OpenAI-compatible
/v1/chat/completions endpoint and grades tool selection, command selection,
parameter extraction and the target device. The tool schema defaults to
suites/tools_nested.json.

Usage:
  python3 bench_toolcall.py --url http://<host>:<port> --model qwen3-8b \
      --out results/qwen3-8b.json [--limit N] [--no-thinking-kwarg]

  # measurement hygiene: repeat the whole suite, warm the engine up first,
  # and pin the tool schema file explicitly
  python3 bench_toolcall.py --url http://<host>:<port> --model qwen3-8b \
      --out results/qwen3-8b.json --repeats 5 --warmup 2 \
      --tools-file tools_flat.json

  # gate the run on board thermal/CPU idleness before each repetition
  python3 bench_toolcall.py --url http://<host>:<port> --model qwen3-8b \
      --out results/qwen3-8b.json --repeats 3 \
      --board-ssh user@<board> --readiness
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

import yaml  # noqa: E402

API_KEY = os.environ.get("BENCH_API_KEY", "")
OMIT_PARAMS = {p.strip() for p in os.environ.get("BENCH_OMIT_PARAMS", "").split(",") if p.strip()}

# Overridden by --max-tokens in main(); call_endpoint() reads this global so
# its signature/defaults stay unchanged for importers.
MAX_TOKENS = 512

SYSTEM_PROMPT = """\
You are a network device management assistant for managed PoE switches.

Conventions:
- Messages may start with @<device_name> (e.g. @poesw1). That is the target device: pass its name
  without the @ as device_name: '@poesw1' -> {"device_name": "poesw1"}.
- Read-only status/monitoring queries -> query_device_info.
- Configuration changes (enable/disable/set/assign/configure) -> configure_device_setting.
- Port troubleshooting, syslog, events -> diagnose_port_issues.
- How-to / documentation questions -> search_documentation.

CRITICAL parameter rules:
- Port numbers use the '1/N' string format: 'port 5' -> {"port": "1/5"}, 'interface 12' -> {"port": "1/12"}.
- VLAN IDs are integers: 'vlan 10' -> {"vlan_id": 10}.
- PoE power limits are integers (watts): '25W' or '25 watts' -> {"power_limit": 25}.
- Speeds and duplex are strings: '1000', '100', '10', 'auto', 'full', 'half'.
- Port descriptions keep the user's text verbatim.

Common command_name values for configure_device_setting:
enable_port (port), disable_port (port), configure_poe_mode (port, mode: enable|disable),
configure_port_speed (port, speed), configure_port_duplex (port, duplex),
configure_access_vlan (port, vlan_id), configure_poe_power_limit (port, power_limit),
set_port_description (port, description), configure_trunk_allowed_vlans (port, action, vlan_list), create_vlan (vlan_id).

Common command_name values for query_device_info:
show_port_status (port), show_all_ports_status, show_poe_status (port), show_all_poe_status,
show_poe_config (port), show_all_poe_config, show_poe_power_budget, show_cable_diagnostics (port),
show_all_cable_diagnostics, show_vlan_config, show_vlan_membership, show_system_info, show_ip_interface.

Always respond with exactly one tool call for actionable requests. Do not ask for
confirmation; the system handles confirmation of configuration changes itself.
"""


HARNESS_VERSION = "2.0"

DEFAULT_TOOLS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "suites", "tools_nested.json")

# Response keys defined by the OpenAI chat-completions format. Anything else a
# server adds (for example its own prefill/decode timing metrics) is kept per
# request under "server_extra" so it can be cross-checked against our timing.
STANDARD_RESP_KEYS = {"id", "object", "created", "model", "choices", "usage",
                      "system_fingerprint"}

# A reply that carries a tool call as plain text in message.content, rather
# than as a structured message.tool_calls entry, is unusable by an agent that
# reads the structured field. Grading is unchanged (such a reply fails); this
# only flags it so the failure mode is visible in the results.
TEXT_TOOL_CALL_RE = re.compile(
    r'<tool_call>|<function=|"name"\s*:\s*"[A-Za-z_]+"\s*,\s*"(arguments|parameters)"',
    re.I)


def load_tools_file(path):
    """Load an OpenAI-format tools array (list of {"type":"function",...})
    from a JSON file, for --tools-file."""
    with open(path) as f:
        return json.load(f)


def load_cases(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    cases = []
    for level, items in sorted(data.items()):
        for it in items or []:
            cases.append(
                {
                    "level": level,
                    "input": it["input"],
                    "expected_tool": it["expected_tool"],
                    "expected_command": it.get("expected_command"),
                    "expected_params": it.get("expected_params") or {},
                }
            )
    return cases


def call_endpoint(url, model, user_msg, tools, timeout, extra_body):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "tools": tools,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }
    body.update(extra_body)
    for p in OMIT_PARAMS:
        body.pop(p, None)
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        json.dumps(body).encode(),
        headers,
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    latency = time.time() - t0
    return resp, latency


def extract_call(resp):
    """Return (tool_name, arguments_dict) or (None, None)."""
    try:
        msg = resp["choices"][0]["message"]
    except (KeyError, IndexError):
        return None, None
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return None, None
    fn = tcs[0].get("function", {})
    args = fn.get("arguments", "{}")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return fn.get("name"), None
    return fn.get("name"), args


def norm(v):
    return str(v).strip().lower()


def grade(case, tool_name, args):
    g = {
        "ok_tool": False,
        "ok_command": False,
        "ok_params_strict": False,
        "ok_params_lenient": False,
        "ok_device": False,
    }
    if tool_name != case["expected_tool"]:
        return g
    g["ok_tool"] = True
    if args is None:
        return g
    g["ok_device"] = norm(args.get("device_name", "")) == "poesw1"
    cmd = args.get("command_name")
    if case["expected_command"] is None:
        g["ok_command"] = True
    else:
        g["ok_command"] = cmd == case["expected_command"]
    got = args.get("parameters")
    if not isinstance(got, dict):
        if case["expected_command"] is None:
            # Flat-schema case (e.g. test_cases_flat.yaml): there's no nested
            # "parameters" dict -- the command's own params were promoted to
            # top-level call arguments alongside device_name.
            got = {k: v for k, v in args.items() if k != "device_name"}
        else:
            got = {}
    exp = case["expected_params"]
    # lenient: every expected key present with string-equal value, extras ignored
    g["ok_params_lenient"] = all(
        k in got and norm(got[k]) == norm(v) for k, v in exp.items()
    )
    # strict: exact key set and type-faithful values
    g["ok_params_strict"] = (
        set(got.keys()) == set(exp.keys())
        and all(got[k] == v for k, v in exp.items())
    )
    return g


# --------------------------------------------------------------------------
# Measurement-hygiene helpers: per-request execution, provenance, and the
# optional board-readiness gate. None of these change the grading contract
# above -- they only add repeatability, warmup, and provenance around it.
# --------------------------------------------------------------------------

SPACER_MSG = "Spacer request sent by the benchmark harness; reply briefly."


def is_empty_response(resp):
    """True for a reply with no tool call, no text and zero token counts. Some
    servers (seen on GenieX v0.8.0) answer a request that is byte-identical to
    the previous one this way. It is an infrastructure artefact, not a model
    answer, so run_case retries it once after a different spacer request."""
    usage = resp.get("usage") or {}
    msg = ((resp.get("choices") or [{}])[0] or {}).get("message") or {}
    return (not msg.get("tool_calls") and not (msg.get("content") or "").strip()
            and not usage.get("completion_tokens") and not usage.get("prompt_tokens"))


def run_case(url, model, case, tools, timeout, extra_body):
    """Run one request for one case and return a single rep-result dict."""
    try:
        resp, latency = call_endpoint(url, model, case["input"], tools, timeout, extra_body)
        retried = False
        if is_empty_response(resp):
            try:
                call_endpoint(url, model, SPACER_MSG, tools, timeout, extra_body)
            except Exception:  # noqa: BLE001
                pass
            resp, latency = call_endpoint(url, model, case["input"], tools, timeout, extra_body)
            retried = True
            if is_empty_response(resp):
                raise RuntimeError("server_empty_response (persisted after one retry)")
        tool_name, call_args = extract_call(resp)
        g = grade(case, tool_name, call_args)
        msg = ((resp.get("choices") or [{}])[0] or {}).get("message") or {}
        content = msg.get("content") or ""
        usage = resp.get("usage") or {}
        server_extra = {k: v for k, v in resp.items() if k not in STANDARD_RESP_KEYS}
        row = {
            "got_tool": tool_name,
            "got_args": call_args,
            "latency_s": round(latency, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "usage": usage,
            "content": content[:4000],
            "text_tool_call": (not msg.get("tool_calls")) and bool(TEXT_TOOL_CALL_RE.search(content)),
            **g,
        }
        if server_extra:
            row["server_extra"] = server_extra
        if retried:
            row["retried_empty_response"] = True
        return row
    except Exception as e:  # noqa: BLE001
        return {
            "got_tool": None,
            "got_args": None,
            "latency_s": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "content": "",
            "text_tool_call": False,
            "ok_tool": False,
            "ok_command": False,
            "ok_params_strict": False,
            "ok_params_lenient": False,
            "ok_device": False,
            "error": f"{type(e).__name__}: {e}",
        }


def server_version(board_ssh, cmd):
    """Run cmd on the board over ssh and return its output: the version of
    the server binary that actually handles the requests. Never raises."""
    if not (board_ssh and cmd):
        return None
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", board_ssh, cmd],
            capture_output=True, text=True, timeout=60,
        )
        out = (r.stdout or "").strip()
        return out[:2000] if out else ((r.stderr or "").strip()[:300] or None)
    except Exception as e:  # noqa: BLE001
        return f"error: {type(e).__name__}: {e}"


THERMAL_SNIPPET = ('for z in /sys/class/thermal/thermal_zone*; do '
                   'echo "$(cat $z/type 2>/dev/null) $(cat $z/temp 2>/dev/null)"; done; '
                   'echo "@@load $(cut -d" " -f1 /proc/loadavg)"')


class ThermalSampler:
    """Background thread that samples the board's thermal zones over ssh every
    interval_s seconds while a run is in progress, so each result carries an
    under-load temperature trace and not only cool before/after snapshots.
    Records the hottest zone overall and the hottest CPU and NPU (nsp) zones.
    Never raises; a failed sample is skipped."""

    def __init__(self, board_ssh, interval_s):
        self.board_ssh = board_ssh
        self.interval_s = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._t0 = None
        self._th = None

    def _sample(self):
        try:
            r = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                 self.board_ssh, THERMAL_SNIPPET],
                capture_output=True, text=True, timeout=20,
            )
        except Exception:  # noqa: BLE001
            return None
        zones = {}
        load1 = None
        for line in (r.stdout or "").splitlines():
            if line.startswith("@@load"):
                try:
                    load1 = float(line.split()[1])
                except (IndexError, ValueError):
                    pass
                continue
            m = re.match(r"^(.*?)\s+(-?\d+)\s*$", line.strip())
            if m and m.group(1).strip():
                try:
                    zones[m.group(1).strip()] = int(m.group(2)) / 1000.0
                except ValueError:
                    pass
        if not zones:
            return None

        def mx(prefix):
            v = [x for k, x in zones.items() if k.startswith(prefix)]
            return round(max(v), 1) if v else None

        return {"t_s": round(time.time() - self._t0, 1), "all": round(max(zones.values()), 1),
                "cpu": mx("cpu"), "nsp": mx("nsp"), "load1": load1}

    def _loop(self):
        while not self._stop.is_set():
            smp = self._sample()
            if smp:
                self.samples.append(smp)
            self._stop.wait(self.interval_s)

    def start(self):
        if self.board_ssh and self.interval_s and self.interval_s > 0:
            self._t0 = time.time()
            self._th = threading.Thread(target=self._loop, daemon=True)
            self._th.start()
        return self

    def stop(self):
        if self._th:
            self._stop.set()
            self._th.join(timeout=30)
        return self.samples

    def summary(self):
        if not self.samples:
            return None
        a = [x["all"] for x in self.samples]
        cpu = [x["cpu"] for x in self.samples if x["cpu"] is not None]
        nsp = [x["nsp"] for x in self.samples if x["nsp"] is not None]
        return {"n_samples": len(a), "interval_s": self.interval_s,
                "max_all_c": max(a), "mean_all_c": round(sum(a) / len(a), 1),
                "max_cpu_c": max(cpu) if cpu else None, "max_nsp_c": max(nsp) if nsp else None}


def percentile_nearest_rank(values, p):
    if not values:
        return None
    s = sorted(values)
    k = max(1, min(len(s), math.ceil(p / 100.0 * len(s))))
    return s[k - 1]


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


def p95_nearest_rank(values):
    if not values:
        return None
    s = sorted(values)
    k = max(1, min(len(s), math.ceil(0.95 * len(s))))
    return s[k - 1]


# --- board provenance / readiness (optional, via ssh) ---------------------

BOARD_SNIPPET = (
    'echo "@@os"; grep -E "^(VERSION_ID|PRETTY_NAME)=" /etc/os-release; '
    'echo "@@kernel"; uname -r; '
    'echo "@@thermal"; for z in /sys/class/thermal/thermal_zone*; do '
    'echo "$(cat $z/type 2>/dev/null) $(cat $z/temp 2>/dev/null)"; done; '
    'echo "@@stat1"; head -1 /proc/stat; sleep 1; echo "@@stat2"; head -1 /proc/stat; '
    'echo "@@load"; cat /proc/loadavg; '
    'echo "@@docker"; docker ps --format "{{.Names}} {{.Image}} {{.Status}}" 2>/dev/null; '
    'echo "@@geniex"; docker exec gx geniex --version 2>/dev/null; '
    'echo "@@end"'
)

_BOARD_MARKERS = {"@@os", "@@kernel", "@@thermal", "@@stat1", "@@stat2",
                   "@@load", "@@docker", "@@geniex", "@@end"}


def parse_board_snapshot(stdout):
    """Tolerantly parse the BOARD_SNIPPET output. Missing pieces are fine;
    this never raises for malformed/partial input."""
    sections = {}
    cur = None
    for line in stdout.splitlines():
        s = line.strip()
        if s in _BOARD_MARKERS:
            cur = s[2:]
            sections[cur] = []
            continue
        if cur is not None:
            sections[cur].append(line)

    def sec(name):
        return sections.get(name, [])

    os_info = {}
    for line in sec("os"):
        line = line.strip()
        if "=" in line:
            k, _, v = line.partition("=")
            os_info[k.strip()] = v.strip().strip('"')

    kernel = next((l.strip() for l in sec("kernel") if l.strip()), None)

    thermal_zones = {}
    for line in sec("thermal"):
        m = re.match(r"^(.*?)\s+(-?\d+)\s*$", line.strip())
        if not m:
            continue
        ztype = m.group(1).strip()
        if not ztype:
            continue
        try:
            thermal_zones[ztype] = round(int(m.group(2)) / 1000.0, 1)
        except ValueError:
            continue

    def max_by_prefix(prefix):
        vals = [v for k, v in thermal_zones.items() if k.startswith(prefix)]
        return max(vals) if vals else None

    thermal_max_c = {
        "cpu": max_by_prefix("cpu"),
        "nsp": max_by_prefix("nsp"),
        "all": max(thermal_zones.values()) if thermal_zones else None,
    }

    cpu_busy = None
    try:
        l1 = next(l for l in sec("stat1") if l.strip().startswith("cpu"))
        l2 = next(l for l in sec("stat2") if l.strip().startswith("cpu"))
        f1 = [int(x) for x in l1.split()[1:]]
        f2 = [int(x) for x in l2.split()[1:]]
        idle_delta = f2[3] - f1[3]
        iowait_delta = f2[4] - f1[4]
        total_delta = sum(f2) - sum(f1)
        if total_delta > 0:
            cpu_busy = round(1 - (idle_delta + iowait_delta) / total_delta, 3)
    except (StopIteration, ValueError, IndexError):
        cpu_busy = None

    loadavg = next((l.strip() for l in sec("load") if l.strip()), None)
    containers = [l for l in sec("docker") if l.strip()]
    geniex_version = [l for l in sec("geniex") if l.strip()]

    return {
        "os": os_info,
        "kernel": kernel,
        "loadavg": loadavg,
        "containers": containers,
        "gx_container_geniex_version": geniex_version,
        "thermal_max_c": thermal_max_c,
        "cpu_busy": cpu_busy,
        "thermal_zones": thermal_zones,
    }


def board_snapshot(board_ssh):
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", board_ssh, BOARD_SNIPPET],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    if r.returncode != 0:
        err = (r.stderr or "").strip() or f"ssh exited {r.returncode}"
        return {"error": err[:300]}
    try:
        return parse_board_snapshot(r.stdout)
    except Exception as e:  # noqa: BLE001
        return {"error": f"parse failed: {type(e).__name__}: {e}"}


def wait_for_readiness(board_ssh, max_temp_c, max_cpu_busy, poll_s, timeout_s, label, readiness_log):
    """Poll the board until thermal_max_c['all'] < max_temp_c and cpu_busy <
    max_cpu_busy, or timeout_s elapses. Returns True/False."""
    t0 = time.time()
    while True:
        snap = board_snapshot(board_ssh)
        temp = busy = None
        if "error" not in snap:
            temp = (snap.get("thermal_max_c") or {}).get("all")
            busy = snap.get("cpu_busy")
        ready = temp is not None and busy is not None and temp < max_temp_c and busy < max_cpu_busy
        temp_disp = f"{temp:.1f}" if isinstance(temp, (int, float)) else "?"
        busy_disp = f"{busy:.2f}" if isinstance(busy, (int, float)) else "?"
        print(f"readiness: temp {temp_disp}C busy {busy_disp} -> {'ready' if ready else 'waiting'}",
              flush=True)
        if ready:
            readiness_log.append({"before": label, "snapshot": snap})
            return True
        if time.time() - t0 >= timeout_s:
            return False
        time.sleep(poll_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cases", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "suites", "test_cases.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="run only first N cases")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--no-thinking-kwarg", action="store_true",
                    help="omit chat_template_kwargs enable_thinking=false")
    ap.add_argument("--tool-subset", default="",
                    help="comma-separated tool names to offer (default: all 12)")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="shuffle cases with random.Random(N) after loading, "
                         "before --limit is applied (default: no shuffle)")
    ap.add_argument("--tools-file", default=DEFAULT_TOOLS_FILE,
                    help="OpenAI-format tools JSON array (default: suites/tools_nested.json)")
    ap.add_argument("--label", default="",
                    help="free-text configuration label stored in the output")
    ap.add_argument("--server-version-cmd", default=None,
                    help="command run on --board-ssh whose output identifies the "
                         "server binary that handles the requests (stored in provenance)")
    ap.add_argument("--thermal-trace-s", type=float, default=20.0,
                    help="with --board-ssh, sample board thermal zones every N seconds "
                         "for the whole run (0 disables)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="run the whole case list this many times in sequence "
                         "(rep 1 all cases, rep 2 all cases, ...)")
    ap.add_argument("--warmup", type=int, default=0,
                    help="send the first case this many times before timing, discard responses")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="max_tokens for each chat completion request")
    ap.add_argument("--board-ssh", default=None,
                    help="USER@HOST to collect board provenance (and, with --readiness, "
                         "to gate the run) over ssh")
    ap.add_argument("--readiness", action="store_true",
                    help="poll --board-ssh before warmup and before each repetition; "
                         "requires --board-ssh")
    ap.add_argument("--readiness-poll-s", type=int, default=10)
    ap.add_argument("--readiness-timeout-s", type=int, default=600)
    ap.add_argument("--max-temp-c", type=float, default=55.0)
    ap.add_argument("--max-cpu-busy", type=float, default=0.30)
    args = ap.parse_args()

    if args.readiness and not args.board_ssh:
        ap.error("--readiness requires --board-ssh")

    global MAX_TOKENS
    MAX_TOKENS = args.max_tokens

    tools = load_tools_file(args.tools_file)
    if args.tool_subset:
        keep = {t.strip() for t in args.tool_subset.split(",")}
        tools = [t for t in tools if t["function"]["name"] in keep]
    cases = load_cases(args.cases)
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(cases)
    if args.limit:
        cases = cases[: args.limit]
    extra = {} if args.no_thinking_kwarg else {
        "chat_template_kwargs": {"enable_thinking": False},
        "enable_think": False,  # GenieX reads this top-level field (ignored by others)
    }

    # --- provenance (gathered up front; board/board_after filled in below) ---
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    tools_source = os.path.relpath(os.path.abspath(args.tools_file), repo_root)
    tools_source_extra = {"tools_sha256": sha256_file(args.tools_file)}
    provenance = {
        "schema_version": 3,
        "harness_version": HARNESS_VERSION,
        "label": args.label,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "server_version": server_version(args.board_ssh, args.server_version_cmd),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "harness_sha256": sha256_file(os.path.abspath(__file__)),
        "cases_file": os.path.relpath(os.path.abspath(args.cases), repo_root),
        "cases_sha256": sha256_file(args.cases),
        "tools_source": tools_source,
        **tools_source_extra,
        "tool_names": [t["function"]["name"] for t in tools],
        "request_config": {
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "extra_body": extra,
            "omit_params": sorted(OMIT_PARAMS),
        },
        "endpoint_models": get_endpoint_models(args.url),
        "board": None,
    }

    board_before = None
    board_after = None
    if args.board_ssh and not args.readiness:
        board_before = board_snapshot(args.board_ssh)

    readiness_log = []
    repeats = args.repeats if args.repeats and args.repeats > 0 else 1
    sampler = ThermalSampler(args.board_ssh, args.thermal_trace_s).start()

    if args.readiness:
        ok = wait_for_readiness(args.board_ssh, args.max_temp_c, args.max_cpu_busy,
                                 args.readiness_poll_s, args.readiness_timeout_s,
                                 "warmup", readiness_log)
        if not ok:
            print(f"readiness: board not ready within {args.readiness_timeout_s}s; "
                  f"aborting without writing {args.out}", flush=True)
            sys.exit(3)

    if args.warmup and cases:
        # Warm up with inputs other than the first timed case: a request that
        # repeats the previous one verbatim can get an empty reply (see
        # is_empty_response), and consecutive warm-ups must differ too.
        pool = [c["input"] for c in reversed(cases[1:])] or [SPACER_MSG]
        for w in range(args.warmup):
            try:
                call_endpoint(args.url, args.model, pool[w % len(pool)], tools, args.timeout, extra)
            except Exception:  # noqa: BLE001
                pass

    results = [{"idx": i, **case, "reps": []} for i, case in enumerate(cases)]

    for rep in range(1, repeats + 1):
        if args.readiness:
            ok = wait_for_readiness(args.board_ssh, args.max_temp_c, args.max_cpu_busy,
                                     args.readiness_poll_s, args.readiness_timeout_s,
                                     f"rep{rep}", readiness_log)
            if not ok:
                print(f"readiness: board not ready within {args.readiness_timeout_s}s; "
                      f"aborting without writing {args.out}", flush=True)
                sys.exit(3)
        for i, case in enumerate(cases):
            rep_row = run_case(args.url, args.model, case, tools, args.timeout, extra)
            results[i]["reps"].append(rep_row)
            ok = rep_row.get("ok_tool") and rep_row.get("ok_command") and rep_row.get("ok_params_lenient")
            prefix = f"r{rep} " if repeats > 1 else ""
            print(f"[{prefix}{i+1}/{len(cases)}] {'PASS' if ok else 'FAIL'} "
                  f"{case['input'][:50]!r} -> {rep_row.get('got_tool')}"
                  f" ({rep_row.get('latency_s', '-')}s)", flush=True)

    sampler.stop()
    if args.board_ssh:
        if args.readiness:
            board_before = readiness_log[0]["snapshot"] if readiness_log else None
        board_after = board_snapshot(args.board_ssh)
    provenance["board"] = board_before
    provenance["board_after"] = board_after

    # --- per-case rollups across reps ---
    for row in results:
        reps = row["reps"]
        r1 = reps[0] if reps else {}
        row["got_tool"] = r1.get("got_tool")
        row["got_args"] = r1.get("got_args")
        row["content"] = r1.get("content", "")
        row["completion_tokens"] = r1.get("completion_tokens")
        row["prompt_tokens"] = r1.get("prompt_tokens")
        row["text_tool_call_reps"] = sum(1 for rr in reps if rr.get("text_tool_call"))
        row["ok_tool"] = r1.get("ok_tool", False)
        row["ok_command"] = r1.get("ok_command", False)
        row["ok_params_strict"] = r1.get("ok_params_strict", False)
        row["ok_params_lenient"] = r1.get("ok_params_lenient", False)
        row["ok_device"] = r1.get("ok_device", False)
        if "error" in r1:
            row["error"] = r1["error"]

        latencies_s = [rr["latency_s"] for rr in reps if rr.get("latency_s") is not None]
        row["latencies_s"] = latencies_s
        if latencies_s:
            row["latency_s"] = round(sum(latencies_s) / len(latencies_s), 2)
            row["latency_mean_s"] = round(statistics.mean(latencies_s), 3)
            row["latency_std_s"] = (
                round(statistics.stdev(latencies_s), 3) if len(latencies_s) >= 2 else None
            )
        else:
            row["latency_s"] = None
            row["latency_mean_s"] = None
            row["latency_std_s"] = None

        e2e_flags = [
            bool(rr.get("ok_tool") and rr.get("ok_command") and rr.get("ok_params_lenient")
                 and rr.get("ok_device"))
            for rr in reps
        ]
        row["pass_count"] = sum(e2e_flags)
        row["n_reps"] = len(reps)
        row["stable"] = len(set(e2e_flags)) <= 1

    # --- summary ---
    # by_level and the overall per-rep percentages are computed together, one
    # pass per rep, so they stay consistent with each other under --repeats:
    # by_level[*] holds the MEAN per-rep count for that level (2 decimals);
    # overall_e2e_pct etc. are the mean of the per-rep overall percentages.
    levels = []
    for r in results:
        if r["level"] not in levels:
            levels.append(r["level"])
    level_n = {lv: sum(1 for r in results if r["level"] == lv) for lv in levels}
    level_rep_counts = {
        lv: {"tool": [], "cmd": [], "params_l": [], "params_s": [], "e2e": []} for lv in levels
    }

    n_cases = len(results)
    e2e_pct_by_rep, tool_sel_pct_by_rep = [], []
    lenient_pct_by_rep, strict_pct_by_rep = [], []
    for rep_idx in range(repeats):
        tool_c = cmd_c = pl_c = ps_c = e2e_c = 0
        lvl_counts = {lv: {"tool": 0, "cmd": 0, "params_l": 0, "params_s": 0, "e2e": 0} for lv in levels}
        for row in results:
            reps = row["reps"]
            if rep_idx >= len(reps):
                continue
            rr = reps[rep_idx]
            ok_tool = bool(rr.get("ok_tool"))
            ok_cmd = ok_tool and bool(rr.get("ok_command"))
            ok_pl = ok_cmd and bool(rr.get("ok_params_lenient"))
            ok_ps = ok_cmd and bool(rr.get("ok_params_strict"))
            ok_e2e = ok_pl and bool(rr.get("ok_device"))
            tool_c += ok_tool
            cmd_c += ok_cmd
            pl_c += ok_pl
            ps_c += ok_ps
            e2e_c += ok_e2e
            lv = row["level"]
            lvl_counts[lv]["tool"] += ok_tool
            lvl_counts[lv]["cmd"] += ok_cmd
            lvl_counts[lv]["params_l"] += ok_pl
            lvl_counts[lv]["params_s"] += ok_ps
            lvl_counts[lv]["e2e"] += ok_e2e
        denom = max(1, n_cases)
        e2e_pct_by_rep.append(round(100 * e2e_c / denom, 1))
        tool_sel_pct_by_rep.append(round(100 * tool_c / denom, 1))
        lenient_pct_by_rep.append(round(100 * pl_c / denom, 1))
        strict_pct_by_rep.append(round(100 * ps_c / denom, 1))
        for lv in levels:
            for key in ("tool", "cmd", "params_l", "params_s", "e2e"):
                level_rep_counts[lv][key].append(lvl_counts[lv][key])

    by_level = {}
    for lv in levels:
        entry = {"n": level_n[lv]}
        for key in ("tool", "cmd", "params_l", "params_s", "e2e"):
            vals = level_rep_counts[lv][key]
            entry[key] = round(sum(vals) / len(vals), 2) if vals else 0.0
        by_level[lv] = entry

    def mean_round1(vals):
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    unstable_cases = [row["idx"] for row in results if not row["stable"]]
    pts = [r["prompt_tokens"] for r in results if isinstance(r.get("prompt_tokens"), (int, float))]
    prompt_tokens_median = statistics.median(pts) if pts else None
    lat = [row["latency_s"] for row in results if row.get("latency_s") is not None]

    summary = {
        "model": args.model,
        "url": args.url,
        "shuffle_seed": args.shuffle_seed,
        "n_cases": len(results),
        "by_level": by_level,
        "overall_e2e_pct": mean_round1(e2e_pct_by_rep),
        "overall_lenient_pct": mean_round1(lenient_pct_by_rep),
        "overall_strict_pct": mean_round1(strict_pct_by_rep),
        "tool_sel_pct": mean_round1(tool_sel_pct_by_rep),
        "median_latency_s": sorted(lat)[len(lat) // 2] if lat else None,
        "mean_latency_s": round(sum(lat) / len(lat), 2) if lat else None,
        "repeats": repeats,
        "warmup": args.warmup,
        "e2e_pct_by_rep": e2e_pct_by_rep,
        "tool_sel_pct_by_rep": tool_sel_pct_by_rep,
        "overall_e2e_std": round(statistics.stdev(e2e_pct_by_rep), 3) if len(e2e_pct_by_rep) >= 2 else None,
        "tool_sel_std": round(statistics.stdev(tool_sel_pct_by_rep), 3) if len(tool_sel_pct_by_rep) >= 2 else None,
        "unstable_cases": unstable_cases,
        "latency_p95_s": p95_nearest_rank(lat),
        "latency_p5_s": percentile_nearest_rank(lat, 5),
        "latency_p50_s": round(statistics.median(lat), 3) if lat else None,
        "prompt_tokens_median": prompt_tokens_median,
        "text_tool_call_reps": sum(r["text_tool_call_reps"] for r in results),
        "label": args.label,
        "thermal": sampler.summary(),
    }

    output = {"summary": summary, "results": results, "provenance": provenance,
              "thermal_trace": sampler.samples}
    if args.readiness:
        output["readiness"] = readiness_log

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
