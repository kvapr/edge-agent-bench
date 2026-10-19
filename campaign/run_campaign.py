#!/usr/bin/env python3
"""edge-agent-bench v2 campaign runner.

Runs every model x serving path x task in campaign/v2_models.json on one
board, strictly one at a time, and is safe to stop and restart:

- a task whose output file already exists and parses is skipped, so a rerun
  resumes where the previous run stopped;
- the NPU serves one model at a time, so before any NPU step the runner stops
  the board's production container (if one is configured) and resets the
  compute DSPs, and it restarts production when NPU work ends, on errors, and
  on SIGINT/SIGTERM;
- a model that fails to load on a path gets a load_failure.json holding the
  server log tail instead of results, and the campaign moves on.

Board details come from the environment so nothing lab-specific is stored:
  EAB_BOARD_SSH        user@host for ssh and scp                    (required)
  EAB_BOARD_IP         host for HTTP                                (required)
  EAB_PROD_CONTAINER   docker container that shares the NPU         (optional)
  EAB_PROD_PORT        its HTTP port, to confirm it serves again   (default 18181)
  EAB_ARTIFACTS        local folder of GGUF files and bundles       (default ~/eab-artifacts)
  EAB_RUNS             local folder for raw results                 (default ~/eab-runs/v2)
  EAB_GENIEX_DIR       GenieX CLI folder on the board

Usage:
  python3 campaign/run_campaign.py --dry-run
  python3 campaign/run_campaign.py [--models a,b] [--paths cpu,npu_llama,compiled] [--mode interleave|block]
  python3 campaign/run_campaign.py --status
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SPEC_PATH = os.path.join(REPO, "campaign", "v2_models.json")
BOARD = os.environ.get("EAB_BOARD_SSH")
IP = os.environ.get("EAB_BOARD_IP")
PROD = os.environ.get("EAB_PROD_CONTAINER") or None
PROD_PORT = int(os.environ.get("EAB_PROD_PORT", "18181"))
ART = os.path.expanduser(os.environ.get("EAB_ARTIFACTS", "~/eab-artifacts"))
RUNS = os.path.expanduser(os.environ.get("EAB_RUNS", "~/eab-runs/v2"))
GXD = os.environ.get("EAB_GENIEX_DIR", "/root/geniex-v080/geniex-cli-linux-arm64-v0.8.0")
DATA = "/root/eab-data"
INC = "/root/eab-incoming"
SERVE_LOG = "/root/eab-serve.log"

PATHS = ("cpu", "npu_llama", "compiled")
PORTS = {"cpu": 18383, "npu_llama": 18384, "compiled": 18384}
SUBSET = "query_device_info,configure_device_setting,diagnose_port_issues,search_documentation"
EXTRA = '{"enable_think": false, "chat_template_kwargs": {"enable_thinking": false}}'
# Protocol (v2): canonical suite 3 repetitions, paraphrase suite 1 repetition,
# throughput at the agent-like prompt lengths and at a chat-like shape.
BENCH = {"canonical": ("suites/test_cases.yaml", 3), "paraphrase": ("suites/test_cases_paraphrase.yaml", 1)}
PROBES = {"probe_agent": ("256,512,1024,2048,2800", 128), "probe_chat": ("100", 300)}
TASKS = ("canonical", "paraphrase", "probe_agent", "probe_chat")
# Diagnostic variants of the canonical suite, run after the matrix (--diagnostics)
# on every configuration whose prompts never contained the tool definitions,
# plus the calibration configurations listed in the spec. Results are written
# to diag_<mode>.json and carry mode=<mode> in their label.
DIAG = {"tools_in_prompt": "bench_tools_in_prompt.py"}
# llama.cpp context matches the 4,096-token compiled bundles; the model stays
# loaded between tasks; the HTP power mode is left at the server default
# (burst) and recorded.
SERVER_OPTS = "--nctx 4096 --keepalive 3600"
GENV = f"ADSP_LIBRARY_PATH={GXD}/qairt/htp-files LD_LIBRARY_PATH={GXD}"
GX = f"env {GENV} {GXD}/geniex --data-dir {DATA} --skip-update"
VERSION_CMD = f"LD_LIBRARY_PATH={GXD} {GXD}/geniex version"
BOUNCE = ('for r in /sys/class/remoteproc/remoteproc*; do n=$(cat $r/name); '
          'if [ "$n" = cdsp ] || [ "$n" = cdsp1 ]; then echo stop > $r/state 2>/dev/null; sleep 1; '
          'echo start > $r/state 2>/dev/null; fi; done; sleep 2')


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def run(cmd, timeout=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def ssh(remote, timeout=120):
    return run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", BOARD, remote], timeout=timeout)


def scp(local, remote_dir, timeout=7200):
    r = run(["scp", "-q", "-o", "BatchMode=yes", local, f"{BOARD}:{remote_dir}/"], timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"scp failed: {r.stderr[-300:]}")


def http_models(port):
    try:
        with urllib.request.urlopen(f"http://{IP}:{port}/v1/models", timeout=8) as r:
            return [m["id"] for m in json.loads(r.read()).get("data", [])]
    except Exception:  # noqa: BLE001
        return None


def disk_free_gb():
    r = ssh("df -k / | tail -1 | awk '{print $4}'", timeout=30)
    try:
        return int(r.stdout.strip()) / 1e6
    except ValueError:
        return 0.0


def sha256(path, cache={}):  # noqa: B006  (memo across calls)
    if path not in cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        cache[path] = h.hexdigest()
    return cache[path]


def bundle_info(path):
    try:
        with zipfile.ZipFile(path) as z:
            cfg = next(n for n in z.namelist() if n.endswith("genie_config.json"))
            d = json.loads(z.read(cfg))["dialog"]
            parts = sorted((i.file_size for i in z.infolist() if "_of_" in i.filename and i.filename.endswith(".bin")),
                           reverse=True)
            return {"context": d["context"]["size"], "parts": len(parts),
                    "largest_part_gb": round(parts[0] / 1e9, 3) if parts else None}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def expected_size(local):
    """Expected byte size of an artifact from the download manifest, or None."""
    try:
        man = json.load(open(os.path.join(ART, "manifest.json")))
    except (OSError, json.JSONDecodeError):
        man = []
    for o in man:
        if os.path.basename(o.get("url", "")) == os.path.basename(local) and o.get("bytes", 0) > 0:
            return o["bytes"]
    return json.load(open(os.path.join(ART, "extra_sizes.json"))).get(os.path.basename(local)) \
        if os.path.exists(os.path.join(ART, "extra_sizes.json")) else None


def wait_complete(local, max_wait_s=12 * 3600):
    """Block until a still-downloading artifact reaches its expected size, so
    a truncated file is never staged."""
    exp = expected_size(local)
    t0 = time.time()
    while True:
        cur = os.path.getsize(local) if os.path.exists(local) else -1
        if (exp is None and cur > 0) or (exp and cur == exp):
            return
        if time.time() - t0 > max_wait_s:
            raise RuntimeError(f"artifact still incomplete: {local} ({cur} of {exp} bytes)")
        log(f"  waiting for download: {os.path.basename(local)} {max(cur, 0) / 1e9:.2f} of {(exp or 0) / 1e9:.2f} GB")
        time.sleep(120)


class Board:
    def __init__(self):
        self.prod_was_up = None
        self.prod_up = None

    def probe_prod(self):
        if not PROD:
            return
        r = ssh(f"docker ps --format '{{{{.Names}}}}' | grep -x {PROD} || true", timeout=30)
        self.prod_up = bool(r.stdout.strip())
        if self.prod_was_up is None:
            self.prod_was_up = self.prod_up
        log(f"production container {PROD}: {'running' if self.prod_up else 'stopped'}")

    def stop_servers(self):
        ssh('pkill -f "geniex-v08[0]"; sleep 2; true', timeout=60)

    def prod_down(self):
        if PROD and self.prod_up is not False:
            log(f"stopping production container {PROD} for NPU work")
            ssh(f"docker stop {PROD} >/dev/null 2>&1; true", timeout=180)
            self.prod_up = False

    def prod_restore(self, force=False):
        if not PROD or self.prod_up is True or not (force or self.prod_was_up):
            return
        log(f"restoring production container {PROD}")
        self.stop_servers()
        ssh(BOUNCE, timeout=120)
        ssh(f"docker start {PROD} >/dev/null 2>&1; true", timeout=180)
        for _ in range(40):
            if http_models(PROD_PORT):
                log("production is serving again")
                self.prod_up = True
                return
            time.sleep(5)
        self.prod_up = True
        log("WARNING: production started but did not list models within 200 s")

    def registered(self, reg):
        return ssh(f"test -d {DATA}/models/qualcomm/{reg} && echo yes", timeout=30).stdout.strip() == "yes"

    def stage(self, m, kind):
        reg = f"{m['name']}-q4_0" if kind == "gguf" else f"{m['name']}-w4a16"
        if self.registered(reg):
            return reg
        local = os.path.join(ART, m[kind])
        wait_complete(local)
        size_gb = os.path.getsize(local) / 1e9
        need = size_gb * (2.2 if kind == "bundle" else 1.2) + 3
        free = disk_free_gb()
        if free < need:
            raise RuntimeError(f"board disk too small to stage {reg}: need {need:.1f} GB, free {free:.1f} GB")
        rdir = f"{INC}/{reg}"
        ssh(f"rm -rf {rdir}; mkdir -p {rdir}", timeout=60)
        log(f"staging {kind} for {m['name']} ({size_gb:.2f} GB)")
        scp(local, rdir)
        src = f"{rdir}/{os.path.basename(local)}" if kind == "bundle" else rdir
        name = f"{reg}:Q4_0" if kind == "gguf" else reg
        mtype = m.get("model_type", "llm")
        r = ssh(f"timeout 1800 {GX} pull {name} --model-hub localfs --local-path {src} "
                f"--model-type {mtype} </dev/null 2>&1 | tail -4; rm -rf {rdir}", timeout=2000)
        if not self.registered(reg):
            raise RuntimeError(f"registration failed for {reg}: {r.stdout[-400:]}")
        return reg

    def unstage(self, reg):
        ssh(f"rm -rf {DATA}/models/qualcomm/{reg}", timeout=120)

    def start_server(self, path):
        self.stop_servers()
        compute = "cpu" if path == "cpu" else "npu"
        if compute == "npu":
            ssh(BOUNCE, timeout=120)
        cmd = (f"nohup {GX} --log info serve --host 0.0.0.0:{PORTS[path]} -c {compute} {SERVER_OPTS} "
               f"> {SERVE_LOG} 2>&1 &")
        ssh(cmd + " echo started", timeout=60)
        return cmd


def served_id(port, reg, bare=False, wait_s=180):
    """The id the server lists for a registered model. GenieX v0.8.0 lists a
    compiled bundle as 'qualcomm/<name>:w4a16' but rejects that exact id
    ("quantization 'W4A16' not found") and accepts the bare 'qualcomm/<name>',
    so bundles are requested by their bare id (bare=True)."""
    deadline = time.time() + wait_s
    while time.time() < deadline:
        ids = http_models(port) or []
        hit = [i for i in ids if i.split("/")[-1].split(":")[0] == reg]
        if hit:
            return hit[0].split(":")[0] if bare else hit[0]
        time.sleep(4)
    return None


def load_check(port, model_id):
    body = {"model": model_id, "messages": [{"role": "user", "content": "Reply with the single word ok."}],
            "max_tokens": 8, "temperature": 0, "enable_think": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://{IP}:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.loads(r.read())
        ok = "choices" in d
        return ok, {"load_and_first_reply_s": round(time.time() - t0, 2), "reply": json.dumps(d)[:300]}
    except urllib.error.HTTPError as e:
        return False, {"http_status": e.code, "body": e.read().decode(errors="replace")[:600]}
    except Exception as e:  # noqa: BLE001
        return False, {"error": f"{type(e).__name__}: {e}"}


def out_path(m, path, name):
    return os.path.join(RUNS, m["name"], path, name)


def is_done(p):
    try:
        d = json.load(open(p))
        return "summary" in d or "lengths" in d or "load_failure" in d
    except (OSError, json.JSONDecodeError):
        return False


def run_task(m, path, task, port, model_id, mode=None):
    op = out_path(m, path, f"diag_{mode}.json" if mode else f"{task}.json")
    if is_done(op):
        log(f"  skip {task} (done)")
        return
    os.makedirs(os.path.dirname(op), exist_ok=True)
    suite_or_probe = task if task in BENCH else task.replace("probe_", "")
    label = f"model={m['name']};path={path};{'suite' if task in BENCH else 'probe'}={suite_or_probe}"
    if mode:
        label += f";mode={mode}"
    common = ["--url", f"http://{IP}:{port}", "--model", model_id, "--board-ssh", BOARD, "--readiness",
              "--server-version-cmd", VERSION_CMD, "--thermal-trace-s", "20", "--label", label, "--out", op]
    if task in BENCH:
        cases, reps = BENCH[task]
        cmd = [sys.executable, os.path.join(REPO, "harness", DIAG[mode] if mode else "bench_toolcall.py"), *common,
               "--cases", os.path.join(REPO, cases), "--tool-subset", SUBSET,
               "--repeats", str(reps), "--warmup", "1", "--timeout", "900"]
    else:
        lengths, gen = PROBES[task]
        cmd = [sys.executable, os.path.join(REPO, "harness", "phase_probe.py"), *common,
               "--lengths", lengths, "--gen-tokens", str(gen), "--repeats", "5", "--warmup", "1",
               "--method", "stream", "--extra-json", EXTRA, "--timeout", "900"]
    for attempt in (1, 2):
        log(f"  run {task} (attempt {attempt})")
        t0 = time.time()
        r = run(cmd, timeout=48 * 3600)
        with open(op + ".log", "a") as f:
            f.write(f"--- attempt {attempt} rc={r.returncode} {now()}\n{r.stdout[-6000:]}\n{r.stderr[-3000:]}\n")
        log(f"  {task} rc={r.returncode} in {(time.time() - t0) / 60:.1f} min")
        if is_done(op):
            return
        if r.returncode == 3:  # readiness gate timed out (board too hot or busy)
            log("  board not ready; waiting 5 min before retrying")
            time.sleep(300)
            continue
        break
    log(f"  WARNING: {task} produced no result file")


def run_path(board, m, path, keep):
    kind = "gguf" if path in ("cpu", "npu_llama") else "bundle"
    if not m.get(kind):
        return
    fail = out_path(m, path, "load_failure.json")
    if all(is_done(out_path(m, path, f"{t}.json")) for t in TASKS) or is_done(fail):
        log(f"{m['name']} / {path}: already complete")
        return
    served = serve(board, m, path, kind, fail, "server.json")
    if served:
        port, model_id = served
        for t in TASKS:
            run_task(m, path, t, port, model_id)
    board.stop_servers()
    # Free board disk once an artifact has no remaining user: the bundle after
    # the compiled path, the GGUF after the NPU llama.cpp path (its last user).
    if not keep and (kind == "bundle" or path == "npu_llama"):
        board.unstage(f"{m['name']}-q4_0" if kind == "gguf" else f"{m['name']}-w4a16")


def run_diag(board, m, path, mode):
    kind = "gguf" if path in ("cpu", "npu_llama") else "bundle"
    if is_done(out_path(m, path, f"diag_{mode}.json")):
        log(f"{m['name']} / {path} / {mode}: already complete")
        return None
    served = serve(board, m, path, kind, out_path(m, path, f"load_failure.diag_{mode}.json"),
                   f"server.diag_{mode}.json", banner=f" / diagnostic {mode}")
    if served:
        run_task(m, path, "canonical", *served, mode=mode)
    board.stop_servers()
    return f"{m['name']}-q4_0" if kind == "gguf" else f"{m['name']}-w4a16"


def tools_missing(m, path):
    """True when the native canonical run's median server-reported prompt was
    too short to contain the tool definitions (threshold from the analysis)."""
    sys.path.insert(0, os.path.join(REPO, "analysis"))
    import analyze  # noqa: E402
    try:
        d = json.load(open(out_path(m, path, "canonical.json")))
    except (OSError, json.JSONDecodeError):
        return False
    toks = sorted(r["prompt_tokens"] for c in d["results"] for r in c["reps"] if r.get("prompt_tokens"))
    return bool(toks) and toks[len(toks) // 2] < analyze.TOOLS_DELIVERED_MIN_PROMPT_TOKENS


def diag_plan(spec, models):
    calib = {tuple(x) for x in spec.get("diagnostic_calibration", [])}
    steps = [(m, p, mode) for m in models for p in PATHS for mode in DIAG
             if m.get("gguf" if p != "compiled" else "bundle")
             and ((m["name"], p) in calib or tools_missing(m, p))]
    # NPU steps first, then CPU, so production is stopped and restored once
    return sorted(steps, key=lambda s: s[1] == "cpu")


def serve(board, m, path, kind, fail, side_name, banner=""):
    """Stage the artifact, start the server for the path and check that the
    model answers. Returns (port, model_id), or None after recording a load
    failure in `fail`."""
    if path == "cpu":
        board.prod_restore()          # production can serve while the CPU works
    else:
        board.prod_down()
    log(f"=== {m['name']} / {path}{banner} ===")
    reg = board.stage(m, kind)
    cmd = board.start_server(path)
    port = PORTS[path]
    model_id = served_id(port, reg, bare=(kind == "bundle"))
    side = {"model": m["name"], "path": path, "server_cmd": cmd.replace(GENV, "<env>"),
            "server_options": SERVER_OPTS, "power_mode": "server default (burst)",
            "server_version": ssh(VERSION_CMD, timeout=60).stdout.strip(),
            "artifact": m[kind], "artifact_sha256": sha256(os.path.join(ART, m[kind])),
            "gguf_source": m.get("gguf_source") if kind == "gguf" else None,
            "bundle": bundle_info(os.path.join(ART, m[kind])) if kind == "bundle" else None,
            "served_model_id": model_id, "started": now()}
    ok, info = (False, {"error": "model never listed by the server"}) if not model_id else load_check(port, model_id)
    side["load_check"] = info
    os.makedirs(os.path.dirname(fail), exist_ok=True)
    json.dump(side, open(out_path(m, path, side_name), "w"), indent=1)
    if not ok:
        tail = ssh(f"grep -aiE 'error|fail|unable|exception' {SERVE_LOG} | tail -15", timeout=60).stdout
        json.dump({"load_failure": True, **side, "server_log_tail": tail[-4000:]}, open(fail, "w"), indent=1)
        log(f"  LOAD FAILURE on {path}: {json.dumps(info)[:200]}")
        return None
    log(f"  loaded {model_id} ({info.get('load_and_first_reply_s')} s to first reply)")
    side["server_log_runtime_lines"] = ssh(
        f"grep -aoE '\\[Optimise\\][^\"]{{0,240}}|HTP arch[^\"]{{0,80}}|Using [0-9]+ device[^\"]{{0,40}}|power.?mode[^\"]{{0,60}}' "
        f"{SERVE_LOG} | sort -u | head -12",
        timeout=60).stdout[-2000:]
    json.dump(side, open(out_path(m, path, side_name), "w"), indent=1)
    return port, model_id


def plan(models, paths, mode):
    steps = []
    if mode == "block":
        steps += [(m, "cpu") for m in models if "cpu" in paths]
        steps += [(m, p) for m in models for p in ("npu_llama", "compiled") if p in paths]
    else:
        steps += [(m, p) for m in models for p in PATHS if p in paths]
    return [(m, p) for m, p in steps if m.get("gguf" if p != "compiled" else "bundle")]


def status():
    for root, _dirs, files in sorted(os.walk(RUNS)):
        done = sorted(f[:-5] for f in files if f.endswith(".json") and f != "server.json" and is_done(os.path.join(root, f)))
        if done:
            print(f"{os.path.relpath(root, RUNS):34} {', '.join(done)}")


def main():
    ap = argparse.ArgumentParser(description="edge-agent-bench v2 campaign runner")
    ap.add_argument("--models", default="")
    ap.add_argument("--paths", default="cpu,npu_llama,compiled")
    ap.add_argument("--mode", choices=("interleave", "block"), default="interleave")
    ap.add_argument("--keep", action="store_true", help="keep staged artifacts on the board")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--diagnostics", action="store_true",
                    help="run the diagnostic variants (see DIAG) instead of the matrix")
    a = ap.parse_args()
    if a.status:
        return status()
    spec = json.load(open(SPEC_PATH))
    models = spec["models"]
    if a.models:
        keep_names = a.models.split(",")
        models = [m for m in models if m["name"] in keep_names]
    paths = [p for p in a.paths.split(",") if p in PATHS]
    steps = plan(models, paths, a.mode)
    if a.diagnostics:
        steps = [s for s in diag_plan(spec, models) if s[1] in paths]
    if a.dry_run or not (BOARD and IP):
        for st in steps:
            print(f"{st[0]['name']:20} {st[1]:10} tasks: {st[2] if a.diagnostics else ', '.join(TASKS)}")
        if not (BOARD and IP):
            print("EAB_BOARD_SSH and EAB_BOARD_IP must be set to run.")
        return
    board = Board()

    def on_signal(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, on_signal)
    log(f"campaign start: {len(steps)} {'diagnostic' if a.diagnostics else 'model/path'} steps, mode={a.mode}")
    board.probe_prod()
    staged = set()
    try:
        for st in steps:
            m, p = st[0], st[1]
            try:
                if a.diagnostics:
                    staged.add(run_diag(board, m, p, st[2]))
                else:
                    run_path(board, m, p, a.keep)
            except (FileNotFoundError, RuntimeError) as e:
                log(f"ERROR {m['name']} / {p}: {e}")
                board.stop_servers()
        log("diagnostics complete" if a.diagnostics else "campaign complete")
    except KeyboardInterrupt as e:
        log(f"interrupted: {e}")
    finally:
        board.stop_servers()
        for reg in staged - {None}:
            if not a.keep:
                board.unstage(reg)
        board.prod_restore()
        log("board left with production restored")


if __name__ == "__main__":
    main()
