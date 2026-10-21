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

Follow-up campaigns (campaign/v2_followup.json) run named phases, each on one
GenieX release with its own data folder on the board and its own results
folder under EAB_RUNS (default ~/eab-runs/v2-followup for --followup); a phase
can be gated on a triage phase, so only steps that now load, receive the tools
and return structured calls get the full protocol:
  python3 campaign/run_campaign.py --followup campaign/v2_followup.json --dry-run
  python3 campaign/run_campaign.py --followup campaign/v2_followup.json [--phase triage]
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
INC = "/root/eab-incoming"
SERVE_LOG = "/root/eab-serve.log"

# gpu (llama.cpp OpenCL) is measured only in follow-up phases; it sits before
# npu_llama so the GGUF's last user is still the NPU llama.cpp step.
PATHS = ("cpu", "gpu", "npu_llama", "compiled")
PORTS = {"cpu": 18383, "gpu": 18385, "npu_llama": 18384, "compiled": 18384}
COMPUTE = {"cpu": "cpu", "gpu": "gpu", "npu_llama": "npu", "compiled": "npu"}
SUBSET = "query_device_info,configure_device_setting,diagnose_port_issues,search_documentation"
# The flat schema offers one tool per command: the 14 the flat cases expect plus
# the two distractors of the nested subset (about 18% more prompt than SUBSET).
FLAT_SUBSET = ("show_port_status,show_all_ports_status,show_poe_status,show_poe_power_budget,"
               "show_system_info,show_vlan_config,enable_port,disable_port,configure_poe_mode,"
               "configure_port_speed,configure_port_duplex,configure_access_vlan,"
               "configure_poe_power_limit,set_port_description,diagnose_port_issues,search_documentation")
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
# Follow-up tasks: the canonical suite with one change each, written to
# <task>.json. triage is a short check of whether a step is now readable.
VARIANTS = {
    "triage": {"repeats": 1, "args": ["--limit", "6"]},
    "canonical_topk1": {"script": "bench_sampling.py", "args": ["--sampling", '{"top_k": 1}']},
    "canonical_seed7": {"script": "bench_sampling.py", "args": ["--sampling", '{"seed": 7}']},
    "canonical_flat": {"cases": "suites/test_cases_flat.yaml", "subset": FLAT_SUBSET,
                       "args": ["--tools-file", os.path.join(REPO, "suites", "tools_flat.json")]},
    "sweep_length": {"script": "sweep_prompt_length.py", "repeats": 1},
}
# llama.cpp context matches the 4,096-token compiled bundles; the model stays
# loaded between tasks; the HTP power mode is left at the server default
# (burst) and recorded.
SERVER_OPTS = "--nctx 4096 --keepalive 3600"
# Every runner-started server has a data folder under /root/eab-data*; the
# bracket keeps pkill from matching its own command line, and production
# (another data folder) is never matched.
SERVER_PATTERN = "geniex --data-dir /root/eab-dat[a]"


def use_geniex(gxd, data):
    """Point the runner at one GenieX install and data folder on the board."""
    global GXD, DATA, GENV, GX, VERSION_CMD
    GXD, DATA = gxd, data
    GENV = f"ADSP_LIBRARY_PATH={GXD}/qairt/htp-files LD_LIBRARY_PATH={GXD}"
    GX = f"env {GENV} {GXD}/geniex --data-dir {DATA} --skip-update"
    VERSION_CMD = f"LD_LIBRARY_PATH={GXD} {GXD}/geniex version"


use_geniex(os.environ.get("EAB_GENIEX_DIR", "/root/geniex-v080/geniex-cli-linux-arm64-v0.8.0"), "/root/eab-data")
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
        ssh(f'pkill -f "{SERVER_PATTERN}"; sleep 2; true', timeout=60)

    def ensure_geniex(self, rel):
        """Install a GenieX CLI release on the board from its archive in
        EAB_ARTIFACTS/geniex/ unless it is already there, checking the
        archive's SHA-256 on both ends. A release that names a platform list
        gets it in its data folder: GenieX v0.9.3 identifies the board's chip
        from Qualcomm AI Hub's platform.json and, without a copy, tries to
        download it, which hangs on a board without internet access."""
        if ssh(f"test -x {rel['dir']}/geniex && echo yes", timeout=30).stdout.strip() != "yes":
            self.install_geniex(rel)
        if rel.get("platform_list"):
            target = f"{DATA}/aihub/platform.json"
            if ssh(f"sha256sum {target} 2>/dev/null", timeout=30).stdout.split()[:1] != [rel["platform_list_sha256"]]:
                local = os.path.join(ART, "geniex", rel["platform_list"])
                if sha256(local) != rel["platform_list_sha256"]:
                    raise RuntimeError(f"platform list {local} does not match its recorded SHA-256")
                log(f"  seeding {target} from {rel['platform_list']}")
                ssh(f"mkdir -p {DATA}/aihub", timeout=30)
                scp(local, f"{DATA}/aihub")
                ssh(f"mv {DATA}/aihub/{rel['platform_list']} {target}", timeout=30)

    def install_geniex(self, rel):
        local = os.path.join(ART, "geniex", rel["archive"])
        if sha256(local) != rel["sha256"]:
            raise RuntimeError(f"GenieX archive {local} does not match its recorded SHA-256")
        parent = os.path.dirname(rel["dir"])
        log(f"installing {rel['archive']} on the board in {parent}")
        ssh(f"mkdir -p {parent}", timeout=60)
        scp(local, parent)
        r = ssh(f"cd {parent} && echo '{rel['sha256']}  {rel['archive']}' | sha256sum -c - && "
                f"tar -xzf {rel['archive']} && rm -f {rel['archive']} && "
                f"LD_LIBRARY_PATH={rel['dir']} {rel['dir']}/geniex version", timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"GenieX install failed: {(r.stdout + r.stderr)[-400:]}")

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
        """True when GenieX finished registering the model; a failed pull
        leaves the folder behind holding only its .lock and .inflight markers."""
        d = f"{DATA}/models/qualcomm/{reg}"
        return ssh(f"test -d {d} && test ! -e {d}/.inflight && ls {d} | grep -q . && echo yes",
                   timeout=30).stdout.strip() == "yes"

    def stage(self, m, kind):
        reg = f"{m['name']}-q4_0" if kind == "gguf" else f"{m['name']}-w4a16"
        if self.registered(reg):
            return reg
        self.unstage(reg)  # clear what a failed pull left behind
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
        compute = COMPUTE[path]
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
    suite = task in BENCH or task in VARIANTS
    label = f"model={m['name']};path={path};{'suite' if suite else 'probe'}={task.replace('probe_', '')}"
    if mode:
        label += f";mode={mode}"
    common = ["--url", f"http://{IP}:{port}", "--model", model_id, "--board-ssh", BOARD, "--readiness",
              "--server-version-cmd", VERSION_CMD, "--thermal-trace-s", "20", "--label", label, "--out", op]
    if suite:
        cases, reps = BENCH.get(task, BENCH["canonical"])
        v = VARIANTS.get(task, {})
        script = DIAG[mode] if mode else v.get("script", "bench_toolcall.py")
        cmd = [sys.executable, os.path.join(REPO, "harness", script), *common,
               "--cases", os.path.join(REPO, v.get("cases", cases)), "--tool-subset", v.get("subset", SUBSET),
               "--repeats", str(v.get("repeats", reps)), "--warmup", "1", "--timeout", "900", *v.get("args", [])]
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


def kind_of(path):
    return "bundle" if path == "compiled" else "gguf"


def reg_of(m, path):
    return f"{m['name']}-w4a16" if path == "compiled" else f"{m['name']}-q4_0"


def run_path(board, m, path, keep, tasks=TASKS):
    kind = kind_of(path)
    if not m.get(kind):
        return
    fail = out_path(m, path, "load_failure.json")
    if all(is_done(out_path(m, path, f"{t}.json")) for t in tasks) or is_done(fail):
        log(f"{m['name']} / {path}: already complete")
        return
    served = serve(board, m, path, kind, fail, "server.json")
    if served:
        port, model_id = served
        for t in tasks:
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
            # read without starting GenieX, whose first call may reach for the network
            "server_config": ssh(f"cat {DATA}/config.json 2>/dev/null; sha256sum {DATA}/aihub/platform.json 2>/dev/null",
                                 timeout=30).stdout.strip(),
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


def set_runs(path):
    global RUNS
    RUNS = path


def followup_steps(phase, models):
    """A phase's "model/path" steps, ordered by model and then by PATHS so a
    GGUF's last user in the phase is still its NPU llama.cpp step."""
    steps = []
    for s in phase["steps"]:
        name, path = s.split("/")
        if name not in models or path not in PATHS or not models[name].get(kind_of(path)):
            raise SystemExit(f"phase {phase['name']}: unknown or impossible step {s}")
        steps.append((models[name], path))
    order = list(models)
    return sorted(steps, key=lambda s: (order.index(s[0]["name"]), PATHS.index(s[1])))


def triage_verdict(m, path, triage_dir):
    """Whether a step was readable in a triage phase, by the report's rule
    (analysis/report_common.unreadable_reason), with the evidence behind it."""
    root = os.path.join(triage_dir, m["name"], path)
    v = {"step": f"{m['name']}/{path}", "readable": False}
    if is_done(os.path.join(root, "load_failure.json")):
        return {**v, "reason": "load failure"}
    try:
        d = json.load(open(os.path.join(root, "triage.json")))
    except (OSError, json.JSONDecodeError):
        return {**v, "reason": "no triage result"}
    sys.path.insert(0, os.path.join(REPO, "analysis"))
    import report_common  # noqa: E402
    reps = [r for c in d["results"] for r in c.get("reps") or []]
    toks = sorted(r["prompt_tokens"] for r in reps if r.get("prompt_tokens"))
    why = report_common.unreadable_reason({"canonical": d, "paraphrase": d})
    return {**v, "readable": why is None, "reason": why or "readable",
            "median_prompt_tokens": toks[len(toks) // 2] if toks else None,
            "structured_calls": sum(bool(r.get("got_tool")) for r in reps),
            "text_calls": sum(bool(r.get("text_tool_call") or (not r.get("got_tool") and
                                    report_common.analyze.parse_text_tool_call(r.get("content"))[0]))
                              for r in reps),
            "replies": len(reps)}


def run_followup(board, spec, models, phases, keep):
    base = RUNS
    try:
        for ph in phases:
            rel = spec["geniex"][ph["geniex"]]
            use_geniex(rel["dir"], rel["data"])
            set_runs(os.path.join(base, ph["name"]))
            log(f"phase {ph['name']}: GenieX {ph['geniex']}, {len(ph['steps'])} steps, tasks {', '.join(ph['tasks'])}")
            board.ensure_geniex(rel)
            used = set()
            for m, p in followup_steps(ph, models):
                if ph.get("gate"):
                    v = triage_verdict(m, p, os.path.join(base, ph["gate"]))
                    if not v["readable"]:
                        log(f"{m['name']} / {p}: skipped ({v['reason']} in phase {ph['gate']})")
                        continue
                used.add(reg_of(m, p))
                try:
                    run_path(board, m, p, keep, tasks=ph["tasks"])
                except (FileNotFoundError, RuntimeError) as e:
                    log(f"ERROR {m['name']} / {p}: {e}")
                    board.stop_servers()
            board.stop_servers()
            if ph["tasks"] == ["triage"]:
                rows = [triage_verdict(m, p, RUNS) for m, p in followup_steps(ph, models)]
                os.makedirs(RUNS, exist_ok=True)
                json.dump(rows, open(os.path.join(RUNS, "triage_summary.json"), "w"), indent=1)
                for r in rows:
                    log(f"  triage {r['step']:28} {r['reason']:24} prompt {r.get('median_prompt_tokens')} "
                        f"structured {r.get('structured_calls')}/{r.get('replies')} text {r.get('text_calls')}")
            if not keep:  # artifacts left by steps that are not their last user
                for reg in used:
                    board.unstage(reg)
    finally:
        set_runs(base)


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
    ap.add_argument("--followup", default="", help="run the phases of a follow-up spec (campaign/v2_followup.json)")
    ap.add_argument("--phase", default="", help="with --followup: run only these phases (comma-separated)")
    a = ap.parse_args()
    if a.followup and "EAB_RUNS" not in os.environ:
        set_runs(os.path.expanduser("~/eab-runs/v2-followup"))
    if a.status:
        return status()
    spec = json.load(open(SPEC_PATH))
    models = spec["models"]
    if a.followup:
        return main_followup(a, json.load(open(a.followup)), {m["name"]: m for m in models})
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


def main_followup(a, spec, models):
    names = [p for p in a.phase.split(",") if p]
    phases = [ph for ph in spec["phases"] if not names or ph["name"] in names]
    if names and len(phases) != len(names):
        raise SystemExit(f"unknown phase in {a.phase}; the spec has {', '.join(ph['name'] for ph in spec['phases'])}")
    if a.dry_run or not (BOARD and IP):
        for ph in phases:
            rel = spec["geniex"][ph["geniex"]]
            gate = f", only steps readable in phase {ph['gate']}" if ph.get("gate") else ""
            print(f"phase {ph['name']}: GenieX {ph['geniex']} ({rel['dir']}, data {rel['data']}), "
                  f"results {os.path.join(RUNS, ph['name'])}{gate}")
            for m, p in followup_steps(ph, models):
                print(f"  {m['name']:20} {p:10} tasks: {', '.join(ph['tasks'])}")
        if not (BOARD and IP):
            print("EAB_BOARD_SSH and EAB_BOARD_IP must be set to run.")
        return
    board = Board()

    def on_signal(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, on_signal)
    log(f"follow-up start: phases {', '.join(ph['name'] for ph in phases)}")
    board.probe_prod()
    try:
        run_followup(board, spec, models, phases, a.keep)
        log("follow-up complete")
    except KeyboardInterrupt as e:
        log(f"interrupted: {e}")
    finally:
        board.stop_servers()
        board.prod_restore()
        log("board left with production restored")


if __name__ == "__main__":
    main()
