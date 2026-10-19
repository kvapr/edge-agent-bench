#!/bin/bash
# Post-matrix re-run: phase probe on all three paths with ONE method (stream), 5 reps. Serialized.
set +e
cd <repo>/harness
B=user@<board-ip>; IP=<board-ip>
GX=/root/geniex-alpha/geniex-cli-linux-arm64-v0.5.1-alpha.2
D=20260910
GGUF_MODEL="qualcomm/qwen3-4b-2507-q4_0:Q4_0"
EXTRA='{"enable_think": false}'
restore() {
  echo "=== RESTORE ==="
  ssh $B 'pkill -f "geniex[-]alpha"; sleep 2; echo alpha-servers-stopped'
  ssh $B 'for r in /sys/class/remoteproc/remoteproc*; do n=$(cat $r/name); if [ "$n" = "cdsp" ] || [ "$n" = "cdsp1" ]; then echo stop > $r/state 2>/dev/null; sleep 1; echo start > $r/state 2>/dev/null; fi; done; sleep 2; echo dsp-bounced'
  ssh $B 'docker start gx >/dev/null 2>&1; sleep 6; docker ps --format "{{.Names}} {{.Status}}"'
  for i in $(seq 1 15); do curl -s --max-time 4 http://$IP:18181/v1/models 2>/dev/null | grep -q qwen3 && { echo qwen-endpoint-restored; break; }; sleep 3; done
}
trap restore EXIT
wait_up() { for i in $(seq 1 30); do curl -s --max-time 4 "http://$IP:$1/v1/models" 2>/dev/null | grep -q "$2" && { echo "server-up :$1"; return 0; }; sleep 3; done; echo "SERVER :$1 NEVER UP"; return 1; }
probe() { python3 phase_probe.py --method stream --repeats 5 --warmup 1 --extra-json "$EXTRA" --board-ssh $B --readiness "$@"; }

echo "=== A2. NPU qairt W4A16 (production gx :18181), stream method ==="
wait_up 18181 qwen3 || exit 1
probe --url http://$IP:18181 --model qualcomm/qwen3-4b-4k --label npu-geniex040-qairt-w4a16 --out results/phase2-qwen3-4b-npu-qairt-w4a16-$D.json 2>&1 | tail -9

echo "=== B2. CPU llama.cpp Q4_0 (alpha :18283), stream method ==="
ssh $B "nohup env LD_LIBRARY_PATH=$GX $GX/geniex --data-dir /root/geniex-data --skip-update serve --host 0.0.0.0:18283 -c cpu > /root/qwen-cpu-serve2.log 2>&1 & echo launched"
wait_up 18283 q4_0 || exit 1
probe --url http://$IP:18283 --model "$GGUF_MODEL" --label cpu-geniex051-llamacpp-q4_0 --out results/phase2-qwen3-4b-cpu-llamacpp-q4_0-$D.json 2>&1 | tail -9
ssh $B 'pkill -f "geniex[-]alpha"; sleep 2; echo cpu-server-stopped'

echo "=== D2. NPU llama.cpp Q4_0 (alpha :18284, gx stopped), stream method ==="
ssh $B 'docker stop gx >/dev/null 2>&1; sleep 3; echo gx-stopped'
ssh $B 'for r in /sys/class/remoteproc/remoteproc*; do n=$(cat $r/name); if [ "$n" = "cdsp" ] || [ "$n" = "cdsp1" ]; then echo stop > $r/state 2>/dev/null; sleep 1; echo start > $r/state 2>/dev/null; fi; done; sleep 2; echo dsp-bounced'
ssh $B "nohup env ADSP_LIBRARY_PATH=$GX/qairt/htp-files LD_LIBRARY_PATH=$GX $GX/geniex --data-dir /root/geniex-data --skip-update serve --host 0.0.0.0:18284 -c npu > /root/qwen-npu-llamacpp-serve2.log 2>&1 & echo launched"
wait_up 18284 q4_0 || exit 1
probe --url http://$IP:18284 --model "$GGUF_MODEL" --label npu-geniex051-llamacpp-q4_0 --out results/phase2-qwen3-4b-npu-llamacpp-q4_0-$D.json 2>&1 | tail -9
echo "=== matrix2 done ==="
