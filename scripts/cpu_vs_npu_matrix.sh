#!/bin/bash
# Controlled CPU vs NPU measurement matrix. Strictly serialized (no concurrency).
set +e
cd <repo>/harness
B=user@<board-ip>; IP=<board-ip>
GX=/root/geniex-alpha/geniex-cli-linux-arm64-v0.5.1-alpha.2
D=20260910
GGUF_MODEL="qualcomm/qwen3-4b-2507-q4_0:Q4_0"
SUB="query_device_info,configure_device_setting,diagnose_port_issues,search_documentation"
EXTRA='{"enable_think": false}'
[ -f phase_probe.py ] || { echo "phase_probe.py missing"; exit 1; }

restore() {
  echo "=== RESTORE ==="
  ssh $B 'pkill -f "geniex[-]alpha"; sleep 2; echo alpha-servers-stopped'
  ssh $B 'for r in /sys/class/remoteproc/remoteproc*; do n=$(cat $r/name); if [ "$n" = "cdsp" ] || [ "$n" = "cdsp1" ]; then echo stop > $r/state 2>/dev/null; sleep 1; echo start > $r/state 2>/dev/null; fi; done; sleep 2; echo dsp-bounced'
  ssh $B 'docker start gx >/dev/null 2>&1; sleep 6; docker ps --format "{{.Names}} {{.Status}}"'
  for i in $(seq 1 15); do curl -s --max-time 4 http://$IP:18181/v1/models 2>/dev/null | grep -q qwen3 && { echo qwen-endpoint-restored; break; }; sleep 3; done
}
trap restore EXIT
wait_up() { for i in $(seq 1 30); do curl -s --max-time 4 "http://$IP:$1/v1/models" 2>/dev/null | grep -q "$2" && { echo "server-up :$1"; return 0; }; sleep 3; done; echo "SERVER :$1 NEVER UP"; return 1; }

echo "=== A. phase probe: production NPU qairt W4A16 (gx :18181) ==="
python3 phase_probe.py --url http://$IP:18181 --model qualcomm/qwen3-4b-4k --repeats 5 --warmup 1 \
  --label npu-geniex040-qairt-w4a16 --extra-json "$EXTRA" --board-ssh $B --readiness \
  --out results/phase-qwen3-4b-npu-qairt-w4a16-$D.json 2>&1 | tail -8

echo "=== B. phase probe: CPU llama.cpp Q4_0 (alpha :18283, already serving) ==="
wait_up 18283 q4_0 || exit 1
python3 phase_probe.py --url http://$IP:18283 --model "$GGUF_MODEL" --repeats 5 --warmup 1 \
  --label cpu-geniex051-llamacpp-q4_0 --extra-json "$EXTRA" --board-ssh $B --readiness \
  --out results/phase-qwen3-4b-cpu-llamacpp-q4_0-$D.json 2>&1 | tail -8

echo "=== C. 65-case bench on CPU path (2 reps) ==="
python3 bench_toolcall.py --url http://$IP:18283 --model "$GGUF_MODEL" --tool-subset "$SUB" \
  --repeats 2 --warmup 1 --board-ssh $B --readiness \
  --out results/qwen3-4b-2507-q4_0-cpu-llamacpp-hygiene-2rep-$D.json 2>&1 | grep -vE "^\[r?[0-9]* ?[0-9]+/" | tail -30

echo "=== D. NPU llama.cpp Q4_0 (same GGUF, same binary, -c npu; gx stopped) ==="
ssh $B 'pkill -f "geniex[-]alpha"; sleep 2; echo cpu-server-stopped'
ssh $B 'docker stop gx >/dev/null 2>&1; sleep 3; echo gx-stopped'
ssh $B 'for r in /sys/class/remoteproc/remoteproc*; do n=$(cat $r/name); if [ "$n" = "cdsp" ] || [ "$n" = "cdsp1" ]; then echo stop > $r/state 2>/dev/null; sleep 1; echo start > $r/state 2>/dev/null; fi; done; sleep 2; echo dsp-bounced'
ssh $B "nohup env ADSP_LIBRARY_PATH=$GX/qairt/htp-files LD_LIBRARY_PATH=$GX $GX/geniex --data-dir /root/geniex-data --skip-update serve --host 0.0.0.0:18284 -c npu > /root/qwen-npu-llamacpp-serve.log 2>&1 & echo launched"
wait_up 18284 q4_0 || exit 1
echo "--- smoke to confirm NPU engagement (CPU busy should be low):"
ssh $B 'head -1 /proc/stat' > /tmp/s1
curl -s --max-time 600 -H 'Content-Type: application/json' -d "{\"model\":\"$GGUF_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"In two sentences, what is PoE?\"}],\"max_tokens\":48,\"temperature\":0,\"enable_think\":false}" http://$IP:18284/v1/chat/completions | python3 -c "import sys,json; d=json.load(sys.stdin); print('usage', d.get('usage',{}).get('completion_tokens'), repr(d['choices'][0]['message']['content'][:80]))"
ssh $B 'head -1 /proc/stat' > /tmp/s2
python3 -c "
f1=[int(x) for x in open('/tmp/s1').read().split()[1:]]; f2=[int(x) for x in open('/tmp/s2').read().split()[1:]]
tot=sum(f2)-sum(f1); idle=(f2[3]-f1[3])+(f2[4]-f1[4]); print(f'board CPU busy during NPU smoke: {1-idle/tot:.2f}')"
python3 phase_probe.py --url http://$IP:18284 --model "$GGUF_MODEL" --repeats 5 --warmup 1 \
  --label npu-geniex051-llamacpp-q4_0 --extra-json "$EXTRA" --board-ssh $B --readiness \
  --out results/phase-qwen3-4b-npu-llamacpp-q4_0-$D.json 2>&1 | tail -8
echo "=== E. 65-case bench on NPU llama.cpp path (2 reps) ==="
python3 bench_toolcall.py --url http://$IP:18284 --model "$GGUF_MODEL" --tool-subset "$SUB" \
  --repeats 2 --warmup 1 --board-ssh $B --readiness \
  --out results/qwen3-4b-2507-q4_0-npu-llamacpp-hygiene-2rep-$D.json 2>&1 | grep -vE "^\[r?[0-9]* ?[0-9]+/" | tail -30
ssh $B 'grep -aiE "hexagon|htp|ngl|offload" /root/qwen-npu-llamacpp-serve.log | head -5' > qwen-npu-llamacpp-serve-$D.log
echo "=== matrix done ==="
