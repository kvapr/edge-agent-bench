#!/bin/bash
# Qwen3-8B row, CPU serving leg (GenieX 0.5.1-alpha CLI, llama.cpp CPU, Q4_0).
# Strictly serialized; the production gx container is left running on :18181.
# The two NPU legs are omitted: the 8B does not load on either NPU path on this
# board (FastRPC per-protection-domain map limit). See the results README for
# the exact load errors. Lab-specific addresses shown as placeholders; set REPO,
# B and IP before running.
set +e
REPO="<repo>"
B="root@<board-ip>"; IP="<board-ip>"
GX=/root/geniex-alpha/geniex-cli-linux-arm64-v0.5.1-alpha.2
D=$(date +%Y%m%d)
OUT="$REPO/results/$(date +%Y-%m-%d)-qcs9075-qwen3-8b"; mkdir -p "$OUT"
SUITES="$REPO/suites"
SUB="query_device_info,configure_device_setting,diagnose_port_issues,search_documentation"
EXTRA='{"enable_think": false}'
GGUF_MODEL="qualcomm/qwen3-8b-q4_0:Q4_0"
REPS=5
cd "$REPO/harness" || exit 1
wait_up(){ for i in $(seq 1 40); do curl -s --max-time 4 "http://$IP:$1/v1/models" 2>/dev/null | grep -q "$2" && { echo "up :$1"; return 0; }; sleep 5; done; echo "NEVER UP :$1"; return 1; }

echo "=== start alpha CPU server (production gx stays up on 18181) ==="
ssh "$B" "nohup env LD_LIBRARY_PATH=$GX $GX/geniex --data-dir /root/geniex-data --skip-update serve --host 0.0.0.0:18283 -c cpu > /root/qwen8b-cpu-serve.log 2>&1 & echo launched"
wait_up 18283 "qwen3-8b-q4_0" || { echo "CPU server never came up"; exit 1; }

echo "=== phase probe 5 rep ==="
python3 phase_probe.py --url "http://$IP:18283" --model "$GGUF_MODEL" --method stream --repeats $REPS --warmup 1 \
  --extra-json "$EXTRA" --board-ssh "$B" --readiness --label cpu-geniex051-llamacpp-q4_0 \
  --out "$OUT/phase-qwen3-8b-cpu-llamacpp-q4_0-$D.json"

echo "=== 65-case bench 5 rep (public tool schema) ==="
python3 bench_toolcall.py --url "http://$IP:18283" --model "$GGUF_MODEL" \
  --cases "$SUITES/test_cases.yaml" --tools-file "$SUITES/tools_nested.json" --tool-subset "$SUB" \
  --repeats $REPS --warmup 1 --board-ssh "$B" --readiness \
  --out "$OUT/qwen3-8b-q4_0-cpu-llamacpp-hygiene-${REPS}rep-$D.json"

echo "=== stop alpha CPU server (leave gx up) ==="
ssh "$B" 'pkill -f "geniex[-]alpha"; sleep 2; echo cpu-server-stopped'
echo "=== CPU LEG DONE ==="
