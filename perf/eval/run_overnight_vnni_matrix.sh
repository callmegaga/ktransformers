#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wy/Work/ktransformers"
EVAL_SCRIPT="$ROOT/kt-kernel/scripts/eval_single_endpoint.py"
COMPARE_SCRIPT="$ROOT/kt-kernel/scripts/compare_eval_results.py"
OUT_ROOT="$ROOT/perf/eval"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$OUT_ROOT/overnight-$STAMP"

mkdir -p "$RUN_DIR"

LIMIT="${LIMIT:-10}"
MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-256}"
TIMEOUT="${TIMEOUT:-600}"
TASKS="${TASKS:-wikitext,c4,hellaswag,arc_challenge,winogrande,piqa,mmlu_pro,gsm8k,ifeval}"

BASELINE_LABEL="baseline-avx2"
BASELINE_MODEL="/home/wy/Work/models/Qwen3.5-35B-A3B-GPTQ-Int4"
BASELINE_CMD=(env KT_GPTQ_INT4_BACKEND=avx2 bash "$ROOT/30b-build.sh")

declare -a LABELS=(
  "baseline-avx2"
  "gptq-avxvnni"
  "fp8-avxvnni-int4"
  "fp8-avxvnni-int8"
)

declare -A MODEL_BY_LABEL=(
  ["baseline-avx2"]="/home/wy/Work/models/Qwen3.5-35B-A3B-GPTQ-Int4"
  ["gptq-avxvnni"]="/home/wy/Work/models/Qwen3.5-35B-A3B-GPTQ-Int4"
  ["fp8-avxvnni-int4"]="/home/wy/Work/models/Qwen3.5-35B-A3B-FP8"
  ["fp8-avxvnni-int8"]="/home/wy/Work/models/Qwen3.5-35B-A3B-FP8"
)

declare -A SCRIPT_BY_LABEL=(
  ["baseline-avx2"]="$ROOT/30b-build.sh"
  ["gptq-avxvnni"]="$ROOT/30b-build-vnni.sh"
  ["fp8-avxvnni-int4"]="$ROOT/30b-build-avxvnni-int4-fp8.sh"
  ["fp8-avxvnni-int8"]="$ROOT/30b-build-avxvnni-int8-fp8.sh"
)

SERVER_PID=""

log() {
  echo "[$(date '+%F %T')] $*"
}

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    log "Stopping server pid=${SERVER_PID}"
    kill -INT "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}

cleanup() {
  stop_server
}

trap cleanup EXIT

wait_for_server() {
  local attempts=0
  until curl -sf http://127.0.0.1:30000/model_info >/dev/null; do
    attempts=$((attempts + 1))
    if [[ $attempts -gt 180 ]]; then
      log "Server readiness timed out"
      return 1
    fi
    sleep 5
  done
}

start_backend() {
  local label="$1"
  local server_log="$RUN_DIR/${label}.server.log"

  stop_server
  log "Starting backend ${label}"

  if [[ "$label" == "$BASELINE_LABEL" ]]; then
    "${BASELINE_CMD[@]}" >"$server_log" 2>&1 &
  else
    bash "${SCRIPT_BY_LABEL[$label]}" >"$server_log" 2>&1 &
  fi

  SERVER_PID=$!
  log "Server pid=${SERVER_PID}, log=$server_log"
  wait_for_server
  curl -s http://127.0.0.1:30000/model_info | tee "$RUN_DIR/${label}.model_info.json" >/dev/null
}

run_eval() {
  local label="$1"
  local model_name="${MODEL_BY_LABEL[$label]}"
  local output_json="$RUN_DIR/${label}.json"
  local eval_log="$RUN_DIR/${label}.eval.log"

  log "Running eval for ${label}"
  python "$EVAL_SCRIPT" \
    --base-url http://127.0.0.1:30000/v1 \
    --model-name "$model_name" \
    --label "$label" \
    --tasks "$TASKS" \
    --limit "$LIMIT" \
    --max-samples-per-task "$MAX_SAMPLES_PER_TASK" \
    --timeout "$TIMEOUT" \
    --output "$output_json" \
    >"$eval_log" 2>&1
}

run_compare() {
  local baseline_json="$RUN_DIR/${BASELINE_LABEL}.json"
  local label="$1"

  if [[ "$label" == "$BASELINE_LABEL" ]]; then
    return 0
  fi

  log "Comparing ${label} against ${BASELINE_LABEL}"
  python "$COMPARE_SCRIPT" \
    --baseline "$baseline_json" \
    --candidate "$RUN_DIR/${label}.json" \
    --output-json "$RUN_DIR/compare-${BASELINE_LABEL}-vs-${label}.json" \
    --output-csv "$RUN_DIR/compare-${BASELINE_LABEL}-vs-${label}.csv" \
    >"$RUN_DIR/compare-${BASELINE_LABEL}-vs-${label}.log" 2>&1
}

log "Run dir: $RUN_DIR"
log "Tasks: $TASKS"
log "LM-Eval limit: $LIMIT"
log "PPL max samples per task: $MAX_SAMPLES_PER_TASK"

for label in "${LABELS[@]}"; do
  start_backend "$label"
  run_eval "$label"
  stop_server
  run_compare "$label"
done

log "Overnight VNNI matrix completed successfully"
