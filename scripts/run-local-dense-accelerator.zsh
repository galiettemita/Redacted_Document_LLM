#!/bin/zsh
set -euo pipefail

SCRIPT_PATH="${0:A}"
SCRIPT_PROJECT_ROOT="${SCRIPT_PATH:h:h}"
PROJECT_ROOT="${DENSE_ACCELERATOR_PROJECT_ROOT:-$SCRIPT_PROJECT_ROOT}"
CORPUS_ROOT="${DENSE_ACCELERATOR_CORPUS_ROOT:-$PROJECT_ROOT/var/corpus}"
BEST_AVAILABLE_ROOT="$CORPUS_ROOT/best-available"
STATE_DIR="${DENSE_ACCELERATOR_STATE_DIR:-$BEST_AVAILABLE_ROOT/dense-accelerator-state}"
CHUNKS_PATH="${DENSE_ACCELERATOR_CHUNKS:-$BEST_AVAILABLE_ROOT/chunks.jsonl}"
CORPUS_MANIFEST_PATH="${DENSE_ACCELERATOR_CORPUS_MANIFEST:-$BEST_AVAILABLE_ROOT/corpus-manifest.json}"
DENSE_DIRECTORY="${DENSE_ACCELERATOR_DENSE_DIRECTORY:-$BEST_AVAILABLE_ROOT/dense}"
SHARD_SIZE="${DENSE_ACCELERATOR_SHARD_SIZE:-68572}"
BATCH_SIZE="${DENSE_ACCELERATOR_BATCH_SIZE:-64}"
ENCODER_THREADS="${DENSE_ACCELERATOR_ENCODER_THREADS:-4}"
SAFETY_GAP_SHARDS="${DENSE_ACCELERATOR_SAFETY_GAP_SHARDS:-4}"
STATE_FILE="$STATE_DIR/state.json"
PID_FILE="$STATE_DIR/runner.pid"
LOG_FILE="$STATE_DIR/runner.log"
REPORT_FILE="$STATE_DIR/report.json"

if [[ -n "${DENSE_ACCELERATOR_UV:-}" ]]; then
  UV_COMMAND="$DENSE_ACCELERATOR_UV"
else
  UV_COMMAND="$(command -v uv 2>/dev/null || true)"
fi
if [[ -n "${DENSE_ACCELERATOR_CAFFEINATE:-}" ]]; then
  CAFFEINATE_COMMAND="$DENSE_ACCELERATOR_CAFFEINATE"
else
  CAFFEINATE_COMMAND="$(command -v caffeinate 2>/dev/null || true)"
fi
if [[ -n "${DENSE_ACCELERATOR_LAUNCHCTL:-}" ]]; then
  LAUNCHCTL_COMMAND="$DENSE_ACCELERATOR_LAUNCHCTL"
else
  LAUNCHCTL_COMMAND="$(command -v launchctl 2>/dev/null || true)"
fi

iso_now() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

write_state() {
  local phase=$1
  local exit_code=${2:-null}
  local reason=${3:-}
  local temporary="$STATE_DIR/.state.json.$$.tmp"
  mkdir -p "$STATE_DIR"
  jq -n \
    --arg phase "$phase" \
    --arg updated_at "$(iso_now)" \
    --arg reason "$reason" \
    --argjson exit_code "$exit_code" \
    '{
      schema_version: "local-dense-accelerator-v1",
      phase: $phase,
      updated_at: $updated_at,
      exit_code: $exit_code,
      reason: (if $reason == "" then null else $reason end)
    }' > "$temporary"
  mv "$temporary" "$STATE_FILE"
}

read_phase() {
  if [[ -f "$STATE_FILE" ]]; then
    jq -er '.phase | strings' "$STATE_FILE" 2>/dev/null || print -- "invalid"
  else
    print -- "not_started"
  fi
}

read_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local value
  value="$(<"$PID_FILE")"
  [[ "$value" == <1-> ]] || return 2
  print -- "$value"
}

classify_pid() {
  local pid=$1
  if ! kill -0 "$pid" 2>/dev/null; then
    print -- "stopped"
    return 0
  fi
  local command_text
  command_text="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  if [[ "$command_text" == *"$SCRIPT_PATH"* && "$command_text" == *"internal-run"* ]]; then
    print -- "running"
  else
    print -- "foreign_pid"
  fi
}

show_status() {
  local phase
  phase="$(read_phase)"
  local pid=""
  local process_status="stopped"
  if pid="$(read_pid 2>/dev/null)"; then
    process_status="$(classify_pid "$pid")"
  else
    local result=$?
    if (( result == 2 )); then
      process_status="invalid_pid"
    fi
    pid=""
  fi
  local base='{}'
  if [[ -f "$STATE_FILE" ]] && jq -e 'type == "object"' "$STATE_FILE" >/dev/null 2>&1; then
    base="$(<"$STATE_FILE")"
  fi
  print -r -- "$base" | jq \
    --arg phase "$phase" \
    --arg process_status "$process_status" \
    --argjson runner_pid "${pid:-null}" \
    '. + {
      schema_version: (.schema_version // "local-dense-accelerator-v1"),
      phase: (.phase // $phase),
      process_status: $process_status,
      runner_pid: $runner_pid
    }'
  [[ "$process_status" != "foreign_pid" && "$process_status" != "invalid_pid" ]]
}

require_runtime() {
  [[ "$PROJECT_ROOT" == /* && -d "$PROJECT_ROOT/backend" ]] || return 64
  [[ "$CHUNKS_PATH" == /* && -f "$CHUNKS_PATH" ]] || return 64
  [[ "$CORPUS_MANIFEST_PATH" == /* && -f "$CORPUS_MANIFEST_PATH" ]] || return 64
  [[ "$DENSE_DIRECTORY" == /* && -d "$DENSE_DIRECTORY" ]] || return 64
  [[ -n "$UV_COMMAND" && -x "$UV_COMMAND" ]] || return 64
  [[ -n "$CAFFEINATE_COMMAND" && -x "$CAFFEINATE_COMMAND" ]] || return 64
  [[ "$SHARD_SIZE" == <1-> && "$BATCH_SIZE" == <1-> ]] || return 64
  [[ "$ENCODER_THREADS" == <1-> && "$SAFETY_GAP_SHARDS" == <1-> ]] || return 64
}

run_internal() {
  require_runtime || return $?
  mkdir -p "$STATE_DIR"
  write_state "running"
  local temporary="$STATE_DIR/.report.json.$$.tmp"
  local run_exit=0
  (
    cd "$PROJECT_ROOT/backend"
    "$UV_COMMAND" run corpus accelerate-best-available-dense \
      --chunks "$CHUNKS_PATH" \
      --corpus-manifest "$CORPUS_MANIFEST_PATH" \
      --output-directory "$DENSE_DIRECTORY" \
      --shard-size "$SHARD_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --encoder-threads "$ENCODER_THREADS" \
      --safety-gap-shards "$SAFETY_GAP_SHARDS"
  ) > "$temporary" || run_exit=$?
  if (( run_exit != 0 )); then
    if [[ -f "$temporary" ]]; then
      mv "$temporary" "$STATE_DIR/output.failed.$(date -u '+%Y%m%dT%H%M%SZ').json"
    fi
    write_state "failed" "$run_exit" "accelerator_command_failed"
    return "$run_exit"
  fi
  if ! jq -e '.schema_version == "dense-embedding-acceleration-v1"' "$temporary" >/dev/null 2>&1; then
    mv "$temporary" "$STATE_DIR/output.invalid.$(date -u '+%Y%m%dT%H%M%SZ').json"
    write_state "failed" 65 "invalid_accelerator_report"
    return 65
  fi
  mv "$temporary" "$REPORT_FILE"
  write_state "completed" 0
}

cleanup_managed_launch() {
  if [[ -n "${DENSE_ACCELERATOR_LAUNCH_LABEL:-}" && -x "$LAUNCHCTL_COMMAND" ]]; then
    "$LAUNCHCTL_COMMAND" remove "$DENSE_ACCELERATOR_LAUNCH_LABEL" >/dev/null 2>&1 || true
  fi
}

run_managed_internal() {
  mkdir -p "$STATE_DIR"
  exec >> "$LOG_FILE" 2>&1
  trap cleanup_managed_launch EXIT
  local temporary_pid="$STATE_DIR/.runner.pid.$$.tmp"
  print -- "$$" > "$temporary_pid"
  mv "$temporary_pid" "$PID_FILE"
  local run_exit=0
  run_internal || run_exit=$?
  cleanup_managed_launch
  trap - EXIT
  return "$run_exit"
}

start_runner() {
  require_runtime || return $?
  mkdir -p "$STATE_DIR"
  local pid=""
  if pid="$(read_pid 2>/dev/null)"; then
    local classification
    classification="$(classify_pid "$pid")"
    if [[ "$classification" == "running" ]]; then
      print -- "Dense accelerator already running with PID $pid."
      return 0
    fi
    if [[ "$classification" == "foreign_pid" ]]; then
      show_status
      return 69
    fi
    mv "$PID_FILE" "$STATE_DIR/runner.pid.stale.$(date -u '+%Y%m%dT%H%M%SZ').$$"
  fi
  if [[ "${DENSE_ACCELERATOR_TEST_MODE:-0}" == "1" || "$(uname -s)" != "Darwin" || ! -x "$LAUNCHCTL_COMMAND" ]]; then
    nohup "$CAFFEINATE_COMMAND" -i "$SCRIPT_PATH" internal-run-managed >> "$LOG_FILE" 2>&1 &
    pid=$!
  else
    local label="com.redacted-document-llm.dense-accelerator.$(date -u '+%Y%m%dT%H%M%SZ').$$"
    "$LAUNCHCTL_COMMAND" submit -l "$label" -- \
      /usr/bin/env \
      DENSE_ACCELERATOR_PROJECT_ROOT="$PROJECT_ROOT" \
      DENSE_ACCELERATOR_CORPUS_ROOT="$CORPUS_ROOT" \
      DENSE_ACCELERATOR_STATE_DIR="$STATE_DIR" \
      DENSE_ACCELERATOR_CHUNKS="$CHUNKS_PATH" \
      DENSE_ACCELERATOR_CORPUS_MANIFEST="$CORPUS_MANIFEST_PATH" \
      DENSE_ACCELERATOR_DENSE_DIRECTORY="$DENSE_DIRECTORY" \
      DENSE_ACCELERATOR_SHARD_SIZE="$SHARD_SIZE" \
      DENSE_ACCELERATOR_BATCH_SIZE="$BATCH_SIZE" \
      DENSE_ACCELERATOR_ENCODER_THREADS="$ENCODER_THREADS" \
      DENSE_ACCELERATOR_SAFETY_GAP_SHARDS="$SAFETY_GAP_SHARDS" \
      DENSE_ACCELERATOR_UV="$UV_COMMAND" \
      DENSE_ACCELERATOR_CAFFEINATE="$CAFFEINATE_COMMAND" \
      DENSE_ACCELERATOR_LAUNCHCTL="$LAUNCHCTL_COMMAND" \
      DENSE_ACCELERATOR_LAUNCH_LABEL="$label" \
      "$CAFFEINATE_COMMAND" -i "$SCRIPT_PATH" internal-run-managed
    for _attempt in {1..40}; do
      if pid="$(read_pid 2>/dev/null)" && [[ "$(classify_pid "$pid")" == "running" ]]; then
        break
      fi
      pid=""
      sleep 0.05
    done
    if [[ -z "$pid" ]]; then
      write_state "launch_failed" 70 "managed_process_did_not_register"
      return 70
    fi
  fi
  print -- "Dense accelerator started with PID $pid."
}

case "${1:-}" in
  status) show_status ;;
  start) start_runner ;;
  internal-run) run_internal ;;
  internal-run-managed) run_managed_internal ;;
  *) print -u2 -- "usage: ${0:t} {status|start|internal-run}"; exit 64 ;;
esac
