#!/bin/zsh
set -euo pipefail

SCRIPT_PATH="${0:A}"
SCRIPT_PROJECT_ROOT="${SCRIPT_PATH:h:h}"
PROJECT_ROOT="${DENSE_FINALIZER_PROJECT_ROOT:-$SCRIPT_PROJECT_ROOT}"
CORPUS_ROOT="${DENSE_FINALIZER_CORPUS_ROOT:-$PROJECT_ROOT/var/corpus}"
BEST_AVAILABLE_ROOT="$CORPUS_ROOT/best-available"
BUILD_STATE_FILE="${DENSE_FINALIZER_BUILD_STATE_FILE:-$BEST_AVAILABLE_ROOT/dense-build-state/state.json}"
STATE_DIR="${DENSE_FINALIZER_STATE_DIR:-$BEST_AVAILABLE_ROOT/dense-finalizer-state}"
DENSE_DIRECTORY="${DENSE_FINALIZER_DENSE_DIRECTORY:-$BEST_AVAILABLE_ROOT/dense}"
CHUNKS_PATH="${DENSE_FINALIZER_CHUNKS:-$BEST_AVAILABLE_ROOT/chunks.jsonl}"
CORPUS_MANIFEST_PATH="${DENSE_FINALIZER_CORPUS_MANIFEST:-$BEST_AVAILABLE_ROOT/corpus-manifest.json}"
DENSE_MANIFEST_PATH="${DENSE_FINALIZER_DENSE_MANIFEST:-$DENSE_DIRECTORY/manifest.json}"
CERTIFICATION_QUERIES_PATH="${DENSE_FINALIZER_CERTIFICATION_QUERIES:-$DENSE_DIRECTORY/recall-queries.txt}"
LATENCY_REPORT_PATH="${DENSE_FINALIZER_LATENCY_REPORT:-$DENSE_DIRECTORY/latency-report.json}"
STATE_FILE="$STATE_DIR/state.json"
PID_FILE="$STATE_DIR/finalizer.pid"
LOG_FILE="$STATE_DIR/finalizer.log"
VERIFICATION_REPORT="$STATE_DIR/verification-report.json"
POLL_SECONDS="${DENSE_FINALIZER_POLL_SECONDS:-60}"

if [[ -n "${DENSE_FINALIZER_DOCKER:-}" ]]; then
  DOCKER_COMMAND="$DENSE_FINALIZER_DOCKER"
else
  DOCKER_COMMAND="$(command -v docker 2>/dev/null || true)"
fi
if [[ -n "${DENSE_FINALIZER_CAFFEINATE:-}" ]]; then
  CAFFEINATE_COMMAND="$DENSE_FINALIZER_CAFFEINATE"
else
  CAFFEINATE_COMMAND="$(command -v caffeinate 2>/dev/null || true)"
fi
LAUNCHCTL_COMMAND="$(command -v launchctl 2>/dev/null || true)"

iso_now() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

require_absolute_directory() {
  local path=$1
  local label=$2
  if [[ "$path" != /* || ! -d "$path" ]]; then
    print -u2 -- "STOP: $label must be an existing absolute directory: $path"
    return 64
  fi
}

require_file() {
  local path=$1
  local label=$2
  if [[ "$path" != /* || ! -f "$path" ]]; then
    print -u2 -- "STOP: $label must be an existing absolute file: $path"
    return 64
  fi
}

require_launch_runtime() {
  require_absolute_directory "$PROJECT_ROOT" "project root"
  require_absolute_directory "$CORPUS_ROOT" "corpus root"
  require_file "$PROJECT_ROOT/compose.yaml" "Compose file"
  if [[ -z "$DOCKER_COMMAND" || ! -x "$DOCKER_COMMAND" ]]; then
    print -u2 -- "STOP: docker executable is unavailable"
    return 64
  fi
  if [[ ! "$POLL_SECONDS" =~ '^[0-9]+([.][0-9]+)?$' || "$POLL_SECONDS" == "0" || "$POLL_SECONDS" == "0.0" ]]; then
    print -u2 -- "STOP: poll seconds must be positive"
    return 64
  fi
}

require_artifacts() {
  require_file "$CHUNKS_PATH" "chunks"
  require_file "$CORPUS_MANIFEST_PATH" "corpus manifest"
  require_absolute_directory "$DENSE_DIRECTORY" "dense artifact directory"
  require_file "$DENSE_MANIFEST_PATH" "dense manifest"
  require_file "$CERTIFICATION_QUERIES_PATH" "certification queries"
}

write_state() {
  local phase=$1
  local exit_code=${2:-null}
  local reason=${3:-}
  local build_phase=${4:-}
  local temporary="$STATE_DIR/.state.json.$$.tmp"
  mkdir -p "$STATE_DIR"
  jq -n \
    --arg phase "$phase" \
    --arg updated_at "$(iso_now)" \
    --arg reason "$reason" \
    --arg build_phase "$build_phase" \
    --argjson exit_code "$exit_code" \
    '{
      schema_version: "local-dense-finalizer-v1",
      phase: $phase,
      updated_at: $updated_at,
      exit_code: $exit_code,
      reason: (if $reason == "" then null else $reason end),
      build_phase: (if $build_phase == "" then null else $build_phase end)
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

read_build_phase() {
  if [[ -f "$BUILD_STATE_FILE" ]]; then
    jq -er '.phase | strings' "$BUILD_STATE_FILE" 2>/dev/null || print -- "invalid"
  else
    print -- "not_started"
  fi
}

read_pid() {
  if [[ ! -f "$PID_FILE" ]]; then
    return 1
  fi
  local value
  value="$(<"$PID_FILE")"
  if [[ "$value" != <1-> ]]; then
    return 2
  fi
  print -- "$value"
}

classify_pid() {
  local pid=$1
  if ! kill -0 "$pid" 2>/dev/null; then
    print -- "stale"
    return 0
  fi
  local command_text
  command_text="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  if [[ "$command_text" == *"$SCRIPT_PATH"* && "$command_text" == *"internal-run"* ]]; then
    print -- "finalizer"
  else
    print -- "foreign"
  fi
}

status_json() {
  local process_status=$1
  local phase=$2
  local pid=${3:-null}
  local base='{}'
  if [[ -f "$STATE_FILE" ]] && jq -e 'type == "object"' "$STATE_FILE" >/dev/null 2>&1; then
    base="$(<"$STATE_FILE")"
  fi
  print -r -- "$base" | jq \
    --arg process_status "$process_status" \
    --arg phase "$phase" \
    --arg build_phase "$(read_build_phase)" \
    --argjson finalizer_pid "$pid" \
    '. + {
      schema_version: (.schema_version // "local-dense-finalizer-v1"),
      phase: (.phase // $phase),
      process_status: $process_status,
      finalizer_pid: $finalizer_pid,
      build_phase: $build_phase
    }'
}

show_status() {
  local phase
  phase="$(read_phase)"
  local pid
  pid="$(read_pid 2>/dev/null)" || {
    local pid_result=$?
    if (( pid_result == 2 )); then
      status_json "invalid_pid" "$phase"
      return 69
    fi
    status_json "stopped" "$phase"
    return 0
  }
  local classification
  classification="$(classify_pid "$pid")"
  if [[ "$classification" == "finalizer" ]]; then
    status_json "running" "$phase" "$pid"
    return 0
  fi
  if [[ "$classification" == "foreign" ]]; then
    status_json "foreign_pid" "$phase" "$pid"
    return 69
  fi
  status_json "stopped" "$phase" "$pid"
}

archive_stale_pid() {
  if [[ -f "$PID_FILE" ]]; then
    mv "$PID_FILE" "$STATE_DIR/finalizer.pid.stale.$(date -u '+%Y%m%dT%H%M%SZ').$$"
  fi
}

start_finalizer() {
  mkdir -p "$STATE_DIR"
  if [[ "$(read_phase)" == "completed" && -f "$VERIFICATION_REPORT" ]]; then
    print -- "Dense finalization already completed."
    show_status
    return 0
  fi
  local pid
  if pid="$(read_pid 2>/dev/null)"; then
    local classification
    classification="$(classify_pid "$pid")"
    if [[ "$classification" == "finalizer" ]]; then
      print -- "Dense finalizer already running with PID $pid."
      return 0
    fi
    if [[ "$classification" == "foreign" ]]; then
      print -u2 -- "STOP: finalizer.pid belongs to a live unrelated process; it was not signaled"
      show_status
      return 69
    fi
    archive_stale_pid
  else
    local pid_result=$?
    if (( pid_result == 2 )); then
      print -u2 -- "STOP: finalizer.pid is not a valid positive PID"
      show_status
      return 69
    fi
  fi
  require_launch_runtime
  if [[ -z "$CAFFEINATE_COMMAND" || ! -x "$CAFFEINATE_COMMAND" ]]; then
    print -u2 -- "STOP: caffeinate executable is unavailable"
    return 64
  fi
  local launch_lock="$STATE_DIR/.launch-lock"
  if ! mkdir "$launch_lock" 2>/dev/null; then
    print -u2 -- "STOP: another finalizer launch owns $launch_lock"
    return 69
  fi
  trap 'rmdir "$launch_lock" 2>/dev/null || true' EXIT INT TERM
  write_state "starting" null "" "$(read_build_phase)"
  local finalizer_pid
  if [[ "$(uname -s)" == "Darwin" && "${DENSE_FINALIZER_TEST_MODE:-0}" != "1" && -x "$LAUNCHCTL_COMMAND" ]]; then
    local launch_label="com.redacted-document-llm.dense-finalizer.$(date -u '+%Y%m%dT%H%M%SZ').$$"
    "$LAUNCHCTL_COMMAND" submit -l "$launch_label" -- \
      /usr/bin/env \
      DENSE_FINALIZER_PROJECT_ROOT="$PROJECT_ROOT" \
      DENSE_FINALIZER_CORPUS_ROOT="$CORPUS_ROOT" \
      DENSE_FINALIZER_BUILD_STATE_FILE="$BUILD_STATE_FILE" \
      DENSE_FINALIZER_STATE_DIR="$STATE_DIR" \
      DENSE_FINALIZER_DENSE_DIRECTORY="$DENSE_DIRECTORY" \
      DENSE_FINALIZER_CHUNKS="$CHUNKS_PATH" \
      DENSE_FINALIZER_CORPUS_MANIFEST="$CORPUS_MANIFEST_PATH" \
      DENSE_FINALIZER_DENSE_MANIFEST="$DENSE_MANIFEST_PATH" \
      DENSE_FINALIZER_CERTIFICATION_QUERIES="$CERTIFICATION_QUERIES_PATH" \
      DENSE_FINALIZER_LATENCY_REPORT="$LATENCY_REPORT_PATH" \
      DENSE_FINALIZER_POLL_SECONDS="$POLL_SECONDS" \
      DENSE_FINALIZER_DOCKER="$DOCKER_COMMAND" \
      DENSE_FINALIZER_CAFFEINATE="$CAFFEINATE_COMMAND" \
      "$CAFFEINATE_COMMAND" -i "$SCRIPT_PATH" internal-run-managed
    local launch_attempt
    finalizer_pid=""
    for launch_attempt in {1..40}; do
      if finalizer_pid="$(read_pid 2>/dev/null)" && [[ "$(classify_pid "$finalizer_pid")" == "finalizer" ]]; then
        break
      fi
      finalizer_pid=""
      sleep 0.05
    done
    if [[ -z "$finalizer_pid" ]]; then
      write_state "launch_failed" 70 "managed_process_did_not_register" "$(read_build_phase)"
      rmdir "$launch_lock"
      trap - EXIT INT TERM
      return 70
    fi
  else
    nohup "$CAFFEINATE_COMMAND" -i "$SCRIPT_PATH" internal-run >> "$LOG_FILE" 2>&1 &
    finalizer_pid=$!
    local temporary_pid="$STATE_DIR/.finalizer.pid.$$.tmp"
    print -- "$finalizer_pid" > "$temporary_pid"
    mv "$temporary_pid" "$PID_FILE"
  fi
  rmdir "$launch_lock"
  trap - EXIT INT TERM
  print -- "Dense finalizer started with PID $finalizer_pid."
}

wait_for_build() {
  local build_phase
  while true; do
    build_phase="$(read_build_phase)"
    case "$build_phase" in
      completed)
        return 0
        ;;
      benchmark_failed|storage_rejected|build_failed|launch_failed)
        write_state "blocked" 68 "dense_build_$build_phase" "$build_phase"
        return 68
        ;;
      *)
        write_state "waiting" null "" "$build_phase"
        sleep "$POLL_SECONDS"
        ;;
    esac
  done
}

run_finalizer() {
  require_launch_runtime
  wait_for_build
  require_artifacts

  write_state "latency_certifying" null "" "completed"
  local latency_output="$STATE_DIR/.latency-command-report.$$.tmp"
  local latency_exit=0
  (
    cd "$PROJECT_ROOT"
    "$DOCKER_COMMAND" compose run --rm --no-deps \
      -e REDACTED_RUNTIME_ROLE=worker \
      -v "$CORPUS_ROOT:/artifacts/corpus:rw" \
      worker corpus certify-best-available-dense-latency \
      --chunks /artifacts/corpus/best-available/chunks.jsonl \
      --dense-directory /artifacts/corpus/best-available/dense \
      --dense-manifest /artifacts/corpus/best-available/dense/manifest.json \
      --certification-queries /artifacts/corpus/best-available/dense/recall-queries.txt \
      --output /artifacts/corpus/best-available/dense/latency-report.json
  ) > "$latency_output" || latency_exit=$?
  if (( latency_exit != 0 )); then
    mv "$latency_output" "$STATE_DIR/latency-output.failed.$(date -u '+%Y%m%dT%H%M%SZ').json"
    write_state "failed" "$latency_exit" "latency_certification_failed" "completed"
    return "$latency_exit"
  fi
  if [[ ! -f "$LATENCY_REPORT_PATH" ]] || ! jq -e 'type == "object"' "$LATENCY_REPORT_PATH" >/dev/null 2>&1 || ! jq -e 'type == "object"' "$latency_output" >/dev/null 2>&1; then
    mv "$latency_output" "$STATE_DIR/latency-output.invalid.$(date -u '+%Y%m%dT%H%M%SZ').json"
    write_state "failed" 65 "invalid_latency_output" "completed"
    return 65
  fi
  mv "$latency_output" "$STATE_DIR/latency-command-report.json"

  write_state "verifying" null "" "completed"
  local verification_output="$STATE_DIR/.verification-report.$$.tmp"
  local verification_exit=0
  (
    cd "$PROJECT_ROOT"
    "$DOCKER_COMMAND" compose run --rm --no-deps \
      -e REDACTED_RUNTIME_ROLE=worker \
      -v "$CORPUS_ROOT:/artifacts/corpus:rw" \
      worker corpus verify-best-available-dense \
      --chunks /artifacts/corpus/best-available/chunks.jsonl \
      --corpus-manifest /artifacts/corpus/best-available/corpus-manifest.json \
      --dense-directory /artifacts/corpus/best-available/dense \
      --dense-manifest /artifacts/corpus/best-available/dense/manifest.json \
      --certification-queries /artifacts/corpus/best-available/dense/recall-queries.txt \
      --latency-report /artifacts/corpus/best-available/dense/latency-report.json
  ) > "$verification_output" || verification_exit=$?
  if (( verification_exit != 0 )); then
    mv "$verification_output" "$STATE_DIR/verification-output.failed.$(date -u '+%Y%m%dT%H%M%SZ').json"
    write_state "failed" "$verification_exit" "artifact_verification_failed" "completed"
    return "$verification_exit"
  fi
  if ! jq -e 'type == "object"' "$verification_output" >/dev/null 2>&1; then
    mv "$verification_output" "$STATE_DIR/verification-output.invalid.$(date -u '+%Y%m%dT%H%M%SZ').json"
    write_state "failed" 65 "invalid_verification_output" "completed"
    return 65
  fi
  mv "$verification_output" "$VERIFICATION_REPORT"
  write_state "completed" 0 "" "completed"
}

run_managed_internal() {
  mkdir -p "$STATE_DIR"
  exec >> "$LOG_FILE" 2>&1
  local temporary_pid="$STATE_DIR/.finalizer.pid.$$.tmp"
  print -- "$$" > "$temporary_pid"
  mv "$temporary_pid" "$PID_FILE"
  run_finalizer
}

usage() {
  print -u2 -- "usage: ${0:t} {status|start|internal-run}"
  return 64
}

command_name=${1:-}
case "$command_name" in
  status)
    mkdir -p "$STATE_DIR"
    show_status
    ;;
  start)
    start_finalizer
    ;;
  internal-run)
    run_finalizer
    ;;
  internal-run-managed)
    run_managed_internal
    ;;
  *)
    usage
    ;;
esac
