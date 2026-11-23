#!/bin/zsh
set -euo pipefail

SCRIPT_PATH="${0:A}"
SCRIPT_PROJECT_ROOT="${SCRIPT_PATH:h:h}"
PROJECT_ROOT="${DENSE_RUNNER_PROJECT_ROOT:-$SCRIPT_PROJECT_ROOT}"
CORPUS_ROOT="${DENSE_RUNNER_CORPUS_ROOT:-$PROJECT_ROOT/var/corpus}"
BEST_AVAILABLE_ROOT="$CORPUS_ROOT/best-available"
STATE_DIR="${DENSE_RUNNER_STATE_DIR:-$BEST_AVAILABLE_ROOT/dense-build-state}"
CHUNKS_PATH="${DENSE_RUNNER_CHUNKS:-$BEST_AVAILABLE_ROOT/chunks.jsonl}"
CORPUS_MANIFEST_PATH="${DENSE_RUNNER_CORPUS_MANIFEST:-$BEST_AVAILABLE_ROOT/corpus-manifest.json}"
BENCHMARK_DIRECTORY="${DENSE_RUNNER_BENCHMARK_DIRECTORY:-$BEST_AVAILABLE_ROOT/dense-benchmark}"
DENSE_DIRECTORY="${DENSE_RUNNER_DENSE_DIRECTORY:-$BEST_AVAILABLE_ROOT/dense}"
APPLICATION_QUERIES_PATH="${DENSE_RUNNER_APPLICATION_QUERIES:-$PROJECT_ROOT/backend/corpus/best_available_application_queries.txt}"
CERTIFICATION_QUERIES_PATH="${DENSE_RUNNER_CERTIFICATION_QUERIES:-$DENSE_DIRECTORY/recall-queries.txt}"
DENSE_MANIFEST_PATH="${DENSE_RUNNER_DENSE_MANIFEST:-$DENSE_DIRECTORY/manifest.json}"
STATE_FILE="$STATE_DIR/state.json"
PID_FILE="$STATE_DIR/runner.pid"
LOG_FILE="$STATE_DIR/runner.log"
BENCHMARK_REPORT="$STATE_DIR/benchmark-report.json"
BUILD_REPORT="$STATE_DIR/build-report.json"
MAX_ARTIFACT_BYTES="${DENSE_RUNNER_MAX_ARTIFACT_BYTES:-17179869184}"
MIN_PROJECTED_FREE_BYTES="${DENSE_RUNNER_MIN_PROJECTED_FREE_BYTES:-4294967296}"
BATCH_SIZE="${DENSE_RUNNER_BATCH_SIZE:-64}"
RECALL_BLOCK_ROWS="${DENSE_RUNNER_RECALL_BLOCK_ROWS:-16384}"

if [[ -n "${DENSE_RUNNER_UV:-}" ]]; then
  UV_COMMAND="$DENSE_RUNNER_UV"
else
  UV_COMMAND="$(command -v uv 2>/dev/null || true)"
fi
if [[ -n "${DENSE_RUNNER_CAFFEINATE:-}" ]]; then
  CAFFEINATE_COMMAND="$DENSE_RUNNER_CAFFEINATE"
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

require_runtime() {
  require_absolute_directory "$PROJECT_ROOT" "project root"
  require_absolute_directory "$BEST_AVAILABLE_ROOT" "best-available corpus root"
  require_file "$CHUNKS_PATH" "chunks"
  require_file "$CORPUS_MANIFEST_PATH" "corpus manifest"
  require_file "$APPLICATION_QUERIES_PATH" "application queries"
  if [[ -z "$UV_COMMAND" || ! -x "$UV_COMMAND" ]]; then
    print -u2 -- "STOP: uv executable is unavailable"
    return 64
  fi
  if [[ "$BATCH_SIZE" != <1-> || "$RECALL_BLOCK_ROWS" != <1-> ]]; then
    print -u2 -- "STOP: batch and recall block sizes must be positive integers"
    return 64
  fi
}

write_state() {
  local phase=$1
  local exit_code=${2:-null}
  local reason=${3:-}
  local selected_shard_size=${4:-null}
  local projected_artifact_bytes=${5:-null}
  local available_bytes=${6:-null}
  local temporary="$STATE_DIR/.state.json.$$.tmp"
  mkdir -p "$STATE_DIR"
  jq -n \
    --arg phase "$phase" \
    --arg updated_at "$(iso_now)" \
    --arg reason "$reason" \
    --argjson exit_code "$exit_code" \
    --argjson selected_shard_size "$selected_shard_size" \
    --argjson projected_artifact_bytes "$projected_artifact_bytes" \
    --argjson available_bytes "$available_bytes" \
    '{
      schema_version: "local-dense-runner-v1",
      phase: $phase,
      updated_at: $updated_at,
      exit_code: $exit_code,
      reason: (if $reason == "" then null else $reason end),
      selected_shard_size: $selected_shard_size,
      projected_artifact_bytes: $projected_artifact_bytes,
      available_bytes: $available_bytes
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
    print -- "runner"
  else
    print -- "foreign"
  fi
}

status_json() {
  local runner_status=$1
  local phase=$2
  local pid=${3:-null}
  local interrupted_phase=""
  local resumable=false
  local restart_required=false
  if [[ "$runner_status" == "stopped" && "$phase" == "building" || "$runner_status" == "stopped" && "$phase" == "build_failed" ]]; then
    interrupted_phase="$phase"
    resumable=true
  elif [[ "$runner_status" == "stopped" && "$phase" == "benchmarking" ]]; then
    interrupted_phase="$phase"
    restart_required=true
  fi
  local base='{}'
  if [[ -f "$STATE_FILE" ]] && jq -e 'type == "object"' "$STATE_FILE" >/dev/null 2>&1; then
    base="$(<"$STATE_FILE")"
  fi
  print -r -- "$base" | jq \
    --arg runner_status "$runner_status" \
    --arg phase "$phase" \
    --arg interrupted_phase "$interrupted_phase" \
    --argjson runner_pid "$pid" \
    --argjson resumable "$resumable" \
    --argjson restart_required "$restart_required" \
    '. + {
      schema_version: (.schema_version // "local-dense-runner-v1"),
      phase: (.phase // $phase),
      runner_status: $runner_status,
      runner_pid: $runner_pid,
      interrupted_phase: (if $interrupted_phase == "" then null else $interrupted_phase end),
      resumable: $resumable,
      restart_required: $restart_required
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
  if [[ "$classification" == "runner" ]]; then
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
    mv "$PID_FILE" "$STATE_DIR/runner.pid.stale.$(date -u '+%Y%m%dT%H%M%SZ').$$"
  fi
}

start_runner() {
  mkdir -p "$STATE_DIR"
  local phase
  phase="$(read_phase)"
  if [[ "$phase" == "completed" && -f "$DENSE_MANIFEST_PATH" ]]; then
    print -- "Dense build already completed."
    show_status
    return 0
  fi
  local pid
  if pid="$(read_pid 2>/dev/null)"; then
    local classification
    classification="$(classify_pid "$pid")"
    if [[ "$classification" == "runner" ]]; then
      print -- "Dense runner already running with PID $pid."
      return 0
    fi
    if [[ "$classification" == "foreign" ]]; then
      print -u2 -- "STOP: runner.pid belongs to a live unrelated process; it was not signaled"
      show_status
      return 69
    fi
    archive_stale_pid
  else
    local pid_result=$?
    if (( pid_result == 2 )); then
      print -u2 -- "STOP: runner.pid is not a valid positive PID"
      show_status
      return 69
    fi
  fi
  require_runtime
  if [[ -z "$CAFFEINATE_COMMAND" || ! -x "$CAFFEINATE_COMMAND" ]]; then
    print -u2 -- "STOP: caffeinate executable is unavailable"
    return 64
  fi
  local launch_lock="$STATE_DIR/.launch-lock"
  if ! mkdir "$launch_lock" 2>/dev/null; then
    print -u2 -- "STOP: another runner launch owns $launch_lock"
    return 69
  fi
  trap 'rmdir "$launch_lock" 2>/dev/null || true' EXIT INT TERM
  if [[ "$phase" != "building" && "$phase" != "build_failed" && "$phase" != "storage_rejected" ]]; then
    write_state "starting"
  fi
  local runner_pid
  if [[ "$(uname -s)" == "Darwin" && "${DENSE_RUNNER_TEST_MODE:-0}" != "1" && -x "$LAUNCHCTL_COMMAND" ]]; then
    local launch_label="com.redacted-document-llm.dense.$(date -u '+%Y%m%dT%H%M%SZ').$$"
    "$LAUNCHCTL_COMMAND" submit -l "$launch_label" -- \
      /usr/bin/env \
      DENSE_RUNNER_PROJECT_ROOT="$PROJECT_ROOT" \
      DENSE_RUNNER_CORPUS_ROOT="$CORPUS_ROOT" \
      DENSE_RUNNER_STATE_DIR="$STATE_DIR" \
      DENSE_RUNNER_CHUNKS="$CHUNKS_PATH" \
      DENSE_RUNNER_CORPUS_MANIFEST="$CORPUS_MANIFEST_PATH" \
      DENSE_RUNNER_BENCHMARK_DIRECTORY="$BENCHMARK_DIRECTORY" \
      DENSE_RUNNER_DENSE_DIRECTORY="$DENSE_DIRECTORY" \
      DENSE_RUNNER_APPLICATION_QUERIES="$APPLICATION_QUERIES_PATH" \
      DENSE_RUNNER_CERTIFICATION_QUERIES="$CERTIFICATION_QUERIES_PATH" \
      DENSE_RUNNER_DENSE_MANIFEST="$DENSE_MANIFEST_PATH" \
      DENSE_RUNNER_MAX_ARTIFACT_BYTES="$MAX_ARTIFACT_BYTES" \
      DENSE_RUNNER_MIN_PROJECTED_FREE_BYTES="$MIN_PROJECTED_FREE_BYTES" \
      DENSE_RUNNER_BATCH_SIZE="$BATCH_SIZE" \
      DENSE_RUNNER_RECALL_BLOCK_ROWS="$RECALL_BLOCK_ROWS" \
      DENSE_RUNNER_UV="$UV_COMMAND" \
      DENSE_RUNNER_CAFFEINATE="$CAFFEINATE_COMMAND" \
      "$CAFFEINATE_COMMAND" -i "$SCRIPT_PATH" internal-run-managed
    local launch_attempt
    runner_pid=""
    for launch_attempt in {1..40}; do
      if runner_pid="$(read_pid 2>/dev/null)" && [[ "$(classify_pid "$runner_pid")" == "runner" ]]; then
        break
      fi
      runner_pid=""
      sleep 0.05
    done
    if [[ -z "$runner_pid" ]]; then
      write_state "launch_failed" 70 "managed_process_did_not_register"
      rmdir "$launch_lock"
      trap - EXIT INT TERM
      return 70
    fi
  else
    nohup "$CAFFEINATE_COMMAND" -i "$SCRIPT_PATH" internal-run >> "$LOG_FILE" 2>&1 &
    runner_pid=$!
    local temporary_pid="$STATE_DIR/.runner.pid.$$.tmp"
    print -- "$runner_pid" > "$temporary_pid"
    mv "$temporary_pid" "$PID_FILE"
  fi
  rmdir "$launch_lock"
  trap - EXIT INT TERM
  print -- "Dense runner started with PID $runner_pid."
}

benchmark_is_valid() {
  [[ -f "$BENCHMARK_REPORT" ]] || return 1
  jq -e '
    .schema_version == "dense-build-benchmark-v2" and
    .sample_rows == 100000 and
    .production_chunk_count == 14549269 and
    (.rows_per_second | type == "number" and . > 0) and
    (.embedding_bytes_per_row | type == "number" and . > 0) and
    (.candidate_bytes_per_row | type == "number" and . > 0) and
    (.candidate_fixed_bytes_per_shard | type == "number" and . > 0) and
    (.projected_shard_count | type == "number" and . > 0) and
    (.offset_bytes_per_row | type == "number" and . > 0) and
    (.projected_full_artifact_bytes | type == "number" and . > 0)
  ' "$BENCHMARK_REPORT" >/dev/null
}

available_bytes() {
  if [[ "${DENSE_RUNNER_TEST_MODE:-0}" == "1" && -n "${DENSE_RUNNER_TEST_AVAILABLE_BYTES:-}" ]]; then
    if [[ "$DENSE_RUNNER_TEST_AVAILABLE_BYTES" != <1-> ]]; then
      print -u2 -- "STOP: test available-byte override must be a positive integer"
      return 64
    fi
    print -- "$DENSE_RUNNER_TEST_AVAILABLE_BYTES"
    return 0
  fi
  df -Pk "$BEST_AVAILABLE_ROOT" | awk 'NR == 2 {printf "%.0f\n", $4 * 1024}'
}

select_build_shape() {
  jq -r --argjson maximum "$MAX_ARTIFACT_BYTES" '
    . as $report |
    ([($report.rows_per_second * 7200 | ceil), 1] | max) as $two_hour_rows |
    ($report.production_chunk_count * (
      $report.embedding_bytes_per_row +
      $report.candidate_bytes_per_row +
      $report.offset_bytes_per_row
    )) as $row_bytes |
    ($report.projected_full_artifact_bytes -
      ($report.production_chunk_count * (
        $report.embedding_bytes_per_row +
        $report.candidate_bytes_per_row +
        $report.offset_bytes_per_row
      )) -
      ($report.candidate_fixed_bytes_per_shard * $report.projected_shard_count)
    ) as $measured_shared_bytes |
    (if $measured_shared_bytes > 0 then $measured_shared_bytes else 0 end) as $shared_bytes |
    (($maximum - $row_bytes - $shared_bytes) /
      $report.candidate_fixed_bytes_per_shard | floor) as $maximum_shards |
    (if $maximum_shards > 0
      then (($report.production_chunk_count / $maximum_shards) | ceil)
      else 1000001
    end) as $storage_minimum_shard_size |
    ([$two_hour_rows, $storage_minimum_shard_size, 1] | max) as $desired_shard_size |
    ([$desired_shard_size, 1000000] | min) as $shard_size |
    (($report.production_chunk_count / $shard_size) | ceil) as $shard_count |
    (($row_bytes +
      ($report.candidate_fixed_bytes_per_shard * $shard_count) +
      $shared_bytes
    ) | ceil) as $projected |
    [$shard_size, $projected] | @tsv
  ' "$BENCHMARK_REPORT"
}

run_benchmark_if_needed() {
  local prior_phase
  prior_phase="$(read_phase)"
  if [[ "$prior_phase" == "building" || "$prior_phase" == "build_failed" || "$prior_phase" == "storage_rejected" ]] && benchmark_is_valid; then
    return 0
  fi
  write_state "benchmarking"
  local temporary="$STATE_DIR/.benchmark-report.json.$$.tmp"
  local benchmark_exit=0
  (
    cd "$PROJECT_ROOT/backend"
    "$UV_COMMAND" run corpus benchmark-best-available-dense \
      --chunks "$CHUNKS_PATH" \
      --corpus-manifest "$CORPUS_MANIFEST_PATH" \
      --output-directory "$BENCHMARK_DIRECTORY" \
      --batch-size "$BATCH_SIZE"
  ) > "$temporary" || benchmark_exit=$?
  if (( benchmark_exit != 0 )); then
    if [[ -f "$temporary" ]]; then
      mv "$temporary" "$STATE_DIR/benchmark-output.failed.$(date -u '+%Y%m%dT%H%M%SZ').json"
    fi
    write_state "benchmark_failed" "$benchmark_exit" "benchmark_command_failed"
    return "$benchmark_exit"
  fi
  if ! jq -e 'type == "object"' "$temporary" >/dev/null 2>&1; then
    mv "$temporary" "$STATE_DIR/benchmark-output.invalid.$(date -u '+%Y%m%dT%H%M%SZ').json"
    write_state "benchmark_failed" 65 "invalid_benchmark_report"
    return 65
  fi
  mv "$temporary" "$BENCHMARK_REPORT"
  if ! benchmark_is_valid; then
    write_state "benchmark_failed" 65 "incompatible_benchmark_report"
    return 65
  fi
}

run_internal() {
  require_runtime
  mkdir -p "$STATE_DIR" "$BENCHMARK_DIRECTORY" "$DENSE_DIRECTORY"
  run_benchmark_if_needed

  local build_shape
  build_shape="$(select_build_shape)"
  local selected_shard_size="${build_shape%%$'\t'*}"
  local projected_artifact_bytes="${build_shape##*$'\t'}"
  local current_available_bytes
  current_available_bytes="$(available_bytes)"
  if [[ "$selected_shard_size" != <1-> || "$projected_artifact_bytes" != <1-> || "$current_available_bytes" != <1-> ]]; then
    write_state "storage_rejected" 66 "invalid_storage_measurement"
    return 66
  fi
  if (( projected_artifact_bytes > MAX_ARTIFACT_BYTES )); then
    write_state "storage_rejected" 66 "artifact_limit" "$selected_shard_size" "$projected_artifact_bytes" "$current_available_bytes"
    return 66
  fi
  if (( current_available_bytes - projected_artifact_bytes < MIN_PROJECTED_FREE_BYTES )); then
    write_state "storage_rejected" 66 "free_storage" "$selected_shard_size" "$projected_artifact_bytes" "$current_available_bytes"
    return 66
  fi

  write_state "building" null "" "$selected_shard_size" "$projected_artifact_bytes" "$current_available_bytes"
  local temporary="$STATE_DIR/.build-report.json.$$.tmp"
  local build_exit=0
  (
    cd "$PROJECT_ROOT/backend"
    "$UV_COMMAND" run corpus build-best-available-dense \
      --chunks "$CHUNKS_PATH" \
      --corpus-manifest "$CORPUS_MANIFEST_PATH" \
      --application-queries "$APPLICATION_QUERIES_PATH" \
      --certification-queries "$CERTIFICATION_QUERIES_PATH" \
      --output-directory "$DENSE_DIRECTORY" \
      --output-manifest "$DENSE_MANIFEST_PATH" \
      --batch-size "$BATCH_SIZE" \
      --shard-size "$selected_shard_size" \
      --recall-block-rows "$RECALL_BLOCK_ROWS"
  ) > "$temporary" || build_exit=$?
  if (( build_exit != 0 )); then
    if [[ -f "$temporary" ]]; then
      mv "$temporary" "$STATE_DIR/build-output.failed.$(date -u '+%Y%m%dT%H%M%SZ').json"
    fi
    write_state "build_failed" "$build_exit" "build_command_failed" "$selected_shard_size" "$projected_artifact_bytes" "$current_available_bytes"
    return "$build_exit"
  fi
  if [[ ! -f "$DENSE_MANIFEST_PATH" ]] || ! jq -e '.schema_version == "sharded-dense-index-manifest-v3"' "$DENSE_MANIFEST_PATH" >/dev/null 2>&1 || ! jq -e 'type == "object"' "$temporary" >/dev/null 2>&1; then
    if [[ -f "$temporary" ]]; then
      mv "$temporary" "$STATE_DIR/build-output.invalid.$(date -u '+%Y%m%dT%H%M%SZ').json"
    fi
    write_state "build_failed" 65 "invalid_build_output" "$selected_shard_size" "$projected_artifact_bytes" "$current_available_bytes"
    return 65
  fi
  mv "$temporary" "$BUILD_REPORT"
  write_state "completed" 0 "" "$selected_shard_size" "$projected_artifact_bytes" "$current_available_bytes"
}

run_managed_internal() {
  mkdir -p "$STATE_DIR"
  exec >> "$LOG_FILE" 2>&1
  local temporary_pid="$STATE_DIR/.runner.pid.$$.tmp"
  print -- "$$" > "$temporary_pid"
  mv "$temporary_pid" "$PID_FILE"
  run_internal
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
    start_runner
    ;;
  internal-run)
    run_internal
    ;;
  internal-run-managed)
    run_managed_internal
    ;;
  *)
    usage
    ;;
esac
