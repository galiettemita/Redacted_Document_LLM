#!/bin/zsh
set -euo pipefail

SCRIPT_PATH="${0:A}"
SCRIPT_PROJECT_ROOT="${SCRIPT_PATH:h:h}"
PROJECT_ROOT="${FULL_ACTIVATOR_PROJECT_ROOT:-$SCRIPT_PROJECT_ROOT}"
STATE_DIR="${FULL_ACTIVATOR_STATE_DIR:-$PROJECT_ROOT/var/corpus/best-available/dense-activation-state}"
FINALIZER_STATE_FILE="${FULL_ACTIVATOR_FINALIZER_STATE_FILE:-$PROJECT_ROOT/var/corpus/best-available/dense-finalizer-state/state.json}"
VERIFICATION_REPORT="${FULL_ACTIVATOR_VERIFICATION_REPORT:-$PROJECT_ROOT/var/corpus/best-available/dense-finalizer-state/verification-report.json}"
ENV_FILE="${FULL_ACTIVATOR_ENV_FILE:-$PROJECT_ROOT/.env}"
STATE_FILE="$STATE_DIR/state.json"
PID_FILE="$STATE_DIR/activator.pid"
LOG_FILE="$STATE_DIR/activator.log"
READINESS_RESPONSE="$STATE_DIR/readiness-response.json"
READINESS_URL="${FULL_ACTIVATOR_READINESS_URL:-http://127.0.0.1:8000/v1/readiness}"
POLL_SECONDS="${FULL_ACTIVATOR_POLL_SECONDS:-60}"
READINESS_TIMEOUT_SECONDS="${FULL_ACTIVATOR_READINESS_TIMEOUT_SECONDS:-600}"
READINESS_PROBE_TIMEOUT_SECONDS="${FULL_ACTIVATOR_READINESS_PROBE_TIMEOUT_SECONDS:-45}"

if [[ -n "${FULL_ACTIVATOR_DOCKER:-}" ]]; then
  DOCKER_COMMAND="$FULL_ACTIVATOR_DOCKER"
else
  DOCKER_COMMAND="$(command -v docker 2>/dev/null || true)"
fi
if [[ -n "${FULL_ACTIVATOR_CURL:-}" ]]; then
  CURL_COMMAND="$FULL_ACTIVATOR_CURL"
else
  CURL_COMMAND="$(command -v curl 2>/dev/null || true)"
fi
if [[ -n "${FULL_ACTIVATOR_CAFFEINATE:-}" ]]; then
  CAFFEINATE_COMMAND="$FULL_ACTIVATOR_CAFFEINATE"
else
  CAFFEINATE_COMMAND="$(command -v caffeinate 2>/dev/null || true)"
fi
if [[ -n "${FULL_ACTIVATOR_LAUNCHCTL:-}" ]]; then
  LAUNCHCTL_COMMAND="$FULL_ACTIVATOR_LAUNCHCTL"
else
  LAUNCHCTL_COMMAND="$(command -v launchctl 2>/dev/null || true)"
fi

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

require_regular_file() {
  local path=$1
  local label=$2
  if [[ "$path" != /* || ! -f "$path" || -L "$path" ]]; then
    print -u2 -- "STOP: $label must be an existing non-symlink absolute file: $path"
    return 64
  fi
}

require_positive_number() {
  local value=$1
  local label=$2
  if [[ ! "$value" =~ '^[0-9]+([.][0-9]+)?$' || "$value" == "0" || "$value" == "0.0" ]]; then
    print -u2 -- "STOP: $label must be positive"
    return 64
  fi
}

require_runtime() {
  require_absolute_directory "$PROJECT_ROOT" "project root" || return $?
  require_regular_file "$PROJECT_ROOT/compose.yaml" "Compose file" || return $?
  require_regular_file "$ENV_FILE" "environment file" || return $?
  if [[ -z "$DOCKER_COMMAND" || ! -x "$DOCKER_COMMAND" ]]; then
    print -u2 -- "STOP: docker executable is unavailable"
    return 64
  fi
  if [[ -z "$CURL_COMMAND" || ! -x "$CURL_COMMAND" ]]; then
    print -u2 -- "STOP: curl executable is unavailable"
    return 64
  fi
  if ! command -v jq >/dev/null 2>&1; then
    print -u2 -- "STOP: jq executable is unavailable"
    return 64
  fi
  require_positive_number "$POLL_SECONDS" "poll seconds" || return $?
  require_positive_number "$READINESS_TIMEOUT_SECONDS" "readiness timeout" || return $?
  require_positive_number "$READINESS_PROBE_TIMEOUT_SECONDS" "readiness probe timeout" || return $?
  if ! awk \
    -v probe="$READINESS_PROBE_TIMEOUT_SECONDS" \
    -v timeout="$READINESS_TIMEOUT_SECONDS" \
    'BEGIN { exit(probe <= timeout ? 0 : 1) }'; then
    print -u2 -- "STOP: readiness probe timeout cannot exceed the overall readiness timeout"
    return 64
  fi
}

write_state() {
  local phase=$1
  local exit_code=${2:-null}
  local reason=${3:-}
  local finalizer_phase=${4:-}
  local temporary="$STATE_DIR/.state.json.$$.tmp"
  mkdir -p "$STATE_DIR"
  jq -n \
    --arg phase "$phase" \
    --arg updated_at "$(iso_now)" \
    --arg reason "$reason" \
    --arg finalizer_phase "$finalizer_phase" \
    --argjson exit_code "$exit_code" \
    '{
      schema_version: "local-full-prediction-activator-v1",
      phase: $phase,
      updated_at: $updated_at,
      exit_code: $exit_code,
      reason: (if $reason == "" then null else $reason end),
      finalizer_phase: (if $finalizer_phase == "" then null else $finalizer_phase end)
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

read_finalizer_phase() {
  if [[ -f "$FINALIZER_STATE_FILE" ]]; then
    jq -er '
      select(.schema_version == "local-dense-finalizer-v1") |
      .phase | strings
    ' "$FINALIZER_STATE_FILE" 2>/dev/null || print -- "invalid"
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
    print -- "activator"
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
    --arg finalizer_phase "$(read_finalizer_phase)" \
    --argjson activator_pid "$pid" \
    '. + {
      schema_version: (.schema_version // "local-full-prediction-activator-v1"),
      phase: (.phase // $phase),
      process_status: $process_status,
      activator_pid: $activator_pid,
      finalizer_phase: $finalizer_phase
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
  if [[ "$classification" == "activator" ]]; then
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
    mv "$PID_FILE" "$STATE_DIR/activator.pid.stale.$(date -u '+%Y%m%dT%H%M%SZ').$$"
  fi
}

start_activator() {
  mkdir -p "$STATE_DIR"
  if [[ "$(read_phase)" == "activated" && -f "$READINESS_RESPONSE" ]]; then
    print -- "Full prediction activation already completed."
    show_status
    return 0
  fi
  local pid
  if pid="$(read_pid 2>/dev/null)"; then
    local classification
    classification="$(classify_pid "$pid")"
    if [[ "$classification" == "activator" ]]; then
      print -- "Full prediction activator already running with PID $pid."
      return 0
    fi
    if [[ "$classification" == "foreign" ]]; then
      print -u2 -- "STOP: activator.pid belongs to a live unrelated process; it was not signaled"
      show_status
      return 69
    fi
    archive_stale_pid
  else
    local pid_result=$?
    if (( pid_result == 2 )); then
      print -u2 -- "STOP: activator.pid is not a valid positive PID"
      show_status
      return 69
    fi
  fi
  require_runtime || return $?
  if [[ -z "$CAFFEINATE_COMMAND" || ! -x "$CAFFEINATE_COMMAND" ]]; then
    print -u2 -- "STOP: caffeinate executable is unavailable"
    return 64
  fi
  local launch_lock="$STATE_DIR/.launch-lock"
  if ! mkdir "$launch_lock" 2>/dev/null; then
    print -u2 -- "STOP: another activator launch owns $launch_lock"
    return 69
  fi
  trap 'rmdir "$launch_lock" 2>/dev/null || true' EXIT INT TERM
  write_state "starting" null "" "$(read_finalizer_phase)"
  local activator_pid
  if [[ "$(uname -s)" == "Darwin" && "${FULL_ACTIVATOR_TEST_MODE:-0}" != "1" && -x "$LAUNCHCTL_COMMAND" ]]; then
    local launch_label="com.redacted-document-llm.full-prediction-activator.$(date -u '+%Y%m%dT%H%M%SZ').$$"
    "$LAUNCHCTL_COMMAND" submit -l "$launch_label" -- \
      /usr/bin/env \
      FULL_ACTIVATOR_PROJECT_ROOT="$PROJECT_ROOT" \
      FULL_ACTIVATOR_STATE_DIR="$STATE_DIR" \
      FULL_ACTIVATOR_FINALIZER_STATE_FILE="$FINALIZER_STATE_FILE" \
      FULL_ACTIVATOR_VERIFICATION_REPORT="$VERIFICATION_REPORT" \
      FULL_ACTIVATOR_ENV_FILE="$ENV_FILE" \
      FULL_ACTIVATOR_READINESS_URL="$READINESS_URL" \
      FULL_ACTIVATOR_POLL_SECONDS="$POLL_SECONDS" \
      FULL_ACTIVATOR_READINESS_TIMEOUT_SECONDS="$READINESS_TIMEOUT_SECONDS" \
      FULL_ACTIVATOR_DOCKER="$DOCKER_COMMAND" \
      FULL_ACTIVATOR_CURL="$CURL_COMMAND" \
      FULL_ACTIVATOR_CAFFEINATE="$CAFFEINATE_COMMAND" \
      FULL_ACTIVATOR_LAUNCHCTL="$LAUNCHCTL_COMMAND" \
      FULL_ACTIVATOR_LAUNCH_LABEL="$launch_label" \
      "$CAFFEINATE_COMMAND" -i "$SCRIPT_PATH" internal-run-managed
    local launch_attempt
    activator_pid=""
    for launch_attempt in {1..40}; do
      if activator_pid="$(read_pid 2>/dev/null)" && [[ "$(classify_pid "$activator_pid")" == "activator" ]]; then
        break
      fi
      activator_pid=""
      sleep 0.05
    done
    if [[ -z "$activator_pid" ]]; then
      write_state "launch_failed" 70 "managed_process_did_not_register" "$(read_finalizer_phase)"
      rmdir "$launch_lock"
      trap - EXIT INT TERM
      return 70
    fi
  else
    nohup "$CAFFEINATE_COMMAND" -i "$SCRIPT_PATH" internal-run >> "$LOG_FILE" 2>&1 &
    activator_pid=$!
    local temporary_pid="$STATE_DIR/.activator.pid.$$.tmp"
    print -- "$activator_pid" > "$temporary_pid"
    mv "$temporary_pid" "$PID_FILE"
  fi
  rmdir "$launch_lock"
  trap - EXIT INT TERM
  print -- "Full prediction activator started with PID $activator_pid."
}

wait_for_finalizer() {
  local finalizer_phase
  while true; do
    finalizer_phase="$(read_finalizer_phase)"
    case "$finalizer_phase" in
      completed)
        if ! jq -e '
          .schema_version == "local-dense-finalizer-v1" and
          .phase == "completed" and
          .exit_code == 0
        ' "$FINALIZER_STATE_FILE" >/dev/null 2>&1; then
          write_state "blocked" 65 "dense_finalizer_state_invalid" "$finalizer_phase"
          return 65
        fi
        return 0
        ;;
      failed|blocked)
        write_state "blocked" 68 "dense_finalizer_$finalizer_phase" "$finalizer_phase"
        return 68
        ;;
      invalid)
        write_state "blocked" 65 "dense_finalizer_state_invalid" "$finalizer_phase"
        return 65
        ;;
      *)
        write_state "waiting" null "" "$finalizer_phase"
        sleep "$POLL_SECONDS"
        ;;
    esac
  done
}

validate_verification_report() {
  if [[ ! -f "$VERIFICATION_REPORT" || -L "$VERIFICATION_REPORT" ]]; then
    write_state "blocked" 65 "verification_report_invalid" "completed"
    return 65
  fi
  if ! jq -e '
    type == "object" and
    .chunk_count == 14549269 and
    (.recall_at_300 | type == "number" and . >= 0.98) and
    (.max_warm_search_seconds | type == "number" and . <= 5.0)
  ' "$VERIFICATION_REPORT" >/dev/null 2>&1; then
    write_state "blocked" 65 "verification_report_invalid" "completed"
    return 65
  fi
}

env_key_count() {
  local key=$1
  awk -F= -v key="$key" '$1 == key { count += 1 } END { print count + 0 }' "$ENV_FILE"
}

env_key_value() {
  local key=$1
  awk -v key="$key" 'index($0, key "=") == 1 { print substr($0, length(key) + 2) }' "$ENV_FILE"
}

validate_approval_config() {
  local master_count
  master_count="$(env_key_count REDACTED_ALLOW_PAID_MODEL_CALLS)"
  if [[ "$master_count" != "1" || "$(env_key_value REDACTED_ALLOW_PAID_MODEL_CALLS)" != "true" ]]; then
    write_state "blocked" 65 "master_approval_not_enabled" "completed"
    return 65
  fi
  local full_count
  full_count="$(env_key_count REDACTED_ALLOW_BEST_AVAILABLE_CALLS)"
  if (( full_count > 1 )); then
    write_state "blocked" 65 "full_basis_approval_ambiguous" "completed"
    return 65
  fi
  if (( full_count == 1 )); then
    local full_value
    full_value="$(env_key_value REDACTED_ALLOW_BEST_AVAILABLE_CALLS)"
    if [[ "$full_value" != "true" && "$full_value" != "false" ]]; then
      write_state "blocked" 65 "full_basis_approval_invalid" "completed"
      return 65
    fi
  fi
}

write_full_basis_config() {
  local mode=$1
  local value=${2:-}
  local temporary="${ENV_FILE:h}/.env.full-prediction-activation.$$.tmp"
  awk -v mode="$mode" -v value="$value" '
    BEGIN { replaced = 0 }
    index($0, "REDACTED_ALLOW_BEST_AVAILABLE_CALLS=") == 1 {
      if (mode == "present") {
        print "REDACTED_ALLOW_BEST_AVAILABLE_CALLS=" value
      }
      replaced = 1
      next
    }
    { print }
    END {
      if (mode == "present" && replaced == 0) {
        print "REDACTED_ALLOW_BEST_AVAILABLE_CALLS=" value
      }
    }
  ' "$ENV_FILE" > "$temporary"
  chmod 600 "$temporary"
  mv "$temporary" "$ENV_FILE"
}

enable_full_basis() {
  if [[ "$(env_key_count REDACTED_ALLOW_BEST_AVAILABLE_CALLS)" == "1" && "$(env_key_value REDACTED_ALLOW_BEST_AVAILABLE_CALLS)" == "true" ]]; then
    return 0
  fi
  write_full_basis_config present true
}

restore_full_basis() {
  local previous_count=$1
  local previous_value=$2
  if [[ "$previous_count" == "0" ]]; then
    write_full_basis_config absent
  else
    write_full_basis_config present "$previous_value"
  fi
}

recreate_services() {
  local output_name=$1
  (
    cd "$PROJECT_ROOT"
    "$DOCKER_COMMAND" compose up -d --force-recreate backend worker
  ) > "$STATE_DIR/$output_name" 2>&1
}

rollback_activation() {
  local previous_count=$1
  local previous_value=$2
  restore_full_basis "$previous_count" "$previous_value" || return 70
  recreate_services "rollback-compose-output.log"
}

readiness_attempt_count() {
  awk \
    -v timeout="$READINESS_TIMEOUT_SECONDS" \
    -v poll="$POLL_SECONDS" \
    -v probe="$READINESS_PROBE_TIMEOUT_SECONDS" '
    BEGIN {
      attempts = int((timeout + poll) / (probe + poll))
      if (attempts < 1) attempts = 1
      print attempts
    }
  '
}

deploy_and_verify() {
  local previous_full_count
  previous_full_count="$(env_key_count REDACTED_ALLOW_BEST_AVAILABLE_CALLS)"
  local previous_full_value=""
  if [[ "$previous_full_count" == "1" ]]; then
    previous_full_value="$(env_key_value REDACTED_ALLOW_BEST_AVAILABLE_CALLS)"
  fi
  write_state "enabling" null "" "completed"
  enable_full_basis
  write_state "deploying" null "" "completed"
  local deploy_exit=0
  recreate_services "compose-output.log" || deploy_exit=$?
  if (( deploy_exit != 0 )); then
    local rollback_exit=0
    rollback_activation "$previous_full_count" "$previous_full_value" || rollback_exit=$?
    local failure_reason="compose_recreate_failed"
    if (( rollback_exit != 0 )); then
      failure_reason="compose_recreate_failed_rollback_recreate_failed"
    fi
    write_state "failed" "$deploy_exit" "$failure_reason" "completed"
    return "$deploy_exit"
  fi

  write_state "verifying_readiness" null "" "completed"
  local maximum_attempts
  maximum_attempts="$(readiness_attempt_count)"
  local attempt
  local temporary="$STATE_DIR/.readiness-response.$$.tmp"
  for (( attempt = 1; attempt <= maximum_attempts; attempt += 1 )); do
    local curl_exit=0
    "$CURL_COMMAND" \
      -fsS \
      --connect-timeout "$READINESS_PROBE_TIMEOUT_SECONDS" \
      --max-time "$READINESS_PROBE_TIMEOUT_SECONDS" \
      "$READINESS_URL" > "$temporary" 2>> "$LOG_FILE" || curl_exit=$?
    if (( curl_exit == 0 )) && jq -e '
      .prediction_bases.best_available_foiarchive_v1.operational == true and
      .prediction_bases.best_available_foiarchive_v1.approved == true and
      .prediction_bases.best_available_foiarchive_v1.ready == true and
      .strongest_ready_basis == "best_available_foiarchive_v1"
    ' "$temporary" >/dev/null 2>&1; then
      mv "$temporary" "$READINESS_RESPONSE"
      write_state "activated" 0 "" "completed"
      return 0
    fi
    if [[ -f "$temporary" ]]; then
      mv "$temporary" "$READINESS_RESPONSE"
    fi
    if (( attempt < maximum_attempts )); then
      sleep "$POLL_SECONDS"
    fi
  done
  local rollback_exit=0
  rollback_activation "$previous_full_count" "$previous_full_value" || rollback_exit=$?
  if (( rollback_exit != 0 )); then
    write_state "failed" "$rollback_exit" "full_basis_readiness_timeout_rollback_failed" "completed"
    return "$rollback_exit"
  fi
  write_state "failed" 69 "full_basis_readiness_timeout" "completed"
  return 69
}

run_internal() {
  require_runtime || return $?
  mkdir -p "$STATE_DIR"
  wait_for_finalizer || return $?
  validate_verification_report || return $?
  validate_approval_config || return $?
  deploy_and_verify
}

run_managed_internal() {
  mkdir -p "$STATE_DIR"
  exec >> "$LOG_FILE" 2>&1
  cleanup_managed_launch() {
    if [[ -n "${FULL_ACTIVATOR_LAUNCH_LABEL:-}" && -x "$LAUNCHCTL_COMMAND" ]]; then
      "$LAUNCHCTL_COMMAND" remove "$FULL_ACTIVATOR_LAUNCH_LABEL" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup_managed_launch EXIT
  local temporary_pid="$STATE_DIR/.activator.pid.$$.tmp"
  print -- "$$" > "$temporary_pid"
  mv "$temporary_pid" "$PID_FILE"
  local run_exit=0
  run_internal || run_exit=$?
  cleanup_managed_launch
  trap - EXIT
  return "$run_exit"
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
    start_activator
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
