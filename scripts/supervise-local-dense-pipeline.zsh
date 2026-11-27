#!/bin/zsh
set -euo pipefail

SCRIPT_PATH="${0:A}"
PROJECT_ROOT="${DENSE_SUPERVISOR_PROJECT_ROOT:-${SCRIPT_PATH:h:h}}"
RUNNER="${DENSE_SUPERVISOR_RUNNER:-$PROJECT_ROOT/scripts/run-local-dense-build.zsh}"
FINALIZER="${DENSE_SUPERVISOR_FINALIZER:-$PROJECT_ROOT/scripts/finalize-local-dense-build.zsh}"
ACTIVATOR="${DENSE_SUPERVISOR_ACTIVATOR:-$PROJECT_ROOT/scripts/activate-local-full-predictions.zsh}"

require_component() {
  local component=$1
  local component_path=$2
  if [[ "$component_path" != /* || ! -x "$component_path" || ! -f "$component_path" ]]; then
    print -u2 -- "$component executable is unavailable: $component_path"
    return 64
  fi
}

read_status() {
  local component=$1
  local component_path=$2
  local output
  local status_exit=0
  output="$("$component_path" status)" || status_exit=$?
  if ! jq -e 'type == "object"' >/dev/null 2>&1 <<< "$output"; then
    print -u2 -- "$component status is invalid"
    return 65
  fi
  print -r -- "$output"
}

start_component() {
  local component=$1
  local component_path=$2
  if ! "$component_path" start >/dev/null; then
    print -u2 -- "$component could not be started"
    return 70
  fi
}

run_supervisor() {
  require_component runner "$RUNNER"
  require_component finalizer "$FINALIZER"
  require_component activator "$ACTIVATOR"

  local runner_status
  local finalizer_status
  local activator_status
  runner_status="$(read_status runner "$RUNNER")"
  finalizer_status="$(read_status finalizer "$FINALIZER")"
  activator_status="$(read_status activator "$ACTIVATOR")"

  if ! jq -e '
    (.phase | type == "string") and
    (.runner_status | type == "string") and
    (.resumable | type == "boolean")
  ' >/dev/null 2>&1 <<< "$runner_status"; then
    print -u2 -- "runner status is invalid"
    return 65
  fi
  if ! jq -e '
    (.phase | type == "string") and
    (.process_status | type == "string")
  ' >/dev/null 2>&1 <<< "$finalizer_status"; then
    print -u2 -- "finalizer status is invalid"
    return 65
  fi
  if ! jq -e '
    (.phase | type == "string") and
    (.process_status | type == "string")
  ' >/dev/null 2>&1 <<< "$activator_status"; then
    print -u2 -- "activator status is invalid"
    return 65
  fi

  local -a actions
  actions=()
  if jq -e '
    .phase == "building" and
    .runner_status == "stopped" and
    .resumable == true
  ' >/dev/null <<< "$runner_status"; then
    start_component runner "$RUNNER"
    actions+=(runner_started)
  fi
  if jq -e '
    .process_status == "stopped" and
    (.phase == "not_started" or .phase == "starting" or .phase == "waiting")
  ' >/dev/null <<< "$finalizer_status"; then
    start_component finalizer "$FINALIZER"
    actions+=(finalizer_started)
  fi
  if jq -e '
    .process_status == "stopped" and
    (.phase == "not_started" or .phase == "starting" or .phase == "waiting")
  ' >/dev/null <<< "$activator_status"; then
    start_component activator "$ACTIVATOR"
    actions+=(activator_started)
  fi

  jq -n \
    --arg checked_at "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
    --args \
    '{
      schema_version: "local-dense-pipeline-supervisor-v1",
      checked_at: $checked_at,
      actions: $ARGS.positional
    }' \
    -- "${actions[@]}"
}

usage() {
  print -u2 -- "usage: ${0:t} run"
  return 64
}

case "${1:-}" in
  run)
    run_supervisor
    ;;
  *)
    usage
    ;;
esac
