#!/bin/zsh
set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the absolute repository path}"
: "${REMOTE:?Set REMOTE to the RunPod SSH destination}"
: "${SSH_PORT:?Set SSH_PORT to the RunPod SSH port}"
: "${SSH_KEY:?Set SSH_KEY to the private SSH key path}"
: "${REMOTE_DENSE:?Set REMOTE_DENSE to the remote dense artifact directory}"

if [[ "$REMOTE_DENSE" != /* || "$REMOTE_DENSE" == *[^A-Za-z0-9_./-]* ]]; then
  print -u2 -- "STOP: REMOTE_DENSE must be a safe absolute path"
  exit 64
fi
if [[ "$REMOTE" != *@* || "$REMOTE" == *[^A-Za-z0-9_@.:-]* ]]; then
  print -u2 -- "STOP: REMOTE must be a safe user@host destination"
  exit 64
fi
if [[ "$SSH_PORT" != <1-> ]]; then
  print -u2 -- "STOP: SSH_PORT must be a positive integer"
  exit 64
fi
if [[ "$SSH_KEY" != /* || "$SSH_KEY" == *[^A-Za-z0-9_./-]* || ! -f "$SSH_KEY" ]]; then
  print -u2 -- "STOP: SSH_KEY must name an existing safe absolute path"
  exit 64
fi

PROJECT_ROOT_CANONICAL="${PROJECT_ROOT:A}"
if [[ ! -d "$PROJECT_ROOT_CANONICAL" ]]; then
  print -u2 -- "STOP: PROJECT_ROOT must name an existing directory"
  exit 64
fi

if [[ -v DENSE_DIR ]]; then
  DENSE_DIR="${DENSE_DIR:a}"
else
  DENSE_DIR="$PROJECT_ROOT_CANONICAL/var/corpus/best-available/dense"
fi
if [[ "$DENSE_DIR" != "$PROJECT_ROOT_CANONICAL"/* ]]; then
  print -u2 -- "STOP: dense artifact directory must stay inside PROJECT_ROOT"
  exit 64
fi

relative_dense="${DENSE_DIR#$PROJECT_ROOT_CANONICAL/}"
current_destination="$PROJECT_ROOT_CANONICAL"
for component in ${(s:/:)relative_dense}; do
  current_destination="$current_destination/$component"
  if [[ -L "$current_destination" ]]; then
    print -u2 -- "STOP: symbolic-link dense-directory component rejected: $current_destination"
    exit 64
  fi
done

HEADROOM_BYTES=$((2 * 1024 * 1024 * 1024))
mkdir -p "$DENSE_DIR"

if [[ "${DENSE_DIR:A}" != "$PROJECT_ROOT_CANONICAL"/* ]]; then
  print -u2 -- "STOP: canonical dense artifact directory escaped PROJECT_ROOT"
  exit 64
fi

reject_symlink_components() {
  local relative_path=$1
  local current="$DENSE_DIR"
  local component
  for component in ${(s:/:)relative_path}; do
    current="$current/$component"
    if [[ -L "$current" ]]; then
      print -u2 -- "STOP: symbolic-link destination component rejected: $relative_path"
      return 1
    fi
  done
}

incoming_tmp="$DENSE_DIR/.incoming-manifest.json.tmp"
incoming="$DENSE_DIR/.incoming-manifest.json"
plan="$DENSE_DIR/.transfer-files.jsonl"
shard_tmp="$DENSE_DIR/.incoming-shard.json.tmp"
trap 'rm -f "$incoming_tmp" "$incoming" "$plan" "$shard_tmp"' EXIT

for internal_path in \
  ".incoming-manifest.json.tmp" \
  ".incoming-manifest.json" \
  ".transfer-files.jsonl" \
  ".incoming-shard.json.tmp" \
  "manifest.json"; do
  reject_symlink_components "$internal_path"
done

ssh -p "$SSH_PORT" -i "$SSH_KEY" -- "$REMOTE" \
  "cat -- '$REMOTE_DENSE/manifest.json'" > "$incoming_tmp"

jq -e -s '
  def sha256:
    type == "string" and length == 64 and test("^[0-9a-f]+$");
  def shard_path:
    type == "string" and length == 11 and startswith("shard-") and
    (.[6:] | test("^[0-9]{5}$"));
  if length != 1 then
    error("expected exactly one dense manifest JSON object")
  else
    .[0]
  end |
  select(
  type == "object" and
  .schema_version == "sharded-dense-index-manifest-v2" and
  .chunk_offsets_path == "chunks.offsets.u64" and
  .candidate_template_path == "candidate-template.faiss" and
  .recall_report_path == "recall-report.json" and
  (.chunk_offsets_size_bytes | type == "number" and . > 0 and floor == .) and
  (.candidate_template_size_bytes | type == "number" and . > 0 and floor == .) and
  (.recall_report_size_bytes | type == "number" and . > 0 and floor == .) and
  (.total_artifact_bytes | type == "number" and . > 0 and floor == .) and
  (.chunk_offsets_sha256 | sha256) and
  (.candidate_template_sha256 | sha256) and
  (.recall_report_sha256 | sha256) and
  (.recall_queries_sha256 | sha256) and
  (.shards | type == "array" and length > 0) and
  (([.shards[].path] | unique | length) == (.shards | length)) and
  all(.shards[];
    (.path | shard_path) and
    (.embeddings_size_bytes | type == "number" and . > 0 and floor == .) and
    (.candidate_index_size_bytes | type == "number" and . > 0 and floor == .) and
    (.embeddings_sha256 | sha256) and
    (.candidate_index_sha256 | sha256)
  ) and
  (
    .chunk_offsets_size_bytes +
    .candidate_template_size_bytes +
    .recall_report_size_bytes +
    ([.shards[] | .embeddings_size_bytes + .candidate_index_size_bytes] | add)
  ) == .total_artifact_bytes
  )
' "$incoming_tmp" > "$incoming"

recall_queries_size=$(
  ssh -p "$SSH_PORT" -i "$SSH_KEY" -- "$REMOTE" \
    "wc -c < '$REMOTE_DENSE/recall-queries.txt'"
)
recall_queries_size="${recall_queries_size//[[:space:]]/}"
if [[ "$recall_queries_size" != <1-> ]]; then
  print -u2 -- "STOP: remote recall query size is invalid"
  exit 65
fi

jq -c '
  [
    {
      path: .chunk_offsets_path,
      size: .chunk_offsets_size_bytes,
      sha256: .chunk_offsets_sha256
    },
    {
      path: .candidate_template_path,
      size: .candidate_template_size_bytes,
      sha256: .candidate_template_sha256
    },
    {
      path: .recall_report_path,
      size: .recall_report_size_bytes,
      sha256: .recall_report_sha256
    }
  ] + [
    .shards[] |
    {
      path: (.path + "/embeddings.f16"),
      size: .embeddings_size_bytes,
      sha256: .embeddings_sha256
    },
    {
      path: (.path + "/candidate.faiss"),
      size: .candidate_index_size_bytes,
      sha256: .candidate_index_sha256
    }
  ] | .[]
' "$incoming" > "$plan"
jq -cn \
  --argjson size "$recall_queries_size" \
  --arg sha256 "$(jq -r '.recall_queries_sha256' "$incoming")" \
  '{path:"recall-queries.txt", size:$size, sha256:$sha256}' >> "$plan"

file_is_verified() {
  local target=$1
  local expected_size=$2
  local expected_sha256=$3
  [[ -f "$target" ]] || return 1
  [[ "$(stat -f '%z' "$target")" == "$expected_size" ]] || return 1
  printf '%s  %s\n' "$expected_sha256" "$target" | shasum -a 256 -c - >/dev/null 2>&1
}

remaining_bytes=0
while IFS= read -r record; do
  relative_path=$(printf '%s' "$record" | jq -er '.path')
  expected_size=$(printf '%s' "$record" | jq -er '.size')
  expected_sha256=$(printf '%s' "$record" | jq -er '.sha256')
  reject_symlink_components "$relative_path"
  target="$DENSE_DIR/$relative_path"
  if file_is_verified "$target" "$expected_size" "$expected_sha256"; then
    continue
  fi
  current_size=0
  if [[ -f "$target" ]]; then
    current_size=$(stat -f '%z' "$target")
  fi
  if (( current_size < expected_size )); then
    remaining_bytes=$((remaining_bytes + expected_size - current_size))
  fi
done < "$plan"

required_bytes=$((remaining_bytes + HEADROOM_BYTES))
available_bytes=$(df -Pk "$DENSE_DIR" | awk 'NR==2 {printf "%.0f\n", $4 * 1024}')
if [[ "$available_bytes" != <1-> || "$available_bytes" -lt "$required_bytes" ]]; then
  print -u2 -- "STOP: need $required_bytes free bytes for remaining files and headroom; only $available_bytes are available"
  exit 66
fi

transfer_file() {
  local relative_path=$1
  local expected_size=$2
  local expected_sha256=$3
  local target="$DENSE_DIR/$relative_path"
  reject_symlink_components "$relative_path"
  mkdir -p "${target:h}"
  reject_symlink_components "$relative_path"
  if file_is_verified "$target" "$expected_size" "$expected_sha256"; then
    return 0
  fi
  rsync -a --progress --partial --inplace --checksum \
    -e "ssh -p $SSH_PORT -i $SSH_KEY" \
    "$REMOTE:$REMOTE_DENSE/$relative_path" "$target"
  if [[ "$(stat -f '%z' "$target")" != "$expected_size" ]]; then
    print -u2 -- "STOP: transferred size mismatch for $relative_path"
    return 1
  fi
  printf '%s  %s\n' "$expected_sha256" "$target" | shasum -a 256 -c -
}

while IFS= read -r record; do
  transfer_file \
    "$(printf '%s' "$record" | jq -er '.path')" \
    "$(printf '%s' "$record" | jq -er '.size')" \
    "$(printf '%s' "$record" | jq -er '.sha256')"
done < "$plan"

jq -c '.shards[]' "$incoming" | while IFS= read -r shard; do
  shard_path=$(printf '%s' "$shard" | jq -er '.path')
  reject_symlink_components "$shard_path/manifest.json"
  mkdir -p "$DENSE_DIR/$shard_path"
  reject_symlink_components "$shard_path/manifest.json"
  printf '%s' "$shard" | jq -S . > "$shard_tmp"
  mv "$shard_tmp" "$DENSE_DIR/$shard_path/manifest.json"
done

mv "$incoming" "$DENSE_DIR/manifest.json"
print -- "Compressed dense transfer verified and installed."
