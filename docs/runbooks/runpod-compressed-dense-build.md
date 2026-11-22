# RunPod compressed dense build runbook

This runbook builds the real `best_available_foiarchive_v1` compressed dense artifacts. It does not call OpenAI and it never uses an uploaded user PDF. The source is the public, pinned History Lab FOIArchive snapshot.

No step that creates storage, starts a Pod, or spends money may be performed until the user approves the exact benchmark cap in dollars. The benchmark approval does not authorize the full build. The full build needs a second approval after the benchmark report exists.

The RunPod labels below were checked against the official documentation on 2026-11-22: [first Pod](https://docs.runpod.io/get-started), [credentials](https://docs.runpod.io/get-started/credentials), [billing](https://docs.runpod.io/accounts-billing/billing), [network volumes](https://docs.runpod.io/storage/network-volumes), [Pod management](https://docs.runpod.io/pods/manage-pods), and the [RunPod CLI](https://docs.runpod.io/runpodctl/overview).

## Fixed safety and scientific rules

- Keep all OpenAI approval flags false. Do not put an OpenAI key on RunPod.
- Do not upload user PDFs, local `.env` files, browser data, or other secrets.
- Use one NVIDIA GPU with at least 48 GiB VRAM.
- Use a 150 GiB RunPod network volume mounted at `/workspace`.
- Run exactly the 100,000-row benchmark first.
- Record the live total hourly rate shown by RunPod before deployment.
- Stop the Pod immediately after the benchmark finishes.
- Keep RunPod auto-pay off, verify that no other billable RunPod resources are active, and treat the entire prepaid account balance as the provider-enforced financial ceiling. RunPod stops running Pods when that balance reaches zero.
- Schedule the same stop deadline from both the Mac and the Pod immediately after creation. A command timeout inside SSH is a third, shorter limit, not a substitute for either stop scheduler.
- Do not run the full build until the user approves its separately calculated maximum cost.
- Publication still requires all checksums, recall@300 of at least `0.98`, artifacts no larger than 28 GiB, and all 100 local warm searches no slower than 5.0 seconds.

## Phase 1: Prepare the account without deploying anything

### Create and secure the account

1. Open [console.runpod.io](https://console.runpod.io/).
2. Click **Sign Up** if no account exists; otherwise click **Log In**.
3. Complete the email-verification message RunPod sends.
4. Open the account/security settings and enable two-factor authentication.
5. Stop here if the account is not verified or two-factor authentication cannot be enabled.

### Configure billing only after a benchmark cap is ready

Adding credits is a financial action. Do not do it during local implementation.

1. In the RunPod left sidebar, click **Billing**.
2. Leave **Auto-pay** off.
3. Under **Notifications**, enable **Low balance alert** and enter a small threshold.
4. After the exact benchmark cap is approved, select only the approved credit amount. Use **Other** if the preset amounts are larger.
5. Add the payment method and complete that one credit purchase.

RunPod deducts compute and storage from prepaid credits. Storage continues to cost money while compute is stopped.

### Add an SSH key—the recommended access method

On the Mac, open Terminal and check for an existing public key:

```bash
ls -l ~/.ssh/id_ed25519.pub
```

If that file does not exist, create it:

```bash
ssh-keygen -t ed25519 -C "YOUR_EMAIL_ADDRESS"
```

Press Enter to accept `~/.ssh/id_ed25519`, then choose a passphrase and record it in the password manager. Never upload or paste the private file `~/.ssh/id_ed25519`.

Add only the public key to RunPod:

1. In the RunPod left sidebar, click **Credentials**.
2. Click the **SSH Public Keys** tab.
3. In Mac Terminal, run `cat ~/.ssh/id_ed25519.pub`.
4. Copy the one line beginning with `ssh-ed25519`.
5. Back in RunPod, click **Add SSH Key**.
6. Name it `redacted-dense-build-mac`.
7. Paste the public-key line and click **Save**.

### Create a restricted API key for the capped deployment

The benchmark Pod must be created through `runpodctl` so its ID can be captured immediately and two independent stop schedulers can be armed. The API key stays on the Mac and is never copied to the Pod or chat.

1. Open **Credentials**.
2. Select **API Keys**.
3. Click **Create API Key**.
4. Name it `redacted-dense-build-temporary`.
5. Select **Restricted**, not **All**.
6. Leave unrelated APIs at **None**. Enable only the minimum Pod and network-volume permissions needed by the approved automation. If the displayed permission controls cannot express that narrow scope, cancel and stop; do not deploy until a narrowly scoped key is available.
7. Click **Create**.
8. Click the new key once to copy it and save it in the password manager. Do not paste it into chat, Git, `.env`, or a Pod.
9. After the work, return to **Credentials** → **API Keys**, click the trash icon, and confirm **Revoke Key**.

On the Mac, install and configure the official CLI:

```bash
brew install runpod/runpodctl/runpodctl
runpodctl doctor
runpodctl version
runpodctl pod create --help
```

When `runpodctl doctor` asks for the API key, paste it directly into Mac Terminal. Do not put it in a shell-history command. Verify that `runpodctl doctor` reports a healthy API-key check.

Do not use `--stop-after` or `--terminate-after`. RunPod [removed both flags in `runpodctl` v2.12.0](https://github.com/runpod/runpodctl/releases/tag/v2.12.0) because the backend never enforced them; an old CLI accepting either flag does not make it a safety control. Upgrade before proceeding if `runpodctl pod create --help` still lists either flag. The current official Pod-management procedure schedules a stop with `sleep ...; runpodctl pod stop ... &`.

## Phase 2: Price the benchmark before creating billable resources

1. In the RunPod console, click **+ New** in the top-right corner and choose **Pod**. Do not deploy.
2. In **Workload**, choose an official RunPod PyTorch template with CUDA 12.8 and PyTorch 2.x. Enable **SSH terminal access**. Do not add secrets or environment variables.
3. In **Compute**, open **Available** and filter to NVIDIA GPUs with at least 48 GiB VRAM. Suitable classes include A40 48 GB, RTX A6000 48 GB, L40/L40S 48 GB, and A100 80 GB.
4. Select one currently available GPU. Do not use **Deploy when available**, because it could start spending later without a person present.
5. In **Storage**, preview a 150 GiB network volume in the same data center. The volume cannot be attached across data centers.
6. In the right-side **Summary**, record the exact displayed GPU rate, storage rate, total running rate, and stopped-storage rate.
7. Open **Pods**, **Serverless**, and **Storage** in separate tabs and verify there are no other active billable resources. Stop here if another resource must remain active because the prepaid balance would no longer be a benchmark-only ceiling.
8. Open **Billing**, verify **Auto-pay** is off, and record the current prepaid balance. RunPod requires at least one hour of credits to deploy a Pod. If the current balance is greater than the maximum amount the user is willing to risk, stop and do not deploy.
9. Close the deploy page without clicking **Deploy Pod**.

State the benchmark proposal in this exact form, filling in real numbers:

> One 100,000-row benchmark on [GPU] at $[running rate]/hour. Both the Mac and Pod will schedule a stop at [exact deadline], and the benchmark command will have a shorter [exact duration] timeout. With auto-pay off and no other active resources, the full prepaid balance of $[current balance] is the provider-enforced account-level maximum; the planned benchmark plus volume allowance is $[planned total].

Proceed only after the user replies with an explicit approval naming both the planned total and the full prepaid-balance maximum. Never describe a local or in-Pod timer as provider-enforced.

## Phase 3: Create the approved volume and Pod

These clicks spend money. They are future actions, not part of local implementation.

### Create the network volume

1. In the left sidebar, click **Storage**.
2. Click **New Network Volume**.
3. Select the same data center used for the priced GPU.
4. Enter the name `redacted-dense-build`.
5. Enter `150` GB.
6. Choose **Standard** unless the approved quote explicitly used High-Performance storage.
7. Recheck the approved cap, then click **Create Network Volume**.

### Deploy the Pod and arm both stop schedulers

Do not click **Deploy Pod** in the console. In Mac Terminal, use `runpodctl template search pytorch`, `runpodctl gpu list`, and `runpodctl network-volume list` to copy the exact already-priced template ID, GPU ID, and network-volume ID. Then set the approved values without including an API key:

```bash
RUNPOD_TEMPLATE_ID='EXACT_PRICED_TEMPLATE_ID'
RUNPOD_GPU_ID='EXACT_PRICED_GPU_ID'
RUNPOD_NETWORK_VOLUME_ID='EXACT_NETWORK_VOLUME_ID'
APPROVED_POD_DURATION='EXACT_APPROVED_DURATION'
APPROVED_POD_SECONDS='EXACT_APPROVED_INTEGER_SECONDS'
export RUNPOD_TEMPLATE_ID RUNPOD_GPU_ID RUNPOD_NETWORK_VOLUME_ID APPROVED_POD_DURATION APPROVED_POD_SECONDS
```

Re-run `runpodctl doctor`, confirm the approved prepaid-balance ceiling and auto-pay setting in the console, and create exactly one Pod without `--wait` so the Pod ID is available before SSH startup:

```bash
set -euo pipefail
POD_CREATED_AT_EPOCH="$(date +%s)"
POD_STOP_DEADLINE_EPOCH="$((POD_CREATED_AT_EPOCH + APPROVED_POD_SECONDS))"
POD_CREATE_JSON="$(runpodctl pod create \
  --template-id "$RUNPOD_TEMPLATE_ID" \
  --gpu-id "$RUNPOD_GPU_ID" \
  --gpu-count 1 \
  --name redacted-dense-benchmark \
  --network-volume-id "$RUNPOD_NETWORK_VOLUME_ID" \
  --container-disk-in-gb 30 \
  --ssh)"
RUNPOD_POD_ID="$(printf '%s' "$POD_CREATE_JSON" | jq -er '.id')"
export RUNPOD_POD_ID POD_STOP_DEADLINE_EPOCH

nohup sh -c '
  remaining=$(( $1 - $(date +%s) ))
  if [ "$remaining" -gt 0 ]; then sleep "$remaining"; fi
  runpodctl pod stop "$2"
' runpod-stop "$POD_STOP_DEADLINE_EPOCH" "$RUNPOD_POD_ID" \
  >"/tmp/redacted-dense-${RUNPOD_POD_ID}-stop.log" 2>&1 &
LOCAL_STOP_SCHEDULER_PID=$!
export LOCAL_STOP_SCHEDULER_PID

runpodctl pod get "$RUNPOD_POD_ID"
```

Confirm the background scheduler is alive with `kill -0 "$LOCAL_STOP_SCHEDULER_PID"`. If creation, JSON parsing, or the scheduler check reports any error, immediately run `runpodctl pod list --all`. A failed client command can leave a billable Pod behind. Stop any `redacted-dense-benchmark` Pod with `runpodctl pod stop POD_ID` before diagnosing. If the live rate is higher than the approved rate, stop it immediately and request a revised cap.

### Connect over SSH

1. Open **Pods** in the left sidebar.
2. Click the running Pod to expand its detail panel.
3. Click **Connect** and copy the displayed **SSH over exposed TCP** command.
4. Paste that command into Mac Terminal. If asked to trust a new host fingerprint, compare it with the RunPod panel before typing `yes`.
5. In the Pod terminal, verify `RUNPOD_POD_ID` equals the ID captured on the Mac and arm the independent in-Pod scheduler against the same absolute deadline:

```bash
EXPECTED_POD_ID='PASTE_THE_CAPTURED_POD_ID'
POD_STOP_DEADLINE_EPOCH='PASTE_THE_APPROVED_DEADLINE_EPOCH'
test "$RUNPOD_POD_ID" = "$EXPECTED_POD_ID"
remaining="$((POD_STOP_DEADLINE_EPOCH - $(date +%s)))"
test "$remaining" -gt 0
nohup sh -c '
  remaining=$(( $1 - $(date +%s) ))
  if [ "$remaining" -gt 0 ]; then sleep "$remaining"; fi
  runpodctl pod stop "$2"
' runpod-stop "$POD_STOP_DEADLINE_EPOCH" "$RUNPOD_POD_ID" \
  >/workspace/redacted-dense-stop.log 2>&1 &
kill -0 $!
```

Stop immediately if the Pod ID is absent/mismatched, the deadline has passed, or the scheduler does not stay alive. Do not start the benchmark with only one scheduler.

## Phase 4: Reproduce the public source and prepare the GPU environment

In the Pod SSH terminal, set up the pinned repository checkout. Replace `APPROVED_BUILD_COMMIT` with the exact pushed commit chosen before deployment.

```bash
mkdir -p /workspace/redacted-build/bin \
  /workspace/redacted-build/uv-python \
  /workspace/redacted-build/uv-cache
export PATH="/workspace/redacted-build/bin:$PATH"
export UV_PYTHON_INSTALL_DIR=/workspace/redacted-build/uv-python
export UV_CACHE_DIR=/workspace/redacted-build/uv-cache
curl -LsSf https://astral.sh/uv/0.12.21/install.sh \
  | env UV_UNMANAGED_INSTALL=/workspace/redacted-build/bin sh
uv python install 3.12

cd /workspace
git clone https://github.com/galiettemita/Redacted_Document_LLM.git
cd Redacted_Document_LLM
git checkout APPROVED_BUILD_COMMIT
cd backend
uv sync --python 3.12 --group dev
uv pip install --python .venv/bin/python --reinstall torch --index-url https://download.pytorch.org/whl/cu128
uv run --no-sync python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

The final command must print the selected NVIDIA GPU. Stop if it reports no CUDA device. The pinned `uv` binary, its managed Python, cache, repository, and virtual environment now live under `/workspace`; this is required because [RunPod clears container-disk data when a Pod is stopped](https://docs.runpod.io/pods/manage-pods#stop-a-pod).

Download the public pinned source directly to the network volume. Do not upload the Mac copy:

```bash
mkdir -p /workspace/redacted-build/source
uv run --no-sync hf download HistoryLab/foiarchive foiarchive-202409.json.gz \
  --repo-type dataset \
  --revision 411cf6f86b38342683e27148be167c6094d1a10c \
  --local-dir /workspace/redacted-build/source
```

Build the deterministic best-available chunks from the frozen audit artifacts:

```bash
mkdir -p /workspace/redacted-build/best-available
uv run --no-sync corpus build-best-available-chunks \
  --source /workspace/redacted-build/source/foiarchive-202409.json.gz \
  --snapshot-lock corpus/historylab-huggingface-2025-snapshot-lock.json \
  --catalog corpus/historylab-huggingface-2025-catalog.json \
  --audit corpus/historylab-huggingface-2025-audit.json \
  --limitations ../4-bulletpoint-limitations.md \
  --output /workspace/redacted-build/best-available/chunks.jsonl \
  --checkpoint /workspace/redacted-build/best-available/chunks-checkpoint.json \
  --manifest /workspace/redacted-build/best-available/corpus-manifest.json
```

## Phase 5: Run only the fixed benchmark, then stop

```bash
set -euo pipefail
APPROVED_BENCHMARK_COMMAND_SECONDS=EXACT_APPROVED_COMMAND_SECONDS
mkdir -p /workspace/redacted-build/benchmark
timeout --signal=TERM --kill-after=300s \
  "${APPROVED_BENCHMARK_COMMAND_SECONDS}s" \
  uv run --no-sync corpus benchmark-best-available-dense \
  --chunks /workspace/redacted-build/best-available/chunks.jsonl \
  --corpus-manifest /workspace/redacted-build/best-available/corpus-manifest.json \
  --output-directory /workspace/redacted-build/benchmark \
  | tee /workspace/redacted-build/dense-benchmark-report.json
```

`set -o pipefail` makes a benchmark or timeout failure remain a failure even if `tee` succeeds. The command timeout kills a hung benchmark. The Mac and in-Pod stop schedulers remain independent backstops if SSH disconnects or the benchmark shell fails, while the approved prepaid balance with auto-pay off is the financial ceiling.

On every exit—success, timeout, error, or interrupt—stop the Pod immediately from another Mac Terminal with `runpodctl pod stop POD_ID`, then verify `runpodctl pod get POD_ID` reports it stopped. The two scheduled stops are backstops, not the normal shutdown. Only after a successful exit:

1. Copy `dense-benchmark-report.json` to the Mac using the SSH host and port from **Connect**.
2. Confirm again that the Pod is stopped. Do not leave the GPU running while reviewing results.
3. Validate the JSON as `DenseBuildBenchmarkReport`.
4. Record the separate `setup_seconds`, `encoding_seconds`, `training_seconds`, `candidate_add_seconds`, and `certification_probe_seconds` measurements, plus `rows_per_second`, `peak_rss_bytes`, `peak_gpu_memory_bytes`, `projected_shard_count`, `projected_full_artifact_bytes`, and `projected_gpu_hours`. The projection scales row work by corpus rows, fixed candidate overhead by shard count, and certification by both rows and the frozen 100-query count.
5. Reject the plan if projected artifacts exceed 28 GiB.

Calculate the separate full-build proposal from the measured report and the live RunPod rate:

- expected compute = `projected_gpu_hours × live total running rate`;
- conservative compute maximum = `max(projected_gpu_hours × 1.5, projected_gpu_hours + 2) × live total running rate`;
- add the stated network-volume allowance through the transfer deadline;
- do not add an invented transfer fee—RunPod's billing documentation currently says data transfer has no fee.

Show both expected and maximum cost. Do not restart or redeploy a GPU until the user explicitly approves the full-build maximum in dollars.

## Phase 6: Full build after the second approval

Restart the Pod only if its GPU is available and its rate is within the approved maximum. Otherwise deploy a new Pod against the same network volume, re-price it, and obtain approval if the cap changes.

```bash
export PATH="/workspace/redacted-build/bin:$PATH"
export UV_PYTHON_INSTALL_DIR=/workspace/redacted-build/uv-python
export UV_CACHE_DIR=/workspace/redacted-build/uv-cache
test -x /workspace/redacted-build/bin/uv
cd /workspace/Redacted_Document_LLM/backend
uv run --no-sync python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
mkdir -p /workspace/redacted-build/best-available/dense
uv run --no-sync corpus build-best-available-dense \
  --chunks /workspace/redacted-build/best-available/chunks.jsonl \
  --corpus-manifest /workspace/redacted-build/best-available/corpus-manifest.json \
  --application-queries corpus/best_available_application_queries.txt \
  --certification-queries /workspace/redacted-build/best-available/dense/recall-queries.txt \
  --output-directory /workspace/redacted-build/best-available/dense \
  --output-manifest /workspace/redacted-build/best-available/dense/manifest.json
```

The build is resumable. If the SSH session disconnects, reconnect and run the same command. Never delete completed shard/checkpoint files to force progress. Report status at least once per hour and stop if the approved maximum could be exceeded.

The top-level manifest is published only after the checksum, storage, and recall gates pass. Require `recall_at_300 >= 0.98` and `total_artifact_bytes <= 30064771072` before transfer.

## Phase 7: Transfer, local latency certification, and verification

Keep the Pod running only for the transfer. From Mac Terminal, use the SSH host and port shown by RunPod. Replace the placeholders below, then run the checked-in native-zsh transfer script. It validates the remote manifest and every relative path, stops on the first SSH/rsync/checksum error, counts only bytes that remain to be downloaded plus 2 GiB headroom, resumes in place, derives each shard manifest from the verified top manifest, and installs `manifest.json` last:

```bash
PROJECT_ROOT=/absolute/path/to/Redacted_Document_LLM
REMOTE=root@RUNPOD_SSH_HOST
SSH_PORT=RUNPOD_SSH_PORT
SSH_KEY="$HOME/.ssh/id_ed25519"
REMOTE_DENSE=/workspace/redacted-build/best-available/dense
export PROJECT_ROOT REMOTE SSH_PORT SSH_KEY REMOTE_DENSE
cd "$PROJECT_ROOT"
./scripts/transfer-compressed-dense.zsh
```

On the Mac, certify all 100 warm searches inside the same worker image that serves predictions. The command has no option that can raise the frozen five-second limit. It mounts the artifact directory read-write only for the certification report; ordinary API and worker services continue mounting it read-only:

```bash
cd "$PROJECT_ROOT"
docker compose build worker
docker compose run --rm --no-deps \
  -e REDACTED_RUNTIME_ROLE=worker \
  -v "$PROJECT_ROOT/var/corpus:/artifacts/corpus:rw" \
  worker corpus certify-best-available-dense-latency \
  --chunks /artifacts/corpus/best-available/chunks.jsonl \
  --dense-directory /artifacts/corpus/best-available/dense \
  --dense-manifest /artifacts/corpus/best-available/dense/manifest.json \
  --certification-queries /artifacts/corpus/best-available/dense/recall-queries.txt \
  --output /artifacts/corpus/best-available/dense/latency-report.json

docker compose run --rm --no-deps \
  -e REDACTED_RUNTIME_ROLE=worker \
  -v "$PROJECT_ROOT/var/corpus:/artifacts/corpus:rw" \
  worker corpus verify-best-available-dense \
  --chunks /artifacts/corpus/best-available/chunks.jsonl \
  --corpus-manifest /artifacts/corpus/best-available/corpus-manifest.json \
  --dense-directory /artifacts/corpus/best-available/dense \
  --dense-manifest /artifacts/corpus/best-available/dense/manifest.json \
  --certification-queries /artifacts/corpus/best-available/dense/recall-queries.txt \
  --latency-report /artifacts/corpus/best-available/dense/latency-report.json
```

Verification must print `"verified": true`. It performs no model/provider call.

## Phase 8: Stop charges safely

After the Mac verification succeeds:

1. Open **Pods**, expand the Pod, click **Stop**, and confirm.
2. Decide whether the network volume must be retained. Keeping it incurs ongoing storage charges.
3. If recovery is no longer needed, expand the stopped Pod, click **Terminate** (trash icon), and confirm. The network volume remains separate.
4. Delete the network volume only after the user explicitly chooses deletion and the local artifacts have been verified again. Volume deletion is destructive and not implied by Pod termination.
5. If an API key was created, open **Credentials** → **API Keys**, click its trash icon, and confirm **Revoke Key**.

Do not enable any paid-model flag at this point. Artifact readiness and provider spending approval are separate gates.
