#!/usr/bin/env bash
set -Eeuo pipefail

# Usage: ./chain_dm.sh [K_chunks=6] [RUN_ID=auto]
# Queues K back-to-back 4h jobs that all resume the same W&B run.

K="${1:-6}"

# Generate a W&B id if not provided
RUN_ID="${2:-$(python - <<'PY'
try:
    import wandb
    print(wandb.util.generate_id())
except Exception:
    import uuid
    print(uuid.uuid4().hex)
PY
)}"

# Basic sanity
if ! [[ "$K" =~ ^[0-9]+$ ]] || (( K < 1 )); then
  echo "Error: K must be a positive integer (got '$K')." >&2
  exit 1
fi

SBATCH_FILE="./slurm_train_dm.sbatch"
if [[ ! -f "$SBATCH_FILE" ]]; then
  echo "Error: cannot find $SBATCH_FILE (run this from your repo root or fix the path)." >&2
  exit 1
fi

echo "[chain] Using WANDB_RUN_ID = $RUN_ID"
echo "[chain] Submitting $K chained jobs via $SBATCH_FILE"

# Submit the first job, capture JobID
jid=$(sbatch --parsable \
  --export=ALL,WANDB_RUN_ID="$RUN_ID",WANDB_ENTITY="${WANDB_ENTITY:-}",WANDB_PROJECT="${WANDB_PROJECT:-}" \
  "$SBATCH_FILE")
echo "[chain] submitted JID $jid"

# Submit the remaining jobs with dependency=afterok (change to afterany if desired)
for i in $(seq 2 "$K"); do
  jid=$(sbatch --parsable \
    --dependency=afterok:"$jid" \
    --export=ALL,WANDB_RUN_ID="$RUN_ID",WANDB_ENTITY="${WANDB_ENTITY:-}",WANDB_PROJECT="${WANDB_PROJECT:-}" \
    "$SBATCH_FILE")
  echo "[chain] submitted JID $jid"
done

echo "[chain] All $K jobs submitted and chained."
