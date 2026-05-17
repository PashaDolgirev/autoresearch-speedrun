#!/bin/bash
# Wrap one experiment with a wall-clock watchdog + retries.
#
# Why: on 8x A100 + torch 2.7.1 + NCCL 2.26.2, ~20% of runs hang silently
# (NCCL collective deadlock; one rank goes idle while others spin). A clean
# run takes ~95 sec, so we hard-kill at HARD_CAP_S (default 240, ~2.5x slack)
# and retry up to NUM_ATTEMPTS-1 times. P(all 3 attempts hang) ~= 0.8%, so
# genuine crashes after this wrapper are very rare.
#
# Usage:   SEED=<n> bash run_with_retry.sh > run.log 2>&1
# Env:     SEED, TIME_BUDGET_S (forwarded to run.sh)
#          HARD_CAP_S      (default 240) — per-attempt timeout in seconds
#          NUM_ATTEMPTS    (default 3)   — total attempts including the first
# Exit:    0       — clean run
#          137     — all attempts hit the watchdog (hang); treat as `crash`
#          other   — Python crash / OOM / other non-hang error (don't retry)

set -uo pipefail
SEED="${SEED:-42}"
TIME_BUDGET_S="${TIME_BUDGET_S:-90}"
HARD_CAP_S="${HARD_CAP_S:-240}"
NUM_ATTEMPTS="${NUM_ATTEMPTS:-3}"

for attempt in $(seq 1 "$NUM_ATTEMPTS"); do
    echo "[run_with_retry] attempt $attempt/$NUM_ATTEMPTS (SEED=$SEED, hard_cap=${HARD_CAP_S}s)"
    timeout --signal=KILL "$HARD_CAP_S" \
        env SEED="$SEED" TIME_BUDGET_S="$TIME_BUDGET_S" bash run.sh
    rc=$?
    case "$rc" in
        0)
            exit 0
            ;;
        137|124)
            # Watchdog fired — NCCL hang. Kill stragglers, brief pause, retry.
            echo "[run_with_retry] attempt $attempt hit the watchdog (rc=$rc); cleaning up and retrying"
            pkill -9 -f train.py 2>/dev/null || true
            sleep 3
            ;;
        *)
            # Genuine crash (OOM, NameError, etc.) — don't retry, surface it.
            echo "[run_with_retry] attempt $attempt failed with rc=$rc (not a hang); aborting"
            exit "$rc"
            ;;
    esac
done

echo "[run_with_retry] all $NUM_ATTEMPTS attempts hit the watchdog; marking as hang"
exit 137
