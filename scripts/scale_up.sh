#!/usr/bin/env bash
# Scale-up run (NOTES.md, "Pre-registration: scale-up"). Stages are separate so a person or agent
# checks each result before paying for the next one.
#
#   scripts/scale_up.sh sft              # 900-step mixed-source SFT from the base model, auto-resumes
#   scripts/scale_up.sh rl <init-val>    # 10 x 10 RL steps on the mix, starting from the SFT result
#   scripts/scale_up.sh final <sampler>  # the selected checkpoint on every frozen test split, once
#
# RANK=16 (default) fits Tinker's free 100 GB checkpoint storage next to the adapters already kept;
# RANK=32 needs paid storage. TINKER_API_KEY must be set in the environment.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.
RANK=${RANK:-16}
SFT=runs/sft_scale
RL=runs/rl_scale
PY=".venv/bin/python"
VAL="repair:data/val.jsonl,repair:data/val_trained.jsonl"
STEPS=900

last_ckpt() {  # prints "<step> <state> <sampler>" of the newest SFT checkpoint
  $PY - "$SFT/checkpoints.jsonl" <<'PYEOF'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1]) if l.strip()][-1]
print(r["step"], r["state_path"], r["sampler_path"])
PYEOF
}

case "${1:-}" in
sft)
  mkdir -p "$SFT"
  for attempt in 1 2 3 4 5 6; do
    if ls "$SFT"/val_step$(printf %05d $STEPS)*val_trained.jsonl >/dev/null 2>&1 && grep -q "\"step\": $STEPS, \"kind\": \"val\"" "$SFT/metrics.jsonl"; then
      echo "[scale_up] SFT complete"; exit 0
    fi
    RESUME=()
    if [ -s "$SFT/checkpoints.jsonl" ]; then
      read -r STEP STATE SAMPLER < <(last_ckpt)
      RESUME=(--load "$STATE" --load-sampler "$SAMPLER" --start-step "$STEP")
      echo "[scale_up] attempt $attempt: resuming from step $STEP"
    elif [ -e "$SFT/metrics.jsonl" ]; then
      mv "$SFT" "$SFT.failed$attempt.$(date +%s)"; mkdir -p "$SFT"   # died before the first checkpoint: start over, keep the log
      echo "[scale_up] attempt $attempt: no checkpoint yet, starting over"
    fi
    $PY -m weightgym.tinker_train sft --log-path "$SFT" --lr 3e-4 --decay 0.8 --steps $STEPS --batch-size 64 \
      --rank "$RANK" --mix repair,repair_trained --train-offset 300000 --trained-offset 0 \
      --eval-every 100 --save-every 100 --val-sets "$VAL" "${RESUME[@]}" >> "$SFT/stdout.log" 2>&1
    code=$?
    [ $code -eq 0 ] && { echo "[scale_up] SFT finished"; exit 0; }
    echo "[scale_up] SFT exited with $code (attempt $attempt); last lines:"; tail -3 "$SFT/stdout.log"
    sleep 60
  done
  echo "[scale_up] SFT gave up after 6 attempts"; exit 1
  ;;
rl)
  INIT_VAL=${2:?mean validation repair rate of the SFT result}
  read -r STEP STATE SAMPLER < <(last_ckpt)
  [ "$STEP" = "$STEPS" ] || { echo "newest SFT checkpoint is step $STEP, not $STEPS"; exit 1; }
  mkdir -p "$RL"
  $PY -m weightgym.rl_chunks --log-path "$RL" --load "$STATE" --init-sampler "$SAMPLER" --init-val "$INIT_VAL" \
    --lr 2e-5 --rank "$RANK" --chunks 10 --chunk-steps 10 --group-size 8 --groups-per-batch 64 \
    --train-offset 600000 --mix repair,repair_trained \
    --val-split data/val.jsonl,data/val_trained.jsonl --select mean_repair >> "$RL/driver.log" 2>&1
  ;;
final)
  ARM=${3:-scale_rl}
  $PY -m weightgym.final_eval --arm "$ARM=${2:?sampler path}" >> runs/final_scale.log 2>&1
  ;;
*)
  sed -n 2,11p "$0"; exit 2
  ;;
esac
