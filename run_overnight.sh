#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CONFIGS="${CONFIGS:-C1 C2 C3 C4 C5}"
MAX_EXAMPLES="${MAX_EXAMPLES:-100}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
LOG_DIR="${LOG_DIR:-outputs/run_logs}"

mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN"
  echo "Set PYTHON_BIN, or run this from a synced project environment."
  exit 1
fi

echo "Starting overnight run at $(date)"
echo "Configs: $CONFIGS"
echo "Max evaluation examples per config: $MAX_EXAMPLES"
echo "Skip existing checkpoints: $SKIP_EXISTING"
echo "Logs: $LOG_DIR"

for CONFIG in $CONFIGS; do
  CONFIG_LOWER="$(echo "$CONFIG" | tr '[:upper:]' '[:lower:]')"
  CHECKPOINT_PATH="checkpoints/$CONFIG_LOWER/model.pt"
  TRAIN_LOG="$LOG_DIR/${CONFIG_LOWER}_train.log"
  EVAL_LOG="$LOG_DIR/${CONFIG_LOWER}_eval.log"

  echo "============================================================"
  echo "[$(date)] Processing $CONFIG"

  if [[ "$SKIP_EXISTING" == "1" && -f "$CHECKPOINT_PATH" ]]; then
    echo "[$(date)] Skipping training for $CONFIG because $CHECKPOINT_PATH exists"
  else
    echo "[$(date)] Training $CONFIG"
    "$PYTHON_BIN" -m src.train --config "$CONFIG" 2>&1 | tee "$TRAIN_LOG"
  fi

  echo "[$(date)] Evaluating $CONFIG"
  "$PYTHON_BIN" -m src.evaluate --config "$CONFIG" --max-examples "$MAX_EXAMPLES" 2>&1 | tee "$EVAL_LOG"
done

echo "============================================================"
echo "[$(date)] Building final ablation summary from available checkpoints"
"$PYTHON_BIN" -m src.evaluate --config all --max-examples "$MAX_EXAMPLES" 2>&1 | tee "$LOG_DIR/ablation_eval.log"

echo "Finished overnight run at $(date)"
