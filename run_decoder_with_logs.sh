#!/usr/bin/env bash
# Run decoder with GPU image and full logging (decoder, Docker stats, GPU stats).
# Usage:
#   ./scripts/run_decoder_with_logs.sh [path-to.pkl] [standard|streaming]
#   USE_GPU=1 ./scripts/run_decoder_with_logs.sh amazon0302.pkl
# Logs: run_logs/decoder_*.log, docker_stats_*.log, gpu_stats_*.log, run_summary_*.log

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${REPO_ROOT}/run_logs"
mkdir -p "$LOG_DIR"
DECODER_LOG="${LOG_DIR}/decoder_${RUN_ID}.log"
STATS_LOG="${LOG_DIR}/docker_stats_${RUN_ID}.log"
GPU_LOG="${LOG_DIR}/gpu_stats_${RUN_ID}.log"
SUMMARY_LOG="${LOG_DIR}/run_summary_${RUN_ID}.log"

# Use GPU image and --gpus all by default (set USE_GPU=0 for CPU-only image)
USE_GPU="${USE_GPU:-1}"
if [[ "$USE_GPU" == "1" ]]; then
  DOCKER_IMAGE="neural-miner-gpu"
  GPU_FLAG="--gpus all"
else
  DOCKER_IMAGE="neural-miner"
  GPU_FLAG=""
fi

PKL="${1:-${REPO_ROOT}/amazon0302.pkl}"
MODE="${2:-streaming}"
if [[ ! -f "$PKL" ]]; then
  echo "Error: $PKL not found. Pass path to .pkl as first argument."
  exit 1
fi
PKL_ABS="$(cd "$(dirname "$PKL")" && pwd)/$(basename "$PKL")"

DATASET_NAME="$(basename "$PKL")"
if [[ "$MODE" == "standard" || "$MODE" == "amazon" ]]; then
  SEARCH_PIPELINE="standard"
  OUT_PATH="results/patterns_amazon_standard.p"
else
  SEARCH_PIPELINE="streaming"
  OUT_PATH="results/patterns_livejournal.p"
fi

echo "========================================" | tee "$SUMMARY_LOG"
echo "Decoder run: $RUN_ID" | tee -a "$SUMMARY_LOG"
echo "Image: $DOCKER_IMAGE | Dataset: $DATASET_NAME | Pipeline: $SEARCH_PIPELINE" | tee -a "$SUMMARY_LOG"
echo "Logs: $DECODER_LOG | $STATS_LOG | $GPU_LOG" | tee -a "$SUMMARY_LOG"
echo "Stats: CPU% + MEM (Docker every 5s) | GPU memory + util (nvidia-smi every 5s)" | tee -a "$SUMMARY_LOG"
echo "Start: $(date -Iseconds)" | tee -a "$SUMMARY_LOG"
echo "========================================" | tee -a "$SUMMARY_LOG"

docker rm -f decoder-run 2>/dev/null || true

# Defaults: 100 trials, 1k neighborhoods (~5 min). Override: N_TRIALS=1000 N_NEIGH=10000 ./scripts/run_decoder_with_logs.sh ...
N_TRIALS="${N_TRIALS:-100}"
N_NEIGH="${N_NEIGH:-1000}"

docker run --name decoder-run $GPU_FLAG \
  -e OMP_NUM_THREADS=2 \
  -e MKL_NUM_THREADS=2 \
  -v "${PKL_ABS}:/app/$(basename "$PKL")" \
  -v "${REPO_ROOT}/results:/app/results" \
  -w /app \
  "$DOCKER_IMAGE" \
  python -m subgraph_mining.decoder \
    --dataset "$DATASET_NAME" \
    --search_pipeline "$SEARCH_PIPELINE" \
    --n_trials "$N_TRIALS" \
    --n_neighborhoods "$N_NEIGH" \
    --min_neighborhood_size 3 \
    --max_neighborhood_size 29 \
    --out_path "$OUT_PATH" \
  > "$DECODER_LOG" 2>&1 &
DECODER_PID=$!

sleep 3
for _ in 1 2 3 4 5 6 7 8 9 10; do
  docker ps --format '{{.Names}}' 2>/dev/null | grep -q decoder-run && break
  kill -0 "$DECODER_PID" 2>/dev/null || { wait $DECODER_PID 2>/dev/null; echo "Container failed to start. Check $DECODER_LOG"; exit 1; }
  sleep 1
done

echo "=== Docker stats (every 5s) for run $RUN_ID ===" > "$STATS_LOG"
echo "=== GPU stats (every 5s) for run $RUN_ID ===" > "$GPU_LOG"
while docker ps --format '{{.Names}}' 2>/dev/null | grep -q decoder-run; do
  echo "--- $(date -Iseconds) ---" >> "$STATS_LOG"
  docker stats decoder-run --no-stream >> "$STATS_LOG" 2>&1
  if command -v nvidia-smi &>/dev/null; then
    echo "--- $(date -Iseconds) ---" >> "$GPU_LOG"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu --format=csv >> "$GPU_LOG" 2>/dev/null || true
  fi
  sleep 5
done

wait $DECODER_PID 2>/dev/null || true
docker rm -f decoder-run 2>/dev/null || true

echo "End: $(date -Iseconds)" | tee -a "$SUMMARY_LOG"
echo "Decoder log: $DECODER_LOG" | tee -a "$SUMMARY_LOG"
echo "Stats log:   $STATS_LOG" | tee -a "$SUMMARY_LOG"
echo "GPU log:     $GPU_LOG" | tee -a "$SUMMARY_LOG"
echo "========================================" | tee -a "$SUMMARY_LOG"
echo "Done. Review $DECODER_LOG, $STATS_LOG, and $GPU_LOG"
