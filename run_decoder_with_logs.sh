#!/usr/bin/env bash
# Run decoder with full logging and Docker stats capture.
# Usage: ./scripts/run_decoder_with_logs.sh [optional: path to livejournal.pkl]
# Logs: run_logs/decoder_YYYYMMDD_HHMMSS.log, run_logs/docker_stats_YYYYMMDD_HHMMSS.log

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${REPO_ROOT}/run_logs"
mkdir -p "$LOG_DIR"
DECODER_LOG="${LOG_DIR}/decoder_${RUN_ID}.log"
STATS_LOG="${LOG_DIR}/docker_stats_${RUN_ID}.log"
SUMMARY_LOG="${LOG_DIR}/run_summary_${RUN_ID}.log"

PKL="${1:-${REPO_ROOT}/livejournal.pkl}"
if [[ ! -f "$PKL" ]]; then
  echo "Error: $PKL not found. Pass path to .pkl as first argument or put livejournal.pkl in repo root."
  exit 1
fi

echo "========================================" | tee "$SUMMARY_LOG"
echo "Decoder run: $RUN_ID" | tee -a "$SUMMARY_LOG"
echo "Logs: $DECODER_LOG | $STATS_LOG" | tee -a "$SUMMARY_LOG"
echo "Start: $(date -Iseconds)" | tee -a "$SUMMARY_LOG"
echo "========================================" | tee -a "$SUMMARY_LOG"

# Remove any existing container with same name
docker rm -f decoder-run 2>/dev/null || true

# Start container in background (no --rm so we can stats it; remove at end)
docker run --name decoder-run \
  -e OMP_NUM_THREADS=2 \
  -e MKL_NUM_THREADS=2 \
  -v "${PKL}:/app/$(basename "$PKL")" \
  -v "${REPO_ROOT}/results:/app/results" \
  -w /app \
  neural-miner \
  python -m subgraph_mining.decoder --dataset "$(basename "$PKL")" --n_trials 1000 --n_neighborhoods 10000 --min_neighborhood_size 3 --max_neighborhood_size 29 --out_path results/patterns_livejournal.p \
  > "$DECODER_LOG" 2>&1 &
DECODER_PID=$!

# Wait until container is running
sleep 3
until docker ps --format '{{.Names}}' 2>/dev/null | grep -q decoder-run; do
  sleep 1
done

# Sample docker stats every 5 seconds until container exits
echo "=== Docker stats (every 5s) for run $RUN_ID ===" > "$STATS_LOG"
while docker ps --format '{{.Names}}' 2>/dev/null | grep -q decoder-run; do
  echo "--- $(date -Iseconds) ---" >> "$STATS_LOG"
  docker stats decoder-run --no-stream >> "$STATS_LOG" 2>&1
  sleep 5
done

wait $DECODER_PID 2>/dev/null || true
docker rm -f decoder-run 2>/dev/null || true

echo "End: $(date -Iseconds)" | tee -a "$SUMMARY_LOG"
echo "Decoder log: $DECODER_LOG" | tee -a "$SUMMARY_LOG"
echo "Stats log:   $STATS_LOG" | tee -a "$SUMMARY_LOG"
echo "========================================" | tee -a "$SUMMARY_LOG"
echo "Done. Review $DECODER_LOG and $STATS_LOG"
