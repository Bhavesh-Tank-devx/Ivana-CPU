#!/usr/bin/env bash
set -euo pipefail
cd /home/ec2-user/ivana

LOGFILE="batch_results/pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p batch_results

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"; }

log "=== Pipeline start (MiniCPM-V 4.6) ==="

# ── 1. Start server in background ──────────────────────────────────────────
log "Starting llama server (download + rebuild if needed)…"
python3.12 serve_llama.py >> "$LOGFILE" 2>&1 &
SERVER_PID=$!
log "Server PID: $SERVER_PID"

# ── 2. Wait for server ready (up to 20 min) ────────────────────────────────
log "Waiting for server at http://localhost:8080 …"
READY=0
for i in $(seq 1 120); do
  if curl -sf http://localhost:8080/v1/models > /dev/null 2>&1; then
    READY=1
    log "Server ready after $((i * 10))s"
    break
  fi
  sleep 10
done

if [ "$READY" -eq 0 ]; then
  log "ERROR: Server did not come up within 20 minutes"
  kill "$SERVER_PID" 2>/dev/null || true
  exit 1
fi

# ── 3. Run batch ────────────────────────────────────────────────────────────
log "Running batch_caption.py …"
python3.12 batch_caption.py 2>&1 | tee -a "$LOGFILE"

# ── 4. Generate HTML ────────────────────────────────────────────────────────
log "Generating HTML report …"
python3.12 generate_html_report.py 2>&1 | tee -a "$LOGFILE"

# ── 5. Shutdown server ──────────────────────────────────────────────────────
log "Shutting down server (PID $SERVER_PID)"
kill "$SERVER_PID" 2>/dev/null || true

log "=== Pipeline complete ==="
