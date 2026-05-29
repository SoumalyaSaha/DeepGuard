#!/bin/bash
# ── Local dev launcher (no Docker) ─────────────────────────────────────────────
# Starts gateway + all model services in background processes.
# Logs go to logs/<service>.log
# Run:  chmod +x start.sh && ./start.sh
# Stop: ./stop.sh

set -e

mkdir -p logs

echo "Starting deepfake detection services..."

# Gateway
uvicorn gateway.main:app --host 0.0.0.0 --port 8000 --reload \
    > logs/gateway.log 2>&1 &
echo "  ✓ Gateway        :8000  (pid $!)"

# NPR image model
cd models/npr && uvicorn main:app --host 0.0.0.0 --port 5001 \
    > ../../logs/npr.log 2>&1 &
echo "  ✓ NPR            :5001  (pid $!)"
cd ../..

# UniversalFakeDetect image model
cd models/ufd && uvicorn main:app --host 0.0.0.0 --port 5004 \
    > ../../logs/ufd.log 2>&1 &
echo "  ✓ UFD            :5004  (pid $!)"
cd ../..

# RawNet2 audio model
cd models/rawnet && uvicorn main:app --host 0.0.0.0 --port 5002 \
    > ../../logs/rawnet.log 2>&1 &
echo "  ✓ RawNet2        :5002  (pid $!)"
cd ../..

# CrossEfficientViT video model
cd models/crossvit && uvicorn main:app --host 0.0.0.0 --port 7001 \
    > ../../logs/crossvit.log 2>&1 &
echo "  ✓ CrossViT       :7001  (pid $!)"
cd ../..

echo ""
echo "All services running. Gateway at http://localhost:8000"
echo "API docs at http://localhost:8000/docs"
echo ""
echo "To stop all: ./stop.sh"

# Save PIDs
jobs -p > logs/pids.txt
