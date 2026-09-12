#!/usr/bin/env bash
# Phase 1b on a CDNA droplet (root). Runs the whole validation chain, logs with ### step markers.
set -uo pipefail
cd "$(dirname "$0")/.."
export EXL3_CTR=docker EXL3_ROCM_ARCH=${EXL3_ROCM_ARCH:-gfx942}
log() { echo "### $(date +%H:%M:%S) $*"; }
log "STEP up (docker pull rocm/pytorch:latest + container start)"
docker pull -q docker.io/rocm/pytorch:latest 2>&1 | tail -1
./rocm/box.sh up 2>&1 || { log "FAILED up"; exit 1; }
log "STEP build"
./rocm/box.sh build 2>&1
log "STEP test (native wave64, gfx942)"
./rocm/box.sh test 2>&1
log "STEP segfault-at-exit recheck"
docker exec exl3rocm bash -lc 'cd /work && PYTHONPATH=/work/rocm python rocm/tests/p2b_min.py >/dev/null 2>&1; echo "p2b_min exit code: $?"'
dmesg 2>/dev/null | grep -iE "segfault|gpu fault|page fault" | tail -3
log "STEP plugin-tests"
./rocm/box.sh plugin-tests 2>&1
log "STEP wave64-flag sanity (no-op on CDNA, must still pass)"
EXL3_ROCM_EXTRA_FLAGS="-mwavefrontsize64" ./rocm/box.sh build 2>&1 | tail -3
./rocm/box.sh test 2>&1
log "DONE"
