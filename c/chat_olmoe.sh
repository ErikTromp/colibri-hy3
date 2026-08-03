#!/bin/bash
# Launch OLMoE-1B-7B chat, memory-safe: hard RAM cap + swap disabled for this
# process (via a systemd-run scope), so a misbehaving run degrades to a clean
# OOM-kill of itself instead of taking the whole machine down.
#
# Usage:
#   SNAP_DIR=/path/to/olmoe_merged ./chat_olmoe.sh
# Override any default via env var, e.g.:  MAX_NEW=600 ./chat_olmoe.sh
set -e
cd "$(dirname "$0")"

if [ -z "$SNAP_DIR" ]; then
  echo "set SNAP_DIR=<converted OLMoE model directory> (see tools/convert_olmoe_merged.py)" >&2
  exit 1
fi

MEMORY_MAX="${MEMORY_MAX:-10G}"
MAX_NEW="${MAX_NEW:-300}"
PILOT="${PILOT:-0}"   # measured faster than PILOT=1 once CACHE=64 gives a ~94% hit rate --
                       # the prefetch thread has little disk-wait left to hide and just
                       # competes with compute for CPU on an 8-core CPU-only box
TEMP="${TEMP:-0.7}"
NUCLEUS="${NUCLEUS:-0.95}"
CTX="${CTX:-4096}"
# 64 = OLMoE's total experts/layer, so the cache holds ALL of them -- no eviction,
# no thrashing. This engine doesn't dedupe repeated expert picks within a single
# prefill batch, so a too-small cache causes constant reload storms; caching every
# expert sidesteps that entirely for a model this size (~6GB cache + ~1.8GB dense
# =~ 7.8GB peak, hence MEMORY_MAX=10G above for margin).
CACHE="${CACHE:-64}"

exec systemd-run --user --scope -p MemoryMax="$MEMORY_MAX" -p MemorySwapMax=0 --collect \
  -- bash -c "SNAP='$SNAP_DIR' CHAT=1 MAX_NEW=$MAX_NEW PILOT=$PILOT TEMP=$TEMP NUCLEUS=$NUCLEUS CTX=$CTX ./olmoe $CACHE 8"
