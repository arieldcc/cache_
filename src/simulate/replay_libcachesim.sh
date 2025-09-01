#!/usr/bin/env bash
if [ -z "$1" ] || [ -z "$2" ]; then echo "Usage: $0 <libcachesim_binary> <trace_csv>"; exit 1; fi
BIN="$1"; TRACE="$2"; OUT="results/libcachesim_metrics.csv"; mkdir -p results
"$BIN" --trace "$TRACE" --policy belady --export "$OUT"
echo "libCacheSim metrics written to $OUT"
