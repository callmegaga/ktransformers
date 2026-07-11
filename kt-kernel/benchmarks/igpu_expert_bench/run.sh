#!/usr/bin/env bash
# Build & run the iGPU expert microbenchmark.
# Requires: Intel oneAPI DPC++ (icpx) + Intel GPU compute runtime.
#   source /opt/intel/oneapi/setvars.sh   # done here if present
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

[ -f /opt/intel/oneapi/setvars.sh ] && source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true

icpx -fsycl -O3 -qopenmp -march=native \
     -fsycl-targets=spir64 \
     -o "$HERE/bench" "$HERE/main.cpp"

echo "built: $HERE/bench"
# args: NE (experts/token, default 320)  iters (default 8)
"$HERE/bench" "$@"
