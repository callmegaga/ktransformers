#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Measured Panther Lake / Qwen3.5-35B-A3B-GPTQ-Int4 profile, not a generic GPU preset.
set -euo pipefail
MODEL=${KT_MODEL_PATH:-/home/wy/models/Qwen3.5-35B-A3B-GPTQ-Int4}
PYTHON=${KT_PYTHON:-/home/wy/miniconda3/envs/kt-kernel/bin/python}
SETVARS=${ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}
# v6 is byte-identical to v5 on controlled iGPU inputs and faster end-to-end.
# Hybrid NVIDIA expert execution is not bitwise repeatable even with v5;
# see doc/zh/SYCL-IGPU-Validation_zh.md. This is not a full PPL/quality claim.
PREFILL_PROFILE=${KT_IGPU_PREFILL_PROFILE:-v6}
case "$PREFILL_PROFILE" in
  v5) PREFILL_LOAD=block; PREFILL_MERGE=scalar; MIN_REVISION=5 ;;
  v6) PREFILL_LOAD=gather; PREFILL_MERGE=vector; MIN_REVISION=6
      echo "v6 vectorized prefill enabled. Hybrid NVIDIA Marlin is not bitwise repeatable; see SYCL-IGPU-Validation_zh.md." >&2 ;;
  *) echo "KT_IGPU_PREFILL_PROFILE must be v5 or v6: $PREFILL_PROFILE" >&2; exit 2 ;;
esac
# CUDA dense paths change reduction order (not quantization). Keep them opt-in
# until broader whole-model quality validation; select BEFORE graph capture.
DENSE_PROFILE=${KT_IGPU_DENSE_PROFILE:-native}
case "$DENSE_PROFILE" in
  native|linear|head-row|head-pad4|both-row|both-pad4) ;;
  *) echo "KT_IGPU_DENSE_PROFILE must be native, linear, head-row, head-pad4, both-row or both-pad4" >&2; exit 2 ;;
esac
export SGLANG_KT_CUDA_DENSE="$DENSE_PROFILE"
[[ -d "$MODEL" ]] || { echo "Model directory not found: $MODEL" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "Python not found: $PYTHON" >&2; exit 1; }
[[ -r "$SETVARS" ]] || { echo "oneAPI environment not found: $SETVARS" >&2; exit 1; }
set +u
source "$SETVARS" --force >/dev/null
set -u
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export KT_SYCL_INT4_BACKEND=xmx KT_SYCL_XMX_MODE=int8 KT_SYCL_XMX_ROWS=8
export KT_SYCL_XMX_GATE_KERNEL=xmx KT_SYCL_XMX_DOWN_KERNEL=xmx
export KT_SYCL_XMX_DECODE_GATE_KERNEL=xmx-split KT_SYCL_XMX_DECODE_DOWN_KERNEL=xmx
export KT_SYCL_XMX_GATE_SPLITS=2 KT_SYCL_XMX_DOWN_SPLITS=1
export KT_SYCL_XMX_PREFILL_ROWS=8 KT_SYCL_XMX_PREFILL_DOWN_ROWS=16 KT_SYCL_XMX_PREFILL_N=32
export KT_SYCL_XMX_PREFILL_LOAD="$PREFILL_LOAD" KT_SYCL_XMX_PREFILL_MERGE="$PREFILL_MERGE"
export KT_SYCL_XMX_FUSED_EXPERT=1 KT_SYCL_XMX_DECODE_MERGE=separate
export KT_SYCL_XMX_FUSED_QUANT=prefill KT_SYCL_XMX_NUMA_DISPATCH=1
export KT_SYCL_XMX_GRAPH=0 KT_SYCL_XMX_HOST_QUANT=0
export KT_SYCL_XMX_TRACE=0 KT_SYCL_XMX_TRACE_EVERY=0 KT_SYCL_PROFILE=0
export SGLANG_DISABLE_CUDNN_CHECK=1 SGLANG_MAMBA_CONV_DTYPE=float16
# Fail explicitly if an old/uncompiled extension is installed. C++ validates
# fixed dimensions, group size and device aspects when model weights load.
"$PYTHON" -c "from kt_kernel import kt_kernel_ext as e; assert hasattr(e.moe, 'SYCLXMXGPTQInt4_MOE') and getattr(e.moe.SYCLXMXGPTQInt4_MOE, 'implementation_revision', lambda: 0)() >= ${MIN_REVISION}, 'Installed extension is too old; rebuild kt-kernel with CPUINFER_USE_SYCL=1'"
if [[ "$DENSE_PROFILE" != native ]]; then
  echo "Opt-in CUDA dense profile: $DENSE_PROFILE; FP16 weights retained, reduction order may differ. See SYCL-IGPU-CUDA-Dense_zh.md." >&2
  "$PYTHON" -c "from sglang.srt.layers.kt_cuda_dense import get_mode; assert get_mode() == '${DENSE_PROFILE}', 'Install the matching third_party/sglang Python sources before selecting CUDA dense optimization'"
fi
exec "$PYTHON" -m sglang.launch_server \
  --model "$MODEL" --kt-weight-path "$MODEL" --kt-method SYCL_GPTQ_INT4 \
  --host "${KT_HOST:-127.0.0.1}" --port "${KT_PORT:-30000}" \
  --kt-cpuinfer 10 --kt-threadpool-count 1 --kt-num-gpu-experts 8 \
  --tensor-parallel-size 1 --dtype float16 --attention-backend triton \
  --chunked-prefill-size 512 --max-prefill-tokens 4096 \
  --max-running-requests 1 --max-total-tokens 8192 --context-length 8192 \
  --mem-fraction-static 0.9 --disable-radix-cache --tool-call-parser qwen3_coder \
  --disable-shared-experts-fusion --watchdog-timeout 1200 --trust-remote-code "$@"
