# iGPU expert microbenchmark — results (Phase 1: decode + prefill)

**Box**: Intel Core Ultra 7 265K (Arrow Lake-S, 20c, AVX2+AVX-VNNI, no AMX) · Intel Graphics Xe-LPG iGPU (no XMX, DP4A) · DDR5 dual-channel (shared CPU/iGPU) · RTX 4060 Ti 16GB dGPU.
**Toolchain**: oneAPI DPC++ (icpx) 2026.1.0 · Level-Zero GPU 12.70.4 / NEO 25.18.
**Workload**: NE=320 experts/token (top-8×40 layers), GPTQ-Int4 sym group_size=128, gate/up (N=512,K=2048)+down (N=2048,K=512). Independent grouped int4×int8 GEMMs (measures per-token weight traffic + MAC volume; CPU/iGPU correctness compare exact, all M: 0 mismatch). CPU = AVX-VNNI reading unpacked int8 (framework's actual GPTQ backend here). iGPU = SYCL reading packed int4, unpacked in-kernel.

## Results

| scenario | CPU (AVX-VNNI, int8) | iGPU (SYCL, int4) | winner |
|---|---|---|---|
| **decode M=1** (bandwidth-bound) | 53.1 tok/s · 53.4 GB/s · 107 GFLOP/s | **91.8 tok/s** · 46.2 GB/s · 185 GFLOP/s | **iGPU +73%** |
| **prefill M=16** (compute-bound) | **762 tok/s · 1534 GFLOP/s** | 273 tok/s · 549 GFLOP/s | CPU ~2.8× |
| **prefill M=64** (compute-bound) | **834 tok/s · 1679 GFLOP/s** | 278 tok/s · 560 GFLOP/s | CPU ~3× |

raw DRAM read BW: iGPU ~50 GB/s, CPU ~61 GB/s. iGPU decode kernel is bandwidth-bound (46/50 ≈ 92%). iGPU compute ceiling ~560 GFLOP/s (Xe-LPG, no XMX) — flat across M. CPU scales to ~1700 GFLOP/s with batch reuse.

## Conclusions
1. **Clean phase split**: decode (memory-bound) → **iGPU wins +73%** (reads int4 = half the bytes, bandwidth-efficient, frees the CPU). prefill (compute-bound) → **CPU wins ~3×** (20 AVX-VNNI cores ≫ tiny no-XMX iGPU).
2. **Maps onto the framework's existing prefill_op / generate_op split**: ideal config = `generate(decode) → iGPU SYCL`, `prefill → CPU (AVX-VNNI)` (or dGPU). No new scheduling concept needed.
3. Decode win is real *and* frees the CPU during token generation (for attention/gate/sampling/overlap with dGPU).
4. iGPU compute ceiling ~560 GFLOP/s confirms it is compute-bound in prefill on this Xe-LPG part; XMX (Panther Lake) would lift this ~10× and make the iGPU competitive for prefill too.

## Power / perf-watt (pending — needs root turbostat)
Sustained-load mode added (`./bench NE iters M {cpu|gpu}`). Measure with:
```
source /opt/intel/oneapi/setvars.sh
sudo turbostat --show PkgWatt,GFXWatt,GFXMHz,Busy% --interval 2 -- ./bench 320 1 1 gpu   # decode on iGPU
sudo turbostat --show PkgWatt,GFXWatt,GFXMHz,Busy% --interval 2 -- ./bench 320 1 1 cpu   # decode on CPU
```
tok/J = (bench's printed tok/s) / (turbostat steady-state PkgWatt). Expect the iGPU path to show much lower PkgWatt (CPU mostly idle) → better tok/J.

## Caveats
- Microbench: independent GEMVs, synthetic weights, no SwiGLU chain / act requant. Real end-to-end gain smaller (attention/gate/overheads, output copy to dGPU).
- Decode +73% compares iGPU-int4 vs the CPU AVX-VNNI int8 path (what runs today); a CPU int4-reading kernel would narrow it.
- Same SYCL code targets XMX/joint_matrix on Panther Lake (expected larger wins, incl. prefill).

## Phase 2 A1 — per-layer reality check (CRITICAL correction)
Built the full fused expert (`fused_expert.cpp`): dynamic top-k routing (skips dGPU experts), gate/up→SiLU→act-quant→down→router-weighted accumulate, single fused kernel (work-group per expert, act in SLM, atomic-accumulated [1,H] partial). Correctness vs CPU ref: max_rel ~1e-5, 0 mismatch. The user's single-fused-kernel design beat a flat multi-kernel variant (0.91 vs 1.43 ms).

**But at the REAL granularity the decode win evaporates.** The framework calls the expert op **per layer** (40×/token, k=8 experts each), not as one 320-expert batch. Overhead decomposition (E=256, k=8, 6 active):
- empty kernel 0.059 ms · memset 0.065 ms · fused compute-only (async) 0.406 ms · fused+memset (sync, real) **0.914 ms/call**
- per-call fixed dispatch/sync overhead ≈ **0.51 ms**; compute ≈ 0.41 ms (6 work-groups → badly under-occupied)
- ⇒ per-token (×40 layers): **~36 ms = 27 tok/s**, *slower* than CPU's 52. Even the unrealistic zero-sync floor is ~62 tok/s (only marginally > CPU).

Root cause: k=8 experts/layer is tiny work; **per-layer GPU dispatch+sync overhead dominates**. The earlier +73% was a batched-microbenchmark artifact that doesn't match the per-layer call pattern. Combined with prefill (CPU wins ~3× on this no-XMX iGPU), **the iGPU expert offload does NOT pay off on Arrow Lake at the real granularity.**

Possible (uncertain) rescues, all requiring the full integration to evaluate: overlap the iGPU kernel with dGPU layer work to hide dispatch (as the CPU path overlaps the CUDA stream); cut dispatch via in-order queue / no per-call memset / SYCL graphs / persistent kernel; fix occupancy by splitting experts across work-groups. Upside likely modest on Arrow Lake.

## Decision: DO NOT proceed to full Phase 2 integration on Arrow Lake (would not beat CPU).
The idea is validated as **not beneficial on this box** for solid measured reasons. Where value remains: (1) the finding itself (saved weeks of integration); (2) **Panther Lake / XMX + wider memory + no dGPU** — compute-bound prefill/batch would win there and dispatch amortizes over bigger work, so the SYCL backend is a forward-looking asset; (3) reusable SYCL kernels + benchmark harness for evaluating future hardware.

Repro: `./run.sh 320 6 1` (decode) · `./bench 320 4 64` (prefill). args: NE iters M [mode].
