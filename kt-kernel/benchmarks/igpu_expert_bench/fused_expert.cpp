// fused_expert — Phase 2 step A1: full fused MoE expert (decode) on iGPU vs CPU ref.
//
// Replicates kt-kernel's AVX2_MOE_BASE::forward_decode semantics exactly:
//   pick k active experts (skip masked = "on dGPU"), per expert:
//     gate = Wgate @ xq ; up = Wup @ xq ; a = silu(gate)*up ;
//     down = Wdown @ quant(a) ; out += rweight * down
//   with GPTQ-Int4 sym group weights + on-the-fly symmetric int8 activation quant.
// Both CPU reference and SYCL do identical math -> must match to fp rounding.
//
// This validates the algorithmic core of the SYCL_GPTQ_INT4 MoE_Interface backend
// before wiring it into the kt-kernel build/pybind (Phase 2 A2).

#include <sycl/sycl.hpp>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace sycl;
static constexpr int H = 2048, I = 512, GS = 128;

static inline double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
static inline float silu(float g) { return g / (1.0f + std::exp(-g)); }

// packed int4 weight for one matrix [K/8][N] uint32 + scales [K/GS][N] float
struct W {
  int N, K, numg;
  uint32_t* q;   // USM [K/8*N]
  float* s;      // USM [numg*N]
};
static W makeW(queue& q, int N, int K, std::mt19937& rng) {
  W w; w.N = N; w.K = K; w.numg = K / GS;
  w.q = malloc_shared<uint32_t>((size_t)(K / 8) * N, q);
  w.s = malloc_shared<float>((size_t)w.numg * N, q);
  std::memset(w.q, 0, (size_t)(K / 8) * N * sizeof(uint32_t));
  std::uniform_int_distribution<int> nib(0, 15);
  std::uniform_real_distribution<float> sc(0.004f, 0.02f);
  for (int k = 0; k < K; ++k)
    for (int n = 0; n < N; ++n)
      w.q[(size_t)(k / 8) * N + n] |= ((uint32_t)nib(rng)) << ((k % 8) * 4);
  for (int i = 0; i < w.numg * N; ++i) w.s[i] = sc(rng);
  return w;
}
static inline int nibble(const uint32_t* q, int N, int k, int n) {
  return (int)((q[(size_t)(k / 8) * N + n] >> ((k % 8) * 4)) & 0xF) - 8;
}

// symmetric per-group int8 quant of a float vector [len] -> qs int8, scale [len/GS]
static void quantize(const float* x, int len, int8_t* qs, float* scale) {
  for (int g = 0; g < len / GS; ++g) {
    float amax = 0;
    for (int t = 0; t < GS; ++t) amax = std::max(amax, std::fabs(x[g * GS + t]));
    float s = amax > 0 ? amax / 127.0f : 0.0f;
    scale[g] = s;
    float inv = s > 0 ? 1.0f / s : 0.0f;
    for (int t = 0; t < GS; ++t) {
      int q = (int)std::lrint(x[g * GS + t] * inv);
      qs[g * GS + t] = (int8_t)std::clamp(q, -127, 127);
    }
  }
}

// dequant GEMV: out[n] = sum_g ( sum_{k in g} aq[k]*W[k,n] ) * ascale[g]*wscale[g,n]
static void gemv_cpu(const W& w, const int8_t* aq, const float* ascale, float* out) {
  for (int n = 0; n < w.N; ++n) {
    float acc = 0;
    for (int g = 0; g < w.numg; ++g) {
      int dot = 0;
      for (int t = 0; t < GS; ++t) { int k = g * GS + t; dot += (int)aq[k] * nibble(w.q, w.N, k, n); }
      acc += (float)dot * ascale[g] * w.s[(size_t)g * w.N + n];
    }
    out[n] = acc;
  }
}

int main(int argc, char** argv) {
  int E = argc > 1 ? std::atoi(argv[1]) : 32;   // total experts
  int k = argc > 2 ? std::atoi(argv[2]) : 8;    // active per token
  int iters = argc > 3 ? std::atoi(argv[3]) : 200;
  queue q{gpu_selector_v};
  printf("Device: %s\nE=%d k=%d H=%d I=%d\n\n", q.get_device().get_info<info::device::name>().c_str(), E, k, H, I);

  std::mt19937 rng(7);
  std::vector<W> gate(E), up(E), down(E);
  for (int e = 0; e < E; ++e) { gate[e] = makeW(q, I, H, rng); up[e] = makeW(q, I, H, rng); down[e] = makeW(q, H, I, rng); }

  // token input (bf16-ish; use fp32 here, quantized identically both sides)
  float* x = malloc_shared<float>(H, q);
  std::uniform_real_distribution<float> xd(-1.f, 1.f);
  for (int h = 0; h < H; ++h) x[h] = xd(rng);

  // routing: k distinct experts, random weights, mark ~1/4 as "on dGPU" (skip)
  std::vector<int64_t> eid(k); std::vector<float> rw(k); std::vector<uint8_t> mask(E, 0);
  { std::vector<int> perm(E); for (int i = 0; i < E; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int j = 0; j < k; ++j) { eid[j] = perm[j]; rw[j] = 0.05f + 0.1f * j; }
    for (int e = 0; e < E; ++e) if ((e % 4) == 0) mask[e] = 1; }  // masked = on dGPU
  auto skip = [&](int64_t e) { return e < 0 || e >= E || mask[e]; };

  // shared activation quant of x (same for all experts)
  int8_t* xq = malloc_shared<int8_t>(H, q); float* xs = malloc_shared<float>(H / GS, q);
  quantize(x, H, xq, xs);

  // ---------- CPU reference ----------
  std::vector<float> out_ref(H, 0.f);
  {
    std::vector<float> gbuf(I), ubuf(I), abuf(I), dbuf(H); std::vector<int8_t> aq(I); std::vector<float> as(I / GS);
    for (int j = 0; j < k; ++j) {
      if (skip(eid[j])) continue; int e = eid[j];
      gemv_cpu(gate[e], xq, xs, gbuf.data());
      gemv_cpu(up[e], xq, xs, ubuf.data());
      for (int i = 0; i < I; ++i) abuf[i] = silu(gbuf[i]) * ubuf[i];
      quantize(abuf.data(), I, aq.data(), as.data());
      gemv_cpu(down[e], aq.data(), as.data(), dbuf.data());
      for (int h = 0; h < H; ++h) out_ref[h] += rw[j] * dbuf[h];
    }
  }

  // ---------- SYCL fused (decode) ----------
  // Pack active (non-skipped) experts contiguously.
  std::vector<int> act; for (int j = 0; j < k; ++j) if (!skip(eid[j])) act.push_back(j);
  int na = act.size();
  // USM device-side handles
  uint32_t **gq = malloc_shared<uint32_t*>(na, q), **uq = malloc_shared<uint32_t*>(na, q), **dq = malloc_shared<uint32_t*>(na, q);
  float **gs = malloc_shared<float*>(na, q), **us = malloc_shared<float*>(na, q), **ds = malloc_shared<float*>(na, q);
  float* rwv = malloc_shared<float>(na, q);
  for (int a = 0; a < na; ++a) { int e = eid[act[a]];
    gq[a] = gate[e].q; uq[a] = up[e].q; dq[a] = down[e].q; gs[a] = gate[e].s; us[a] = up[e].s; ds[a] = down[e].s; rwv[a] = rw[act[a]]; }
  float* out_gpu = malloc_shared<float>(H, q);
  int8_t* xq_d = xq; float* xs_d = xs;
  const int numgH = H / GS, numgI = I / GS;
  constexpr int WG = 256;  // threads per work-group (one work-group = one active expert)

  // Single fused kernel: work-group per active expert. gate/up -> SiLU -> quant(act in SLM)
  // -> down -> atomic-accumulate rw*down into out_gpu[hidden]. One launch, one [1,H] partial.
  auto fused = [&]() {
    q.memset(out_gpu, 0, H * sizeof(float)).wait();
    q.submit([&](handler& h) {
       local_accessor<float, 1> actf(I, h);        // silu(gate)*up
       local_accessor<int8_t, 1> actq(I, h);       // quantized activation
       local_accessor<float, 1> ascl(numgI, h);    // activation group scales
       h.parallel_for(nd_range<1>((size_t)na * WG, WG), [=](nd_item<1> it) {
         int a = it.get_group(0), lid = it.get_local_id(0);
         // phase 1: gate/up GEMV + SiLU*up -> actf (SLM)
         for (int i = lid; i < I; i += WG) {
           float gsum = 0, usum = 0;
           for (int g = 0; g < numgH; ++g) {
             int dg = 0, du = 0, kp0 = g * (GS / 8);
             for (int kp = kp0; kp < kp0 + GS / 8; ++kp) {
               uint32_t pg = gq[a][(size_t)kp * I + i], pu = uq[a][(size_t)kp * I + i];
               const int8_t* xb = xq_d + (size_t)kp * 8;
#pragma unroll
               for (int bb = 0; bb < 8; ++bb) {
                 int xv = xb[bb];
                 dg += xv * ((int)((pg >> (bb * 4)) & 0xF) - 8);
                 du += xv * ((int)((pu >> (bb * 4)) & 0xF) - 8);
               }
             }
             gsum += (float)dg * xs_d[g] * gs[a][(size_t)g * I + i];
             usum += (float)du * xs_d[g] * us[a][(size_t)g * I + i];
           }
           float gg = gsum / (1.0f + sycl::exp(-gsum));
           actf[i] = gg * usum;
         }
         it.barrier(access::fence_space::local_space);
         // phase 2: quantize activation per group (SLM)
         for (int g = lid; g < numgI; g += WG) {
           float amax = 0;
           for (int t = 0; t < GS; ++t) amax = sycl::fmax(amax, sycl::fabs(actf[g * GS + t]));
           float s = amax > 0 ? amax / 127.0f : 0.f;
           ascl[g] = s;
           float inv = s > 0 ? 1.0f / s : 0.f;
           for (int t = 0; t < GS; ++t)
             actq[g * GS + t] = (int8_t)sycl::clamp((int)sycl::rint(actf[g * GS + t] * inv), -127, 127);
         }
         it.barrier(access::fence_space::local_space);
         // phase 3: down GEMV + atomic accumulate rw*down into out
         float rwa = rwv[a];
         for (int hh = lid; hh < H; hh += WG) {
           float acc = 0;
           for (int g = 0; g < numgI; ++g) {
             int dot = 0, kp0 = g * (GS / 8);
             for (int kp = kp0; kp < kp0 + GS / 8; ++kp) {
               uint32_t pd = dq[a][(size_t)kp * H + hh];
               const int8_t* ab = &actq[kp * 8];
#pragma unroll
               for (int bb = 0; bb < 8; ++bb) dot += (int)ab[bb] * ((int)((pd >> (bb * 4)) & 0xF) - 8);
             }
             acc += (float)dot * ascl[g] * ds[a][(size_t)g * H + hh];
           }
           sycl::atomic_ref<float, memory_order::relaxed, memory_scope::device,
                            access::address_space::global_space>
               ar(out_gpu[hh]);
           ar.fetch_add(rwa * acc);
         }
       });
     }).wait();
  };
  fused();

  // ---------- alt design: flat (many work-items, higher occupancy) ----------
  // 3 kernels but tiny global act intermediate; range spreads work across all EUs.
  float* g_act = malloc_shared<float>((size_t)na * I, q);
  int8_t* g_aq = malloc_shared<int8_t>((size_t)na * I, q);
  float* g_as = malloc_shared<float>((size_t)na * numgI, q);
  float* out_flat = malloc_shared<float>(H, q);
  auto flat = [&]() {
    q.memset(out_flat, 0, H * sizeof(float)).wait();
    q.parallel_for(range<2>(na, I), [=](id<2> id) {  // gate/up/act
       int a = id[0], i = id[1];
       float gsum = 0, usum = 0;
       for (int g = 0; g < numgH; ++g) {
         int dg = 0, du = 0, kp0 = g * (GS / 8);
         for (int kp = kp0; kp < kp0 + GS / 8; ++kp) {
           uint32_t pg = gq[a][(size_t)kp * I + i], pu = uq[a][(size_t)kp * I + i];
           const int8_t* xb = xq_d + (size_t)kp * 8;
#pragma unroll
           for (int bb = 0; bb < 8; ++bb) { int xv = xb[bb];
             dg += xv * ((int)((pg >> (bb * 4)) & 0xF) - 8); du += xv * ((int)((pu >> (bb * 4)) & 0xF) - 8); }
         }
         gsum += (float)dg * xs_d[g] * gs[a][(size_t)g * I + i];
         usum += (float)du * xs_d[g] * us[a][(size_t)g * I + i];
       }
       float gg = gsum / (1.0f + sycl::exp(-gsum));
       g_act[(size_t)a * I + i] = gg * usum;
     }).wait();
    q.parallel_for(range<2>(na, numgI), [=](id<2> id) {  // quant
       int a = id[0], g = id[1]; float amax = 0;
       for (int t = 0; t < GS; ++t) amax = sycl::fmax(amax, sycl::fabs(g_act[(size_t)a * I + g * GS + t]));
       float s = amax > 0 ? amax / 127.0f : 0.f; g_as[(size_t)a * numgI + g] = s; float inv = s > 0 ? 1.0f / s : 0.f;
       for (int t = 0; t < GS; ++t)
         g_aq[(size_t)a * I + g * GS + t] = (int8_t)sycl::clamp((int)sycl::rint(g_act[(size_t)a * I + g * GS + t] * inv), -127, 127);
     }).wait();
    q.parallel_for(range<2>(na, H), [=](id<2> id) {  // down + atomic accumulate
       int a = id[0], hh = id[1]; float acc = 0;
       for (int g = 0; g < numgI; ++g) {
         int dot = 0, kp0 = g * (GS / 8);
         for (int kp = kp0; kp < kp0 + GS / 8; ++kp) {
           uint32_t pd = dq[a][(size_t)kp * H + hh]; const int8_t* ab = g_aq + (size_t)a * I + kp * 8;
#pragma unroll
           for (int bb = 0; bb < 8; ++bb) dot += (int)ab[bb] * ((int)((pd >> (bb * 4)) & 0xF) - 8);
         }
         acc += (float)dot * g_as[(size_t)a * numgI + g] * ds[a][(size_t)g * H + hh];
       }
       sycl::atomic_ref<float, memory_order::relaxed, memory_scope::device, access::address_space::global_space> ar(out_flat[hh]);
       ar.fetch_add(rwv[a] * acc);
     }).wait();
  };
  flat();
  { double mr = 0; for (int h = 0; h < H; ++h) mr = std::max(mr, (double)std::fabs(out_flat[h] - out_gpu[h]) / (std::fabs(out_gpu[h]) + 1e-3));
    printf("[flat vs fused] max_rel=%.2e (should match)\n", mr); }

  // ---------- correctness ----------
  double maxrel = 0; int bad = 0;
  for (int h = 0; h < H; ++h) { double r = std::fabs(out_ref[h] - out_gpu[h]) / (std::fabs(out_ref[h]) + 1e-3);
    maxrel = std::max(maxrel, r); if (r > 2e-3) bad++; }
  printf("[Correctness] fused expert CPU-ref vs iGPU: max_rel=%.2e mismatch=%d/%d  (active experts=%d/%d)\n",
         maxrel, bad, H, na, k);

  // ---------- latency ----------
  double best = 1e30; for (int it = 0; it < iters; ++it) { double t0 = now_ms(); fused(); best = std::min(best, now_ms() - t0); }
  double bestf = 1e30; for (int it = 0; it < iters; ++it) { double t0 = now_ms(); flat(); bestf = std::min(bestf, now_ms() - t0); }
  printf("[Latency] fused(WG/expert) %.3f ms/call (%.1f/s) | flat(multi-kernel) %.3f ms/call (%.1f/s)  [%d experts]\n",
         best, 1000.0 / best, bestf, 1000.0 / bestf, na);
  printf("[Per-token est] x40 layers: fused %.1f ms (%.1f tok/s) | flat %.1f ms (%.1f tok/s)\n",
         best * 40, 1000.0 / (best * 40), bestf * 40, 1000.0 / (bestf * 40));
  // ---------- overhead decomposition ----------
  {
    int R = 300;
    double t0 = now_ms(); for (int i = 0; i < R; ++i) q.single_task([=]() {}).wait(); double emptyms = (now_ms() - t0) / R;
    t0 = now_ms(); for (int i = 0; i < R; ++i) q.memset(out_gpu, 0, H * sizeof(float)).wait(); double memms = (now_ms() - t0) / R;
    // fused compute kernel only (no memset, no per-call wait): submit R, one wait
    q.memset(out_gpu, 0, H * sizeof(float)).wait();
    t0 = now_ms();
    for (int r = 0; r < R; ++r)
      q.submit([&](handler& hh) {
        local_accessor<float, 1> actf(I, hh); local_accessor<int8_t, 1> actq(I, hh); local_accessor<float, 1> ascl(numgI, hh);
        hh.parallel_for(nd_range<1>((size_t)na * WG, WG), [=](nd_item<1> it) {
          int a = it.get_group(0), lid = it.get_local_id(0);
          for (int i = lid; i < I; i += WG) {
            float gsum = 0, usum = 0;
            for (int g = 0; g < numgH; ++g) { int dg = 0, du = 0, kp0 = g * (GS / 8);
              for (int kp = kp0; kp < kp0 + GS / 8; ++kp) { uint32_t pg = gq[a][(size_t)kp * I + i], pu = uq[a][(size_t)kp * I + i];
                const int8_t* xb = xq_d + (size_t)kp * 8;
#pragma unroll
                for (int bb = 0; bb < 8; ++bb) { int xv = xb[bb]; dg += xv * ((int)((pg >> (bb*4)) & 0xF) - 8); du += xv * ((int)((pu >> (bb*4)) & 0xF) - 8); } }
              gsum += (float)dg * xs_d[g] * gs[a][(size_t)g*I+i]; usum += (float)du * xs_d[g] * us[a][(size_t)g*I+i]; }
            float gg = gsum / (1.0f + sycl::exp(-gsum)); actf[i] = gg * usum; }
          it.barrier(access::fence_space::local_space);
          for (int g = lid; g < numgI; g += WG) { float amax = 0;
            for (int t = 0; t < GS; ++t) amax = sycl::fmax(amax, sycl::fabs(actf[g*GS+t]));
            float s = amax > 0 ? amax/127.0f : 0.f; ascl[g] = s; float inv = s > 0 ? 1.0f/s : 0.f;
            for (int t = 0; t < GS; ++t) actq[g*GS+t] = (int8_t)sycl::clamp((int)sycl::rint(actf[g*GS+t]*inv), -127, 127); }
          it.barrier(access::fence_space::local_space);
          float rwa = rwv[a];
          for (int hh2 = lid; hh2 < H; hh2 += WG) { float acc = 0;
            for (int g = 0; g < numgI; ++g) { int dot = 0, kp0 = g * (GS / 8);
              for (int kp = kp0; kp < kp0 + GS / 8; ++kp) { uint32_t pd = dq[a][(size_t)kp*H+hh2]; const int8_t* ab = &actq[kp*8];
#pragma unroll
                for (int bb = 0; bb < 8; ++bb) dot += (int)ab[bb] * ((int)((pd >> (bb*4)) & 0xF) - 8); }
              acc += (float)dot * ascl[g] * ds[a][(size_t)g*H+hh2]; }
            sycl::atomic_ref<float, memory_order::relaxed, memory_scope::device, access::address_space::global_space> ar(out_gpu[hh2]);
            ar.fetch_add(rwa * acc); } });
      });
    q.wait();
    double koms = (now_ms() - t0) / R;
    printf("[Overhead] empty-kernel %.3f ms | memset %.3f ms | fused-kernel-only(async) %.3f ms | fused+memset(sync) %.3f ms\n",
           emptyms, memms, koms, best);
    printf("  => per-call fixed overhead ~%.3f ms; compute ~%.3f ms. 40 layers async-compute-only: %.1f ms (%.1f tok/s)\n",
           best - koms, koms, koms * 40, 1000.0 / (koms * 40));
  }
  printf("\nNote: real framework calls the expert op per layer (40x/token, k=8 each) -> launch/occupancy\n"
         "dominated. The 'async-compute-only' row shows the floor if per-call sync overhead were removed.\n");
  return 0;
}
