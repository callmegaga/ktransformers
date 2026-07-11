// fp8_fused — can an optimized SYCL FP8 fused expert kernel match CPU AVX2 FP8 on Arrow Lake?
// iGPU: ONE fused kernel/layer (work-group per expert, LUT FP8 decode, act in SLM, atomic
// accumulate). CPU: AVX2 FP8 (256-entry LUT + i32gather + FMA), OpenMP over experts.
// Model: Qwen3.5-35B-A3B-FP8  H=2048 I=512  k=8 experts/layer  40 layers  block 128.

#include <sycl/sycl.hpp>
#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
using namespace sycl;

static constexpr int H = 2048, I = 512, BS = 128;

static float LUT[256];
static void init_lut() {
  for (int i = 0; i < 256; i++) {
    int s = (i >> 7) & 1, e = (i >> 3) & 0xF, m = i & 0x7;
    float v;
    if (e == 0 && m == 0) v = 0.f;
    else if (e == 0) v = std::ldexp((float)m / 8.f, -6);
    else if (e == 15 && m == 7) v = 0.f;
    else v = std::ldexp(1.f + (float)m / 8.f, e - 7);
    LUT[i] = s ? -v : v;
  }
}
static inline double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
static inline float bf16f(uint16_t b) { uint32_t x = (uint32_t)b << 16; float f; std::memcpy(&f, &x, 4); return f; }
static inline uint16_t fbf16(float f) { uint32_t x; std::memcpy(&x, &f, 4); uint32_t r = (x >> 16) & 1; x += 0x7fff + r; return (uint16_t)(x >> 16); }
static inline float silu(float g) { return g / (1.f + std::exp(-g)); }

// one FP8 block-scaled matrix for one expert-projection: W[N][K] fp8 + scale[N/BS][K/BS]
struct W { int N, K, nbk; uint8_t* q; float* s; };
static W mkW(queue& qu, int N, int K, std::mt19937& rng) {
  W w; w.N = N; w.K = K; w.nbk = (K + BS - 1) / BS;
  w.q = malloc_shared<uint8_t>((size_t)N * K, qu);
  w.s = malloc_shared<float>((size_t)((N + BS - 1) / BS) * w.nbk, qu);
  std::uniform_int_distribution<int> fd(0, 255); std::uniform_real_distribution<float> sc(0.004f, 0.02f);
  for (size_t i = 0; i < (size_t)N * K; i++) { int v = fd(rng); if (v == 0x7f || v == 0xff) v = 0x38; w.q[i] = (uint8_t)v; }
  for (size_t i = 0; i < (size_t)((N + BS - 1) / BS) * w.nbk; i++) w.s[i] = sc(rng);
  return w;
}

// ---- CPU AVX2 FP8 (LUT + gather + FMA) ----
static inline __m256 fp8x8(const uint8_t* p) {
  __m128i b = _mm_loadl_epi64((const __m128i*)p);
  return _mm256_i32gather_ps(LUT, _mm256_cvtepu8_epi32(b), 4);
}
static inline __m256 ldbf(const uint16_t* p) {
  __m128i b = _mm_loadu_si128((const __m128i*)p);
  return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b), 16));
}
static inline float hs(__m256 v) { __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1); lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo); return _mm_cvtss_f32(lo); }
static void gemv_cpu(const W& w, const uint16_t* a, float* out) {
  for (int n = 0; n < w.N; n++) {
    const uint8_t* br = w.q + (size_t)n * w.K; int nb = n / BS; float sum = 0;
    for (int kb = 0; kb < w.K; kb += BS) {
      float sc = w.s[(size_t)nb * w.nbk + kb / BS]; __m256 acc = _mm256_setzero_ps();
      int k = 0; for (; k + 8 <= BS && kb + k < w.K; k += 8) acc = _mm256_fmadd_ps(ldbf(a + kb + k), fp8x8(br + kb + k), acc);
      out[n] += hs(acc) * sc;  // note: out preset to 0 by caller
    }
  }
}

int main(int argc, char** argv) {
  int k = argc > 1 ? std::atoi(argv[1]) : 8;    // experts/layer on this device
  int iters = argc > 2 ? std::atoi(argv[2]) : 100;
  int LAYERS = 40;
  init_lut();
  queue qu{gpu_selector_v};
  printf("Device: %s\nFP8 fused: k=%d experts/layer, H=%d I=%d, x%d layers\n\n",
         qu.get_device().get_info<info::device::name>().c_str(), k, H, I, LAYERS);

  std::mt19937 rng(9);
  std::vector<W> gate(k), up(k), down(k);
  for (int e = 0; e < k; e++) { gate[e] = mkW(qu, I, H, rng); up[e] = mkW(qu, I, H, rng); down[e] = mkW(qu, H, I, rng); }
  uint16_t* x = malloc_shared<uint16_t>(H, qu);
  std::uniform_real_distribution<float> xd(-1.f, 1.f);
  for (int h = 0; h < H; h++) x[h] = fbf16(xd(rng));
  std::vector<float> rw(k); for (int e = 0; e < k; e++) rw[e] = 0.05f + 0.02f * e;

  // device handles
  uint8_t **gq = malloc_shared<uint8_t*>(k, qu), **uq = malloc_shared<uint8_t*>(k, qu), **dq = malloc_shared<uint8_t*>(k, qu);
  float **gs = malloc_shared<float*>(k, qu), **us = malloc_shared<float*>(k, qu), **ds = malloc_shared<float*>(k, qu), *rwv = malloc_shared<float>(k, qu);
  for (int e = 0; e < k; e++) { gq[e]=gate[e].q; uq[e]=up[e].q; dq[e]=down[e].q; gs[e]=gate[e].s; us[e]=up[e].s; ds[e]=down[e].s; rwv[e]=rw[e]; }
  float* lut = malloc_shared<float>(256, qu); std::memcpy(lut, LUT, sizeof(LUT));
  float* out_g = malloc_shared<float>(H, qu);
  uint16_t* xg = x;
  const int nbkH = H / BS, nbkI = I / BS;
  constexpr int WG = 256;

  auto fused = [&]() {
    qu.memset(out_g, 0, H * sizeof(float)).wait();
    qu.submit([&](handler& h) {
      local_accessor<float, 1> act(I, h);
      h.parallel_for(nd_range<1>((size_t)k * WG, WG), [=](nd_item<1> it) {
        int e = it.get_group(0), lid = it.get_local_id(0);
        const uint8_t* wg = gq[e]; const uint8_t* wu = uq[e]; const float* sg = gs[e]; const float* su = us[e];
        // phase1: gate/up (bf16 x FP8-LUT, block scale) -> SiLU -> act (SLM)
        for (int i = lid; i < I; i += WG) {
          int ib = i / BS; float g = 0, u = 0;
          for (int kb = 0; kb < H; kb += BS) {
            float sgv = sg[(size_t)ib * nbkH + kb / BS], suv = su[(size_t)ib * nbkH + kb / BS];
            float pg = 0, pu = 0;
            const uint8_t* rg = wg + (size_t)i * H + kb; const uint8_t* ru = wu + (size_t)i * H + kb;
            for (int t = 0; t < BS; t++) { float av = bf16f(xg[kb + t]); pg += av * lut[rg[t]]; pu += av * lut[ru[t]]; }
            g += pg * sgv; u += pu * suv;
          }
          act[i] = silu(g) * u;
        }
        it.barrier(access::fence_space::local_space);
        // phase2: down (act x FP8-LUT) -> atomic accumulate rw*down
        const uint8_t* wd = dq[e]; const float* sd = ds[e]; float rwa = rwv[e];
        for (int hh = lid; hh < H; hh += WG) {
          int hb = hh / BS; float acc = 0;
          for (int kb = 0; kb < I; kb += BS) {
            float sdv = sd[(size_t)hb * nbkI + kb / BS]; float p = 0;
            const uint8_t* rd = wd + (size_t)hh * I + kb;
            for (int t = 0; t < BS; t++) p += act[kb + t] * lut[rd[t]];
            acc += p * sdv;
          }
          atomic_ref<float, memory_order::relaxed, memory_scope::device, access::address_space::global_space> ar(out_g[hh]);
          ar.fetch_add(rwa * acc);
        }
      });
    }).wait();
  };

  // ---- CPU reference (AVX2 FP8, OMP over experts) ----
  std::vector<float> out_c(H, 0.f);
  auto cpu = [&]() {
    std::fill(out_c.begin(), out_c.end(), 0.f);
    std::vector<std::vector<float>> partial(omp_get_max_threads(), std::vector<float>(H, 0.f));
#pragma omp parallel for schedule(dynamic)
    for (int e = 0; e < k; e++) {
      std::vector<float> g(I, 0.f), u(I, 0.f), a(H > I ? H : I, 0.f); std::vector<uint16_t> abf(I);
      gemv_cpu(gate[e], xg, g.data()); gemv_cpu(up[e], xg, u.data());
      for (int i = 0; i < I; i++) abf[i] = fbf16(silu(g[i]) * u[i]);
      std::vector<float> d(H, 0.f); gemv_cpu(down[e], abf.data(), d.data());
      auto& pp = partial[omp_get_thread_num()];
      for (int hh = 0; hh < H; hh++) pp[hh] += rw[e] * d[hh];
    }
    for (auto& pp : partial) for (int hh = 0; hh < H; hh++) out_c[hh] += pp[hh];
  };

  fused(); cpu();
  double mr = 0; for (int hh = 0; hh < H; hh++) mr = std::max(mr, (double)std::fabs(out_c[hh] - out_g[hh]) / (std::fabs(out_c[hh]) + 1e-3));
  printf("[correctness] iGPU vs CPU-ref max_rel=%.2e\n", mr);

  double bg = 1e30; for (int i = 0; i < iters; i++) { double t = now_ms(); fused(); bg = std::min(bg, now_ms() - t); }
  double bc = 1e30; for (int i = 0; i < iters; i++) { double t = now_ms(); cpu(); bc = std::min(bc, now_ms() - t); }
  double wbytes = (double)k * (2.0 * I * H + (double)H * I);  // fp8 weight bytes/layer
  printf("[per-layer]  iGPU %.3f ms (%.1f GB/s) | CPU %.3f ms (%.1f GB/s)\n",
         bg, wbytes / 1e9 / (bg / 1e3), bc, wbytes / 1e9 / (bc / 1e3));
  printf("[per-token x%d layers]  iGPU %.1f ms = %.1f tok/s | CPU %.1f ms = %.1f tok/s\n",
         LAYERS, bg * LAYERS, 1000.0 / (bg * LAYERS), bc * LAYERS, 1000.0 / (bc * LAYERS));
  printf("\n(CPU here = a simple AVX2 FP8 ref, not the tuned kt-kernel; treat as ballpark.)\n");
  return 0;
}
