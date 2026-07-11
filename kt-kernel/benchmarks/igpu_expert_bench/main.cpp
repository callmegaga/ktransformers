// igpu_expert_bench — Phase 1 microbenchmark (decode + prefill + power mode)
//
// Q: on THIS box (Arrow Lake-S, DDR5 shared with iGPU), when MoE experts
// overflow to the "CPU bucket", how does the Intel iGPU (SYCL, reads int4)
// compare to the CPU (AVX-VNNI, reads int8) for the GPTQ-Int4 expert GEMM?
//
// Modes:
//   ./bench [NE] [iters] [M] [mode]
//     NE    experts per token stack (default 320 = top-8*40 layers)
//     iters timing repeats (default 6)
//     M     tokens: 1 = decode (bandwidth-bound), >1 = prefill (compute-bound)
//     mode  all (default) | cpu | gpu   (cpu/gpu = sustained ~12s load for turbostat)
//
// Simplification: gate/up/down measured as independent grouped int4xint8
// GEMMs (no SwiGLU chain / act requant). Faithful to per-token weight traffic
// (decode) and MAC volume (prefill); keeps CPU/iGPU correctness compare exact.

#include <sycl/sycl.hpp>

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace sycl;

static constexpr int H = 2048;  // hidden_size
static constexpr int I = 512;   // moe_intermediate_size
static constexpr int GS = 128;  // group_size
static constexpr int MT = 8;    // prefill m-tile (accumulators per work-item)

static inline double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static inline int hsum256(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v), hi = _mm256_extracti128_si256(v, 1);
  __m128i s = _mm_add_epi32(lo, hi);
  s = _mm_hadd_epi32(s, s);
  s = _mm_hadd_epi32(s, s);
  return _mm_cvtsi128_si32(s);
}

struct Batch {
  int njobs, N, K, M, numg;
  uint32_t* w4 = nullptr;      // [njobs][K/8][N] packed int4 (iGPU)
  int8_t* w8 = nullptr;        // [njobs][N][K]   int8       (CPU)
  float* wscale = nullptr;     // [njobs][numg][N]
  int16_t* wsums = nullptr;    // [njobs][numg][N]
  int8_t* aq = nullptr;        // [njobs][M][K]
  float* ascale = nullptr;     // [njobs][M][numg]
  float* out_cpu = nullptr;    // [njobs][M][N]
  float* out_gpu = nullptr;    // [njobs][M][N]
  size_t bytes8() const { return (size_t)njobs * N * K; }
  size_t bytes4() const { return (size_t)njobs * (K / 8) * N * 4; }
};

static Batch make_batch(queue& q, int njobs, int N, int K, int M) {
  Batch b;
  b.njobs = njobs; b.N = N; b.K = K; b.M = M; b.numg = K / GS;
  b.w4 = malloc_shared<uint32_t>((size_t)njobs * (K / 8) * N, q);
  b.w8 = malloc_shared<int8_t>((size_t)njobs * N * K, q);
  b.wscale = malloc_shared<float>((size_t)njobs * b.numg * N, q);
  b.wsums = malloc_shared<int16_t>((size_t)njobs * b.numg * N, q);
  b.aq = malloc_shared<int8_t>((size_t)njobs * M * K, q);
  b.ascale = malloc_shared<float>((size_t)njobs * M * b.numg, q);
  b.out_cpu = malloc_shared<float>((size_t)njobs * M * N, q);
  b.out_gpu = malloc_shared<float>((size_t)njobs * M * N, q);
  std::memset(b.w4, 0, (size_t)njobs * (K / 8) * N * sizeof(uint32_t));

  const int numg = b.numg;
#pragma omp parallel for schedule(dynamic, 1)
  for (int j = 0; j < njobs; ++j) {
    std::mt19937 rng(1234u + (unsigned)j);
    std::uniform_int_distribution<int> nib(0, 15), act(-127, 127);
    std::uniform_real_distribution<float> sc(0.004f, 0.02f);
    std::vector<long> gsum((size_t)numg * N, 0);
    for (int k = 0; k < K; ++k)
      for (int n = 0; n < N; ++n) {
        int nibble = nib(rng), val = nibble - 8;
        b.w8[(size_t)j * N * K + (size_t)n * K + k] = (int8_t)val;
        b.w4[(size_t)j * (K / 8) * N + (size_t)(k / 8) * N + n] |= ((uint32_t)nibble) << ((k % 8) * 4);
        gsum[(size_t)(k / GS) * N + n] += val;
      }
    for (int g = 0; g < numg; ++g)
      for (int n = 0; n < N; ++n) {
        b.wsums[(size_t)j * numg * N + (size_t)g * N + n] = (int16_t)gsum[(size_t)g * N + n];
        b.wscale[(size_t)j * numg * N + (size_t)g * N + n] = sc(rng);
      }
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) b.aq[((size_t)j * M + m) * K + k] = (int8_t)act(rng);
      for (int g = 0; g < numg; ++g) b.ascale[((size_t)j * M + m) * numg + g] = sc(rng);
    }
  }
  return b;
}

static void free_batch(queue& q, Batch& b) {
  for (void* p : {(void*)b.w4, (void*)b.w8, (void*)b.wscale, (void*)b.wsums, (void*)b.aq, (void*)b.ascale,
                  (void*)b.out_cpu, (void*)b.out_gpu})
    if (p) free(p, q);
}

// ---- CPU AVX-VNNI (reads int8), handles M>=1 ----
static void cpu_gemm(const Batch& b, int j0, int j1) {
  const int N = b.N, K = b.K, numg = b.numg, M = b.M;
#pragma omp parallel for schedule(dynamic, 8)
  for (int jn = j0 * N; jn < j1 * N; ++jn) {
    int j = jn / N, n = jn % N;
    const int8_t* w = b.w8 + (size_t)j * N * K + (size_t)n * K;  // weight row (cached across m)
    const float* wsc = b.wscale + (size_t)j * numg * N;
    const int16_t* wsm = b.wsums + (size_t)j * numg * N;
    for (int m = 0; m < M; ++m) {
      const int8_t* a = b.aq + ((size_t)j * M + m) * K;
      const float* asc = b.ascale + ((size_t)j * M + m) * numg;
      float oacc = 0.f;
      for (int g = 0; g < numg; ++g) {
        __m256i acc = _mm256_setzero_si256();
        int kb = g * GS;
        for (int t = 0; t < GS; t += 32) {
          __m256i av = _mm256_loadu_si256((const __m256i*)(a + kb + t));
          av = _mm256_xor_si256(av, _mm256_set1_epi8((char)0x80));
          __m256i wv = _mm256_loadu_si256((const __m256i*)(w + kb + t));
          acc = _mm256_dpbusd_avx_epi32(acc, av, wv);
        }
        int dot = hsum256(acc) - 128 * (int)wsm[(size_t)g * N + n];
        oacc += (float)dot * asc[g] * wsc[(size_t)g * N + n];
      }
      b.out_cpu[((size_t)j * M + m) * N + n] = oacc;
    }
  }
}

// ---- iGPU SYCL decode (M==1): reads int4, load-once + unpack-8 ----
static event gpu_decode(queue& q, const Batch& b, int j0, int j1, const std::vector<event>& deps = {}) {
  const int N = b.N, K = b.K, numg = b.numg, Kp = b.K / 8;
  uint32_t* w4 = b.w4; int8_t* aq = b.aq; float* wsc = b.wscale; float* asc = b.ascale; float* out = b.out_gpu;
  const int nj = j1 - j0;
  return q.submit([&](handler& h) {
    h.depends_on(deps);
    h.parallel_for(range<2>(nj, N), [=](id<2> idx) {
      int j = j0 + (int)idx[0], n = (int)idx[1];
      const uint32_t* w = w4 + (size_t)j * Kp * N;
      const int8_t* a = aq + (size_t)j * K;
      const float* as = asc + (size_t)j * numg;
      const float* ws = wsc + (size_t)j * numg * N;
      float oacc = 0.f;
      const int ppg = GS / 8;
      for (int g = 0; g < numg; ++g) {
        int dot = 0, kp0 = g * ppg;
        for (int kp = kp0; kp < kp0 + ppg; ++kp) {
          uint32_t packed = w[(size_t)kp * N + n];
          const int8_t* a8 = a + (size_t)kp * 8;
#pragma unroll
          for (int bb = 0; bb < 8; ++bb) dot += (int)a8[bb] * ((int)((packed >> (bb * 4)) & 0xF) - 8);
        }
        oacc += (float)dot * as[g] * ws[(size_t)g * N + n];
      }
      out[(size_t)j * N + n] = oacc;
    });
  });
}

// ---- iGPU SYCL prefill (M>1): m-tiled, weight loaded once per pack ----
static event gpu_prefill(queue& q, const Batch& b) {
  const int N = b.N, K = b.K, numg = b.numg, Kp = b.K / 8, M = b.M;
  uint32_t* w4 = b.w4; int8_t* aq = b.aq; float* wsc = b.wscale; float* asc = b.ascale; float* out = b.out_gpu;
  int ntiles = (M + MT - 1) / MT;
  return q.parallel_for(range<3>(b.njobs, ntiles, N), [=](id<3> idx) {
    int j = (int)idx[0], m0 = (int)idx[1] * MT, n = (int)idx[2];
    const uint32_t* w = w4 + (size_t)j * Kp * N;
    const float* ws = wsc + (size_t)j * numg * N;  // per-job weight scales (was missing j offset)
    float oacc[MT];
#pragma unroll
    for (int mm = 0; mm < MT; ++mm) oacc[mm] = 0.f;
    const int ppg = GS / 8;
    for (int g = 0; g < numg; ++g) {
      int iacc[MT];
#pragma unroll
      for (int mm = 0; mm < MT; ++mm) iacc[mm] = 0;
      int kp0 = g * ppg;
      for (int kp = kp0; kp < kp0 + ppg; ++kp) {
        uint32_t packed = w[(size_t)kp * N + n];
        int wv[8];
#pragma unroll
        for (int bb = 0; bb < 8; ++bb) wv[bb] = (int)((packed >> (bb * 4)) & 0xF) - 8;
        for (int mm = 0; mm < MT; ++mm) {
          int m = m0 + mm;
          if (m >= M) break;
          const int8_t* a8 = aq + ((size_t)j * M + m) * K + (size_t)kp * 8;
          int s = 0;
#pragma unroll
          for (int bb = 0; bb < 8; ++bb) s += (int)a8[bb] * wv[bb];
          iacc[mm] += s;
        }
      }
      for (int mm = 0; mm < MT; ++mm) {
        int m = m0 + mm;
        if (m >= M) break;
        oacc[mm] += (float)iacc[mm] * asc[((size_t)j * M + m) * numg + g] * ws[(size_t)g * N + n];
      }
    }
    for (int mm = 0; mm < MT; ++mm) {
      int m = m0 + mm;
      if (m >= M) break;
      out[((size_t)j * M + m) * N + n] = oacc[mm];
    }
  });
}

template <class F>
static double best_ms(F&& f, int iters) {
  double best = 1e30;
  for (int i = 0; i < iters; ++i) {
    double t0 = now_ms();
    f();
    best = std::min(best, now_ms() - t0);
  }
  return best;
}

int main(int argc, char** argv) {
  int NE = argc > 1 ? std::atoi(argv[1]) : 320;
  int iters = argc > 2 ? std::atoi(argv[2]) : 6;
  int M = argc > 3 ? std::atoi(argv[3]) : 1;
  std::string mode = argc > 4 ? argv[4] : "all";

  queue q{gpu_selector_v};
  printf("Device : %s\n", q.get_device().get_info<info::device::name>().c_str());
  int nth = 0;
#pragma omp parallel
  { if (omp_get_thread_num() == 0) nth = omp_get_num_threads(); }
  printf("Experts=%d  M=%d (%s)  CPU threads=%d  mode=%s\n\n", NE, M, M == 1 ? "decode" : "prefill", nth,
         mode.c_str());

  Batch gate = make_batch(q, NE, I, H, M);
  Batch up = make_batch(q, NE, I, H, M);
  Batch down = make_batch(q, NE, H, I, M);
  Batch* B[3] = {&gate, &up, &down};

  const double GB = 1e9;
  double w8pt = gate.bytes8() + up.bytes8() + down.bytes8();
  double w4pt = gate.bytes4() + up.bytes4() + down.bytes4();
  double macs = (double)NE * 3.0 * (double)I * H * M;  // gate+up+down MACs
  double gflop = 2.0 * macs / 1e9;

  auto run_cpu = [&] { for (auto* b : B) cpu_gemm(*b, 0, b->njobs); };
  auto run_gpu = [&] {
    if (M == 1) for (auto* b : B) gpu_decode(q, *b, 0, b->njobs);
    else for (auto* b : B) gpu_prefill(q, *b);
    q.wait();
  };

  // ---- power mode: sustained ~12s load for external turbostat ----
  if (mode == "cpu" || mode == "gpu") {
    printf("[power] sustained %s load ~12s (wrap with: sudo turbostat --Summary "
           "--show PkgWatt,GFXWatt,GFXMHz,Busy%% -- <this cmd>)\n", mode.c_str());
    double t0 = now_ms(); long reps = 0;
    while (now_ms() - t0 < 12000.0) { if (mode == "cpu") run_cpu(); else run_gpu(); reps++; }
    double el = now_ms() - t0;
    printf("[power] %s reps=%ld  tok/s=%.1f  (tokens=%d/rep)\n", mode.c_str(), reps, reps * M * 1000.0 / el, M);
    for (auto* b : B) free_batch(q, *b);
    return 0;
  }

  // ---- Test0: raw DRAM read BW ----
  {
    size_t NB = (size_t)1 << 30;
    int8_t* buf = malloc_shared<int8_t>(NB, q);
    for (size_t i = 0; i < NB; i += 4096) buf[i] = 1;
    int64_t* acc = malloc_shared<int64_t>(1, q); acc[0] = 0;
    double gms = best_ms([&] {
      q.submit([&](handler& h) {
         h.parallel_for(range<1>(NB / 16), reduction(acc, (int64_t)0, std::plus<>()), [=](id<1> i, auto& sum) {
           const int8_t* p = buf + i * 16; int s = 0;
           for (int t = 0; t < 16; ++t) s += p[t];
           sum += s;
         });
       }).wait();
    }, iters);
    double cms = best_ms([&] {
      long s = 0;
#pragma omp parallel for reduction(+ : s) schedule(static)
      for (size_t i = 0; i < NB; ++i) s += buf[i];
      if (s == 0x7fffffff) printf("x");
    }, iters);
    printf("[Test0] raw read BW   iGPU %6.1f GB/s | CPU %6.1f GB/s\n\n",
           NB / GB / (gms / 1e3), NB / GB / (cms / 1e3));
    free(buf, q); free(acc, q);
  }

  // ---- correctness (down) ----
  {
    cpu_gemm(down, 0, down.njobs);
    if (M == 1) gpu_decode(q, down, 0, down.njobs).wait(); else gpu_prefill(q, down).wait();
    double maxrel = 0; long bad = 0, tot = (long)down.njobs * M * down.N, first = -1;
    for (long i = 0; i < tot; ++i) {
      double a = down.out_cpu[i], g = down.out_gpu[i];
      double rel = std::fabs(a - g) / (std::fabs(a) + 1e-3);
      maxrel = std::max(maxrel, rel); if (rel > 1e-3) { bad++; if (first < 0) first = i; }
    }
    printf("[Correctness] CPU vs iGPU (down): max_rel=%.2e mismatch=%ld/%ld\n", maxrel, bad, tot);
    if (first >= 0) {
      int N = down.N; long jj = first / ((long)M * N), rem = first % ((long)M * N), mm = rem / N, nn = rem % N;
      printf("    first mismatch i=%ld -> j=%ld m=%ld n=%ld : cpu=%.4f gpu=%.4f\n", first, jj, mm, nn,
             down.out_cpu[first], down.out_gpu[first]);
    }
    printf("\n");
  }

  // ---- Test1: throughput ----
  double cpu_ms = best_ms(run_cpu, iters);
  double gpu_ms = best_ms(run_gpu, iters);
  printf("[Test1] %s: all %d experts, M=%d\n", M == 1 ? "decode" : "prefill", NE, M);
  printf("  CPU  AVX-VNNI: %8.2f ms | %7.1f tok/s | %6.1f GB/s(int8) | %6.1f GFLOP/s\n",
         cpu_ms, M * 1000.0 / cpu_ms, w8pt / GB / (cpu_ms / 1e3), gflop / (cpu_ms / 1e3));
  printf("  iGPU SYCL   : %8.2f ms | %7.1f tok/s | %6.1f GB/s(int4) | %6.1f GFLOP/s\n",
         gpu_ms, M * 1000.0 / gpu_ms, w4pt / GB / (gpu_ms / 1e3), gflop / (gpu_ms / 1e3));

  if (M == 1) {  // co-issue only for decode
    double best = 1e30, bf = 0;
    for (double frac : {0.1, 0.2, 0.3, 0.4, 0.5}) {
      int sp = (int)(NE * frac);
      double t = best_ms([&] {
        std::vector<event> evs;
        for (auto* b : B) evs.push_back(gpu_decode(q, *b, sp, b->njobs));
        for (auto* b : B) cpu_gemm(*b, 0, sp);
        for (auto& e : evs) e.wait();
      }, iters);
      if (t < best) { best = t; bf = frac; }
    }
    printf("  co-issue    : %8.2f ms | %7.1f tok/s | (CPU frac=%.0f%%)\n", best, 1000.0 / best, bf * 100);
  }

  for (auto* b : B) free_batch(q, *b);
  return 0;
}
