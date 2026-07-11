// fp8_compiler_test — does icpx miscompile the AVX2 FP8 gemm numeric path vs g++?
// Replicates kt-kernel avx2/fp8-moe.hpp gemm_fp8 (FP8 E4M3 LUT + i32gather + bf16->fp32
// + 128-block-scaled FMA reduction). Build the SAME source with g++, icpx (default
// fast-math), icpx -fp-model=precise; each prints a checksum of the gemm output.
// If icpx-default diverges from g++ and precise matches -> fast-math is the culprit.

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

static float lut[256];
static void init_lut() {
  for (int i = 0; i < 256; i++) {
    int s = (i >> 7) & 1, e = (i >> 3) & 0xF, m = i & 0x7;
    float v;
    if (e == 0 && m == 0) v = 0.f;
    else if (e == 0) v = std::ldexp((float)m / 8.f, -6);
    else if (e == 15 && m == 7) v = 0.f;
    else v = std::ldexp(1.f + (float)m / 8.f, e - 7);
    lut[i] = s ? -v : v;
  }
}
static inline __m256 fp8x8_to_fp32x8(const uint8_t* src) {
  __m128i bytes = _mm_loadl_epi64((const __m128i*)src);
  __m256i idx = _mm256_cvtepu8_epi32(bytes);
  return _mm256_i32gather_ps(lut, idx, 4);
}
static inline __m256 load_bf16_to_fp32(const uint16_t* p) {
  __m128i b16 = _mm_loadu_si128((const __m128i*)p);
  __m256i b32 = _mm256_cvtepu16_epi32(b16);
  return _mm256_castsi256_ps(_mm256_slli_epi32(b32, 16));
}
static inline float hsum(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}
static inline uint16_t fp32_to_bf16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  if ((x & 0x7fffffff) > 0x7f800000) return (uint16_t)((x >> 16) | 0x40);  // NaN
  uint32_t r = (x >> 16) & 1;
  x += 0x7fff + r;
  return (uint16_t)(x >> 16);
}
static inline float bf16_to_fp32(uint16_t b) {
  uint32_t x = (uint32_t)b << 16;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

// block-scaled fp8 gemm (mirrors avx2::gemm_fp8): C[m,n] = sum_k A[m,k]*dequant(B[n,k])*scale[n/BS,k/BS]
static void gemm_fp8(int m, int n, int k, const uint16_t* a, const uint8_t* b, const float* d,
                     float* c, int BS = 128) {
  int nbk = (k + BS - 1) / BS;
  for (int ni = 0; ni < n; ni++) {
    const uint8_t* brow = b + (size_t)ni * k;
    int nb = ni / BS;
    for (int mi = 0; mi < m; mi++) {
      const uint16_t* arow = a + (size_t)mi * k;
      float sum = 0.f;
      for (int kb = 0; kb < k; kb += BS) {
        int klen = (kb + BS <= k) ? BS : (k - kb);
        float scale = d[nb * nbk + kb / BS];
        __m256 acc = _mm256_setzero_ps();
        int ki = 0;
        for (; ki + 8 <= klen; ki += 8)
          acc = _mm256_fmadd_ps(load_bf16_to_fp32(arow + kb + ki), fp8x8_to_fp32x8(brow + kb + ki), acc);
        float bs = hsum(acc);
        for (; ki < klen; ki++) bs += bf16_to_fp32(arow[kb + ki]) * lut[brow[kb + ki]];
        sum += bs * scale;
      }
      c[(size_t)mi * n + ni] = sum;
    }
  }
}

int main() {
  init_lut();
  const int M = 4, N = 512, K = 512, BS = 128;
  std::mt19937 rng(123);
  std::uniform_int_distribution<int> fp8d(0, 255), bf(1, 60000);
  std::uniform_real_distribution<float> sc(0.004f, 0.02f);
  std::vector<uint16_t> a(M * K);
  std::vector<uint8_t> b(N * K);
  std::vector<float> d(((N + BS - 1) / BS) * ((K + BS - 1) / BS));
  std::vector<float> c(M * N);
  for (auto& x : a) x = (uint16_t)bf(rng);
  for (auto& x : b) { int v = fp8d(rng); if (v == 0x7f || v == 0xff) v = 0x38; x = (uint8_t)v; }  // avoid NaN
  for (auto& x : d) x = sc(rng);

  gemm_fp8(M, N, K, a.data(), b.data(), d.data(), c.data(), BS);

  // checksum + bf16 roundtrip of output (as the real kernel does via BufferC::to_mat)
  double sum = 0, amax = 0;
  int nan = 0, inf = 0;
  for (int i = 0; i < M * N; i++) {
    float v = bf16_to_fp32(fp32_to_bf16(c[i]));  // fp32->bf16->fp32 like the kernel output path
    if (std::isnan(v)) nan++;
    if (std::isinf(v)) inf++;
    sum += v;
    amax = std::max(amax, (double)std::fabs(v));
  }
  printf("gemm checksum=%.6f  amax=%.6f  nan=%d inf=%d  (M=%d N=%d K=%d)\n", sum, amax, nan, inf, M, N, K);
  return 0;
}
