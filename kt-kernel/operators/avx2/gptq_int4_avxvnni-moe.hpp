/**
 * @Description  : AVX-VNNI-256 GPTQ-Int4 MoE operator (symmetric quantization)
 * @Author       : Codex
 * @Date         : 2026-04-05
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * This backend keeps the GPTQ weight layout unchanged:
 *   qweight [K/8, N] int32 + scales [K/group_size, N] fp32
 *
 * To use AVX-VNNI-256 effectively, activations are quantized on the fly to
 * group-wise int8. We then use dpbusd on biased unsigned activations and
 * signed int4 weights unpacked to int8, followed by compensation and rescale.
 **/
#ifndef CPUINFER_OPERATOR_AVX2_GPTQ_INT4_AVXVNNI_MOE_H
#define CPUINFER_OPERATOR_AVX2_GPTQ_INT4_AVXVNNI_MOE_H

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "avx2_bf16_utils.hpp"
#include "moe_base.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define KT_AVXVNNI256_TARGET __attribute__((target("avx2,avxvnni,fma")))
#else
#define KT_AVXVNNI256_TARGET
#endif

namespace avxvnni {

static constexpr int MAX_SUPPORTED_GROUP_SIZE = 256;

static inline int hsum_epi32_avx2(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i sum = _mm_add_epi32(lo, hi);
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return _mm_cvtsi128_si32(sum);
}

// Quantize one activation group to biased uint8 for dpbusd.
// Returns the activation scale. A return value of 0 means the group is all zero.
static inline float quantize_activation_group_u8(const ggml_bf16_t* src, int group_size, uint8_t* dst) {
  float absmax = 0.0f;

  for (int i = 0; i < group_size; ++i) {
    absmax = std::max(absmax, std::fabs(GGML_BF16_TO_FP32(src[i])));
  }

  if (absmax <= std::numeric_limits<float>::min()) {
    std::memset(dst, 0x80, (size_t)group_size);
    return 0.0f;
  }

  const float scale = absmax / 127.0f;
  const float inv_scale = 1.0f / scale;
  for (int i = 0; i < group_size; ++i) {
    int q = (int)std::lrint(GGML_BF16_TO_FP32(src[i]) * inv_scale);
    q = std::clamp(q, -127, 127);
    dst[i] = (uint8_t)(((uint8_t)(int8_t)q) ^ 0x80);
  }
  return scale;
}

struct GemmKernelAVXVNNI256GPTQInt4 {
  using dt = ggml_bf16_t;
  using output_t = float;
  static constexpr int M_STEP = 1;
  static constexpr int N_STEP = 8;
  static constexpr int K_STEP = 8;
  static constexpr int N_BLOCK = 64;
  static constexpr int K_BLOCK = 128;
  static constexpr double ELEMENT_SIZE = 0.5;

  static void config() {}

  static int recommended_nth(int n) { return std::max(1, n / N_BLOCK); }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) { return avx2::split_range(n, ith, nth); }

  struct BufferA {
    ggml_bf16_t* data = nullptr;
    size_t max_m = 0;
    size_t k = 0;

    BufferA() = default;
    BufferA(size_t m, size_t k_, void* ptr) : data((ggml_bf16_t*)ptr), max_m(m), k(k_) {}

    static size_t required_size(size_t m, size_t k) { return m * k * sizeof(ggml_bf16_t); }

    void set_data(void* ptr) { data = (ggml_bf16_t*)ptr; }

    void from_mat(int m, const ggml_bf16_t* src, int ith, int nth) {
      if (ith == 0 && nth == 1) {
        std::memcpy(data, src, (size_t)m * k * sizeof(ggml_bf16_t));
      } else {
        auto [m_start, m_end] = avx2::split_range(m, ith, nth);
        std::memcpy(data + m_start * k, src + m_start * k,
                    (size_t)(m_end - m_start) * k * sizeof(ggml_bf16_t));
      }
    }
  };

  struct BufferB {
    int8_t* qweight_s8 = nullptr;  // [N, K] unpacked signed int8 weights
    float* scales = nullptr;
    int16_t* weight_sums = nullptr;
    int n = 0;
    int k = 0;
    int group_size = 128;
    int num_groups = 0;
    int k_packed = 0;

    BufferB() = default;
    BufferB(size_t n_, size_t k_, int gs, void* ptr) : n((int)n_), k((int)k_), group_size(gs) {
      if (group_size <= 0 || (group_size % 32) != 0) {
        throw std::runtime_error("AVX-VNNI GPTQ INT4 requires group_size to be a positive multiple of 32");
      }
      if (group_size > MAX_SUPPORTED_GROUP_SIZE) {
        throw std::runtime_error("AVX-VNNI GPTQ INT4 requires group_size <= 256");
      }
      if ((k % 8) != 0 || (k % group_size) != 0) {
        throw std::runtime_error("AVX-VNNI GPTQ INT4 requires k to be divisible by both 8 and group_size");
      }
      k_packed = k / 8;
      num_groups = k / group_size;
      qweight_s8 = (int8_t*)ptr;
      scales = (float*)((uint8_t*)ptr + (size_t)k * n * sizeof(int8_t));
      weight_sums = (int16_t*)((uint8_t*)scales + (size_t)num_groups * n * sizeof(float));
    }

    static size_t required_size(size_t n, size_t k, int gs) {
      const size_t num_groups = k / gs;
      return k * n * sizeof(int8_t) + num_groups * n * sizeof(float) + num_groups * n * sizeof(int16_t);
    }

    void from_mat(const uint32_t* src_qweight, const float* src_scales, int ith, int nth) {
      auto [n_start, n_end] = avx2::split_range(n, ith, nth);
      const int n_len = n_end - n_start;
      for (int g = 0; g < num_groups; ++g) {
        std::memcpy(scales + g * n + n_start, src_scales + g * n + n_start, (size_t)n_len * sizeof(float));
      }

      const int group_packed = group_size / 8;
      for (int ni = n_start; ni < n_end; ++ni) {
        int8_t* dst_col = qweight_s8 + (size_t)ni * k;
        for (int g = 0; g < num_groups; ++g) {
          int sum = 0;
          const uint32_t* group_base = src_qweight + (size_t)g * group_packed * n + ni;
          for (int kr = 0; kr < group_packed; ++kr) {
            const uint32_t packed = group_base[kr * n];
            for (int nib = 0; nib < 8; ++nib) {
              const int8_t value = (int8_t)(((packed >> (nib * 4)) & 0xF) - 8);
              dst_col[g * group_size + kr * 8 + nib] = value;
              sum += value;
            }
          }
          weight_sums[g * n + ni] = (int16_t)sum;
        }
      }
    }
  };

  struct BufferC {
    float* data = nullptr;
    size_t max_m = 0;
    size_t n = 0;

    BufferC() = default;
    BufferC(size_t m, size_t n_, void* ptr) : data((float*)ptr), max_m(m), n(n_) {}

    static size_t required_size(size_t m, size_t n) { return m * n * sizeof(float); }

    void set_data(void* ptr) { data = (float*)ptr; }

    void to_mat(int m, ggml_bf16_t* dst, int ith, int nth) {
      auto [n_start, n_end] = avx2::split_range((int)n, ith, nth);
      for (int mi = 0; mi < m; ++mi) {
        float* src_row = data + mi * n;
        ggml_bf16_t* dst_row = dst + mi * n;
        int j = n_start;
        for (; j + 8 <= n_end; j += 8) {
          avx2::store_fp32_to_bf16(dst_row + j, _mm256_loadu_ps(src_row + j));
        }
        for (; j < n_end; ++j) {
          dst_row[j] = GGML_FP32_TO_BF16(src_row[j]);
        }
      }
    }
  };
};

KT_AVXVNNI256_TARGET
static inline void gemm_gptq_sym_int4_avxvnni256(int m, int n, int k, GemmKernelAVXVNNI256GPTQInt4::BufferA& a,
                                                 GemmKernelAVXVNNI256GPTQInt4::BufferB& b,
                                                 GemmKernelAVXVNNI256GPTQInt4::BufferC& c, int ith, int nth) {
  (void)k;
  auto [n_start, n_end] = avx2::split_range(n, ith, nth);
  const int group_size = b.group_size;
  const int num_groups = b.num_groups;

  alignas(32) std::array<uint8_t, MAX_SUPPORTED_GROUP_SIZE> a_u8{};

  for (int mi = 0; mi < m; ++mi) {
    const ggml_bf16_t* a_row = a.data + (size_t)mi * a.k;
    float* c_row = c.data + (size_t)mi * n;
    std::fill(c_row + n_start, c_row + n_end, 0.0f);

    for (int g = 0; g < num_groups; ++g) {
      const int k_base = g * group_size;
      const float a_scale = quantize_activation_group_u8(a_row + k_base, group_size, a_u8.data());
      if (a_scale == 0.0f) {
        continue;
      }

      for (int ni = n_start; ni < n_end; ++ni) {
        __m256i acc = _mm256_setzero_si256();
        const int8_t* w_col = b.qweight_s8 + (size_t)ni * b.k + k_base;
        for (int kk = 0; kk < group_size; kk += 32) {
          const __m256i a_vec = _mm256_load_si256((const __m256i*)(a_u8.data() + kk));
          const __m256i w_vec = _mm256_loadu_si256((const __m256i*)(w_col + kk));
          acc = _mm256_dpbusd_avx_epi32(acc, a_vec, w_vec);
        }

        const int dot = hsum_epi32_avx2(acc) - 128 * (int)b.weight_sums[g * n + ni];
        c_row[ni] += (float)dot * a_scale * b.scales[g * n + ni];
      }
    }
  }
}

static inline bool env_flag_enabled(const char* name, bool default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }

  const char c0 = value[0];
  const char c1 = value[1];
  if (c0 == '0' || c0 == 'f' || c0 == 'F' || c0 == 'n' || c0 == 'N' ||
      ((c0 == 'o' || c0 == 'O') && (c1 == 'f' || c1 == 'F'))) {
    return false;
  }
  if (c0 == '1' || c0 == 't' || c0 == 'T' || c0 == 'y' || c0 == 'Y' ||
      ((c0 == 'o' || c0 == 'O') && (c1 == 'n' || c1 == 'N'))) {
    return true;
  }
  return default_value;
}

static inline uint64_t env_u64(const char* name, uint64_t default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }

  char* end = nullptr;
  const uint64_t parsed = std::strtoull(value, &end, 10);
  if (end == value || parsed == 0) {
    return default_value;
  }
  return parsed;
}

static inline void quantize_activation_groups_u8(const ggml_bf16_t* src, int num_groups, int group_size,
                                                 uint8_t* dst, float* scales) {
  for (int g = 0; g < num_groups; ++g) {
    scales[g] = quantize_activation_group_u8(src + (size_t)g * group_size, group_size,
                                             dst + (size_t)g * group_size);
  }
}

static inline float silu_mul_scalar(float gate, float up) {
  return gate * (1.0f / (1.0f + std::exp(-gate))) * up;
}

template <typename BufferB>
KT_AVXVNNI256_TARGET
static inline float dot_prequantized_u8_s8_avxvnni256(const uint8_t* a_u8, const float* a_scales,
                                                      const BufferB& b, int ni) {
  const int group_size = b.group_size;
  float result = 0.0f;

  for (int g = 0; g < b.num_groups; ++g) {
    const float a_scale = a_scales[g];
    if (a_scale == 0.0f) {
      continue;
    }

    const int k_base = g * group_size;
    const int8_t* w_col = b.qweight_s8 + (size_t)ni * b.k + k_base;
    __m256i acc = _mm256_setzero_si256();
    for (int kk = 0; kk < group_size; kk += 32) {
      const __m256i a_vec = _mm256_loadu_si256((const __m256i*)(a_u8 + k_base + kk));
      const __m256i w_vec = _mm256_loadu_si256((const __m256i*)(w_col + kk));
      acc = _mm256_dpbusd_avx_epi32(acc, a_vec, w_vec);
    }

    const int dot = hsum_epi32_avx2(acc) - 128 * (int)b.weight_sums[g * b.n + ni];
    result += (float)dot * a_scale * b.scales[g * b.n + ni];
  }

  return result;
}

template <typename BufferB>
KT_AVXVNNI256_TARGET
static inline void fused_gate_up_activation_avxvnni256(int intermediate_size, const uint8_t* input_u8,
                                                       const float* input_scales,
                                                       const BufferB& gate_b,
                                                       const BufferB& up_b,
                                                       ggml_bf16_t* activation_bf16, int ith, int nth) {
  auto [n_start, n_end] = avx2::split_range(intermediate_size, ith, nth);
  alignas(16) ggml_bf16_t gate_tmp[8];
  alignas(16) ggml_bf16_t up_tmp[8];

  int ni = n_start;
  for (; ni + 8 <= n_end; ni += 8) {
    for (int lane = 0; lane < 8; ++lane) {
      const int col = ni + lane;
      gate_tmp[lane] = GGML_FP32_TO_BF16(dot_prequantized_u8_s8_avxvnni256(input_u8, input_scales, gate_b, col));
      up_tmp[lane] = GGML_FP32_TO_BF16(dot_prequantized_u8_s8_avxvnni256(input_u8, input_scales, up_b, col));
    }

    const __m256 gate_val = avx2::load_bf16_to_fp32(gate_tmp);
    const __m256 up_val = avx2::load_bf16_to_fp32(up_tmp);
    const __m256 result = avx2::act_fn(gate_val, up_val);
    avx2::store_fp32_to_bf16(activation_bf16 + ni, result);
  }

  for (; ni < n_end; ++ni) {
    const ggml_bf16_t gate = GGML_FP32_TO_BF16(dot_prequantized_u8_s8_avxvnni256(input_u8, input_scales, gate_b, ni));
    const ggml_bf16_t up = GGML_FP32_TO_BF16(dot_prequantized_u8_s8_avxvnni256(input_u8, input_scales, up_b, ni));
    activation_bf16[ni] = GGML_FP32_TO_BF16(silu_mul_scalar(GGML_BF16_TO_FP32(gate), GGML_BF16_TO_FP32(up)));
  }
}

template <typename BufferB>
KT_AVXVNNI256_TARGET
static inline void fused_down_avxvnni256(int hidden_size, const uint8_t* activation_u8, const float* activation_scales,
                                         const BufferB& down_b,
                                         ggml_bf16_t* down_bf16, int ith, int nth) {
  auto [n_start, n_end] = avx2::split_range(hidden_size, ith, nth);
  alignas(32) float down_tmp[8];

  int ni = n_start;
  for (; ni + 8 <= n_end; ni += 8) {
    for (int lane = 0; lane < 8; ++lane) {
      down_tmp[lane] = dot_prequantized_u8_s8_avxvnni256(activation_u8, activation_scales, down_b, ni + lane);
    }
    avx2::store_fp32_to_bf16(down_bf16 + ni, _mm256_load_ps(down_tmp));
  }

  for (; ni < n_end; ++ni) {
    down_bf16[ni] = GGML_FP32_TO_BF16(dot_prequantized_u8_s8_avxvnni256(activation_u8, activation_scales, down_b, ni));
  }
}

template <typename BufferB>
KT_AVXVNNI256_TARGET
static inline void fused_down_weighted_sum_avxvnni256(int hidden_size, const uint8_t* const* activation_u8,
                                                      const float* const* activation_scales,
                                                      const BufferB* const* down_bs, const float* weights,
                                                      int active_count, float* output, int ith, int nth) {
  auto [n_start, n_end] = avx2::split_range(hidden_size, ith, nth);
  alignas(16) ggml_bf16_t down_tmp[8];

  int ni = n_start;
  for (; ni + 8 <= n_end; ni += 8) {
    __m256 acc = _mm256_setzero_ps();
    for (int expert = 0; expert < active_count; ++expert) {
      for (int lane = 0; lane < 8; ++lane) {
        down_tmp[lane] = GGML_FP32_TO_BF16(dot_prequantized_u8_s8_avxvnni256(
            activation_u8[expert], activation_scales[expert], *down_bs[expert], ni + lane));
      }
      const __m256 down = avx2::load_bf16_to_fp32(down_tmp);
      const __m256 weight = _mm256_set1_ps(weights[expert]);
      acc = _mm256_fmadd_ps(down, weight, acc);
    }
    _mm256_storeu_ps(output + ni, acc);
  }

  for (; ni < n_end; ++ni) {
    float acc = 0.0f;
    for (int expert = 0; expert < active_count; ++expert) {
      const ggml_bf16_t down = GGML_FP32_TO_BF16(dot_prequantized_u8_s8_avxvnni256(
          activation_u8[expert], activation_scales[expert], *down_bs[expert], ni));
      acc += GGML_BF16_TO_FP32(down) * weights[expert];
    }
    output[ni] = acc;
  }
}

}  // namespace avxvnni

template <class T = avxvnni::GemmKernelAVXVNNI256GPTQInt4>
class AVXVNNI256_GPTQ_INT4_MOE_TP : public AVX2_MOE_BASE<T, AVXVNNI256_GPTQ_INT4_MOE_TP<T>> {
  using Base = AVX2_MOE_BASE<T, AVXVNNI256_GPTQ_INT4_MOE_TP<T>>;
  using Base::config_;
  using Base::down_ba_;
  using Base::down_bb_;
  using Base::down_bc_;
  using Base::gate_bb_;
  using Base::gate_bc_;
  using Base::gate_up_ba_;
  using Base::m_local_num_;
  using Base::tp_part_idx;
  using Base::up_bb_;
  using Base::up_bc_;

  bool fused_decode_enabled_ = false;
  int fused_hidden_groups_ = 0;
  int fused_intermediate_groups_ = 0;
  std::vector<uint8_t> fused_input_u8_;
  std::vector<float> fused_input_scales_;
  std::vector<uint8_t> fused_activation_u8_;
  std::vector<float> fused_activation_scales_;
  std::vector<const uint8_t*> fused_active_activation_u8_;
  std::vector<const float*> fused_active_activation_scales_;
  std::vector<const typename T::BufferB*> fused_active_down_b_;
  std::vector<float> fused_active_weights_;

  bool fused_profile_enabled_ = false;
  uint64_t fused_profile_interval_ = 200;
  uint64_t fused_profile_calls_ = 0;
  uint64_t fused_profile_input_quant_us_ = 0;
  uint64_t fused_profile_gate_up_us_ = 0;
  uint64_t fused_profile_activation_quant_us_ = 0;
  uint64_t fused_profile_down_merge_us_ = 0;
  uint64_t fused_profile_total_us_ = 0;

  void init_fused_decode() {
    const int group_size = config_.quant_config.group_size;
    const bool env_enabled = avxvnni::env_flag_enabled("KT_AVXVNNI_FUSED_MOE", true);
    fused_profile_enabled_ = avxvnni::env_flag_enabled("KT_AVXVNNI_FUSED_MOE_PROFILE", false);
    fused_profile_interval_ = avxvnni::env_u64("KT_AVXVNNI_FUSED_MOE_PROFILE_INTERVAL", 200);
    const bool shape_supported = group_size > 0 && (config_.hidden_size % group_size) == 0 &&
                                 (config_.intermediate_size % group_size) == 0;

    fused_decode_enabled_ = env_enabled && shape_supported;
    if (shape_supported) {
      fused_hidden_groups_ = config_.hidden_size / group_size;
      fused_intermediate_groups_ = config_.intermediate_size / group_size;
      fused_input_u8_.resize((size_t)config_.hidden_size);
      fused_input_scales_.resize((size_t)fused_hidden_groups_);
      fused_activation_u8_.resize((size_t)config_.num_experts_per_tok * config_.intermediate_size);
      fused_activation_scales_.resize((size_t)config_.num_experts_per_tok * fused_intermediate_groups_);
      fused_active_activation_u8_.resize((size_t)config_.num_experts_per_tok);
      fused_active_activation_scales_.resize((size_t)config_.num_experts_per_tok);
      fused_active_down_b_.resize((size_t)config_.num_experts_per_tok);
      fused_active_weights_.resize((size_t)config_.num_experts_per_tok);
    }

    printf("AVXVNNI256_GPTQ_INT4_MOE_TP %d fused_decode=%s (KT_AVXVNNI_FUSED_MOE=%s, profile=%s)\n",
           tp_part_idx, fused_decode_enabled_ ? "on" : "off", env_enabled ? "on" : "off",
           fused_profile_enabled_ ? "on" : "off");
  }

  bool fused_decode_request_supported(int k, const int64_t* expert_ids) const {
    if (k > config_.num_experts_per_tok) {
      return false;
    }

    for (int i = 0; i < k; ++i) {
      if (config_.should_skip_expert(expert_ids[i])) {
        continue;
      }
      for (int j = 0; j < i; ++j) {
        if (!config_.should_skip_expert(expert_ids[j]) && expert_ids[i] == expert_ids[j]) {
          return false;
        }
      }
    }

    return true;
  }

  void report_fused_profile(int active_count, uint64_t input_quant_us, uint64_t gate_up_us,
                            uint64_t activation_quant_us, uint64_t down_merge_us, uint64_t total_us) {
    if (!fused_profile_enabled_) {
      return;
    }

    ++fused_profile_calls_;
    fused_profile_input_quant_us_ += input_quant_us;
    fused_profile_gate_up_us_ += gate_up_us;
    fused_profile_activation_quant_us_ += activation_quant_us;
    fused_profile_down_merge_us_ += down_merge_us;
    fused_profile_total_us_ += total_us;

    if (fused_profile_calls_ > 5 && (fused_profile_calls_ % fused_profile_interval_) != 0) {
      return;
    }

    const double denom = (double)fused_profile_calls_;
    printf("KT_AVXVNNI_FUSED_MOE_PROFILE layer=%d tp=%d call=%llu active=%d "
           "last_us{input_quant=%llu gate_up=%llu activation_quant=%llu down_merge=%llu total=%llu} "
           "avg_us{input_quant=%.1f gate_up=%.1f activation_quant=%.1f down_merge=%.1f total=%.1f}\n",
           config_.layer_idx, tp_part_idx, (unsigned long long)fused_profile_calls_, active_count,
           (unsigned long long)input_quant_us, (unsigned long long)gate_up_us,
           (unsigned long long)activation_quant_us, (unsigned long long)down_merge_us,
           (unsigned long long)total_us, fused_profile_input_quant_us_ / denom, fused_profile_gate_up_us_ / denom,
           fused_profile_activation_quant_us_ / denom, fused_profile_down_merge_us_ / denom,
           fused_profile_total_us_ / denom);
  }

 public:
  using typename Base::input_t;
  using typename Base::output_t;

  AVXVNNI256_GPTQ_INT4_MOE_TP() = default;
  AVXVNNI256_GPTQ_INT4_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {
    init_fused_decode();
  }

  void derived_init() {
#if defined(__GNUC__) || defined(__clang__)
    if (!__builtin_cpu_supports("avxvnni")) {
      throw std::runtime_error("AVX-VNNI-256 GPTQ_INT4 backend requires CPU support for avx_vnni");
    }
#endif
    auto& qc = config_.quant_config;
    if (qc.group_size == 0 || (qc.group_size % 32) != 0) {
      throw std::runtime_error("AVX-VNNI-256 GPTQ_INT4 requires group_size to be a positive multiple of 32");
    }
    if (qc.group_size > avxvnni::MAX_SUPPORTED_GROUP_SIZE) {
      throw std::runtime_error("AVX-VNNI-256 GPTQ_INT4 requires group_size <= 256");
    }
    printf("Created AVXVNNI256_GPTQ_INT4_MOE_TP %d at numa %d (group_size=%d)\n", tp_part_idx,
           numa_node_of_cpu(sched_getcpu()), qc.group_size);
  }

  ~AVXVNNI256_GPTQ_INT4_MOE_TP() = default;

  GeneralMOEConfig& mutable_config() { return config_; }

  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    return T::BufferB::required_size(n, k, config_.quant_config.group_size);
  }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    (void)qlen;
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];
    avxvnni::gemm_gptq_sym_int4_avxvnni256(m, config_.intermediate_size, config_.hidden_size, *ba, *bb, *bc, ith, nth);
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    (void)qlen;
    int m = m_local_num_[expert_idx];
    avxvnni::gemm_gptq_sym_int4_avxvnni256(m, config_.hidden_size, config_.intermediate_size, *down_ba_[expert_idx],
                                          *down_bb_[expert_idx], *down_bc_[expert_idx], ith, nth);
  }

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    if (fused_decode_enabled_ && qlen == 1 && fused_decode_request_supported(k, expert_ids)) {
      forward_decode_fused(k, expert_ids, weights, input, output);
      return;
    }
    Base::forward(qlen, k, expert_ids, weights, input, output);
  }

  void forward_decode_fused(int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    const int group_size = config_.quant_config.group_size;
    const ggml_bf16_t* input_bf16 = (const ggml_bf16_t*)input;
    using Clock = std::chrono::steady_clock;
    Clock::time_point total_start;
    Clock::time_point stage_start;
    uint64_t input_quant_us = 0;
    uint64_t gate_up_us = 0;
    uint64_t activation_quant_us = 0;
    uint64_t down_merge_us = 0;
    uint64_t total_us = 0;
    if (fused_profile_enabled_) {
      total_start = Clock::now();
      stage_start = total_start;
    }

    int activated_expert = 0;
    std::fill(this->m_local_num_.begin(), this->m_local_num_.end(), 0);
    for (int j = 0; j < k; ++j) {
      if (config_.should_skip_expert(expert_ids[j])) {
        continue;
      }

      const int expert_idx = (int)expert_ids[j];
      this->m_expert_id_map_[activated_expert] = expert_idx;
      this->m_local_pos_[0][j] = 0;
      this->m_local_num_[expert_idx] = 1;
      this->m_local_gate_output_ptr_[expert_idx] =
          this->m_local_gate_output_ + (size_t)activated_expert * config_.intermediate_size;
      this->m_local_down_output_ptr_[expert_idx] =
          this->m_local_down_output_ + (size_t)activated_expert * config_.hidden_size;
      fused_active_activation_u8_[activated_expert] =
          fused_activation_u8_.data() + (size_t)activated_expert * config_.intermediate_size;
      fused_active_activation_scales_[activated_expert] =
          fused_activation_scales_.data() + (size_t)activated_expert * fused_intermediate_groups_;
      fused_active_down_b_[activated_expert] = down_bb_[expert_idx].get();
      fused_active_weights_[activated_expert] = weights[j];
      ++activated_expert;
    }

    avxvnni::quantize_activation_groups_u8(input_bf16, fused_hidden_groups_, group_size, fused_input_u8_.data(),
                                           fused_input_scales_.data());
    if (fused_profile_enabled_) {
      auto now = Clock::now();
      input_quant_us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(now - stage_start).count();
      stage_start = now;
    }

    if (activated_expert > 0) {
      const int gate_up_nth = T::recommended_nth(config_.intermediate_size);
      pool->do_work_stealing_job(
          gate_up_nth * activated_expert, [](int) { T::config(); },
          [this, gate_up_nth](int task_id) {
            const int active_idx = task_id / gate_up_nth;
            const int ith = task_id % gate_up_nth;
            const int expert_idx = this->m_expert_id_map_[active_idx];
            ggml_bf16_t* activation_bf16 =
                this->m_local_gate_output_ + (size_t)active_idx * this->config_.intermediate_size;

            avxvnni::fused_gate_up_activation_avxvnni256(
                this->config_.intermediate_size, this->fused_input_u8_.data(), this->fused_input_scales_.data(),
                *this->gate_bb_[expert_idx], *this->up_bb_[expert_idx], activation_bf16, ith, gate_up_nth);
          },
          nullptr);
    }

    if (fused_profile_enabled_) {
      auto now = Clock::now();
      gate_up_us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(now - stage_start).count();
      stage_start = now;
    }

    if (activated_expert > 0) {
      pool->do_work_stealing_job(
          activated_expert, nullptr,
          [this, group_size](int active_idx) {
            ggml_bf16_t* activation_bf16 =
                this->m_local_gate_output_ + (size_t)active_idx * this->config_.intermediate_size;
            avxvnni::quantize_activation_groups_u8(activation_bf16, this->fused_intermediate_groups_, group_size,
                                                   this->fused_activation_u8_.data() +
                                                       (size_t)active_idx * this->config_.intermediate_size,
                                                   this->fused_activation_scales_.data() +
                                                       (size_t)active_idx * this->fused_intermediate_groups_);
          },
          nullptr);
    }

    float* out = (float*)output;
    if (fused_profile_enabled_) {
      auto now = Clock::now();
      activation_quant_us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(now - stage_start).count();
      stage_start = now;
    }

    const int down_nth = T::recommended_nth(config_.hidden_size);
    if (activated_expert > 0) {
      pool->do_work_stealing_job(
          down_nth, [](int) { T::config(); },
          [this, out, activated_expert, down_nth](int ith) {
            avxvnni::fused_down_weighted_sum_avxvnni256(
                this->config_.hidden_size, this->fused_active_activation_u8_.data(),
                this->fused_active_activation_scales_.data(), this->fused_active_down_b_.data(),
                this->fused_active_weights_.data(), activated_expert, out, ith, down_nth);
          },
          nullptr);
    } else {
      std::fill(out, out + config_.hidden_size, 0.0f);
    }

    if (fused_profile_enabled_) {
      auto now = Clock::now();
      down_merge_us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(now - stage_start).count();
      total_us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(now - total_start).count();
      report_fused_profile(activated_expert, input_quant_us, gate_up_us, activation_quant_us, down_merge_us, total_us);
    }
  }

  void load_weights() {
    int group_size = config_.quant_config.group_size;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (config_.gate_scale == nullptr) {
      throw std::runtime_error("GPTQ INT4 MOE requires scale pointers.");
    }

    int gate_up_k = config_.hidden_size;
    int gate_up_n = config_.intermediate_size;
    size_t qw_elems = (size_t)(gate_up_k / 8) * gate_up_n;
    size_t sc_elems = (size_t)(gate_up_k / group_size) * gate_up_n;

    int nth = T::recommended_nth(gate_up_n);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, qw_elems, sc_elems](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;

          gate_bb_[expert_idx]->from_mat((uint32_t*)config_.gate_proj + logical * qw_elems,
                                         (float*)config_.gate_scale + logical * sc_elems, ith, nth);

          up_bb_[expert_idx]->from_mat((uint32_t*)config_.up_proj + logical * qw_elems,
                                       (float*)config_.up_scale + logical * sc_elems, ith, nth);
        },
        nullptr);

    int down_k = config_.intermediate_size;
    int down_n = config_.hidden_size;
    size_t down_qw_elems = (size_t)(down_k / 8) * down_n;
    size_t down_sc_elems = (size_t)(down_k / group_size) * down_n;

    nth = T::recommended_nth(down_n);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, down_qw_elems, down_sc_elems](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;

          down_bb_[expert_idx]->from_mat((uint32_t*)config_.down_proj + logical * down_qw_elems,
                                         (float*)config_.down_scale + logical * down_sc_elems, ith, nth);
        },
        nullptr);
  }

  void write_weights_to_buffer(int gpu_tp_count, [[maybe_unused]] int cpu_tp_count, int expert_id,
                               const GeneralMOEConfig& full_config, const std::vector<uintptr_t>& w13_weight_ptrs,
                               [[maybe_unused]] const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               [[maybe_unused]] const std::vector<uintptr_t>& w2_scale_ptrs) const {
    (void)gpu_tp_count;
    (void)expert_id;
    (void)full_config;
    (void)w13_weight_ptrs;
    (void)w2_weight_ptrs;
    throw std::runtime_error("AVX-VNNI-256 GPTQ INT4 write_weights_to_buffer not yet implemented");
  }
};

template <typename K>
class TP_MOE<AVXVNNI256_GPTQ_INT4_MOE_TP<K>> : public TP_MOE_Common<AVXVNNI256_GPTQ_INT4_MOE_TP<K>> {
 public:
  using Base = TP_MOE_Common<AVXVNNI256_GPTQ_INT4_MOE_TP<K>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    const int group_size = config.quant_config.group_size;
    if (group_size == 0) {
      throw std::runtime_error("GPTQ INT4 requires group_size > 0");
    }

    if (config.gate_projs.empty() && config.gate_proj == nullptr) {
      throw std::runtime_error("no weight source");
    }
    const bool use_per_expert_ptrs = !config.gate_projs.empty();

    const int full_intermediate = config.intermediate_size;
    const int full_hidden = config.hidden_size;

    const int gate_up_k_packed = full_hidden / 8;
    const int gate_up_num_groups = full_hidden / group_size;
    const size_t full_gate_up_qw_elems = (size_t)gate_up_k_packed * full_intermediate;
    const size_t full_gate_up_sc_elems = (size_t)gate_up_num_groups * full_intermediate;

    const int down_k_packed = full_intermediate / 8;
    const int down_num_groups = full_intermediate / group_size;
    const size_t full_down_qw_elems = (size_t)down_k_packed * full_hidden;
    const size_t full_down_sc_elems = (size_t)down_num_groups * full_hidden;

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->mutable_config();
      const int tp_intermediate = tpc.intermediate_size;

      const size_t tp_gate_up_qw_elems = (size_t)gate_up_k_packed * tp_intermediate;
      const size_t tp_gate_up_sc_elems = (size_t)gate_up_num_groups * tp_intermediate;

      tpc.gate_proj = new uint32_t[tpc.expert_num * tp_gate_up_qw_elems];
      tpc.up_proj = new uint32_t[tpc.expert_num * tp_gate_up_qw_elems];
      tpc.gate_scale = new float[tpc.expert_num * tp_gate_up_sc_elems];
      tpc.up_scale = new float[tpc.expert_num * tp_gate_up_sc_elems];

      const int tp_down_k_packed = tp_intermediate / 8;
      const int tp_down_num_groups = tp_intermediate / group_size;
      const size_t tp_down_qw_elems = (size_t)tp_down_k_packed * full_hidden;
      const size_t tp_down_sc_elems = (size_t)tp_down_num_groups * full_hidden;

      tpc.down_proj = new uint32_t[tpc.expert_num * tp_down_qw_elems];
      tpc.down_scale = new float[tpc.expert_num * tp_down_sc_elems];

      const int gate_up_n_offset = i * tp_intermediate;
      const int down_k_offset_packed = i * tp_down_k_packed;
      const int down_group_offset = i * tp_down_num_groups;

      pool->get_subpool(i)->do_work_stealing_job(
          tpc.expert_num, nullptr,
          [&, &tpc](int expert_id_) {
            const size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

            const uint32_t* gate_qw_src;
            const uint32_t* up_qw_src;
            const uint32_t* down_qw_src;
            const float* gate_sc_src;
            const float* up_sc_src;
            const float* down_sc_src;

            if (use_per_expert_ptrs) {
              gate_qw_src = (const uint32_t*)config.gate_projs[0][expert_id];
              up_qw_src = (const uint32_t*)config.up_projs[0][expert_id];
              down_qw_src = (const uint32_t*)config.down_projs[0][expert_id];
              gate_sc_src = (const float*)config.gate_scales[0][expert_id];
              up_sc_src = (const float*)config.up_scales[0][expert_id];
              down_sc_src = (const float*)config.down_scales[0][expert_id];
            } else {
              gate_qw_src = (const uint32_t*)config.gate_proj + expert_id * full_gate_up_qw_elems;
              up_qw_src = (const uint32_t*)config.up_proj + expert_id * full_gate_up_qw_elems;
              down_qw_src = (const uint32_t*)config.down_proj + expert_id * full_down_qw_elems;
              gate_sc_src = (const float*)config.gate_scale + expert_id * full_gate_up_sc_elems;
              up_sc_src = (const float*)config.up_scale + expert_id * full_gate_up_sc_elems;
              down_sc_src = (const float*)config.down_scale + expert_id * full_down_sc_elems;
            }

            uint32_t* gate_qw_dst = (uint32_t*)tpc.gate_proj + expert_id * tp_gate_up_qw_elems;
            uint32_t* up_qw_dst = (uint32_t*)tpc.up_proj + expert_id * tp_gate_up_qw_elems;
            float* gate_sc_dst = (float*)tpc.gate_scale + expert_id * tp_gate_up_sc_elems;
            float* up_sc_dst = (float*)tpc.up_scale + expert_id * tp_gate_up_sc_elems;

            for (int kr = 0; kr < gate_up_k_packed; ++kr) {
              std::memcpy(gate_qw_dst + kr * tp_intermediate, gate_qw_src + kr * full_intermediate + gate_up_n_offset,
                          (size_t)tp_intermediate * sizeof(uint32_t));
              std::memcpy(up_qw_dst + kr * tp_intermediate, up_qw_src + kr * full_intermediate + gate_up_n_offset,
                          (size_t)tp_intermediate * sizeof(uint32_t));
            }

            for (int g = 0; g < gate_up_num_groups; ++g) {
              std::memcpy(gate_sc_dst + g * tp_intermediate, gate_sc_src + g * full_intermediate + gate_up_n_offset,
                          (size_t)tp_intermediate * sizeof(float));
              std::memcpy(up_sc_dst + g * tp_intermediate, up_sc_src + g * full_intermediate + gate_up_n_offset,
                          (size_t)tp_intermediate * sizeof(float));
            }

            uint32_t* down_qw_dst = (uint32_t*)tpc.down_proj + expert_id * tp_down_qw_elems;
            for (int kr = 0; kr < tp_down_k_packed; ++kr) {
              std::memcpy(down_qw_dst + kr * full_hidden, down_qw_src + (down_k_offset_packed + kr) * full_hidden,
                          (size_t)full_hidden * sizeof(uint32_t));
            }

            float* down_sc_dst = (float*)tpc.down_scale + expert_id * tp_down_sc_elems;
            for (int g = 0; g < tp_down_num_groups; ++g) {
              std::memcpy(down_sc_dst + g * full_hidden, down_sc_src + (down_group_offset + g) * full_hidden,
                          (size_t)full_hidden * sizeof(float));
            }
          },
          nullptr);
    });

    pool->dispense_backend()->do_numa_job([&, this](int i) { tps[i]->load_weights(); });

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->mutable_config();
      delete[] (uint32_t*)tpc.gate_proj;
      delete[] (uint32_t*)tpc.up_proj;
      delete[] (uint32_t*)tpc.down_proj;
      delete[] (float*)tpc.gate_scale;
      delete[] (float*)tpc.up_scale;
      delete[] (float*)tpc.down_scale;
    });

    this->weights_loaded = true;
  }

  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_id, const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    (void)gpu_tp_count;
    (void)expert_id;
    (void)w13_weight_ptrs;
    (void)w13_scale_ptrs;
    (void)w2_weight_ptrs;
    (void)w2_scale_ptrs;
    throw std::runtime_error("AVX-VNNI-256 GPTQ INT4 write_weight_scale_to_buffer not yet implemented");
  }

  void merge_results(int qlen, void* output, bool incremental) override {
    auto& config = this->config;
    auto& tp_count = this->tp_count;
    auto& local_output_numa = this->local_output_numa;
    auto& tp_configs = this->tp_configs;

    auto merge_fn = [this, output, incremental, &config, &tp_count, &local_output_numa, &tp_configs](int token_nth) {
      float* merge_to = local_output_numa[0] + token_nth * tp_configs[0].hidden_size;
      if (incremental) {
        for (int e = 0; e < config.hidden_size; e += 16) {
          __m256 x0, x1;
          avx2::load_16xbf16_to_2x8xfp32((ggml_bf16_t*)output + token_nth * config.hidden_size + e, &x0, &x1);
          *((__m256*)(merge_to + e)) = _mm256_add_ps(*((__m256*)(merge_to + e)), x0);
          *((__m256*)(merge_to + e + 8)) = _mm256_add_ps(*((__m256*)(merge_to + e + 8)), x1);
        }
      }

      for (int i = 1; i < tp_count; i++) {
        float* merge_from = local_output_numa[i] + token_nth * tp_configs[i].hidden_size;
        for (int e = 0; e < tp_configs[i].hidden_size; e += 8) {
          *((__m256*)(merge_to + e)) = _mm256_add_ps(*((__m256*)(merge_to + e)), *((__m256*)(merge_from + e)));
        }
      }

      for (int e = 0; e < config.hidden_size; e += 16) {
        __m256 x0 = *(__m256*)(merge_to + e);
        __m256 x1 = *(__m256*)(merge_to + e + 8);
        avx2::store_2x8xfp32_to_16xbf16(&x0, &x1, (ggml_bf16_t*)output + token_nth * config.hidden_size + e);
      }
    };

    auto pool = config.pool;
    if (qlen < 10) {
      for (int i = 0; i < qlen; i++) merge_fn(i);
    } else {
      pool->do_work_stealing_job(qlen, nullptr, merge_fn, nullptr);
    }
  }

  void merge_results(int qlen, void* output) override { merge_results(qlen, output, false); }
};

#undef KT_AVXVNNI256_TARGET

#endif  // CPUINFER_OPERATOR_AVX2_GPTQ_INT4_AVXVNNI_MOE_H
