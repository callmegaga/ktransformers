/**
 * @Description  : AVX-VNNI-256 GPTQ-Int8 MoE operator (symmetric quantization)
 * @Author       : Codex
 * @Date         : 2026-04-09
 * @Version      : 1.0.0
 *
 * Real GPTQ-8bit checkpoints store:
 *   qweight [K/4, N] int32   (packing 4 x uint8 along K)
 *   scales  [K/group_size, N] fp16/fp32
 *   qzeros  [K/group_size, N/4] int32
 *
 * For the symmetric models used here, qzeros is constant 0x7f7f7f7f.
 * To keep the runtime kernel simple and VNNI-friendly, weights are unpacked to
 * signed int8 in [-128, 127] using (raw_u8 - 128). The exact original
 * quantized value is then recovered in GEMM by adding the activation sum once
 * per K-group, because:
 *
 *   raw_u8 - 127 = (raw_u8 - 128) + 1
 **/
#ifndef CPUINFER_OPERATOR_AVX2_GPTQ_INT8_AVXVNNI_MOE_H
#define CPUINFER_OPERATOR_AVX2_GPTQ_INT8_AVXVNNI_MOE_H

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

#include "avx2_bf16_utils.hpp"
#include "moe_base.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define KT_AVXVNNI256_GPTQ_INT8_TARGET __attribute__((target("avx2,avxvnni,fma")))
#else
#define KT_AVXVNNI256_GPTQ_INT8_TARGET
#endif

namespace avxvnni_gptq_int8 {

static inline int hsum_epi32_avx2(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i sum = _mm_add_epi32(lo, hi);
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return _mm_cvtsi128_si32(sum);
}

static inline float quantize_activation_group_u8_sum(const ggml_bf16_t* src, int group_size, uint8_t* dst, int* sum_q) {
  alignas(32) float tmp[256];
  float absmax = 0.0f;
  int sum = 0;

  for (int i = 0; i < group_size; ++i) {
    const float v = GGML_BF16_TO_FP32(src[i]);
    tmp[i] = v;
    absmax = std::max(absmax, std::fabs(v));
  }

  if (absmax <= std::numeric_limits<float>::min()) {
    std::memset(dst, 0x80, (size_t)group_size);
    *sum_q = 0;
    return 0.0f;
  }

  const float scale = absmax / 127.0f;
  const float inv_scale = 1.0f / scale;
  for (int i = 0; i < group_size; ++i) {
    int q = (int)std::lrint(tmp[i] * inv_scale);
    q = std::clamp(q, -127, 127);
    dst[i] = (uint8_t)(((uint8_t)(int8_t)q) ^ 0x80);
    sum += q;
  }
  *sum_q = sum;
  return scale;
}

struct GemmKernelAVXVNNI256GPTQInt8 {
  using dt = ggml_bf16_t;
  using output_t = float;
  static constexpr int M_STEP = 1;
  static constexpr int N_STEP = 8;
  static constexpr int K_STEP = 32;
  static constexpr int N_BLOCK = 64;
  static constexpr int K_BLOCK = 128;
  static constexpr double ELEMENT_SIZE = 1.0;

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
    int8_t* qweight_s8 = nullptr;   // [N, K], unpacked as (raw_u8 - 128)
    float* scales = nullptr;        // [N, num_groups]
    int32_t* weight_sums = nullptr; // [N, num_groups]
    int n = 0;
    int k = 0;
    int group_size = 32;
    int num_groups = 0;
    int k_packed = 0;

    BufferB() = default;
    BufferB(size_t n_, size_t k_, int gs, void* ptr) : n((int)n_), k((int)k_), group_size(gs) {
      if (group_size <= 0 || (group_size % 32) != 0) {
        throw std::runtime_error("AVX-VNNI GPTQ INT8 requires group_size to be a positive multiple of 32");
      }
      if ((k % 4) != 0 || (k % group_size) != 0) {
        throw std::runtime_error("AVX-VNNI GPTQ INT8 requires k to be divisible by both 4 and group_size");
      }
      k_packed = k / 4;
      num_groups = k / group_size;
      qweight_s8 = (int8_t*)ptr;
      scales = (float*)((uint8_t*)ptr + (size_t)k * n * sizeof(int8_t));
      weight_sums = (int32_t*)((uint8_t*)scales + (size_t)n * num_groups * sizeof(float));
    }

    static size_t required_size(size_t n, size_t k, int gs) {
      const size_t num_groups = k / gs;
      return n * k * sizeof(int8_t) + n * num_groups * sizeof(float) + n * num_groups * sizeof(int32_t);
    }

    void from_mat(const uint32_t* src_qweight, const float* src_scales, int ith, int nth) {
      auto [n_start, n_end] = avx2::split_range(n, ith, nth);

      for (int ni = n_start; ni < n_end; ++ni) {
        int8_t* dst_row = qweight_s8 + (size_t)ni * k;
        for (int kp = 0; kp < k_packed; ++kp) {
          uint32_t packed = src_qweight[(size_t)kp * n + ni];
          for (int byte_idx = 0; byte_idx < 4; ++byte_idx) {
            const uint8_t raw = (uint8_t)((packed >> (byte_idx * 8)) & 0xFF);
            dst_row[kp * 4 + byte_idx] = (int8_t)((int)raw - 128);
          }
        }

        int32_t* sums_row = weight_sums + (size_t)ni * num_groups;
        float* scales_row = scales + (size_t)ni * num_groups;
        for (int g = 0; g < num_groups; ++g) {
          scales_row[g] = src_scales[(size_t)g * n + ni];
          int32_t sum = 0;
          const int8_t* group_row = dst_row + (size_t)g * group_size;
          for (int kk = 0; kk < group_size; ++kk) {
            sum += group_row[kk];
          }
          sums_row[g] = sum;
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

KT_AVXVNNI256_GPTQ_INT8_TARGET
static inline void gemm_gptq_sym_int8_avxvnni256(int m, int n, int k, GemmKernelAVXVNNI256GPTQInt8::BufferA& a,
                                                 GemmKernelAVXVNNI256GPTQInt8::BufferB& b,
                                                 GemmKernelAVXVNNI256GPTQInt8::BufferC& c, int ith, int nth) {
  (void)k;
  auto [n_start, n_end] = avx2::split_range(n, ith, nth);
  const int group_size = b.group_size;
  const int num_groups = b.num_groups;

  alignas(32) std::array<uint8_t, 256> a_u8{};

  for (int mi = 0; mi < m; ++mi) {
    const ggml_bf16_t* a_row = a.data + (size_t)mi * a.k;
    float* c_row = c.data + (size_t)mi * n;
    std::fill(c_row + n_start, c_row + n_end, 0.0f);

    for (int g = 0; g < num_groups; ++g) {
      const int k_base = g * group_size;
      int act_sum = 0;
      const float a_scale = quantize_activation_group_u8_sum(a_row + k_base, group_size, a_u8.data(), &act_sum);
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

        const int32_t dot = hsum_epi32_avx2(acc) - 128 * b.weight_sums[(size_t)ni * num_groups + g] + act_sum;
        c_row[ni] += (float)dot * a_scale * b.scales[(size_t)ni * num_groups + g];
      }
    }
  }
}

}  // namespace avxvnni_gptq_int8

template <class T = avxvnni_gptq_int8::GemmKernelAVXVNNI256GPTQInt8>
class AVXVNNI256_GPTQ_INT8_MOE_TP : public AVX2_MOE_BASE<T, AVXVNNI256_GPTQ_INT8_MOE_TP<T>> {
  using Base = AVX2_MOE_BASE<T, AVXVNNI256_GPTQ_INT8_MOE_TP<T>>;
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

 public:
  using typename Base::input_t;
  using typename Base::output_t;

  AVXVNNI256_GPTQ_INT8_MOE_TP() = default;
  AVXVNNI256_GPTQ_INT8_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
#if defined(__GNUC__) || defined(__clang__)
    if (!__builtin_cpu_supports("avxvnni")) {
      throw std::runtime_error("AVX-VNNI-256 GPTQ_INT8 backend requires CPU support for avx_vnni");
    }
#endif
    auto& qc = config_.quant_config;
    if (qc.group_size == 0 || (qc.group_size % 32) != 0) {
      throw std::runtime_error("AVX-VNNI-256 GPTQ_INT8 requires group_size to be a positive multiple of 32");
    }
    printf("Created AVXVNNI256_GPTQ_INT8_MOE_TP %d at numa %d (group_size=%d)\n", tp_part_idx,
           numa_node_of_cpu(sched_getcpu()), qc.group_size);
  }

  ~AVXVNNI256_GPTQ_INT8_MOE_TP() = default;

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
    avxvnni_gptq_int8::gemm_gptq_sym_int8_avxvnni256(m, config_.intermediate_size, config_.hidden_size, *ba, *bb, *bc,
                                                     ith, nth);
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    (void)qlen;
    int m = m_local_num_[expert_idx];
    avxvnni_gptq_int8::gemm_gptq_sym_int8_avxvnni256(m, config_.hidden_size, config_.intermediate_size,
                                                     *down_ba_[expert_idx], *down_bb_[expert_idx], *down_bc_[expert_idx],
                                                     ith, nth);
  }

  void load_weights() {
    int group_size = config_.quant_config.group_size;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (config_.gate_scale == nullptr && config_.gate_scales.empty()) {
      throw std::runtime_error("GPTQ INT8 MOE requires scale pointers.");
    }

    const int gate_up_k = config_.hidden_size;
    const int gate_up_n = config_.intermediate_size;
    const size_t qw_elems = (size_t)(gate_up_k / 4) * gate_up_n;
    const size_t sc_elems = (size_t)(gate_up_k / group_size) * gate_up_n;

    int nth = T::recommended_nth(gate_up_n);
    if (!config_.gate_projs.empty()) {
      pool->do_work_stealing_job(
          nth * config_.expert_num, nullptr,
          [this, nth, physical_to_logical_map](int task_id) {
            uint64_t expert_idx = task_id / nth;
            uint64_t logical = expert_map(physical_to_logical_map, expert_idx);
            int ith = task_id % nth;

            gate_bb_[expert_idx]->from_mat((const uint32_t*)config_.gate_projs[0][logical],
                                           (const float*)config_.gate_scales[0][logical], ith, nth);
            up_bb_[expert_idx]->from_mat((const uint32_t*)config_.up_projs[0][logical],
                                         (const float*)config_.up_scales[0][logical], ith, nth);
          },
          nullptr);
    } else {
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
    }

    const int down_k = config_.intermediate_size;
    const int down_n = config_.hidden_size;
    const size_t down_qw_elems = (size_t)(down_k / 4) * down_n;
    const size_t down_sc_elems = (size_t)(down_k / group_size) * down_n;

    nth = T::recommended_nth(down_n);
    if (!config_.down_projs.empty()) {
      pool->do_work_stealing_job(
          nth * config_.expert_num, nullptr,
          [this, nth, physical_to_logical_map](int task_id) {
            uint64_t expert_idx = task_id / nth;
            uint64_t logical = expert_map(physical_to_logical_map, expert_idx);
            int ith = task_id % nth;

            down_bb_[expert_idx]->from_mat((const uint32_t*)config_.down_projs[0][logical],
                                           (const float*)config_.down_scales[0][logical], ith, nth);
          },
          nullptr);
    } else {
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
    throw std::runtime_error("AVX-VNNI-256 GPTQ INT8 write_weights_to_buffer not yet implemented");
  }
};

template <typename K>
class TP_MOE<AVXVNNI256_GPTQ_INT8_MOE_TP<K>> : public TP_MOE<AVX2_MOE_BASE<K, AVXVNNI256_GPTQ_INT8_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AVX2_MOE_BASE<K, AVXVNNI256_GPTQ_INT8_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    const int group_size = config.quant_config.group_size;
    if (group_size == 0) {
      throw std::runtime_error("GPTQ INT8 requires group_size > 0");
    }

    if (config.gate_projs.empty() && config.gate_proj == nullptr) {
      throw std::runtime_error("no weight source");
    }
    const bool use_per_expert_ptrs = !config.gate_projs.empty();

    const int full_intermediate = config.intermediate_size;
    const int full_hidden = config.hidden_size;

    const int gate_up_k_packed = full_hidden / 4;
    const int gate_up_num_groups = full_hidden / group_size;
    const size_t full_gate_up_qw_elems = (size_t)gate_up_k_packed * full_intermediate;
    const size_t full_gate_up_sc_elems = (size_t)gate_up_num_groups * full_intermediate;

    const int down_k_packed = full_intermediate / 4;
    const int down_num_groups = full_intermediate / group_size;
    const size_t full_down_qw_elems = (size_t)down_k_packed * full_hidden;
    const size_t full_down_sc_elems = (size_t)down_num_groups * full_hidden;

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      const int tp_intermediate = tpc.intermediate_size;

      const size_t tp_gate_up_qw_elems = (size_t)gate_up_k_packed * tp_intermediate;
      const size_t tp_gate_up_sc_elems = (size_t)gate_up_num_groups * tp_intermediate;

      tpc.gate_proj = new uint32_t[tpc.expert_num * tp_gate_up_qw_elems];
      tpc.up_proj = new uint32_t[tpc.expert_num * tp_gate_up_qw_elems];
      tpc.gate_scale = new float[tpc.expert_num * tp_gate_up_sc_elems];
      tpc.up_scale = new float[tpc.expert_num * tp_gate_up_sc_elems];

      const int tp_down_k_packed = tp_intermediate / 4;
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
      auto& tpc = tps[i]->config_;
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
    throw std::runtime_error("AVX-VNNI-256 GPTQ INT8 write_weight_scale_to_buffer not yet implemented");
  }
};

#undef KT_AVXVNNI256_GPTQ_INT8_TARGET

#endif  // CPUINFER_OPERATOR_AVX2_GPTQ_INT8_AVXVNNI_MOE_H
