/**
 * @Description  : AVX-VNNI-256 INT4 MoE operator
 * @Author       : Codex
 * @Date         : 2026-04-05
 * @Version      : 1.0.0
 *
 * Disk format keeps packed signed-int4 weights (two weights per byte).
 * Runtime expands them to signed int8 so AVX-VNNI dpbusd can be used with
 * biased uint8 activations. This matches AMXINT4 symmetric quantization:
 * - Activations: per-row symmetric int8 quantization
 * - Weights: per-output-channel symmetric int4 quantization
 */
#ifndef CPUINFER_OPERATOR_AVX2_INT4_AVXVNNI_MOE_H
#define CPUINFER_OPERATOR_AVX2_INT4_AVXVNNI_MOE_H

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "moe_base.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define KT_AVXVNNI256_INT4_TARGET __attribute__((target("avx2,avxvnni,fma")))
#else
#define KT_AVXVNNI256_INT4_TARGET
#endif

namespace avxvnni_int4 {

static inline int hsum_epi32_avx2(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i sum = _mm_add_epi32(lo, hi);
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return _mm_cvtsi128_si32(sum);
}

static inline float quantize_activation_row_u8(const ggml_bf16_t* src, int k, uint8_t* dst) {
  float absmax = 0.0f;
  for (int i = 0; i < k; ++i) {
    absmax = std::max(absmax, std::fabs(GGML_BF16_TO_FP32(src[i])));
  }

  if (absmax <= std::numeric_limits<float>::min()) {
    std::memset(dst, 0x80, (size_t)k);
    return 0.0f;
  }

  const float scale = absmax / 127.0f;
  const float inv_scale = 1.0f / scale;
  for (int i = 0; i < k; ++i) {
    int q = (int)std::lrint(GGML_BF16_TO_FP32(src[i]) * inv_scale);
    q = std::clamp(q, -127, 127);
    dst[i] = (uint8_t)(((uint8_t)(int8_t)q) ^ 0x80);
  }
  return scale;
}

static inline uint8_t pack_signed_int4_pair(int8_t lo, int8_t hi) {
  uint8_t lo_nib = (uint8_t)lo & 0x0F;
  uint8_t hi_nib = ((uint8_t)hi & 0x0F) << 4;
  return (uint8_t)(lo_nib | hi_nib);
}

static inline int8_t unpack_signed_int4(uint8_t nibble) {
  nibble &= 0x0F;
  return (nibble & 0x08) ? (int8_t)(nibble - 16) : (int8_t)nibble;
}

struct GemmKernelAVXVNNI256Int4 {
  using dt = ggml_bf16_t;
  using output_t = float;
  static constexpr int M_STEP = 1;
  static constexpr int N_STEP = 8;
  static constexpr int K_STEP = 32;
  static constexpr int N_BLOCK = 64;
  static constexpr int K_BLOCK = 256;
  static constexpr double ELEMENT_SIZE = 0.5;

  static void config() {}

  static std::string name() { return "AVXVNNI_INT4"; }

  static int recommended_nth(int n) { return std::max(1, n / N_BLOCK); }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) { return avx2::split_range(n, ith, nth); }

  struct BufferA {
    uint8_t* data = nullptr;
    float* scales = nullptr;
    size_t max_m = 0;
    size_t k = 0;

    BufferA() = default;
    BufferA(size_t m, size_t k_, void* ptr) : max_m(m), k(k_) { set_data(ptr); }

    static size_t required_size(size_t m, size_t k) { return m * k * sizeof(uint8_t) + m * sizeof(float); }

    void set_data(void* ptr) {
      data = (uint8_t*)ptr;
      scales = (float*)((uint8_t*)ptr + max_m * k * sizeof(uint8_t));
    }

    void from_mat(int m, const ggml_bf16_t* src, int ith, int nth) {
      auto [m_start, m_end] = avx2::split_range(m, ith, nth);
      for (int mi = m_start; mi < m_end; ++mi) {
        scales[mi] = quantize_activation_row_u8(src + (size_t)mi * k, (int)k, data + (size_t)mi * k);
      }
    }
  };

  struct BufferB {
    int8_t* b = nullptr;         // unpacked signed int4 values in [-8, 7]
    float* d = nullptr;          // dequant scale = absmax / 7
    int32_t* weight_sums = nullptr;
    int n = 0;
    int k = 0;
    static constexpr bool SCALE = true;

    BufferB() = default;
    BufferB(size_t n_, size_t k_, void* ptr) : n((int)n_), k((int)k_) {
      if ((k % K_STEP) != 0 || (k % 2) != 0) {
        throw std::runtime_error("AVX-VNNI INT4 requires k to be a multiple of 32 and even");
      }
      set_data(ptr);
    }

    static size_t required_size(size_t n, size_t k) {
      return n * k * sizeof(int8_t) + n * sizeof(float) + n * sizeof(int32_t);
    }

    static size_t packed_weight_bytes(size_t n, size_t k) { return n * k / 2; }

    void set_data(void* ptr) {
      b = (int8_t*)ptr;
      d = (float*)((uint8_t*)ptr + (size_t)n * k * sizeof(int8_t));
      weight_sums = (int32_t*)((uint8_t*)d + (size_t)n * sizeof(float));
    }

    void recompute_weight_sums(int n_start, int n_end) {
      for (int ni = n_start; ni < n_end; ++ni) {
        int32_t sum = 0;
        const int8_t* row = b + (size_t)ni * k;
        for (int kk = 0; kk < k; ++kk) {
          sum += row[kk];
        }
        weight_sums[ni] = sum;
      }
    }

    void from_packed_mat(const uint8_t* src_weight, const float* src_scale, int ith, int nth) {
      auto [n_start, n_end] = avx2::split_range(n, ith, nth);
      const size_t row_bytes = (size_t)k / 2;
      for (int ni = n_start; ni < n_end; ++ni) {
        const uint8_t* src_row = src_weight + (size_t)ni * row_bytes;
        int8_t* dst_row = b + (size_t)ni * k;
        for (int kb = 0; kb < k / 2; ++kb) {
          const uint8_t packed = src_row[kb];
          dst_row[kb * 2] = unpack_signed_int4(packed & 0x0F);
          dst_row[kb * 2 + 1] = unpack_signed_int4((packed >> 4) & 0x0F);
        }
      }
      std::memcpy(d + n_start, src_scale + n_start, (size_t)(n_end - n_start) * sizeof(float));
      recompute_weight_sums(n_start, n_end);
    }

    void from_mat(const ggml_bf16_t* src, int ith, int nth) {
      auto [n_start, n_end] = avx2::split_range(n, ith, nth);
      for (int ni = n_start; ni < n_end; ++ni) {
        const ggml_bf16_t* src_row = src + (size_t)ni * k;
        int8_t* dst_row = b + (size_t)ni * k;
        float absmax = 0.0f;
        for (int kk = 0; kk < k; ++kk) {
          absmax = std::max(absmax, std::fabs(GGML_BF16_TO_FP32(src_row[kk])));
        }

        if (absmax <= std::numeric_limits<float>::min()) {
          std::memset(dst_row, 0, (size_t)k);
          d[ni] = 0.0f;
          weight_sums[ni] = 0;
          continue;
        }

        const float scale = absmax / 7.0f;
        const float inv_scale = 1.0f / scale;
        int32_t sum = 0;
        for (int kk = 0; kk < k; ++kk) {
          int q = (int)std::lrint(GGML_BF16_TO_FP32(src_row[kk]) * inv_scale);
          q = std::clamp(q, -8, 7);
          dst_row[kk] = (int8_t)q;
          sum += q;
        }
        d[ni] = scale;
        weight_sums[ni] = sum;
      }
    }

    void serialize_packed(uint8_t* dst_packed, int n_start, int n_end) const {
      const size_t row_bytes = (size_t)k / 2;
      for (int ni = n_start; ni < n_end; ++ni) {
        const int8_t* src_row = b + (size_t)ni * k;
        uint8_t* dst_row = dst_packed + (size_t)(ni - n_start) * row_bytes;
        for (int kb = 0; kb < k / 2; ++kb) {
          dst_row[kb] = pack_signed_int4_pair(src_row[kb * 2], src_row[kb * 2 + 1]);
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

KT_AVXVNNI256_INT4_TARGET
static inline void gemm_int4_avxvnni256(int m, int n, int k, GemmKernelAVXVNNI256Int4::BufferA& a,
                                        GemmKernelAVXVNNI256Int4::BufferB& b, GemmKernelAVXVNNI256Int4::BufferC& c,
                                        int ith, int nth) {
  auto [n_start, n_end] = avx2::split_range(n, ith, nth);

  for (int mi = 0; mi < m; ++mi) {
    const uint8_t* a_row = a.data + (size_t)mi * a.k;
    const float a_scale = a.scales[mi];
    float* c_row = c.data + (size_t)mi * n;

    if (a_scale == 0.0f) {
      std::fill(c_row + n_start, c_row + n_end, 0.0f);
      continue;
    }

    for (int ni = n_start; ni < n_end; ++ni) {
      const int8_t* w_row = b.b + (size_t)ni * b.k;
      __m256i acc = _mm256_setzero_si256();
      for (int kk = 0; kk < k; kk += 32) {
        __m256i a_vec = _mm256_loadu_si256((const __m256i*)(a_row + kk));
        __m256i w_vec = _mm256_loadu_si256((const __m256i*)(w_row + kk));
        acc = _mm256_dpbusd_avx_epi32(acc, a_vec, w_vec);
      }
      const int dot = hsum_epi32_avx2(acc) - 128 * b.weight_sums[ni];
      c_row[ni] = (float)dot * a_scale * b.d[ni];
    }
  }
}

}  // namespace avxvnni_int4

template <class T = avxvnni_int4::GemmKernelAVXVNNI256Int4>
class AVXVNNI256_INT4_MOE_TP : public AVX2_MOE_BASE<T, AVXVNNI256_INT4_MOE_TP<T>> {
  using Base = AVX2_MOE_BASE<T, AVXVNNI256_INT4_MOE_TP<T>>;
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

  inline void write_weights(std::filesystem::path prefix, std::string mat_class, typename T::BufferB* buffer,
                            int expert_idx, size_t quant_size, size_t scale_size) {
    std::vector<uint8_t> packed(T::BufferB::packed_weight_bytes((size_t)buffer->n, (size_t)buffer->k));
    buffer->serialize_packed(packed.data(), 0, buffer->n);

    std::ofstream of(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                               std::to_string(quant_size) + "Byte_quant_.kt"),
                     std::ios::binary);
    if (!of.is_open()) {
      throw std::runtime_error("failed to open quant output file");
    }
    of.write((char*)packed.data(), (std::streamsize)packed.size());
    of.close();

    of.open(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" + std::to_string(scale_size) +
                      "Byte_scale_.kt"),
            std::ios::binary);
    if (!of.is_open()) {
      throw std::runtime_error("failed to open scale output file");
    }
    of.write((char*)buffer->d, scale_size);
  }

  inline void read_weights(std::filesystem::path prefix, std::string mat_class, typename T::BufferB* buffer,
                           int expert_idx, size_t quant_size, size_t scale_size) {
    std::vector<uint8_t> packed(quant_size);
    std::ifstream f(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                              std::to_string(quant_size) + "Byte_quant_.kt"),
                    std::ios::binary);
    if (!f.is_open()) {
      throw std::runtime_error("failed to open quant input file");
    }
    f.read((char*)packed.data(), (std::streamsize)packed.size());
    f.close();

    f.open(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" + std::to_string(scale_size) +
                     "Byte_scale_.kt"),
           std::ios::binary);
    if (!f.is_open()) {
      throw std::runtime_error("failed to open scale input file");
    }
    f.read((char*)buffer->d, scale_size);
    buffer->from_packed_mat(packed.data(), buffer->d, 0, 1);
  }

 public:
  using typename Base::input_t;
  using typename Base::output_t;

  AVXVNNI256_INT4_MOE_TP() = default;
  AVXVNNI256_INT4_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
#if defined(__GNUC__) || defined(__clang__)
    if (!__builtin_cpu_supports("avxvnni")) {
      throw std::runtime_error("AVX-VNNI-256 INT4 backend requires CPU support for avx_vnni");
    }
#endif
    printf("Created AVXVNNI256_INT4_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
  }

  ~AVXVNNI256_INT4_MOE_TP() = default;

  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const { return T::BufferB::required_size(n, k); }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, data);
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
    avxvnni_int4::gemm_int4_avxvnni256(m, config_.intermediate_size, config_.hidden_size, *ba, *bb, *bc, ith, nth);
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    (void)qlen;
    int m = m_local_num_[expert_idx];
    avxvnni_int4::gemm_int4_avxvnni256(m, config_.hidden_size, config_.intermediate_size, *down_ba_[expert_idx],
                                       *down_bb_[expert_idx], *down_bc_[expert_idx], ith, nth);
  }

  void load_weights() {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;

    if (!config_.gate_projs.empty()) {
      pool->do_work_stealing_job(
          config_.expert_num, nullptr,
          [this, physical_to_logical_map](int expert_id) {
            uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_id);
            gate_bb_[expert_id]->from_packed_mat((const uint8_t*)config_.gate_projs[tp_part_idx][logical_expert_id],
                                                 (const float*)config_.gate_scales[tp_part_idx][logical_expert_id], 0, 1);
            up_bb_[expert_id]->from_packed_mat((const uint8_t*)config_.up_projs[tp_part_idx][logical_expert_id],
                                               (const float*)config_.up_scales[tp_part_idx][logical_expert_id], 0, 1);
            down_bb_[expert_id]->from_packed_mat((const uint8_t*)config_.down_projs[tp_part_idx][logical_expert_id],
                                                 (const float*)config_.down_scales[tp_part_idx][logical_expert_id], 0, 1);
          },
          nullptr);
      return;
    }

    std::filesystem::path prefix = config_.path;
    prefix = prefix / ("_layer_" + std::to_string(config_.layer_idx)) / ("_numa_" + std::to_string(tp_part_idx));

    const size_t gate_up_quant = (size_t)config_.intermediate_size * config_.hidden_size / 2;
    const size_t down_quant = (size_t)config_.hidden_size * config_.intermediate_size / 2;

    if (config_.load) {
      for (int expert_id = 0; expert_id < config_.expert_num; ++expert_id) {
        uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_id);
        read_weights(prefix, "_gate_", gate_bb_[expert_id].get(), (int)logical_expert_id, gate_up_quant,
                     (size_t)config_.intermediate_size * sizeof(float));
        read_weights(prefix, "_up_", up_bb_[expert_id].get(), (int)logical_expert_id, gate_up_quant,
                     (size_t)config_.intermediate_size * sizeof(float));
        read_weights(prefix, "_down_", down_bb_[expert_id].get(), (int)logical_expert_id, down_quant,
                     (size_t)config_.hidden_size * sizeof(float));
      }
      return;
    }

    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          int expert_id = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_id);
          int ith = task_id % nth;
          gate_bb_[logical_expert_id]->from_mat(
              (const ggml_bf16_t*)config_.gate_proj + (size_t)logical_expert_id * config_.intermediate_size * config_.hidden_size,
              ith, nth);
          up_bb_[logical_expert_id]->from_mat(
              (const ggml_bf16_t*)config_.up_proj + (size_t)logical_expert_id * config_.intermediate_size * config_.hidden_size,
              ith, nth);
        },
        nullptr);

    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          int expert_id = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_id);
          int ith = task_id % nth;
          down_bb_[logical_expert_id]->from_mat(
              (const ggml_bf16_t*)config_.down_proj + (size_t)logical_expert_id * config_.hidden_size * config_.intermediate_size,
              ith, nth);
        },
        nullptr);

    if (config_.save) {
      std::filesystem::create_directories(prefix);
      pool->do_work_stealing_job(
          config_.expert_num * 3, nullptr,
          [this, physical_to_logical_map, prefix, gate_up_quant, down_quant](int task_id) {
            int expert_id = task_id / 3;
            uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_id);
            if ((task_id % 3) == 0) {
              write_weights(prefix, "_up_", up_bb_[logical_expert_id].get(), (int)logical_expert_id, gate_up_quant,
                            (size_t)config_.intermediate_size * sizeof(float));
            } else if ((task_id % 3) == 1) {
              write_weights(prefix, "_gate_", gate_bb_[logical_expert_id].get(), (int)logical_expert_id, gate_up_quant,
                            (size_t)config_.intermediate_size * sizeof(float));
            } else {
              write_weights(prefix, "_down_", down_bb_[logical_expert_id].get(), (int)logical_expert_id, down_quant,
                            (size_t)config_.hidden_size * sizeof(float));
            }
          },
          nullptr);
    }
  }

  void write_weights_to_buffer(int gpu_tp_count, [[maybe_unused]] int cpu_tp_count, int expert_id,
                               const GeneralMOEConfig& full_config, const std::vector<uintptr_t>& w13_weight_ptrs,
                               const std::vector<uintptr_t>& w13_scale_ptrs, const std::vector<uintptr_t>& w2_weight_ptrs,
                               const std::vector<uintptr_t>& w2_scale_ptrs) const {
    (void)gpu_tp_count;
    (void)expert_id;
    (void)full_config;
    (void)w13_weight_ptrs;
    (void)w13_scale_ptrs;
    (void)w2_weight_ptrs;
    (void)w2_scale_ptrs;
    throw std::runtime_error("AVX-VNNI INT4 write_weights_to_buffer not yet implemented");
  }
};

template <typename K>
class TP_MOE<AVXVNNI256_INT4_MOE_TP<K>> : public TP_MOE<AVX2_MOE_BASE<K, AVXVNNI256_INT4_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AVX2_MOE_BASE<K, AVXVNNI256_INT4_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    if (!config.gate_projs.empty()) {
      DO_TPS_LOAD_WEIGHTS(pool);
      this->weights_loaded = true;
      return;
    }

    if (config.gate_proj != nullptr) {
      for (int i = 0; i < tp_count; ++i) {
        auto& tpc = tps[i]->config_;
        size_t gate_up_elcount = (size_t)tpc.intermediate_size * tpc.hidden_size;
        tpc.gate_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        tpc.up_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        tpc.down_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        if (!tpc.load) {
          pool->get_subpool(i)->do_work_stealing_job(
              tpc.expert_num, nullptr,
              [&](int expert_id_) {
                size_t expert_id = expert_map(physical_to_logical_map, expert_id_);
                std::memcpy((ggml_bf16_t*)tpc.gate_proj + expert_id * gate_up_elcount,
                            (ggml_bf16_t*)config.gate_proj + expert_id * config.intermediate_size * config.hidden_size +
                                i * gate_up_elcount,
                            sizeof(ggml_bf16_t) * gate_up_elcount);
                std::memcpy((ggml_bf16_t*)tpc.up_proj + expert_id * gate_up_elcount,
                            (ggml_bf16_t*)config.up_proj + expert_id * config.intermediate_size * config.hidden_size +
                                i * gate_up_elcount,
                            sizeof(ggml_bf16_t) * gate_up_elcount);
                for (size_t col = 0; col < (size_t)config.hidden_size; ++col) {
                  std::memcpy((ggml_bf16_t*)tpc.down_proj + expert_id * tpc.hidden_size * tpc.intermediate_size +
                                  col * tpc.intermediate_size,
                              (ggml_bf16_t*)config.down_proj + expert_id * config.intermediate_size * config.hidden_size +
                                  col * config.intermediate_size + i * tpc.intermediate_size,
                              sizeof(ggml_bf16_t) * tpc.intermediate_size);
                }
              },
              nullptr);
        }
      }

      DO_TPS_LOAD_WEIGHTS(pool);

      for (int i = 0; i < tp_count; ++i) {
        auto& tpc = tps[i]->config_;
        delete[] (ggml_bf16_t*)tpc.gate_proj;
        delete[] (ggml_bf16_t*)tpc.up_proj;
        delete[] (ggml_bf16_t*)tpc.down_proj;
      }

      this->weights_loaded = true;
      return;
    }

    if (config.path != "") {
      DO_TPS_LOAD_WEIGHTS(pool);
      this->weights_loaded = true;
      return;
    }

    throw std::runtime_error("no weight source");
  }
};

#endif  // CPUINFER_OPERATOR_AVX2_INT4_AVXVNNI_MOE_H
