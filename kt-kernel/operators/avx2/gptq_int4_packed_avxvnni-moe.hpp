/**
 * AVX-VNNI-256 GPTQ INT4 MoE backend with output-major packed weights.
 *
 * The persistent layout matches the SYCL backend:
 *   qweight [N, K/8] uint32_t
 *   scales  [N, K/group_size] float
 *
 * INT4 nibbles are expanded to signed INT8 in registers immediately before
 * dpbusd. This avoids the persistent [N, K] INT8 copy used by the legacy VNNI
 * backend and is the CPU building block for shared CPU/iGPU expert weights.
 */
#ifndef CPUINFER_OPERATOR_AVX2_GPTQ_INT4_PACKED_AVXVNNI_MOE_H
#define CPUINFER_OPERATOR_AVX2_GPTQ_INT4_PACKED_AVXVNNI_MOE_H

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "gptq_int4_avxvnni-moe.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define KT_PACKED_AVXVNNI256_TARGET __attribute__((target("avx2,avxvnni,fma")))
#else
#define KT_PACKED_AVXVNNI256_TARGET
#endif

namespace avxvnni_packed {

static constexpr int MAX_SUPPORTED_GROUP_SIZE = 256;

struct GemmKernelAVXVNNI256PackedGPTQInt4 {
  using dt = ggml_bf16_t;
  using output_t = float;
  using BufferA = avxvnni::GemmKernelAVXVNNI256GPTQInt4::BufferA;
  using BufferC = avxvnni::GemmKernelAVXVNNI256GPTQInt4::BufferC;

  static constexpr int M_STEP = 1;
  static constexpr int N_STEP = 8;
  static constexpr int K_STEP = 32;
  static constexpr int N_BLOCK = 64;
  static constexpr int K_BLOCK = 128;
  static constexpr double ELEMENT_SIZE = 0.5;

  static void config() {}
  static int recommended_nth(int n) { return std::max(1, n / N_BLOCK); }
  static std::pair<int, int> split_range_n(int n, int ith, int nth) { return avx2::split_range(n, ith, nth); }

  struct BufferB {
    uint32_t* qweight = nullptr;     // [N, K/8]
    float* scales = nullptr;         // [N, K/group_size]
    int16_t* weight_sums = nullptr;  // [N, K/group_size]
    int n = 0;
    int k = 0;
    int group_size = 128;
    int num_groups = 0;
    int k_packed = 0;

    BufferB() = default;
    BufferB(size_t n_, size_t k_, int group_size_, void* pointer)
        : n(static_cast<int>(n_)), k(static_cast<int>(k_)), group_size(group_size_) {
      if (group_size <= 0 || (group_size % 32) != 0 || group_size > MAX_SUPPORTED_GROUP_SIZE) {
        throw std::runtime_error("Packed AVX-VNNI GPTQ INT4 requires group_size to be a multiple of 32 up to 256");
      }
      if ((k % 32) != 0 || (k % group_size) != 0) {
        throw std::runtime_error("Packed AVX-VNNI GPTQ INT4 requires K divisible by 32 and group_size");
      }
      k_packed = k / 8;
      num_groups = k / group_size;
      if (pointer != nullptr) bind_view(pointer);
    }

    static size_t required_size(size_t n, size_t k, int group_size) {
      const size_t k_packed = k / 8;
      const size_t num_groups = k / group_size;
      return n * k_packed * sizeof(uint32_t) + n * num_groups * sizeof(float) + n * num_groups * sizeof(int16_t);
    }

    void bind_view(void* pointer) {
      qweight = static_cast<uint32_t*>(pointer);
      scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(pointer) +
                                        static_cast<size_t>(n) * k_packed * sizeof(uint32_t));
      weight_sums = reinterpret_cast<int16_t*>(reinterpret_cast<uint8_t*>(scales) +
                                               static_cast<size_t>(n) * num_groups * sizeof(float));
    }

    void bind_view(uint32_t* qweight_pointer, float* scale_pointer, int16_t* weight_sum_pointer) {
      qweight = qweight_pointer;
      scales = scale_pointer;
      weight_sums = weight_sum_pointer;
    }

    void from_mat(const uint32_t* source_qweight, const float* source_scales, int ith, int nth) {
      auto [n_begin, n_end] = avx2::split_range(n, ith, nth);
      const int packed_per_group = group_size / 8;
      for (int output = n_begin; output < n_end; ++output) {
        uint32_t* destination = qweight + static_cast<size_t>(output) * k_packed;
        for (int packed_k = 0; packed_k < k_packed; ++packed_k) {
          destination[packed_k] = source_qweight[static_cast<size_t>(packed_k) * n + output];
        }

        for (int group = 0; group < num_groups; ++group) {
          const size_t output_group = static_cast<size_t>(output) * num_groups + group;
          scales[output_group] = source_scales[static_cast<size_t>(group) * n + output];
          int sum = 0;
          for (int packed_offset = 0; packed_offset < packed_per_group; ++packed_offset) {
            const uint32_t word = destination[group * packed_per_group + packed_offset];
            for (int nibble = 0; nibble < 8; ++nibble) {
              sum += static_cast<int>((word >> (nibble * 4)) & 0x0fu) - 8;
            }
          }
          weight_sums[output_group] = static_cast<int16_t>(sum);
        }
      }
    }
  };
};

KT_PACKED_AVXVNNI256_TARGET
static inline __m256i unpack_32_int4(const uint32_t* source) {
  const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source));
  const __m128i mask = _mm_set1_epi8(0x0f);
  const __m128i offset = _mm_set1_epi8(8);
  const __m128i low = _mm_and_si128(packed, mask);
  const __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
  const __m128i values_0_15 = _mm_sub_epi8(_mm_unpacklo_epi8(low, high), offset);
  const __m128i values_16_31 = _mm_sub_epi8(_mm_unpackhi_epi8(low, high), offset);
  return _mm256_set_m128i(values_16_31, values_0_15);
}

KT_PACKED_AVXVNNI256_TARGET
static inline void gemm_gptq_sym_int4_packed_avxvnni256(int m, int n, int k,
                                                        GemmKernelAVXVNNI256PackedGPTQInt4::BufferA& activation,
                                                        GemmKernelAVXVNNI256PackedGPTQInt4::BufferB& weight,
                                                        GemmKernelAVXVNNI256PackedGPTQInt4::BufferC& output, int ith,
                                                        int nth) {
  (void)k;
  auto [n_begin, n_end] = avx2::split_range(n, ith, nth);
  const int group_size = weight.group_size;
  const int num_groups = weight.num_groups;
  const int packed_per_group = group_size / 8;
  alignas(32) std::array<uint8_t, MAX_SUPPORTED_GROUP_SIZE> activation_u8{};

  for (int row = 0; row < m; ++row) {
    const ggml_bf16_t* activation_row = activation.data + static_cast<size_t>(row) * activation.k;
    float* output_row = output.data + static_cast<size_t>(row) * n;
    std::fill(output_row + n_begin, output_row + n_end, 0.0f);

    for (int group = 0; group < num_groups; ++group) {
      const int k_base = group * group_size;
      const float activation_scale =
          avxvnni::quantize_activation_group_u8(activation_row + k_base, group_size, activation_u8.data());
      if (activation_scale == 0.0f) continue;

      for (int column = n_begin; column < n_end; ++column) {
        const uint32_t* packed_weight = weight.qweight + static_cast<size_t>(column) * weight.k_packed +
                                        static_cast<size_t>(group) * packed_per_group;
        __m256i accumulator = _mm256_setzero_si256();
        for (int packed_offset = 0; packed_offset < packed_per_group; packed_offset += 4) {
          const __m256i activation_vector =
              _mm256_load_si256(reinterpret_cast<const __m256i*>(activation_u8.data() + packed_offset * 8));
          const __m256i weight_vector = unpack_32_int4(packed_weight + packed_offset);
          accumulator = _mm256_dpbusd_avx_epi32(accumulator, activation_vector, weight_vector);
        }

        const size_t output_group = static_cast<size_t>(column) * num_groups + group;
        const int dot =
            avxvnni::hsum_epi32_avx2(accumulator) - 128 * static_cast<int>(weight.weight_sums[output_group]);
        output_row[column] += static_cast<float>(dot) * activation_scale * weight.scales[output_group];
      }
    }
  }
}

}  // namespace avxvnni_packed

template <class T = avxvnni_packed::GemmKernelAVXVNNI256PackedGPTQInt4>
class AVXVNNI256_PACKED_GPTQ_INT4_MOE_TP : public AVX2_MOE_BASE<T, AVXVNNI256_PACKED_GPTQ_INT4_MOE_TP<T>> {
  using Base = AVX2_MOE_BASE<T, AVXVNNI256_PACKED_GPTQ_INT4_MOE_TP<T>>;
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

  AVXVNNI256_PACKED_GPTQ_INT4_MOE_TP() = default;
  AVXVNNI256_PACKED_GPTQ_INT4_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  GeneralMOEConfig& mutable_config() { return config_; }
  const std::shared_ptr<typename T::BufferB>& gate_weight(int expert) const { return gate_bb_[expert]; }
  const std::shared_ptr<typename T::BufferB>& up_weight(int expert) const { return up_bb_[expert]; }
  const std::shared_ptr<typename T::BufferB>& down_weight(int expert) const { return down_bb_[expert]; }

  void derived_init() {
#if defined(__GNUC__) || defined(__clang__)
    if (!__builtin_cpu_supports("avxvnni")) {
      throw std::runtime_error("Packed AVX-VNNI-256 GPTQ INT4 backend requires avx_vnni");
    }
#endif
    const auto& quant = config_.quant_config;
    if (quant.group_size <= 0 || (quant.group_size % 32) != 0 ||
        quant.group_size > avxvnni_packed::MAX_SUPPORTED_GROUP_SIZE) {
      throw std::runtime_error("Packed AVX-VNNI-256 GPTQ INT4 requires group_size 32, 64, 96, ..., 256");
    }
    printf("Created AVXVNNI256_PACKED_GPTQ_INT4_MOE_TP %d at numa %d (group_size=%d)\n", tp_part_idx,
           numa_node_of_cpu(sched_getcpu()), quant.group_size);
  }

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

  void do_gate_up_gemm(bool do_up, int expert, int ith, int nth, int qlen) {
    (void)qlen;
    auto& weight = do_up ? up_bb_[expert] : gate_bb_[expert];
    auto& output = do_up ? up_bc_[expert] : gate_bc_[expert];
    avxvnni_packed::gemm_gptq_sym_int4_packed_avxvnni256(m_local_num_[expert], config_.intermediate_size,
                                                         config_.hidden_size, *gate_up_ba_[expert], *weight, *output,
                                                         ith, nth);
  }

  void do_down_gemm(int expert, int ith, int nth, int qlen) {
    (void)qlen;
    avxvnni_packed::gemm_gptq_sym_int4_packed_avxvnni256(m_local_num_[expert], config_.hidden_size,
                                                         config_.intermediate_size, *down_ba_[expert],
                                                         *down_bb_[expert], *down_bc_[expert], ith, nth);
  }

  void load_weights() {
    const int group_size = config_.quant_config.group_size;
    const uint64_t* physical_to_logical = static_cast<const uint64_t*>(config_.physical_to_logical_map);
    const bool per_expert = !config_.gate_projs.empty();
    if (!per_expert && (config_.gate_proj == nullptr || config_.gate_scale == nullptr)) {
      throw std::runtime_error("Packed AVX-VNNI GPTQ INT4 requires GPTQ qweight and scale sources");
    }

    auto pool = config_.pool->get_subpool(tp_part_idx);
    const size_t gate_up_qweight_elements = static_cast<size_t>(config_.hidden_size / 8) * config_.intermediate_size;
    const size_t gate_up_scale_elements =
        static_cast<size_t>(config_.hidden_size / group_size) * config_.intermediate_size;
    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical, per_expert, gate_up_qweight_elements, gate_up_scale_elements](int task) {
          const int expert = task / nth;
          const size_t logical = expert_map(physical_to_logical, expert);
          const int ith = task % nth;
          const uint32_t* gate_qweight =
              per_expert ? static_cast<const uint32_t*>(config_.gate_projs[0][logical])
                         : static_cast<const uint32_t*>(config_.gate_proj) + logical * gate_up_qweight_elements;
          const uint32_t* up_qweight =
              per_expert ? static_cast<const uint32_t*>(config_.up_projs[0][logical])
                         : static_cast<const uint32_t*>(config_.up_proj) + logical * gate_up_qweight_elements;
          const float* gate_scale =
              per_expert ? static_cast<const float*>(config_.gate_scales[0][logical])
                         : static_cast<const float*>(config_.gate_scale) + logical * gate_up_scale_elements;
          const float* up_scale = per_expert
                                      ? static_cast<const float*>(config_.up_scales[0][logical])
                                      : static_cast<const float*>(config_.up_scale) + logical * gate_up_scale_elements;
          gate_bb_[expert]->from_mat(gate_qweight, gate_scale, ith, nth);
          up_bb_[expert]->from_mat(up_qweight, up_scale, ith, nth);
        },
        nullptr);

    const size_t down_qweight_elements = static_cast<size_t>(config_.intermediate_size / 8) * config_.hidden_size;
    const size_t down_scale_elements =
        static_cast<size_t>(config_.intermediate_size / group_size) * config_.hidden_size;
    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical, per_expert, down_qweight_elements, down_scale_elements](int task) {
          const int expert = task / nth;
          const size_t logical = expert_map(physical_to_logical, expert);
          const int ith = task % nth;
          const uint32_t* down_qweight =
              per_expert ? static_cast<const uint32_t*>(config_.down_projs[0][logical])
                         : static_cast<const uint32_t*>(config_.down_proj) + logical * down_qweight_elements;
          const float* down_scale = per_expert
                                        ? static_cast<const float*>(config_.down_scales[0][logical])
                                        : static_cast<const float*>(config_.down_scale) + logical * down_scale_elements;
          down_bb_[expert]->from_mat(down_qweight, down_scale, ith, nth);
        },
        nullptr);
  }

  void write_weights_to_buffer(int, int, int, const GeneralMOEConfig&, const std::vector<uintptr_t>&,
                               const std::vector<uintptr_t>&, const std::vector<uintptr_t>&,
                               const std::vector<uintptr_t>&) const {
    throw std::runtime_error("Packed AVX-VNNI GPTQ INT4 does not support write_weights_to_buffer");
  }
};

template <typename Kernel>
class TP_MOE<AVXVNNI256_PACKED_GPTQ_INT4_MOE_TP<Kernel>>
    : public TP_MOE<AVX2_MOE_BASE<Kernel, AVXVNNI256_PACKED_GPTQ_INT4_MOE_TP<Kernel>>> {
 public:
  using Base = TP_MOE<AVX2_MOE_BASE<Kernel, AVXVNNI256_PACKED_GPTQ_INT4_MOE_TP<Kernel>>>;
  using Base::Base;

  void load_weights() override {
    if (this->tp_count != 1) {
      throw std::runtime_error("Packed AVX-VNNI GPTQ INT4 currently supports tensor_parallel_size=1 only");
    }
    this->tps[0]->config_.physical_to_logical_map = this->config.physical_to_logical_map;
    this->tps[0]->load_weights();
    this->weights_loaded = true;
  }

  void write_weight_scale_to_buffer(int, int, const std::vector<uintptr_t>&, const std::vector<uintptr_t>&,
                                    const std::vector<uintptr_t>&, const std::vector<uintptr_t>&) {
    throw std::runtime_error("Packed AVX-VNNI GPTQ INT4 does not support write_weight_scale_to_buffer");
  }
};

#undef KT_PACKED_AVXVNNI256_TARGET

#endif  // CPUINFER_OPERATOR_AVX2_GPTQ_INT4_PACKED_AVXVNNI_MOE_H
