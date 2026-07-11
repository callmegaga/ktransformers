/**
 * @Description  : SYCL FP8 MoE operator for integrated/discrete GPU experiments
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * FP8 E4M3 weights with 128x128 block-wise float32 scales.
 * This backend reuses the AVX2 MoE base for routing/activation/merge and
 * offloads the gate/up/down GEMMs to a SYCL device.
 **/
#ifndef CPUINFER_OPERATOR_SYCL_FP8_MOE_H
#define CPUINFER_OPERATOR_SYCL_FP8_MOE_H

#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../avx2/moe_base.hpp"

namespace sycl_fp8 {

inline int div_up(int a, int b) { return (a + b - 1) / b; }

inline bool trace_alloc() {
  const char* env = std::getenv("KT_SYCL_TRACE_ALLOC");
  return env != nullptr && env[0] != '\0' && std::string(env) != "0";
}

inline size_t env_size_mb(const char* name, size_t fallback_mb) {
  const char* env = std::getenv(name);
  if (env == nullptr || env[0] == '\0') {
    return fallback_mb;
  }
  char* end = nullptr;
  unsigned long long value = std::strtoull(env, &end, 10);
  if (end == env) {
    return fallback_mb;
  }
  return static_cast<size_t>(value);
}

inline std::atomic<size_t>& current_usm_bytes() {
  static std::atomic<size_t> bytes{0};
  return bytes;
}

inline size_t max_usm_bytes() {
  static const size_t bytes = env_size_mb("KT_SYCL_MAX_USM_MB", 0) * 1024ULL * 1024ULL;
  return bytes;
}

inline size_t current_usm_mb() {
  return current_usm_bytes().load(std::memory_order_relaxed) / (1024ULL * 1024ULL);
}

inline sycl::queue& queue() {
  static sycl::queue q([] {
    auto async_handler = [](sycl::exception_list exceptions) {
      for (const auto& e : exceptions) {
        try {
          std::rethrow_exception(e);
        } catch (const sycl::exception& ex) {
          std::fprintf(stderr, "SYCL asynchronous exception: %s\n", ex.what());
        }
      }
    };
    const char* filter = std::getenv("KT_SYCL_DEVICE_FILTER");
    try {
      if (filter != nullptr && filter[0] != '\0') {
        return sycl::queue(sycl::ext::oneapi::filter_selector(filter), async_handler);
      }
      return sycl::queue(sycl::gpu_selector_v, async_handler);
    } catch (const sycl::exception& ex) {
      throw std::runtime_error(
          std::string("SYCL FP8 MoE failed to create a SYCL queue. By default it selects a GPU device. ")
          + "Set KT_SYCL_DEVICE_FILTER=level_zero:gpu to force Intel iGPU, or KT_SYCL_DEVICE_FILTER=opencl:cpu "
          + "for correctness testing. If the GPU is not listed by sycl-ls, check /dev/dri/renderD* permissions "
          + "(the user usually needs the render group). Original SYCL error: " + ex.what());
    }
  }());
  return q;
}

template <typename T>
inline T* malloc_shared_elems(size_t elems, const char* name, size_t* bytes_out = nullptr) {
  elems = std::max<size_t>(elems, 1);
  const size_t bytes = elems * sizeof(T);
  const size_t max_bytes = max_usm_bytes();
  const size_t before = current_usm_bytes().load(std::memory_order_relaxed);
  if (max_bytes != 0 && before + bytes > max_bytes) {
    throw std::runtime_error(std::string("SYCL USM allocation would exceed KT_SYCL_MAX_USM_MB while allocating ")
                             + name + ": requested=" + std::to_string(bytes / (1024ULL * 1024ULL))
                             + " MiB, current=" + std::to_string(before / (1024ULL * 1024ULL))
                             + " MiB, limit=" + std::to_string(max_bytes / (1024ULL * 1024ULL)) + " MiB");
  }
  T* ptr = sycl::malloc_shared<T>(elems, queue());
  if (ptr == nullptr) {
    throw std::runtime_error(std::string("SYCL malloc_shared failed for ") + name);
  }
  current_usm_bytes().fetch_add(bytes, std::memory_order_relaxed);
  if (bytes_out != nullptr) {
    *bytes_out = bytes;
  }
  if (trace_alloc()) {
    std::printf("[SYCL_FP8] malloc_shared %-28s %8zu MiB, total=%zu MiB\n", name,
                static_cast<size_t>(bytes / (1024ULL * 1024ULL)), current_usm_mb());
    std::fflush(stdout);
  }
  return ptr;
}

inline void free_usm(void* ptr, size_t bytes = 0) {
  if (ptr != nullptr) {
    sycl::free(ptr, queue());
    if (bytes != 0) {
      current_usm_bytes().fetch_sub(bytes, std::memory_order_relaxed);
      if (trace_alloc()) {
        std::printf("[SYCL_FP8] free_usm %8zu MiB, total=%zu MiB\n",
                    static_cast<size_t>(bytes / (1024ULL * 1024ULL)), current_usm_mb());
        std::fflush(stdout);
      }
    }
  }
}

static inline float bf16_bits_to_fp32(uint16_t v) {
  const uint32_t bits = static_cast<uint32_t>(v) << 16;
  return sycl::bit_cast<float>(bits);
}

static inline float fp8_e4m3_to_fp32(uint8_t raw) {
  const int sign = (raw >> 7) & 1;
  const int exp = (raw >> 3) & 0xF;
  const int man = raw & 0x7;

  float val = 0.0f;
  if (exp == 0 && man == 0) {
    val = 0.0f;
  } else if (exp == 0) {
    val = sycl::ldexp(static_cast<float>(man) / 8.0f, -6);
  } else if (exp == 15 && man == 7) {
    val = 0.0f;
  } else {
    val = sycl::ldexp(1.0f + static_cast<float>(man) / 8.0f, exp - 7);
  }
  return sign ? -val : val;
}

struct GemmKernelSYCLFP8 {
  using dt = ggml_bf16_t;
  using output_t = float;
  static constexpr int M_STEP = 1;
  static constexpr int N_STEP = 1;
  static constexpr int K_STEP = 1;
  static constexpr int BLOCK_SIZE = 128;
  static constexpr int N_BLOCK = 128;
  static constexpr int K_BLOCK = 128;
  static constexpr double ELEMENT_SIZE = 1.0;

  static void config() {
    static std::once_flag once;
    std::call_once(once, [] {
      auto& q = queue();
      const auto device = q.get_device();
      const bool has_shared = device.get_info<sycl::info::device::usm_shared_allocations>();
      if (!has_shared) {
        throw std::runtime_error("SYCL FP8 MoE requires a device with USM shared allocation support.");
      }
      const auto name = device.get_info<sycl::info::device::name>();
      const auto backend = device.get_backend();
      std::printf("Created SYCL_FP8_MOE on device: %s (backend=%d)\n", name.c_str(), static_cast<int>(backend));
    });
  }

  static int recommended_nth(int) {
    return 1;
  }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    return avx2::split_range(n, ith, nth);
  }

  struct BufferA {
    uint16_t* data = nullptr;
    size_t max_m = 0;
    size_t k = 0;
    size_t capacity_m = 0;
    size_t bytes = 0;

    BufferA() = default;
    BufferA(size_t m, size_t k_, void*) : max_m(m), k(k_) {}
    BufferA(const BufferA&) = delete;
    BufferA& operator=(const BufferA&) = delete;
    ~BufferA() { free_usm(data, bytes); }

    static size_t required_size(size_t, size_t) {
      return 1;
    }

    void ensure_capacity(size_t m) {
      if (m <= capacity_m) {
        return;
      }
      free_usm(data, bytes);
      data = nullptr;
      bytes = 0;
      capacity_m = 0;
      data = malloc_shared_elems<uint16_t>(m * k, "SYCL FP8 BufferA", &bytes);
      capacity_m = m;
    }

    void set_data(void*) {
      ensure_capacity(max_m);
    }

    void from_mat(int m, const ggml_bf16_t* src, int ith, int nth) {
      ensure_capacity(static_cast<size_t>(m));
      if (static_cast<size_t>(m) > capacity_m) {
        throw std::runtime_error("SYCL FP8 BufferA capacity exceeded");
      }
      if (ith == 0 && nth == 1) {
        std::memcpy(data, src, (size_t)m * k * sizeof(uint16_t));
      } else {
        auto [m_start, m_end] = avx2::split_range(m, ith, nth);
        std::memcpy(data + (size_t)m_start * k, src + (size_t)m_start * k,
                    (size_t)(m_end - m_start) * k * sizeof(uint16_t));
      }
    }
  };

  struct BufferB {
    uint8_t* b = nullptr;
    float* d = nullptr;
    size_t n = 0;
    size_t k = 0;
    int block_size = BLOCK_SIZE;
    size_t b_bytes = 0;
    size_t d_bytes = 0;

    BufferB() = default;
    BufferB(size_t n_, size_t k_, int bs, void*) : n(n_), k(k_), block_size(bs) {
    }
    BufferB(const BufferB&) = delete;
    BufferB& operator=(const BufferB&) = delete;
    ~BufferB() {
      free_usm(b, b_bytes);
      free_usm(d, d_bytes);
    }

    static size_t required_size(size_t, size_t, int) {
      return 1;
    }

    void ensure_allocated() {
      if (b != nullptr && d != nullptr) {
        return;
      }
      if (b == nullptr) {
        b = malloc_shared_elems<uint8_t>(n * k, "SYCL FP8 BufferB weights", &b_bytes);
      }
      if (d == nullptr) {
        const size_t scale_elems = (size_t)div_up((int)n, block_size) * div_up((int)k, block_size);
        d = malloc_shared_elems<float>(scale_elems, "SYCL FP8 BufferB scales", &d_bytes);
      }
    }

    void from_mat(const uint8_t* src_weights, const float* src_scales, int ith, int nth) {
      ensure_allocated();
      auto [n_start, n_end] = avx2::split_range((int)n, ith, nth);
      std::memcpy(b + (size_t)n_start * k, src_weights + (size_t)n_start * k,
                  (size_t)(n_end - n_start) * k);

      const int n_blocks_k = div_up((int)k, block_size);
      const int nb_start = n_start / block_size;
      const int nb_end = div_up(n_end, block_size);
      std::memcpy(d + (size_t)nb_start * n_blocks_k, src_scales + (size_t)nb_start * n_blocks_k,
                  (size_t)(nb_end - nb_start) * n_blocks_k * sizeof(float));
    }
  };

  struct BufferC {
    float* data = nullptr;
    size_t max_m = 0;
    size_t n = 0;
    size_t capacity_m = 0;
    size_t bytes = 0;

    BufferC() = default;
    BufferC(size_t m, size_t n_, void*) : max_m(m), n(n_) {}
    BufferC(const BufferC&) = delete;
    BufferC& operator=(const BufferC&) = delete;
    ~BufferC() { free_usm(data, bytes); }

    static size_t required_size(size_t, size_t) {
      return 1;
    }

    void ensure_capacity(size_t m) {
      if (m <= capacity_m) {
        return;
      }
      free_usm(data, bytes);
      data = nullptr;
      bytes = 0;
      capacity_m = 0;
      data = malloc_shared_elems<float>(m * n, "SYCL FP8 BufferC", &bytes);
      capacity_m = m;
    }

    void set_data(void*) {
      ensure_capacity(max_m);
    }

    void to_mat(int m, ggml_bf16_t* dst, int ith, int nth) {
      ensure_capacity(static_cast<size_t>(m));
      if (static_cast<size_t>(m) > capacity_m) {
        throw std::runtime_error("SYCL FP8 BufferC capacity exceeded");
      }
      auto [n_start, n_end] = avx2::split_range((int)n, ith, nth);
      for (int mi = 0; mi < m; mi++) {
        const float* src_row = data + (size_t)mi * n;
        ggml_bf16_t* dst_row = dst + (size_t)mi * n;
        for (int j = n_start; j < n_end; j++) {
          dst_row[j] = GGML_FP32_TO_BF16(src_row[j]);
        }
      }
    }
  };
};

static inline void gemm_fp8_sycl(int m, int n, int k, GemmKernelSYCLFP8::BufferA& a,
                                 GemmKernelSYCLFP8::BufferB& b, GemmKernelSYCLFP8::BufferC& c, int ith, int nth) {
  if (m <= 0 || n <= 0 || k <= 0) {
    return;
  }
  auto [n_start, n_end] = avx2::split_range(n, ith, nth);
  if (n_start >= n_end) {
    return;
  }

  auto& q = queue();
  const int block_size = b.block_size;
  const int n_blocks_k = div_up(k, block_size);
  const int n_len = n_end - n_start;

  auto* a_data = a.data;
  auto* b_data = b.b;
  auto* scales = b.d;
  auto* c_data = c.data;
  const size_t a_ld = a.k;
  const size_t b_ld = b.k;
  const size_t c_ld = c.n;

  sycl::event event = q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<2>((size_t)m, (size_t)n_len), [=](sycl::id<2> id) {
      const int mi = static_cast<int>(id[0]);
      const int ni = n_start + static_cast<int>(id[1]);
      const uint16_t* a_row = a_data + (size_t)mi * a_ld;
      const uint8_t* b_row = b_data + (size_t)ni * b_ld;
      const int n_block_idx = ni / block_size;

      float sum = 0.0f;
      for (int kb = 0; kb < k; kb += block_size) {
        const int k_len = (kb + block_size <= k) ? block_size : (k - kb);
        const int k_block_idx = kb / block_size;
        const float scale = scales[(size_t)n_block_idx * n_blocks_k + k_block_idx];
        float block_sum = 0.0f;
        for (int ki = 0; ki < k_len; ki++) {
          block_sum += bf16_bits_to_fp32(a_row[kb + ki]) * fp8_e4m3_to_fp32(b_row[kb + ki]);
        }
        sum += block_sum * scale;
      }

      c_data[(size_t)mi * c_ld + ni] = sum;
    });
  });
  event.wait_and_throw();
}

}  // namespace sycl_fp8

template <class T = sycl_fp8::GemmKernelSYCLFP8>
class SYCL_FP8_MOE_TP : public AVX2_MOE_BASE<T, SYCL_FP8_MOE_TP<T>> {
  using Base = AVX2_MOE_BASE<T, SYCL_FP8_MOE_TP<T>>;
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

  SYCL_FP8_MOE_TP() = default;
  SYCL_FP8_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    T::config();
    auto& quant_config = config_.quant_config;
    if (quant_config.group_size == 0 || quant_config.zero_point) {
      throw std::runtime_error("SYCL FP8 MoE only supports block-wise FP8 (group_size > 0, no zero_point)");
    }
    std::printf("Created SYCL_FP8_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
  }

  ~SYCL_FP8_MOE_TP() = default;

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

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int) {
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];
    sycl_fp8::gemm_fp8_sycl(m, config_.intermediate_size, config_.hidden_size, *ba, *bb, *bc, ith, nth);
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int) {
    int m = m_local_num_[expert_idx];
    sycl_fp8::gemm_fp8_sycl(m, config_.hidden_size, config_.intermediate_size, *down_ba_[expert_idx],
                            *down_bb_[expert_idx], *down_bc_[expert_idx], ith, nth);
  }

  void load_weights() {
    auto& quant_config = config_.quant_config;
    const int group_size = quant_config.group_size;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (config_.gate_scale == nullptr) {
      throw std::runtime_error("SYCL FP8 MOE requires scale pointers.");
    }

    std::printf("[SYCL_FP8] layer %d TP %d load_weights begin: experts=%d, gpu_experts=%d, H=%d, I=%d, current_usm=%zu MiB\n",
                config_.layer_idx, tp_part_idx, config_.expert_num, config_.num_gpu_experts, config_.hidden_size,
                config_.intermediate_size, sycl_fp8::current_usm_mb());
    std::fflush(stdout);

    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, group_size](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          if (config_.should_skip_expert(logical_expert_id)) return;
          int ith = task_id % nth;

          size_t weight_offset = logical_expert_id * config_.intermediate_size * config_.hidden_size;
          size_t scale_offset = logical_expert_id * sycl_fp8::div_up(config_.hidden_size, group_size) *
                                sycl_fp8::div_up(config_.intermediate_size, group_size);

          gate_bb_[expert_idx]->from_mat((uint8_t*)config_.gate_proj + weight_offset,
                                         (float*)config_.gate_scale + scale_offset, ith, nth);
          up_bb_[expert_idx]->from_mat((uint8_t*)config_.up_proj + weight_offset,
                                       (float*)config_.up_scale + scale_offset, ith, nth);
        },
        nullptr);

    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, group_size](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          if (config_.should_skip_expert(logical_expert_id)) return;
          int ith = task_id % nth;

          size_t weight_offset = logical_expert_id * config_.intermediate_size * config_.hidden_size;
          size_t scale_offset = logical_expert_id * sycl_fp8::div_up(config_.hidden_size, group_size) *
                                sycl_fp8::div_up(config_.intermediate_size, group_size);

          down_bb_[expert_idx]->from_mat((uint8_t*)config_.down_proj + weight_offset,
                                         (float*)config_.down_scale + scale_offset, ith, nth);
        },
        nullptr);

    std::printf("[SYCL_FP8] layer %d TP %d load_weights done: current_usm=%zu MiB\n", config_.layer_idx, tp_part_idx,
                sycl_fp8::current_usm_mb());
    std::fflush(stdout);
  }

  void write_weights_to_buffer(int, int, int, const GeneralMOEConfig&, const std::vector<uintptr_t>&,
                               const std::vector<uintptr_t>&, const std::vector<uintptr_t>&,
                               const std::vector<uintptr_t>&) const {
    throw std::runtime_error("SYCL FP8 MoE does not support write_weights_to_buffer yet.");
  }
};

template <typename K>
class TP_MOE<SYCL_FP8_MOE_TP<K>> : public TP_MOE<AVX2_MOE_BASE<K, SYCL_FP8_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AVX2_MOE_BASE<K, SYCL_FP8_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    const int group_size = config.quant_config.group_size;
    if (group_size == 0 || config.quant_config.zero_point) {
      throw std::runtime_error("SYCL FP8 MoE only supports block-wise (group_size > 0, zero_point=false)");
    }

    if (config.gate_projs.empty() && config.gate_proj == nullptr) {
      throw std::runtime_error("no weight source");
    }
    const bool use_per_expert_ptrs = !config.gate_projs.empty();

    const size_t full_weight_elems = (size_t)config.intermediate_size * config.hidden_size;
    const size_t full_scale_elems =
        (size_t)sycl_fp8::div_up(config.hidden_size, group_size) * sycl_fp8::div_up(config.intermediate_size, group_size);

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      const size_t tp_weight_elems = (size_t)tpc.intermediate_size * tpc.hidden_size;
      const size_t tp_scale_elems = (size_t)sycl_fp8::div_up(tpc.intermediate_size, group_size) *
                                    sycl_fp8::div_up(tpc.hidden_size, group_size);

      tpc.gate_proj = new uint8_t[tpc.expert_num * tp_weight_elems];
      tpc.up_proj = new uint8_t[tpc.expert_num * tp_weight_elems];
      tpc.down_proj = new uint8_t[tpc.expert_num * tp_weight_elems];
      tpc.gate_scale = new float[tpc.expert_num * tp_scale_elems];
      tpc.up_scale = new float[tpc.expert_num * tp_scale_elems];
      tpc.down_scale = new float[tpc.expert_num * tp_scale_elems];

      const size_t gate_up_weight_src_offset = (size_t)i * tp_weight_elems;
      const size_t gate_up_scale_src_offset = (size_t)i * tp_scale_elems;
      const size_t down_weight_src_col_offset = (size_t)i * (size_t)tpc.intermediate_size;
      const size_t down_scale_src_block_k_offset = down_weight_src_col_offset / (size_t)group_size;

      pool->get_subpool(i)->do_work_stealing_job(
          tpc.expert_num, nullptr,
          [&](int expert_id_) {
            const size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

            uint8_t* gate_dst = (uint8_t*)tpc.gate_proj + expert_id * tp_weight_elems;
            uint8_t* up_dst = (uint8_t*)tpc.up_proj + expert_id * tp_weight_elems;
            uint8_t* down_dst = (uint8_t*)tpc.down_proj + expert_id * tp_weight_elems;
            float* gate_scale_dst = (float*)tpc.gate_scale + expert_id * tp_scale_elems;
            float* up_scale_dst = (float*)tpc.up_scale + expert_id * tp_scale_elems;
            float* down_scale_dst = (float*)tpc.down_scale + expert_id * tp_scale_elems;

            const uint8_t* gate_src;
            const uint8_t* up_src;
            const uint8_t* down_src;
            const float* gate_scale_src;
            const float* up_scale_src;
            const float* down_scale_src;

            if (use_per_expert_ptrs) {
              gate_src = (const uint8_t*)config.gate_projs[0][expert_id] + gate_up_weight_src_offset;
              up_src = (const uint8_t*)config.up_projs[0][expert_id] + gate_up_weight_src_offset;
              down_src = (const uint8_t*)config.down_projs[0][expert_id];
              gate_scale_src = (const float*)config.gate_scales[0][expert_id] + gate_up_scale_src_offset;
              up_scale_src = (const float*)config.up_scales[0][expert_id] + gate_up_scale_src_offset;
              down_scale_src = (const float*)config.down_scales[0][expert_id];
            } else {
              gate_src = (const uint8_t*)config.gate_proj + expert_id * full_weight_elems + gate_up_weight_src_offset;
              up_src = (const uint8_t*)config.up_proj + expert_id * full_weight_elems + gate_up_weight_src_offset;
              down_src = (const uint8_t*)config.down_proj + expert_id * full_weight_elems;
              gate_scale_src = (const float*)config.gate_scale + expert_id * full_scale_elems + gate_up_scale_src_offset;
              up_scale_src = (const float*)config.up_scale + expert_id * full_scale_elems + gate_up_scale_src_offset;
              down_scale_src = (const float*)config.down_scale + expert_id * full_scale_elems;
            }

            std::memcpy(gate_dst, gate_src, tp_weight_elems);
            std::memcpy(up_dst, up_src, tp_weight_elems);
            std::memcpy(gate_scale_dst, gate_scale_src, sizeof(float) * tp_scale_elems);
            std::memcpy(up_scale_dst, up_scale_src, sizeof(float) * tp_scale_elems);

            for (int row = 0; row < config.hidden_size; row++) {
              const size_t src_row_offset = (size_t)row * (size_t)config.intermediate_size + down_weight_src_col_offset;
              const size_t dst_row_offset = (size_t)row * (size_t)tpc.intermediate_size;
              std::memcpy(down_dst + dst_row_offset, down_src + src_row_offset, (size_t)tpc.intermediate_size);
            }

            const int n_blocks_n = sycl_fp8::div_up(config.hidden_size, group_size);
            const int full_n_blocks_k = sycl_fp8::div_up(config.intermediate_size, group_size);
            const int tp_n_blocks_k = sycl_fp8::div_up(tpc.intermediate_size, group_size);
            for (int bn = 0; bn < n_blocks_n; bn++) {
              const float* src = down_scale_src + (size_t)bn * full_n_blocks_k + down_scale_src_block_k_offset;
              float* dst = down_scale_dst + (size_t)bn * tp_n_blocks_k;
              std::memcpy(dst, src, sizeof(float) * tp_n_blocks_k);
            }
          },
          nullptr);
    });

    pool->dispense_backend()->do_numa_job([&, this](int i) { tps[i]->load_weights(); });

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      delete[] (uint8_t*)tpc.gate_proj;
      delete[] (uint8_t*)tpc.up_proj;
      delete[] (uint8_t*)tpc.down_proj;
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
    if (this->weights_loaded == false) throw std::runtime_error("Not Loaded");
    if (this->tps.empty()) throw std::runtime_error("No TP parts initialized");

    this->config.pool->dispense_backend()->do_numa_job([&, this](int i) {
      this->tps[i]->write_weights_to_buffer(gpu_tp_count, this->tp_count, expert_id, this->config, w13_weight_ptrs,
                                            w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs);
    });
  }
};

#endif  // CPUINFER_OPERATOR_SYCL_FP8_MOE_H
