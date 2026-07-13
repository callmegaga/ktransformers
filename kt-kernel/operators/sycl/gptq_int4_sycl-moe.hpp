/**
 * @Description  : SYCL GPTQ-Int4 MoE operator for integrated/discrete Intel GPU experiments
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * GPTQ symmetric int4 weights (zero_point = 8): qweight [K/8, N] uint32 (8 nibbles/word
 * along K) + scales [K/group_size, N] float. Dequant: ((nibble - 8) * scale).
 * Mirrors the AVX2 GPTQ-Int4 backend (operators/avx2/gptq_int4-moe.hpp) for weight
 * layout / TP split, and the SYCL FP8 backend (operators/sycl/fp8-moe.hpp) for the
 * SYCL queue / USM / MoE_Interface integration. The gate/up/down GEMMs run on a SYCL
 * device; routing / activation / merge reuse AVX2_MOE_BASE.
 *
 * Milestone 1 (this file): per-GEMM SYCL kernels — correct + integrates like FP8.
 * Milestone 2 (later): single fused kernel/layer (see benchmarks/igpu_expert_bench/
 * fused_expert.cpp) to remove per-GEMM dispatch overhead and beat the CPU on int4.
 **/
#ifndef CPUINFER_OPERATOR_SYCL_GPTQ_INT4_MOE_H
#define CPUINFER_OPERATOR_SYCL_GPTQ_INT4_MOE_H

#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../avx2/moe_base.hpp"

namespace sycl_int4 {

inline int div_up(int a, int b) { return (a + b - 1) / b; }

inline int env_int(const char* name, int fallback = 0) {
  const char* v = std::getenv(name);
  if (v == nullptr || v[0] == '\0') return fallback;
  char* end = nullptr;
  long parsed = std::strtol(v, &end, 10);
  return end == v ? fallback : (int)parsed;
}

inline bool env_flag(const char* name, bool fallback = false) {
  const char* v = std::getenv(name);
  if (v == nullptr || v[0] == '\0') return fallback;
  return !(std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 || std::strcmp(v, "FALSE") == 0 ||
           std::strcmp(v, "off") == 0 || std::strcmp(v, "OFF") == 0);
}

inline bool env_eq(const char* name, const char* expected) {
  const char* v = std::getenv(name);
  return v != nullptr && std::strcmp(v, expected) == 0;
}

inline bool queue_profiling_enabled() { return env_flag("KT_SYCL_QUEUE_PROFILING", false); }
inline bool queue_in_order_enabled() { return env_flag("KT_SYCL_QUEUE_IN_ORDER", false); }
inline bool device_weights_enabled() { return env_flag("KT_SYCL_INT4_DEVICE_WEIGHTS", false); }
inline bool device_scratch_enabled() { return env_flag("KT_SYCL_INT4_DEVICE_SCRATCH", false); }
inline bool fast_silu_enabled() { return env_flag("KT_SYCL_INT4_FAST_SILU", false); }
inline bool pre_kernel_ping_enabled() { return env_flag("KT_SYCL_INT4_PRE_KERNEL_PING", false); }
inline bool shape_ping_enabled() { return env_flag("KT_SYCL_INT4_SHAPE_PING", false); }
inline bool weight_ping_enabled() { return env_flag("KT_SYCL_INT4_WEIGHT_PING", false); }

inline uint64_t now_us() {
  return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct EventTiming {
  uint64_t submit_start_us = 0;
  uint64_t start_end_us = 0;
  uint64_t submit_end_us = 0;
};

inline EventTiming event_timing_us(const sycl::event& ev) {
  EventTiming timing;
  try {
    const auto submit = ev.get_profiling_info<sycl::info::event_profiling::command_submit>();
    const auto start = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto end = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
    timing.submit_start_us = start > submit ? (uint64_t)((start - submit) / 1000) : 0;
    timing.start_end_us = end > start ? (uint64_t)((end - start) / 1000) : 0;
    timing.submit_end_us = end > submit ? (uint64_t)((end - submit) / 1000) : 0;
  } catch (const sycl::exception&) {
  }
  return timing;
}

// ---- SYCL queue + USM helpers (self-contained; mirrors sycl_fp8) ----
inline sycl::queue& queue() {
  static sycl::queue q([] {
    auto handler = [](sycl::exception_list es) {
      for (auto& e : es) {
        try { std::rethrow_exception(e); }
        catch (const sycl::exception& ex) { std::fprintf(stderr, "SYCL async exception: %s\n", ex.what()); }
      }
    };
    const char* filter = std::getenv("KT_SYCL_DEVICE_FILTER");
    const auto make_queue = [&](const sycl::property_list& props) {
      if (filter && filter[0]) return sycl::queue(sycl::ext::oneapi::filter_selector(filter), handler, props);
      return sycl::queue(sycl::gpu_selector_v, handler, props);
    };
    try {
      const bool profiling = queue_profiling_enabled();
      const bool in_order = queue_in_order_enabled();
      if (profiling && in_order) {
        sycl::property_list props{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}};
        return make_queue(props);
      }
      if (profiling) {
        sycl::property_list props{sycl::property::queue::enable_profiling{}};
        return make_queue(props);
      }
      if (in_order) {
        sycl::property_list props{sycl::property::queue::in_order{}};
        return make_queue(props);
      }
      sycl::property_list props{};
      return make_queue(props);
    } catch (const sycl::exception& ex) {
      throw std::runtime_error(std::string("SYCL GPTQ-Int4 MoE failed to create a queue. "
          "Set KT_SYCL_DEVICE_FILTER=level_zero:gpu or opencl:gpu for Intel iGPU, or opencl:cpu for correctness testing. "
          "Original: ") + ex.what());
    }
  }());
  return q;
}

template <typename T>
inline T* usm_alloc(size_t elems, const char* name) {
  elems = std::max<size_t>(elems, 1);
  T* p = sycl::malloc_shared<T>(elems, queue());
  if (!p) throw std::runtime_error(std::string("SYCL malloc_shared failed for ") + name);
  return p;
}
template <typename T>
inline T* usm_alloc_device(size_t elems, const char* name) {
  elems = std::max<size_t>(elems, 1);
  T* p = sycl::malloc_device<T>(elems, queue());
  if (!p) throw std::runtime_error(std::string("SYCL malloc_device failed for ") + name);
  return p;
}
inline void usm_free(void* p) { if (p) sycl::free(p, queue()); }

static inline float bf16_to_fp32(uint16_t v) {
  const uint32_t bits = static_cast<uint32_t>(v) << 16;
  return sycl::bit_cast<float>(bits);
}

static inline uint16_t fp32_to_bf16_device(float v) {
  uint32_t bits = sycl::bit_cast<uint32_t>(v);
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return (uint16_t)(bits >> 16);
}

static inline float bf16_to_fp32_host(uint16_t v) {
  uint32_t bits = static_cast<uint32_t>(v) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

static inline void quantize_bf16_i8_groups(const uint16_t* src, int len, int group_size, int8_t* dst,
                                           float* scales) {
  const int ng = len / group_size;
  for (int g = 0; g < ng; ++g) {
    float amax = 0.0f;
    const int base = g * group_size;
    for (int t = 0; t < group_size; ++t) {
      amax = std::max(amax, std::fabs(bf16_to_fp32_host(src[base + t])));
    }
    const float scale = amax > 0.0f ? amax / 127.0f : 0.0f;
    scales[g] = scale;
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    for (int t = 0; t < group_size; ++t) {
      int qv = (int)std::lrint(bf16_to_fp32_host(src[base + t]) * inv);
      dst[base + t] = (int8_t)std::clamp(qv, -127, 127);
    }
  }
}

// Per-instance device scratch for the fused decode path. Heap-allocated and stashed in
// AVX2_MOE_BASE::fused_scratch_ (see note there: Derived may not add data members).
struct FusedScratch {
  uint16_t* x = nullptr;    // input activations [H] bf16 (USM)
  int8_t* xq = nullptr;     // quantized input activations [H] i8 (USM)
  float* xs = nullptr;      // input activation scales [H/group_size] fp32 (USM)
  float* out = nullptr;     // accumulated output [H] fp32 (USM)
  float* ping_sink = nullptr; // diagnostic sink [1] (USM)
  int8_t* xq_dev = nullptr; // device copy of xq for decode diagnostics
  float* xs_dev = nullptr;  // device copy of xs
  float* out_dev = nullptr; // device output scratch
  float* rw_dev = nullptr;  // device router weights
  uint32_t **gq_dev = nullptr, **uq_dev = nullptr, **dq_dev = nullptr;
  float **gs_dev = nullptr, **us_dev = nullptr, **ds_dev = nullptr;
  float* flat_act = nullptr; // flat path activation [maxk, I] fp32 (USM)
  int8_t* flat_aq = nullptr; // flat path quantized activation [maxk, I] i8 (USM)
  float* flat_as = nullptr;  // flat path activation scales [maxk, I/group_size] fp32 (USM)
  uint32_t *stage_gq = nullptr, *stage_uq = nullptr, *stage_dq = nullptr; // active weights in device USM
  float *stage_gs = nullptr, *stage_us = nullptr, *stage_ds = nullptr;
  int stage_cap = 0;
  uint32_t *cache_gq = nullptr, *cache_uq = nullptr, *cache_dq = nullptr; // reusable device weight cache
  float *cache_gs = nullptr, *cache_us = nullptr, *cache_ds = nullptr;
  int cache_cap = 0;
  std::vector<int> cache_expert_ids;
  std::vector<uint64_t> cache_lru;
  uint64_t cache_tick = 0;
  bool cache_preloaded = false;
  uint16_t **gu_a = nullptr, **gu_out = nullptr; // batched gate+up activation pointer tables
  uint32_t **gu_gq = nullptr, **gu_uq = nullptr;
  float **gu_gs = nullptr, **gu_us = nullptr;
  int gu_cap = 0;
  std::vector<sycl::event> gate_up_pipeline_events;
  std::vector<int> gate_up_pipeline_experts;
  int gate_up_pipeline_active = 0;
  bool gate_up_pipeline_pending = false;
  float* rw = nullptr;      // router weights [maxk] (USM)
  uint32_t **gq = nullptr, **uq = nullptr, **dq = nullptr;  // per-active-expert weight ptrs
  float **gs = nullptr, **us = nullptr, **ds = nullptr;     // per-active-expert scale ptrs
  uint64_t calls = 0, total_us = 0, submit_us = 0, stage_submit_us = 0, stage_wait_us = 0;
  uint64_t control_submit_us = 0, output_copy_us = 0;
  uint64_t kernel_wait_us = 0, memset_event_us = 0, control_event_us = 0, control_device_us = 0;
  uint64_t ping_event_us = 0, ping_device_us = 0, shape_event_us = 0, shape_device_us = 0;
  uint64_t weight_event_us = 0, weight_device_us = 0;
  uint64_t cache_hits = 0, cache_misses = 0, cache_submit_us = 0, cache_event_us = 0, cache_device_us = 0;
  uint64_t kernel_event_us = 0, device_kernel_us = 0;
  uint64_t kernel_submit_start_us = 0, sycl_submit_us = 0, sycl_memset_submit_us = 0;
  uint64_t sycl_kernel_submit_us = 0, sycl_wait_us = 0, copy_us = 0;
  uint64_t active_sum = 0;
};

struct GemmKernelSYCLGPTQInt4 {
  using dt = ggml_bf16_t;
  using output_t = float;
  static constexpr int M_STEP = 1;
  static constexpr int N_STEP = 1;
  static constexpr int K_STEP = 1;
  static constexpr double ELEMENT_SIZE = 0.5;  // int4

  static void config() {
    static std::once_flag once;
    std::call_once(once, [] {
      auto& q = queue();
      const auto dev = q.get_device();
      if (!dev.get_info<sycl::info::device::usm_shared_allocations>())
        throw std::runtime_error("SYCL GPTQ-Int4 MoE requires USM shared allocation support.");
      if (device_weights_enabled() && !dev.get_info<sycl::info::device::usm_device_allocations>())
        throw std::runtime_error("SYCL GPTQ-Int4 device-weight mode requires USM device allocation support.");
      std::printf("Created SYCL_GPTQ_INT4_MOE on device: %s (queue_profiling=%d queue_in_order=%d device_weights=%d)\n",
                  dev.get_info<sycl::info::device::name>().c_str(), queue_profiling_enabled() ? 1 : 0,
                  queue_in_order_enabled() ? 1 : 0, device_weights_enabled() ? 1 : 0);
    });
  }

  static int recommended_nth(int) { return 1; }
  static std::pair<int, int> split_range_n(int n, int ith, int nth) { return avx2::split_range(n, ith, nth); }

  // BufferA: bf16 activations [M, K] in USM
  struct BufferA {
    uint16_t* data = nullptr;
    size_t max_m = 0, k = 0, cap_m = 0;
    BufferA() = default;
    BufferA(size_t m, size_t k_, void*) : max_m(m), k(k_) {}
    BufferA(const BufferA&) = delete;
    BufferA& operator=(const BufferA&) = delete;
    ~BufferA() { usm_free(data); }
    static size_t required_size(size_t, size_t) { return 1; }
    void ensure(size_t m) {
      if (m <= cap_m) return;
      usm_free(data); data = usm_alloc<uint16_t>(m * k, "int4 BufferA"); cap_m = m;
    }
    void set_data(void*) { ensure(max_m); }
    void from_mat(int m, const ggml_bf16_t* src, int ith, int nth) {
      ensure((size_t)m);
      if (ith == 0 && nth == 1) {
        std::memcpy(data, src, (size_t)m * k * sizeof(uint16_t));
      } else {
        auto [ms, me] = avx2::split_range(m, ith, nth);
        std::memcpy(data + (size_t)ms * k, src + (size_t)ms * k, (size_t)(me - ms) * k * sizeof(uint16_t));
      }
    }
  };

  // BufferB: GPTQ int4 weights [K/8, N] uint32 + scales [K/gs, N] float, in USM
  struct BufferB {
    uint32_t* qw = nullptr;   // [K/8, N]
    float* scales = nullptr;   // [K/gs, N]
    int n = 0, k = 0, group_size = 128, num_groups = 0, k_packed = 0;
    bool device_storage = false;
    BufferB() = default;
    BufferB(size_t n_, size_t k_, int gs, void*) : n((int)n_), k((int)k_), group_size(gs),
                                                   device_storage(device_weights_enabled()) {
      if (group_size <= 0 || (k % 8) != 0 || (k % group_size) != 0)
        throw std::runtime_error("SYCL GPTQ-Int4: k must be divisible by 8 and group_size");
      k_packed = k / 8; num_groups = k / group_size;
    }
    BufferB(const BufferB&) = delete;
    BufferB& operator=(const BufferB&) = delete;
    ~BufferB() { usm_free(qw); usm_free(scales); }
    static size_t required_size(size_t, size_t, int) { return 1; }
    size_t qweight_bytes() const { return (size_t)k_packed * n * sizeof(uint32_t); }
    size_t scales_bytes() const { return (size_t)num_groups * n * sizeof(float); }
    void ensure() {
      if (device_storage) {
        if (!qw) qw = usm_alloc_device<uint32_t>((size_t)k_packed * n, "int4 BufferB qweight");
        if (!scales) scales = usm_alloc_device<float>((size_t)num_groups * n, "int4 BufferB scales");
      } else {
        if (!qw) qw = usm_alloc<uint32_t>((size_t)k_packed * n, "int4 BufferB qweight");
        if (!scales) scales = usm_alloc<float>((size_t)num_groups * n, "int4 BufferB scales");
      }
    }
    // src layout matches AVX2 GPTQ: qweight [K/8, N] uint32, scales [K/gs, N] float
    void from_mat(const uint32_t* src_qw, const float* src_sc, int ith, int nth) {
      ensure();
      auto [ns, ne] = avx2::split_range(n, ith, nth);
      const int nlen = ne - ns;
      if (device_storage) {
        if (ns != 0 || nlen != n) {
          throw std::runtime_error("SYCL GPTQ-Int4 device-weight upload expects unsplit full matrices.");
        }
        auto& q = queue();
        q.memcpy(qw, src_qw, qweight_bytes()).wait_and_throw();
        q.memcpy(scales, src_sc, scales_bytes()).wait_and_throw();
        return;
      }
      for (int kp = 0; kp < k_packed; ++kp)
        std::memcpy(qw + (size_t)kp * n + ns, src_qw + (size_t)kp * n + ns, (size_t)nlen * sizeof(uint32_t));
      for (int g = 0; g < num_groups; ++g)
        std::memcpy(scales + (size_t)g * n + ns, src_sc + (size_t)g * n + ns, (size_t)nlen * sizeof(float));
    }
  };

  // BufferC: fp32 output [M, N] in USM
  struct BufferC {
    float* data = nullptr;
    size_t max_m = 0, n = 0, cap_m = 0;
    BufferC() = default;
    BufferC(size_t m, size_t n_, void*) : max_m(m), n(n_) {}
    BufferC(const BufferC&) = delete;
    BufferC& operator=(const BufferC&) = delete;
    ~BufferC() { usm_free(data); }
    static size_t required_size(size_t, size_t) { return 1; }
    void ensure(size_t m) {
      if (m <= cap_m) return;
      usm_free(data); data = usm_alloc<float>(m * n, "int4 BufferC"); cap_m = m;
    }
    void set_data(void*) { ensure(max_m); }
    void to_mat(int m, ggml_bf16_t* dst, int ith, int nth) {
      ensure((size_t)m);
      auto [ns, ne] = avx2::split_range((int)n, ith, nth);
      for (int mi = 0; mi < m; ++mi) {
        const float* s = data + (size_t)mi * n;
        ggml_bf16_t* d = dst + (size_t)mi * n;
        for (int j = ns; j < ne; ++j) d[j] = GGML_FP32_TO_BF16(s[j]);
      }
    }
  };
};

// C[m,n] = sum_g ( sum_{k in g} bf16(a[m,k]) * (nibble(qw,k,n) - 8) ) * scale[g,n]
static inline void gemm_gptq_int4_sycl(int m, int n, int k, GemmKernelSYCLGPTQInt4::BufferA& a,
                                       GemmKernelSYCLGPTQInt4::BufferB& b, GemmKernelSYCLGPTQInt4::BufferC& c,
                                       int ith, int nth) {
  if (m <= 0 || n <= 0 || k <= 0) return;
  auto [ns, ne] = avx2::split_range(n, ith, nth);
  if (ns >= ne) return;
  auto& q = queue();
  const int N = b.n, gs = b.group_size, numg = b.num_groups;
  uint16_t* a_data = a.data; uint32_t* qw = b.qw; float* sc = b.scales; float* c_data = c.data;
  const size_t a_ld = a.k, c_ld = c.n;
  const int nlen = ne - ns;
  q.submit([&](sycl::handler& h) {
     h.parallel_for(sycl::range<2>((size_t)m, (size_t)nlen), [=](sycl::id<2> idx) {
       const int mi = (int)idx[0];
       const int ni = ns + (int)idx[1];
       const uint16_t* arow = a_data + (size_t)mi * a_ld;
       float acc = 0.f;
       for (int g = 0; g < numg; ++g) {
         float gdot = 0.f;
         const int kbase = g * gs;
         for (int t = 0; t < gs; ++t) {
           const int kk = kbase + t;
           const uint32_t packed = qw[(size_t)(kk >> 3) * N + ni];
           const int w = (int)((packed >> ((kk & 7) * 4)) & 0xF) - 8;
           gdot += bf16_to_fp32(arow[kk]) * (float)w;
         }
         acc += gdot * sc[(size_t)g * N + ni];
       }
       c_data[(size_t)mi * c_ld + ni] = acc;
     });
     }).wait_and_throw();
}

template <int SG>
static inline sycl::event gemm_gptq_int4_sycl_subgroup_submit(int m, int n, int k,
                                                              GemmKernelSYCLGPTQInt4::BufferA& a,
                                                              GemmKernelSYCLGPTQInt4::BufferB& b,
                                                              GemmKernelSYCLGPTQInt4::BufferC& c, int ith, int nth,
                                                              const sycl::event* dependency = nullptr) {
  if (m <= 0 || n <= 0 || k <= 0) return sycl::event{};
  auto [ns, ne] = avx2::split_range(n, ith, nth);
  if (ns >= ne) return sycl::event{};
  auto& q = queue();
  const int N = b.n, gs = b.group_size, numg = b.num_groups;
  uint16_t* a_data = a.data;
  uint32_t* qw = b.qw;
  float* sc = b.scales;
  float* c_data = c.data;
  const size_t a_ld = a.k, c_ld = c.n;
  const int nlen = ne - ns;
  const size_t groups = (size_t)m * (size_t)nlen;
  return q.submit([&](sycl::handler& h) {
     if (dependency != nullptr) h.depends_on(*dependency);
     h.parallel_for(sycl::nd_range<1>(groups * (size_t)SG, (size_t)SG), [=](sycl::nd_item<1> it)
                    [[sycl::reqd_sub_group_size(SG)]] {
       const int gid = (int)it.get_group(0);
       const int lane = (int)it.get_local_id(0);
       const int mi = gid / nlen;
       const int ni = ns + (gid - mi * nlen);
       const uint16_t* arow = a_data + (size_t)mi * a_ld;
       float acc = 0.f;
       for (int g = 0; g < numg; ++g) {
         float partial = 0.f;
         const int kbase = g * gs;
         for (int t = lane; t < gs; t += SG) {
           const int kk = kbase + t;
           const uint32_t packed = qw[(size_t)(kk >> 3) * N + ni];
           const int w = (int)((packed >> ((kk & 7) * 4)) & 0xF) - 8;
           partial += bf16_to_fp32(arow[kk]) * (float)w;
         }
         const float group_sum = sycl::reduce_over_group(it.get_sub_group(), partial, sycl::plus<float>());
         if (lane == 0) acc += group_sum * sc[(size_t)g * N + ni];
       }
       if (lane == 0) c_data[(size_t)mi * c_ld + ni] = acc;
     });
   });
}

template <int SG>
static inline void gemm_gptq_int4_sycl_subgroup(int m, int n, int k, GemmKernelSYCLGPTQInt4::BufferA& a,
                                                GemmKernelSYCLGPTQInt4::BufferB& b,
                                                GemmKernelSYCLGPTQInt4::BufferC& c, int ith, int nth) {
  if (m <= 0 || n <= 0 || k <= 0) return;
  auto [ns, ne] = avx2::split_range(n, ith, nth);
  if (ns >= ne) return;
  gemm_gptq_int4_sycl_subgroup_submit<SG>(m, n, k, a, b, c, ith, nth).wait_and_throw();
}

static inline void gemm_gptq_int4_sycl_subgroup_dispatch(int sg, int m, int n, int k,
                                                         GemmKernelSYCLGPTQInt4::BufferA& a,
                                                         GemmKernelSYCLGPTQInt4::BufferB& b,
                                                         GemmKernelSYCLGPTQInt4::BufferC& c, int ith, int nth) {
  switch (sg) {
    case 8:
      gemm_gptq_int4_sycl_subgroup<8>(m, n, k, a, b, c, ith, nth);
      break;
    case 32:
      gemm_gptq_int4_sycl_subgroup<32>(m, n, k, a, b, c, ith, nth);
      break;
    case 16:
    default:
      gemm_gptq_int4_sycl_subgroup<16>(m, n, k, a, b, c, ith, nth);
      break;
  }
}

static inline sycl::event gemm_gptq_int4_sycl_subgroup_submit_dispatch(
    int sg, int m, int n, int k, GemmKernelSYCLGPTQInt4::BufferA& a, GemmKernelSYCLGPTQInt4::BufferB& b,
    GemmKernelSYCLGPTQInt4::BufferC& c, int ith, int nth, const sycl::event* dependency = nullptr) {
  switch (sg) {
    case 8:
      return gemm_gptq_int4_sycl_subgroup_submit<8>(m, n, k, a, b, c, ith, nth, dependency);
    case 32:
      return gemm_gptq_int4_sycl_subgroup_submit<32>(m, n, k, a, b, c, ith, nth, dependency);
    case 16:
    default:
      return gemm_gptq_int4_sycl_subgroup_submit<16>(m, n, k, a, b, c, ith, nth, dependency);
  }
}

template <int SG>
static inline sycl::event gate_up_activation_gptq_int4_sycl_subgroup_submit(
    int m, int n, int k, GemmKernelSYCLGPTQInt4::BufferA& a, GemmKernelSYCLGPTQInt4::BufferB& gate_b,
    GemmKernelSYCLGPTQInt4::BufferB& up_b, GemmKernelSYCLGPTQInt4::BufferA& out_ba, float swiglu_limit,
    float swiglu_alpha, bool fast_silu) {
  if (m <= 0 || n <= 0 || k <= 0) return sycl::event{};
  if (gate_b.n != n || up_b.n != n || gate_b.k != k || up_b.k != k ||
      gate_b.group_size != up_b.group_size) {
    throw std::runtime_error("SYCL GPTQ-Int4 gate_up activation fuse: incompatible gate/up shapes");
  }
  auto& q = queue();
  const int N = gate_b.n, gs = gate_b.group_size, numg = gate_b.num_groups;
  uint16_t* a_data = a.data;
  uint32_t* gqw = gate_b.qw;
  uint32_t* uqw = up_b.qw;
  float* gsc = gate_b.scales;
  float* usc = up_b.scales;
  uint16_t* out = out_ba.data;
  const size_t a_ld = a.k;
  const size_t out_ld = out_ba.k;
  const size_t groups = (size_t)m * (size_t)n;
  return q.submit([&](sycl::handler& h) {
     h.parallel_for(sycl::nd_range<1>(groups * (size_t)SG, (size_t)SG), [=](sycl::nd_item<1> it)
                    [[sycl::reqd_sub_group_size(SG)]] {
       const int gid = (int)it.get_group(0);
       const int lane = (int)it.get_local_id(0);
       const int mi = gid / n;
       const int ni = gid - mi * n;
       const uint16_t* arow = a_data + (size_t)mi * a_ld;
       float gate_acc = 0.f;
       float up_acc = 0.f;
       for (int g = 0; g < numg; ++g) {
         float gate_partial = 0.f;
         float up_partial = 0.f;
         const int kbase = g * gs;
         for (int t = lane; t < gs; t += SG) {
           const int kk = kbase + t;
           const int shift = (kk & 7) * 4;
           const float xv = bf16_to_fp32(arow[kk]);
           const uint32_t pg = gqw[(size_t)(kk >> 3) * N + ni];
           const uint32_t pu = uqw[(size_t)(kk >> 3) * N + ni];
           gate_partial += xv * (float)((int)((pg >> shift) & 0xFu) - 8);
           up_partial += xv * (float)((int)((pu >> shift) & 0xFu) - 8);
         }
         const float gate_sum = sycl::reduce_over_group(it.get_sub_group(), gate_partial, sycl::plus<float>());
         const float up_sum = sycl::reduce_over_group(it.get_sub_group(), up_partial, sycl::plus<float>());
         if (lane == 0) {
           gate_acc += gate_sum * gsc[(size_t)g * N + ni];
           up_acc += up_sum * usc[(size_t)g * N + ni];
         }
       }
       if (lane == 0) {
         float gv = gate_acc;
         float uv = up_acc;
         float act = 0.f;
         if (swiglu_alpha > 0.f) {
           if (swiglu_limit > 0.f) {
             gv = sycl::fmin(sycl::fmax(gv, -swiglu_limit), swiglu_limit);
             uv = sycl::fmin(sycl::fmax(uv, -swiglu_limit), swiglu_limit);
           }
           const float x = -gv * swiglu_alpha;
           const float sig = fast_silu ? 1.f / (1.f + sycl::native::exp(x)) : 1.f / (1.f + sycl::exp(x));
           act = gv * sig * (uv + 1.f);
         } else {
           if (swiglu_limit > 0.f) {
             gv = sycl::fmin(gv, swiglu_limit);
             uv = sycl::fmin(sycl::fmax(uv, -swiglu_limit), swiglu_limit);
           }
           const float x = -gv;
           const float sig = fast_silu ? 1.f / (1.f + sycl::native::exp(x)) : 1.f / (1.f + sycl::exp(x));
           act = gv * sig * uv;
         }
         out[(size_t)mi * out_ld + ni] = fp32_to_bf16_device(act);
       }
     });
   });
}

template <int SG>
static inline void gate_up_activation_gptq_int4_sycl_subgroup(
    int m, int n, int k, GemmKernelSYCLGPTQInt4::BufferA& a, GemmKernelSYCLGPTQInt4::BufferB& gate_b,
    GemmKernelSYCLGPTQInt4::BufferB& up_b, GemmKernelSYCLGPTQInt4::BufferA& out_ba, float swiglu_limit,
    float swiglu_alpha, bool fast_silu) {
  gate_up_activation_gptq_int4_sycl_subgroup_submit<SG>(m, n, k, a, gate_b, up_b, out_ba, swiglu_limit,
                                                        swiglu_alpha, fast_silu)
      .wait_and_throw();
}

static inline void gate_up_activation_gptq_int4_sycl_subgroup_dispatch(
    int sg, int m, int n, int k, GemmKernelSYCLGPTQInt4::BufferA& a, GemmKernelSYCLGPTQInt4::BufferB& gate_b,
    GemmKernelSYCLGPTQInt4::BufferB& up_b, GemmKernelSYCLGPTQInt4::BufferA& out_ba, float swiglu_limit,
    float swiglu_alpha, bool fast_silu) {
  switch (sg) {
    case 8:
      gate_up_activation_gptq_int4_sycl_subgroup<8>(m, n, k, a, gate_b, up_b, out_ba, swiglu_limit, swiglu_alpha,
                                                    fast_silu);
      break;
    case 32:
      gate_up_activation_gptq_int4_sycl_subgroup<32>(m, n, k, a, gate_b, up_b, out_ba, swiglu_limit, swiglu_alpha,
                                                     fast_silu);
      break;
    case 16:
    default:
      gate_up_activation_gptq_int4_sycl_subgroup<16>(m, n, k, a, gate_b, up_b, out_ba, swiglu_limit, swiglu_alpha,
                                                     fast_silu);
      break;
  }
}

static inline sycl::event gate_up_activation_gptq_int4_sycl_subgroup_submit_dispatch(
    int sg, int m, int n, int k, GemmKernelSYCLGPTQInt4::BufferA& a, GemmKernelSYCLGPTQInt4::BufferB& gate_b,
    GemmKernelSYCLGPTQInt4::BufferB& up_b, GemmKernelSYCLGPTQInt4::BufferA& out_ba, float swiglu_limit,
    float swiglu_alpha, bool fast_silu) {
  switch (sg) {
    case 8:
      return gate_up_activation_gptq_int4_sycl_subgroup_submit<8>(m, n, k, a, gate_b, up_b, out_ba, swiglu_limit,
                                                                  swiglu_alpha, fast_silu);
    case 32:
      return gate_up_activation_gptq_int4_sycl_subgroup_submit<32>(m, n, k, a, gate_b, up_b, out_ba, swiglu_limit,
                                                                   swiglu_alpha, fast_silu);
    case 16:
    default:
      return gate_up_activation_gptq_int4_sycl_subgroup_submit<16>(m, n, k, a, gate_b, up_b, out_ba, swiglu_limit,
                                                                   swiglu_alpha, fast_silu);
  }
}

template <int SG>
static inline sycl::event gate_up_activation_gptq_int4_sycl_subgroup_q8_submit(
    int m, int n, int k, const int8_t* xq, const float* xs, GemmKernelSYCLGPTQInt4::BufferB& gate_b,
    GemmKernelSYCLGPTQInt4::BufferB& up_b, GemmKernelSYCLGPTQInt4::BufferA& out_ba, float swiglu_limit,
    float swiglu_alpha, bool fast_silu) {
  if (m <= 0 || n <= 0 || k <= 0) return sycl::event{};
  if (m != 1) {
    throw std::runtime_error("SYCL GPTQ-Int4 q8 gate_up activation fuse currently supports decode m=1 only");
  }
  if (gate_b.n != n || up_b.n != n || gate_b.k != k || up_b.k != k ||
      gate_b.group_size != up_b.group_size) {
    throw std::runtime_error("SYCL GPTQ-Int4 q8 gate_up activation fuse: incompatible gate/up shapes");
  }
  auto& q = queue();
  const int N = gate_b.n, gs = gate_b.group_size, numg = gate_b.num_groups;
  const int packed_per_group = gs / 8;
  uint32_t* gqw = gate_b.qw;
  uint32_t* uqw = up_b.qw;
  float* gsc = gate_b.scales;
  float* usc = up_b.scales;
  uint16_t* out = out_ba.data;
  const size_t out_ld = out_ba.k;
  return q.submit([&](sycl::handler& h) {
     h.parallel_for(sycl::nd_range<1>((size_t)n * (size_t)SG, (size_t)SG), [=](sycl::nd_item<1> it)
                    [[sycl::reqd_sub_group_size(SG)]] {
       const int ni = (int)it.get_group(0);
       const int lane = (int)it.get_local_id(0);
       float gate_acc = 0.f;
       float up_acc = 0.f;
       for (int g = 0; g < numg; ++g) {
         int gate_partial = 0;
         int up_partial = 0;
         const int kp_base = g * packed_per_group;
         for (int kpi = lane; kpi < packed_per_group; kpi += SG) {
           const int kp = kp_base + kpi;
           const uint32_t pg = gqw[(size_t)kp * N + ni];
           const uint32_t pu = uqw[(size_t)kp * N + ni];
           const int8_t* xb = xq + (size_t)kp * 8;
#pragma unroll
           for (int bb = 0; bb < 8; ++bb) {
             const int xv = (int)xb[bb];
             const int shift = bb * 4;
             gate_partial += xv * ((int)((pg >> shift) & 0xFu) - 8);
             up_partial += xv * ((int)((pu >> shift) & 0xFu) - 8);
           }
         }
         const int gate_sum = sycl::reduce_over_group(it.get_sub_group(), gate_partial, sycl::plus<int>());
         const int up_sum = sycl::reduce_over_group(it.get_sub_group(), up_partial, sycl::plus<int>());
         if (lane == 0) {
           gate_acc += (float)gate_sum * xs[g] * gsc[(size_t)g * N + ni];
           up_acc += (float)up_sum * xs[g] * usc[(size_t)g * N + ni];
         }
       }
       if (lane == 0) {
         float gv = gate_acc;
         float uv = up_acc;
         float act = 0.f;
         if (swiglu_alpha > 0.f) {
           if (swiglu_limit > 0.f) {
             gv = sycl::fmin(sycl::fmax(gv, -swiglu_limit), swiglu_limit);
             uv = sycl::fmin(sycl::fmax(uv, -swiglu_limit), swiglu_limit);
           }
           const float x = -gv * swiglu_alpha;
           const float sig = fast_silu ? 1.f / (1.f + sycl::native::exp(x)) : 1.f / (1.f + sycl::exp(x));
           act = gv * sig * (uv + 1.f);
         } else {
           if (swiglu_limit > 0.f) {
             gv = sycl::fmin(gv, swiglu_limit);
             uv = sycl::fmin(sycl::fmax(uv, -swiglu_limit), swiglu_limit);
           }
           const float x = -gv;
           const float sig = fast_silu ? 1.f / (1.f + sycl::native::exp(x)) : 1.f / (1.f + sycl::exp(x));
           act = gv * sig * uv;
         }
         out[ni] = fp32_to_bf16_device(act);
       }
     });
   });
}

static inline sycl::event gate_up_activation_gptq_int4_sycl_subgroup_q8_submit_dispatch(
    int sg, int m, int n, int k, const int8_t* xq, const float* xs, GemmKernelSYCLGPTQInt4::BufferB& gate_b,
    GemmKernelSYCLGPTQInt4::BufferB& up_b, GemmKernelSYCLGPTQInt4::BufferA& out_ba, float swiglu_limit,
    float swiglu_alpha, bool fast_silu) {
  switch (sg) {
    case 8:
      return gate_up_activation_gptq_int4_sycl_subgroup_q8_submit<8>(m, n, k, xq, xs, gate_b, up_b, out_ba,
                                                                     swiglu_limit, swiglu_alpha, fast_silu);
    case 32:
      return gate_up_activation_gptq_int4_sycl_subgroup_q8_submit<32>(m, n, k, xq, xs, gate_b, up_b, out_ba,
                                                                      swiglu_limit, swiglu_alpha, fast_silu);
    case 16:
    default:
      return gate_up_activation_gptq_int4_sycl_subgroup_q8_submit<16>(m, n, k, xq, xs, gate_b, up_b, out_ba,
                                                                      swiglu_limit, swiglu_alpha, fast_silu);
  }
}

template <int SG>
static inline void gate_up_activation_gptq_int4_sycl_subgroup_batched(
    int active, int m, int n, int k, uint16_t** a_ptrs, uint32_t** gate_qw_ptrs, uint32_t** up_qw_ptrs,
    float** gate_sc_ptrs, float** up_sc_ptrs, uint16_t** out_ptrs, int group_size, float swiglu_limit,
    float swiglu_alpha, bool fast_silu) {
  if (active <= 0 || m <= 0 || n <= 0 || k <= 0) return;
  auto& q = queue();
  const int gs = group_size;
  const int numg = k / gs;
  const size_t per_active = (size_t)m * (size_t)n;
  const size_t groups = (size_t)active * per_active;
  q.submit([&](sycl::handler& h) {
     h.parallel_for(sycl::nd_range<1>(groups * (size_t)SG, (size_t)SG), [=](sycl::nd_item<1> it)
                    [[sycl::reqd_sub_group_size(SG)]] {
       const size_t gid = it.get_group(0);
       const int lane = (int)it.get_local_id(0);
       const int aidx = (int)(gid / per_active);
       const size_t rem = gid - (size_t)aidx * per_active;
       const int mi = (int)(rem / (size_t)n);
       const int ni = (int)(rem - (size_t)mi * (size_t)n);
       const uint16_t* arow = a_ptrs[aidx] + (size_t)mi * (size_t)k;
       const uint32_t* gqw = gate_qw_ptrs[aidx];
       const uint32_t* uqw = up_qw_ptrs[aidx];
       const float* gsc = gate_sc_ptrs[aidx];
       const float* usc = up_sc_ptrs[aidx];
       float gate_acc = 0.f;
       float up_acc = 0.f;
       for (int g = 0; g < numg; ++g) {
         float gate_partial = 0.f;
         float up_partial = 0.f;
         const int kbase = g * gs;
         for (int t = lane; t < gs; t += SG) {
           const int kk = kbase + t;
           const int shift = (kk & 7) * 4;
           const float xv = bf16_to_fp32(arow[kk]);
           const uint32_t pg = gqw[(size_t)(kk >> 3) * (size_t)n + (size_t)ni];
           const uint32_t pu = uqw[(size_t)(kk >> 3) * (size_t)n + (size_t)ni];
           gate_partial += xv * (float)((int)((pg >> shift) & 0xFu) - 8);
           up_partial += xv * (float)((int)((pu >> shift) & 0xFu) - 8);
         }
         const float gate_sum = sycl::reduce_over_group(it.get_sub_group(), gate_partial, sycl::plus<float>());
         const float up_sum = sycl::reduce_over_group(it.get_sub_group(), up_partial, sycl::plus<float>());
         if (lane == 0) {
           gate_acc += gate_sum * gsc[(size_t)g * (size_t)n + (size_t)ni];
           up_acc += up_sum * usc[(size_t)g * (size_t)n + (size_t)ni];
         }
       }
       if (lane == 0) {
         float gv = gate_acc;
         float uv = up_acc;
         float act = 0.f;
         if (swiglu_alpha > 0.f) {
           if (swiglu_limit > 0.f) {
             gv = sycl::fmin(sycl::fmax(gv, -swiglu_limit), swiglu_limit);
             uv = sycl::fmin(sycl::fmax(uv, -swiglu_limit), swiglu_limit);
           }
           const float x = -gv * swiglu_alpha;
           const float sig = fast_silu ? 1.f / (1.f + sycl::native::exp(x)) : 1.f / (1.f + sycl::exp(x));
           act = gv * sig * (uv + 1.f);
         } else {
           if (swiglu_limit > 0.f) {
             gv = sycl::fmin(gv, swiglu_limit);
             uv = sycl::fmin(sycl::fmax(uv, -swiglu_limit), swiglu_limit);
           }
           const float x = -gv;
           const float sig = fast_silu ? 1.f / (1.f + sycl::native::exp(x)) : 1.f / (1.f + sycl::exp(x));
           act = gv * sig * uv;
         }
         out_ptrs[aidx][(size_t)mi * (size_t)n + (size_t)ni] = fp32_to_bf16_device(act);
       }
     });
   }).wait_and_throw();
}

static inline void gate_up_activation_gptq_int4_sycl_subgroup_batched_dispatch(
    int sg, int active, int m, int n, int k, uint16_t** a_ptrs, uint32_t** gate_qw_ptrs, uint32_t** up_qw_ptrs,
    float** gate_sc_ptrs, float** up_sc_ptrs, uint16_t** out_ptrs, int group_size, float swiglu_limit,
    float swiglu_alpha, bool fast_silu) {
  switch (sg) {
    case 8:
      gate_up_activation_gptq_int4_sycl_subgroup_batched<8>(active, m, n, k, a_ptrs, gate_qw_ptrs, up_qw_ptrs,
                                                            gate_sc_ptrs, up_sc_ptrs, out_ptrs, group_size,
                                                            swiglu_limit, swiglu_alpha, fast_silu);
      break;
    case 32:
      gate_up_activation_gptq_int4_sycl_subgroup_batched<32>(active, m, n, k, a_ptrs, gate_qw_ptrs, up_qw_ptrs,
                                                             gate_sc_ptrs, up_sc_ptrs, out_ptrs, group_size,
                                                             swiglu_limit, swiglu_alpha, fast_silu);
      break;
    case 16:
    default:
      gate_up_activation_gptq_int4_sycl_subgroup_batched<16>(active, m, n, k, a_ptrs, gate_qw_ptrs, up_qw_ptrs,
                                                             gate_sc_ptrs, up_sc_ptrs, out_ptrs, group_size,
                                                             swiglu_limit, swiglu_alpha, fast_silu);
      break;
  }
}

inline void trace_gemm_limited(const char* kind, int layer_idx, int expert_idx, int m, int n, int k) {
  static std::atomic<int> remaining(env_int("KT_SYCL_INT4_TRACE_GEMM_LIMIT", 0));
  int old = remaining.fetch_sub(1);
  if (old <= 0) return;
  std::printf("[SYCL_GPTQ_INT4 gemm] kind=%s layer=%d expert=%d m=%d n=%d k=%d remaining=%d\n", kind, layer_idx,
              expert_idx, m, n, k, old - 1);
}

struct PerGemmTraceState {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> total_us{0};
  std::atomic<uint64_t> m_sum{0};
};

inline PerGemmTraceState& per_gemm_trace_state(int layer_idx, int kind_idx) {
  static PerGemmTraceState states[128][4];
  const int l = std::clamp(layer_idx, 0, 127);
  const int k = std::clamp(kind_idx, 0, 3);
  return states[l][k];
}

inline void trace_per_gemm_timing(const char* kind, int kind_idx, int layer_idx, int expert_idx, int m, int n, int k,
                                  uint64_t elapsed_us, const char* kernel = "scalar") {
  const int trace_every = env_int("KT_SYCL_INT4_TRACE_EVERY", 0);
  if (trace_every <= 0) return;
  auto& state = per_gemm_trace_state(layer_idx, kind_idx);
  const uint64_t calls = state.calls.fetch_add(1, std::memory_order_relaxed) + 1;
  const uint64_t total = state.total_us.fetch_add(elapsed_us, std::memory_order_relaxed) + elapsed_us;
  const uint64_t m_total = state.m_sum.fetch_add((uint64_t)std::max(m, 0), std::memory_order_relaxed) +
                           (uint64_t)std::max(m, 0);
  if ((calls % (uint64_t)trace_every) != 0) return;
  std::printf(
      "[SYCL_GPTQ_INT4 per_gemm] kernel=%s kind=%s layer=%d calls=%llu avg_gemm=%.3fms avg_m=%.2f last_expert=%d last_m=%d "
      "n=%d k=%d\n",
      kernel, kind, layer_idx, (unsigned long long)calls, (double)total / (double)calls / 1000.0,
      (double)m_total / (double)calls, expert_idx, m, n, k);
}

struct GateUpBatchTraceState {
  uint64_t calls = 0;
  uint64_t total_us = 0;
  uint64_t active_sum = 0;
};

inline GateUpBatchTraceState& gate_up_batch_trace_state(int layer_idx) {
  static GateUpBatchTraceState states[128];
  return states[std::clamp(layer_idx, 0, 127)];
}

inline void trace_gate_up_batch_timing(int layer_idx, int active, int n, int k, int sg, uint64_t elapsed_us) {
  const int trace_every = env_int("KT_SYCL_INT4_TRACE_EVERY", 0);
  if (trace_every <= 0) return;
  auto& state = gate_up_batch_trace_state(layer_idx);
  state.calls++;
  state.total_us += elapsed_us;
  state.active_sum += (uint64_t)std::max(active, 0);
  if ((state.calls % (uint64_t)trace_every) != 0) return;
  const double inv = 1.0 / (double)state.calls;
  const double avg_active = (double)state.active_sum * inv;
  const double avg_total_ms = (double)state.total_us * inv / 1000.0;
  std::printf(
      "[SYCL_GPTQ_INT4 gate_up_batch] kernel=subgroup_fused layer=%d calls=%llu avg_total=%.3fms "
      "avg_per_active=%.3fms avg_active=%.2f n=%d k=%d sg=%d\n",
      layer_idx, (unsigned long long)state.calls, avg_total_ms,
      avg_active > 0.0 ? avg_total_ms / avg_active : 0.0, avg_active, n, k, sg);
}

inline void trace_gate_up_async_timing(int layer_idx, int active, int n, int k, int sg, uint64_t elapsed_us,
                                       const char* kernel = "subgroup_fused") {
  const int trace_every = env_int("KT_SYCL_INT4_TRACE_EVERY", 0);
  if (trace_every <= 0) return;
  auto& state = gate_up_batch_trace_state(layer_idx);
  state.calls++;
  state.total_us += elapsed_us;
  state.active_sum += (uint64_t)std::max(active, 0);
  if ((state.calls % (uint64_t)trace_every) != 0) return;
  const double inv = 1.0 / (double)state.calls;
  const double avg_active = (double)state.active_sum * inv;
  const double avg_total_ms = (double)state.total_us * inv / 1000.0;
  std::printf(
      "[SYCL_GPTQ_INT4 gate_up_async] kernel=%s layer=%d calls=%llu avg_total=%.3fms "
      "avg_per_active=%.3fms avg_active=%.2f n=%d k=%d sg=%d\n",
      kernel, layer_idx, (unsigned long long)state.calls, avg_total_ms,
      avg_active > 0.0 ? avg_total_ms / avg_active : 0.0, avg_active, n, k, sg);
}

inline void trace_gate_up_pipeline_submit_timing(int layer_idx, int active, int n, int k, int sg,
                                                 uint64_t elapsed_us,
                                                 const char* kernel = "subgroup_fused") {
  const int trace_every = env_int("KT_SYCL_INT4_TRACE_EVERY", 0);
  if (trace_every <= 0) return;
  auto& state = gate_up_batch_trace_state(layer_idx);
  state.calls++;
  state.total_us += elapsed_us;
  state.active_sum += (uint64_t)std::max(active, 0);
  if ((state.calls % (uint64_t)trace_every) != 0) return;
  const double inv = 1.0 / (double)state.calls;
  const double avg_active = (double)state.active_sum * inv;
  const double avg_submit_ms = (double)state.total_us * inv / 1000.0;
  std::printf(
      "[SYCL_GPTQ_INT4 gate_up_pipeline] kernel=%s layer=%d calls=%llu avg_submit=%.3fms "
      "avg_per_active=%.3fms avg_active=%.2f n=%d k=%d sg=%d\n",
      kernel, layer_idx, (unsigned long long)state.calls, avg_submit_ms,
      avg_active > 0.0 ? avg_submit_ms / avg_active : 0.0, avg_active, n, k, sg);
}

struct DownAsyncTraceState {
  uint64_t calls = 0;
  uint64_t total_us = 0;
  uint64_t active_sum = 0;
};

inline DownAsyncTraceState& down_async_trace_state(int layer_idx) {
  static DownAsyncTraceState states[128];
  return states[std::clamp(layer_idx, 0, 127)];
}

inline void trace_down_async_timing(int layer_idx, int active, int n, int k, int sg, uint64_t elapsed_us,
                                    bool pipeline = false) {
  const int trace_every = env_int("KT_SYCL_INT4_TRACE_EVERY", 0);
  if (trace_every <= 0) return;
  auto& state = down_async_trace_state(layer_idx);
  state.calls++;
  state.total_us += elapsed_us;
  state.active_sum += (uint64_t)std::max(active, 0);
  if ((state.calls % (uint64_t)trace_every) != 0) return;
  const double inv = 1.0 / (double)state.calls;
  const double avg_active = (double)state.active_sum * inv;
  const double avg_total_ms = (double)state.total_us * inv / 1000.0;
  std::printf(
      "[SYCL_GPTQ_INT4 down_async] kernel=subgroup layer=%d calls=%llu avg_total=%.3fms "
      "avg_per_active=%.3fms avg_active=%.2f n=%d k=%d sg=%d pipeline=%d\n",
      layer_idx, (unsigned long long)state.calls, avg_total_ms,
      avg_active > 0.0 ? avg_total_ms / avg_active : 0.0, avg_active, n, k, sg, pipeline ? 1 : 0);
}

}  // namespace sycl_int4

template <class T = sycl_int4::GemmKernelSYCLGPTQInt4>
class SYCL_GPTQ_INT4_MOE_TP : public AVX2_MOE_BASE<T, SYCL_GPTQ_INT4_MOE_TP<T>> {
  using Base = AVX2_MOE_BASE<T, SYCL_GPTQ_INT4_MOE_TP<T>>;
  using Base::config_;
  using Base::down_ba_;
  using Base::down_bb_;
  using Base::down_bc_;
  using Base::gate_bb_;
  using Base::gate_bc_;
  using Base::gate_up_ba_;
  using Base::fused_scratch_;
  using Base::m_expert_id_map_;
  using Base::m_local_down_output_ptr_;
  using Base::m_local_num_;
  using Base::tp_part_idx;
  using Base::up_bb_;
  using Base::up_bc_;

 public:
  using typename Base::input_t;
  using typename Base::output_t;

  SYCL_GPTQ_INT4_MOE_TP() = default;
  SYCL_GPTQ_INT4_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    T::config();
    auto& qc = config_.quant_config;
    if (qc.group_size == 0 || (qc.group_size % 8) != 0)
      throw std::runtime_error("SYCL GPTQ-Int4 requires group_size > 0 and divisible by 8");
    std::printf("Created SYCL_GPTQ_INT4_MOE_TP %d at numa %d (group_size=%d)\n", tp_part_idx,
                numa_node_of_cpu(sched_getcpu()), qc.group_size);
  }
  ~SYCL_GPTQ_INT4_MOE_TP() = default;

  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    return T::BufferB::required_size(n, k, config_.quant_config.group_size);
  }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* d) const {
    return std::make_shared<typename T::BufferA>(m, k, d);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* d) const {
    return std::make_shared<typename T::BufferB>(n, k, config_.quant_config.group_size, d);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* d) const {
    return std::make_shared<typename T::BufferC>(m, n, d);
  }

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];
    const char* kind = do_up ? "up" : "gate";
    sycl_int4::trace_gemm_limited(kind, config_.layer_idx, expert_idx, m, config_.intermediate_size,
                                  config_.hidden_size);
    const bool use_subgroup =
        qlen == 1 && (sycl_int4::env_eq("KT_SYCL_INT4_PER_GEMM_KERNEL", "subgroup") ||
                      sycl_int4::env_eq("KT_SYCL_INT4_PER_GEMM_KERNEL", "sg"));
    const int subgroup_size = sycl_int4::env_int("KT_SYCL_INT4_PER_GEMM_SUBGROUP", 32);
    const bool trace = qlen == 1 && sycl_int4::env_int("KT_SYCL_INT4_TRACE_EVERY", 0) > 0;
    const uint64_t t0 = trace ? sycl_int4::now_us() : 0;
    if (use_subgroup) {
      sycl_int4::gemm_gptq_int4_sycl_subgroup_dispatch(subgroup_size, m, config_.intermediate_size,
                                                       config_.hidden_size, *ba, *bb, *bc, ith, nth);
    } else {
      sycl_int4::gemm_gptq_int4_sycl(m, config_.intermediate_size, config_.hidden_size, *ba, *bb, *bc, ith, nth);
    }
    if (trace) {
      sycl_int4::trace_per_gemm_timing(kind, do_up ? 1 : 0, config_.layer_idx, expert_idx, m,
                                       config_.intermediate_size, config_.hidden_size, sycl_int4::now_us() - t0,
                                       use_subgroup ? "subgroup" : "scalar");
    }
  }
  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];
    sycl_int4::trace_gemm_limited("down", config_.layer_idx, expert_idx, m, config_.hidden_size,
                                  config_.intermediate_size);
    const bool use_subgroup =
        qlen == 1 && (sycl_int4::env_eq("KT_SYCL_INT4_PER_GEMM_KERNEL", "subgroup") ||
                      sycl_int4::env_eq("KT_SYCL_INT4_PER_GEMM_KERNEL", "sg"));
    const int subgroup_size = sycl_int4::env_int("KT_SYCL_INT4_PER_GEMM_SUBGROUP", 32);
    const bool trace = qlen == 1 && sycl_int4::env_int("KT_SYCL_INT4_TRACE_EVERY", 0) > 0;
    const uint64_t t0 = trace ? sycl_int4::now_us() : 0;
    if (use_subgroup) {
      sycl_int4::gemm_gptq_int4_sycl_subgroup_dispatch(subgroup_size, m, config_.hidden_size,
                                                       config_.intermediate_size, *down_ba_[expert_idx],
                                                       *down_bb_[expert_idx], *down_bc_[expert_idx], ith, nth);
    } else {
      sycl_int4::gemm_gptq_int4_sycl(m, config_.hidden_size, config_.intermediate_size, *down_ba_[expert_idx],
                                     *down_bb_[expert_idx], *down_bc_[expert_idx], ith, nth);
    }
    if (trace) {
      sycl_int4::trace_per_gemm_timing("down", 2, config_.layer_idx, expert_idx, m, config_.hidden_size,
                                       config_.intermediate_size, sycl_int4::now_us() - t0,
                                       use_subgroup ? "subgroup" : "scalar");
    }
  }

  bool use_fused_down_decode() const {
    const bool use_subgroup = sycl_int4::env_eq("KT_SYCL_INT4_PER_GEMM_KERNEL", "subgroup") ||
                              sycl_int4::env_eq("KT_SYCL_INT4_PER_GEMM_KERNEL", "sg");
    return use_subgroup && sycl_int4::env_flag("KT_SYCL_INT4_DOWN_ASYNC", true);
  }

  void decode_down_projection(int activated_expert, int qlen) {
    if (qlen != 1 || activated_expert <= 0) return;
    T::config();
    const int subgroup_size = sycl_int4::env_int("KT_SYCL_INT4_PER_GEMM_SUBGROUP", 32);
    const bool trace = sycl_int4::env_int("KT_SYCL_INT4_TRACE_EVERY", 0) > 0;
    auto* s = reinterpret_cast<sycl_int4::FusedScratch*>(fused_scratch_);
    const bool use_pipeline = sycl_int4::env_flag("KT_SYCL_INT4_GATE_UP_DOWN_PIPELINE", false);
    const bool has_pipeline =
        use_pipeline && s != nullptr && s->gate_up_pipeline_pending &&
        s->gate_up_pipeline_active == activated_expert &&
        (int)s->gate_up_pipeline_events.size() >= activated_expert &&
        (int)s->gate_up_pipeline_experts.size() >= activated_expert;
    std::vector<sycl::event> events;
    events.reserve((size_t)activated_expert);
    sycl_int4::trace_gemm_limited("down_async", config_.layer_idx, -1, qlen, config_.hidden_size,
                                  config_.intermediate_size);
    const uint64_t t0 = trace ? sycl_int4::now_us() : 0;
    for (int task_id = 0; task_id < activated_expert; ++task_id) {
      const int expert_idx = m_expert_id_map_[task_id];
      const int m = m_local_num_[expert_idx];
      const sycl::event* dependency =
          has_pipeline && s->gate_up_pipeline_experts[(size_t)task_id] == expert_idx
              ? &s->gate_up_pipeline_events[(size_t)task_id]
              : nullptr;
      events.push_back(sycl_int4::gemm_gptq_int4_sycl_subgroup_submit_dispatch(
          subgroup_size, m, config_.hidden_size, config_.intermediate_size, *down_ba_[expert_idx],
          *down_bb_[expert_idx], *down_bc_[expert_idx], 0, 1, dependency));
    }
    for (auto& ev : events) ev.wait_and_throw();
    if (s != nullptr) {
      s->gate_up_pipeline_pending = false;
      s->gate_up_pipeline_active = 0;
    }
    for (int task_id = 0; task_id < activated_expert; ++task_id) {
      const int expert_idx = m_expert_id_map_[task_id];
      down_bc_[expert_idx]->to_mat(qlen, m_local_down_output_ptr_[expert_idx], 0, 1);
    }
    if (trace) {
      sycl_int4::trace_down_async_timing(config_.layer_idx, activated_expert, config_.hidden_size,
                                         config_.intermediate_size, subgroup_size, sycl_int4::now_us() - t0,
                                         has_pipeline);
    }
  }

  bool use_fused_gate_up_decode() const {
    return sycl_int4::env_flag("KT_SYCL_INT4_GATE_UP_FUSE", false) ||
           sycl_int4::env_eq("KT_SYCL_INT4_DECODE_MODE", "per_gemm_gateup") ||
           sycl_int4::env_eq("KT_SYCL_INT4_DECODE_MODE", "per_gemm_gateup_batch") ||
           sycl_int4::env_eq("KT_SYCL_INT4_DECODE_MODE", "gateup") ||
           sycl_int4::env_eq("KT_SYCL_INT4_DECODE_MODE", "gateup_batch") ||
           sycl_int4::env_eq("KT_SYCL_INT4_DECODE_MODE", "gate_up") ||
           sycl_int4::env_eq("KT_SYCL_INT4_DECODE_MODE", "gate_up_batch");
  }

  void decode_gate_up_activation(int activated_expert, int qlen) {
    if (qlen != 1 || activated_expert <= 0) return;
    const int subgroup_size = sycl_int4::env_int("KT_SYCL_INT4_PER_GEMM_SUBGROUP", 32);
    const bool trace = sycl_int4::env_int("KT_SYCL_INT4_TRACE_EVERY", 0) > 0;
    const bool fast_silu = sycl_int4::fast_silu_enabled();
    const bool use_q8_gate_up = sycl_int4::env_flag("KT_SYCL_INT4_GATE_UP_Q8", false);
    const int gate_up_subgroup_size =
        use_q8_gate_up ? sycl_int4::env_int("KT_SYCL_INT4_GATE_UP_Q8_SUBGROUP", subgroup_size) : subgroup_size;
    const bool use_batch = sycl_int4::env_flag("KT_SYCL_INT4_GATE_UP_BATCH", false) ||
                           sycl_int4::env_eq("KT_SYCL_INT4_DECODE_MODE", "per_gemm_gateup_batch") ||
                           sycl_int4::env_eq("KT_SYCL_INT4_DECODE_MODE", "gateup_batch") ||
                           sycl_int4::env_eq("KT_SYCL_INT4_DECODE_MODE", "gate_up_batch");
    if (use_batch) {
      auto* s = reinterpret_cast<sycl_int4::FusedScratch*>(fused_scratch_);
      if (s == nullptr) {
        s = new sycl_int4::FusedScratch();
        fused_scratch_ = s;
      }
      if (s->gu_cap < activated_expert) {
        sycl_int4::usm_free(s->gu_a);
        sycl_int4::usm_free(s->gu_out);
        sycl_int4::usm_free(s->gu_gq);
        sycl_int4::usm_free(s->gu_uq);
        sycl_int4::usm_free(s->gu_gs);
        sycl_int4::usm_free(s->gu_us);
        const int cap = std::max(activated_expert, 1);
        s->gu_a = sycl_int4::usm_alloc<uint16_t*>(cap, "gate_up batch a ptrs");
        s->gu_out = sycl_int4::usm_alloc<uint16_t*>(cap, "gate_up batch out ptrs");
        s->gu_gq = sycl_int4::usm_alloc<uint32_t*>(cap, "gate_up batch gate q ptrs");
        s->gu_uq = sycl_int4::usm_alloc<uint32_t*>(cap, "gate_up batch up q ptrs");
        s->gu_gs = sycl_int4::usm_alloc<float*>(cap, "gate_up batch gate scale ptrs");
        s->gu_us = sycl_int4::usm_alloc<float*>(cap, "gate_up batch up scale ptrs");
        s->gu_cap = cap;
      }
      for (int task_id = 0; task_id < activated_expert; ++task_id) {
        const int expert_idx = m_expert_id_map_[task_id];
        s->gu_a[task_id] = gate_up_ba_[expert_idx]->data;
        s->gu_out[task_id] = down_ba_[expert_idx]->data;
        s->gu_gq[task_id] = gate_bb_[expert_idx]->qw;
        s->gu_uq[task_id] = up_bb_[expert_idx]->qw;
        s->gu_gs[task_id] = gate_bb_[expert_idx]->scales;
        s->gu_us[task_id] = up_bb_[expert_idx]->scales;
      }
      sycl_int4::trace_gemm_limited("gate_up_batch", config_.layer_idx, -1, qlen, config_.intermediate_size,
                                    config_.hidden_size);
      const uint64_t t0 = trace ? sycl_int4::now_us() : 0;
      sycl_int4::gate_up_activation_gptq_int4_sycl_subgroup_batched_dispatch(
          subgroup_size, activated_expert, qlen, config_.intermediate_size, config_.hidden_size, s->gu_a, s->gu_gq,
          s->gu_uq, s->gu_gs, s->gu_us, s->gu_out, config_.quant_config.group_size, config_.swiglu_limit,
          config_.swiglu_alpha, fast_silu);
      if (trace) {
        sycl_int4::trace_gate_up_batch_timing(config_.layer_idx, activated_expert, config_.intermediate_size,
                                              config_.hidden_size, subgroup_size, sycl_int4::now_us() - t0);
      }
      return;
    }
    sycl_int4::FusedScratch* q8_scratch = nullptr;
    if (use_q8_gate_up) {
      const int first_expert = m_expert_id_map_[0];
      q8_scratch = reinterpret_cast<sycl_int4::FusedScratch*>(fused_scratch_);
      if (q8_scratch == nullptr) {
        q8_scratch = new sycl_int4::FusedScratch();
        fused_scratch_ = q8_scratch;
      }
      if (q8_scratch->xq == nullptr) q8_scratch->xq = sycl_int4::usm_alloc<int8_t>(config_.hidden_size, "gate_up q8 xq");
      if (q8_scratch->xs == nullptr)
        q8_scratch->xs =
            sycl_int4::usm_alloc<float>(config_.hidden_size / config_.quant_config.group_size, "gate_up q8 xs");
      sycl_int4::quantize_bf16_i8_groups(gate_up_ba_[first_expert]->data, config_.hidden_size,
                                         config_.quant_config.group_size, q8_scratch->xq, q8_scratch->xs);
    }
    const bool use_async = sycl_int4::env_flag("KT_SYCL_INT4_GATE_UP_ASYNC", true);
    if (use_async) {
      const bool use_pipeline = sycl_int4::env_flag("KT_SYCL_INT4_GATE_UP_DOWN_PIPELINE", false) &&
                                sycl_int4::env_flag("KT_SYCL_INT4_DOWN_ASYNC", true);
      sycl_int4::FusedScratch* s = nullptr;
      std::vector<sycl::event> local_events;
      std::vector<sycl::event>* events = &local_events;
      if (use_pipeline) {
        s = reinterpret_cast<sycl_int4::FusedScratch*>(fused_scratch_);
        if (s == nullptr) {
          s = new sycl_int4::FusedScratch();
          fused_scratch_ = s;
        }
        s->gate_up_pipeline_events.clear();
        s->gate_up_pipeline_experts.clear();
        s->gate_up_pipeline_events.reserve((size_t)activated_expert);
        s->gate_up_pipeline_experts.reserve((size_t)activated_expert);
        s->gate_up_pipeline_active = activated_expert;
        s->gate_up_pipeline_pending = false;
        events = &s->gate_up_pipeline_events;
      }
      events->reserve((size_t)activated_expert);
      sycl_int4::trace_gemm_limited("gate_up_async", config_.layer_idx, -1, qlen, config_.intermediate_size,
                                    config_.hidden_size);
      const uint64_t t0 = trace ? sycl_int4::now_us() : 0;
      for (int task_id = 0; task_id < activated_expert; ++task_id) {
        const int expert_idx = m_expert_id_map_[task_id];
        const int m = m_local_num_[expert_idx];
        if (use_q8_gate_up) {
          events->push_back(sycl_int4::gate_up_activation_gptq_int4_sycl_subgroup_q8_submit_dispatch(
              gate_up_subgroup_size, m, config_.intermediate_size, config_.hidden_size, q8_scratch->xq,
              q8_scratch->xs, *gate_bb_[expert_idx], *up_bb_[expert_idx], *down_ba_[expert_idx],
              config_.swiglu_limit, config_.swiglu_alpha, fast_silu));
        } else {
          events->push_back(sycl_int4::gate_up_activation_gptq_int4_sycl_subgroup_submit_dispatch(
              subgroup_size, m, config_.intermediate_size, config_.hidden_size, *gate_up_ba_[expert_idx],
              *gate_bb_[expert_idx], *up_bb_[expert_idx], *down_ba_[expert_idx], config_.swiglu_limit,
              config_.swiglu_alpha, fast_silu));
        }
        if (use_pipeline) s->gate_up_pipeline_experts.push_back(expert_idx);
      }
      if (use_pipeline) {
        s->gate_up_pipeline_pending = true;
        if (trace) {
          sycl_int4::trace_gate_up_pipeline_submit_timing(config_.layer_idx, activated_expert,
                                                          config_.intermediate_size, config_.hidden_size,
                                                          use_q8_gate_up ? gate_up_subgroup_size : subgroup_size,
                                                          sycl_int4::now_us() - t0,
                                                          use_q8_gate_up ? "subgroup_fused_q8"
                                                                         : "subgroup_fused");
        }
        return;
      }
      for (auto& ev : *events) ev.wait_and_throw();
      if (trace) {
        sycl_int4::trace_gate_up_async_timing(config_.layer_idx, activated_expert, config_.intermediate_size,
                                              config_.hidden_size,
                                              use_q8_gate_up ? gate_up_subgroup_size : subgroup_size,
                                              sycl_int4::now_us() - t0,
                                              use_q8_gate_up ? "subgroup_fused_q8" : "subgroup_fused");
      }
      return;
    }
    for (int task_id = 0; task_id < activated_expert; ++task_id) {
      const int expert_idx = m_expert_id_map_[task_id];
      const int m = m_local_num_[expert_idx];
      sycl_int4::trace_gemm_limited("gate_up_act", config_.layer_idx, expert_idx, m, config_.intermediate_size,
                                    config_.hidden_size);
      const uint64_t t0 = trace ? sycl_int4::now_us() : 0;
      if (use_q8_gate_up) {
        sycl_int4::gate_up_activation_gptq_int4_sycl_subgroup_q8_submit_dispatch(
            gate_up_subgroup_size, m, config_.intermediate_size, config_.hidden_size, q8_scratch->xq,
            q8_scratch->xs, *gate_bb_[expert_idx], *up_bb_[expert_idx], *down_ba_[expert_idx],
            config_.swiglu_limit, config_.swiglu_alpha, fast_silu)
            .wait_and_throw();
      } else {
        sycl_int4::gate_up_activation_gptq_int4_sycl_subgroup_dispatch(
            subgroup_size, m, config_.intermediate_size, config_.hidden_size, *gate_up_ba_[expert_idx],
            *gate_bb_[expert_idx], *up_bb_[expert_idx], *down_ba_[expert_idx], config_.swiglu_limit,
            config_.swiglu_alpha, fast_silu);
      }
      if (trace) {
        sycl_int4::trace_per_gemm_timing("gate_up_act", 3, config_.layer_idx, expert_idx, m,
                                         config_.intermediate_size, config_.hidden_size,
                                         sycl_int4::now_us() - t0,
                                         use_q8_gate_up ? "subgroup_fused_q8" : "subgroup_fused");
      }
    }
  }

  // ---- Milestone 2: single fused decode kernel (work-group per active expert) ----
  // One SYCL launch per layer instead of 3*k per-GEMM launches: gate/up -> SiLU (act in
  // SLM) -> down -> router-weighted atomic accumulate. Reads int4 directly (half the bytes
  // of the CPU int8 path) with in-kernel nibble unpack. Requires threadpool_count == 1
  // (one device, no TP shard of the intermediate dim; the launch scripts set this).
  static constexpr bool kFused = true;

  bool use_fused_decode() const {
    return !(use_fused_gate_up_decode() || sycl_int4::env_eq("KT_SYCL_INT4_DECODE_MODE", "per_gemm") ||
             sycl_int4::env_eq("KT_SYCL_INT4_DECODE_MODE", "gemm"));
  }

  void fused_decode(int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    const int H = config_.hidden_size, I = config_.intermediate_size, gs = config_.quant_config.group_size;
    const int ngH = H / gs, ngI = I / gs;
    auto& q = sycl_int4::queue();
    const int trace_every = sycl_int4::env_int("KT_SYCL_INT4_TRACE_EVERY", 0);
    const bool trace = trace_every > 0;
    const bool profile_events = trace && sycl_int4::queue_profiling_enabled();
    const bool fast_silu = sycl_int4::fast_silu_enabled();
    const bool pre_kernel_ping = sycl_int4::pre_kernel_ping_enabled();
    const bool shape_ping = sycl_int4::shape_ping_enabled();
    const bool weight_ping = sycl_int4::weight_ping_enabled();
    const uint64_t t0 = trace ? sycl_int4::now_us() : 0;
    const int maxk = config_.expert_num;  // safe upper bound (>= any active-expert count)
    auto* s = reinterpret_cast<sycl_int4::FusedScratch*>(fused_scratch_);
    if (s == nullptr) {
      s = new sycl_int4::FusedScratch();
      fused_scratch_ = s;
    }
    if (s->x == nullptr) {
      s->x = sycl_int4::usm_alloc<uint16_t>(H, "fused x");
      s->xq = sycl_int4::usm_alloc<int8_t>(H, "fused xq");
      s->xs = sycl_int4::usm_alloc<float>(ngH, "fused xs");
      s->out = sycl_int4::usm_alloc<float>(H, "fused out");
      s->ping_sink = sycl_int4::usm_alloc<float>(1, "fused ping sink");
      s->flat_act = sycl_int4::usm_alloc<float>((size_t)maxk * I, "flat act");
      s->flat_aq = sycl_int4::usm_alloc<int8_t>((size_t)maxk * I, "flat aq");
      s->flat_as = sycl_int4::usm_alloc<float>((size_t)maxk * ngI, "flat as");
      s->gq = sycl_int4::usm_alloc<uint32_t*>(maxk, "fused gq");
      s->uq = sycl_int4::usm_alloc<uint32_t*>(maxk, "fused uq");
      s->dq = sycl_int4::usm_alloc<uint32_t*>(maxk, "fused dq");
      s->gs = sycl_int4::usm_alloc<float*>(maxk, "fused gs");
      s->us = sycl_int4::usm_alloc<float*>(maxk, "fused us");
      s->ds = sycl_int4::usm_alloc<float*>(maxk, "fused ds");
      s->rw = sycl_int4::usm_alloc<float>(maxk, "fused rw");
    }
    const bool device_scratch = sycl_int4::device_scratch_enabled();
    if (device_scratch && s->xq_dev == nullptr) {
      s->xq_dev = sycl_int4::usm_alloc_device<int8_t>(H, "fused xq device");
      s->xs_dev = sycl_int4::usm_alloc_device<float>(ngH, "fused xs device");
      s->out_dev = sycl_int4::usm_alloc_device<float>(H, "fused out device");
      s->rw_dev = sycl_int4::usm_alloc_device<float>(maxk, "fused rw device");
      s->gq_dev = sycl_int4::usm_alloc_device<uint32_t*>(maxk, "fused gq device");
      s->uq_dev = sycl_int4::usm_alloc_device<uint32_t*>(maxk, "fused uq device");
      s->dq_dev = sycl_int4::usm_alloc_device<uint32_t*>(maxk, "fused dq device");
      s->gs_dev = sycl_int4::usm_alloc_device<float*>(maxk, "fused gs device");
      s->us_dev = sycl_int4::usm_alloc_device<float*>(maxk, "fused us device");
      s->ds_dev = sycl_int4::usm_alloc_device<float*>(maxk, "fused ds device");
    }
	    int na = 0;
    std::vector<int> active_expert_ids;
    active_expert_ids.reserve((size_t)k);
    for (int j = 0; j < k && na < maxk; ++j) {
      int64_t e = expert_ids[j];
      if (config_.should_skip_expert(e)) continue;
      s->gq[na] = gate_bb_[e]->qw; s->uq[na] = up_bb_[e]->qw; s->dq[na] = down_bb_[e]->qw;
      s->gs[na] = gate_bb_[e]->scales; s->us[na] = up_bb_[e]->scales; s->ds[na] = down_bb_[e]->scales;
      s->rw[na] = weights[j];
      active_expert_ids.push_back((int)e);
      ++na;
    }
    const bool stage_active = sycl_int4::env_flag("KT_SYCL_INT4_STAGE_ACTIVE", false);
    std::vector<sycl::event> stage_events;
    const uint64_t t_stage_begin = trace ? sycl_int4::now_us() : 0;
    if (stage_active && na > 0) {
      const int stage_cap = std::max({na, k, config_.num_experts_per_tok, 1});
      const size_t gu_qw_elems = (size_t)(H / 8) * I;
      const size_t gu_sc_elems = (size_t)ngH * I;
      const size_t d_qw_elems = (size_t)(I / 8) * H;
      const size_t d_sc_elems = (size_t)ngI * H;
      if (s->stage_cap < stage_cap) {
        sycl_int4::usm_free(s->stage_gq);
        sycl_int4::usm_free(s->stage_uq);
        sycl_int4::usm_free(s->stage_dq);
        sycl_int4::usm_free(s->stage_gs);
        sycl_int4::usm_free(s->stage_us);
        sycl_int4::usm_free(s->stage_ds);
        s->stage_gq = sycl_int4::usm_alloc_device<uint32_t>((size_t)stage_cap * gu_qw_elems, "stage gate q");
        s->stage_uq = sycl_int4::usm_alloc_device<uint32_t>((size_t)stage_cap * gu_qw_elems, "stage up q");
        s->stage_dq = sycl_int4::usm_alloc_device<uint32_t>((size_t)stage_cap * d_qw_elems, "stage down q");
        s->stage_gs = sycl_int4::usm_alloc_device<float>((size_t)stage_cap * gu_sc_elems, "stage gate scales");
        s->stage_us = sycl_int4::usm_alloc_device<float>((size_t)stage_cap * gu_sc_elems, "stage up scales");
        s->stage_ds = sycl_int4::usm_alloc_device<float>((size_t)stage_cap * d_sc_elems, "stage down scales");
        s->stage_cap = stage_cap;
      }
      for (int a = 0; a < na; ++a) {
        const uint32_t* src_gq = s->gq[a];
        const uint32_t* src_uq = s->uq[a];
        const uint32_t* src_dq = s->dq[a];
        const float* src_gs = s->gs[a];
        const float* src_us = s->us[a];
        const float* src_ds = s->ds[a];
        uint32_t* dst_gq = s->stage_gq + (size_t)a * gu_qw_elems;
        uint32_t* dst_uq = s->stage_uq + (size_t)a * gu_qw_elems;
        uint32_t* dst_dq = s->stage_dq + (size_t)a * d_qw_elems;
        float* dst_gs = s->stage_gs + (size_t)a * gu_sc_elems;
        float* dst_us = s->stage_us + (size_t)a * gu_sc_elems;
        float* dst_ds = s->stage_ds + (size_t)a * d_sc_elems;
        stage_events.push_back(q.memcpy(dst_gq, src_gq, gu_qw_elems * sizeof(uint32_t)));
        stage_events.push_back(q.memcpy(dst_uq, src_uq, gu_qw_elems * sizeof(uint32_t)));
        stage_events.push_back(q.memcpy(dst_dq, src_dq, d_qw_elems * sizeof(uint32_t)));
        stage_events.push_back(q.memcpy(dst_gs, src_gs, gu_sc_elems * sizeof(float)));
        stage_events.push_back(q.memcpy(dst_us, src_us, gu_sc_elems * sizeof(float)));
        stage_events.push_back(q.memcpy(dst_ds, src_ds, d_sc_elems * sizeof(float)));
        s->gq[a] = dst_gq;
        s->uq[a] = dst_uq;
        s->dq[a] = dst_dq;
        s->gs[a] = dst_gs;
        s->us[a] = dst_us;
        s->ds[a] = dst_ds;
      }
    }
    const uint64_t t_after_stage_submit = trace ? sycl_int4::now_us() : 0;
    const bool stage_wait = stage_active && sycl_int4::env_flag("KT_SYCL_INT4_STAGE_WAIT", false);
    if (stage_wait) q.wait_and_throw();
    const uint64_t t_after_stage_wait = trace ? sycl_int4::now_us() : t_after_stage_submit;
    const bool force_bf16 = sycl_int4::env_eq("KT_SYCL_INT4_DECODE_MODE", "bf16") ||
                            !sycl_int4::env_flag("KT_SYCL_INT4_DECODE_QUANT", true);
    const bool use_flat = !force_bf16 && !sycl_int4::env_eq("KT_SYCL_INT4_DECODE_MODE", "fused");
    const bool host_zero = sycl_int4::env_flag("KT_SYCL_INT4_HOST_ZERO", true);
    if (use_flat) {
      sycl_int4::quantize_bf16_i8_groups((const uint16_t*)input, H, gs, s->xq, s->xs);
      if (na == 0) { std::memset(output, 0, (size_t)H * sizeof(float)); return; }

      int8_t* xq = s->xq;
      float* xs = s->xs;
      float* out = s->out;
      float* act = s->flat_act;
      int8_t* aq = s->flat_aq;
      float* as = s->flat_as;
      float* rw = s->rw;
      uint32_t **gq = s->gq, **uq = s->uq, **dq = s->dq;
      float **gsc = s->gs, **usc = s->us, **dsc = s->ds;
      const uint64_t t_before_submit = trace ? sycl_int4::now_us() : 0;
      const int packed_per_group = gs / 8;
      const size_t flat_wg = (size_t)std::clamp(sycl_int4::env_int("KT_SYCL_INT4_FLAT_WG", 64), 1, 256);
      const size_t gate_total = (size_t)na * I;
      const size_t actq_total = (size_t)na * ngI;
      const size_t down_total = (size_t)na * H;
      const auto round_wg = [flat_wg](size_t n) { return ((n + flat_wg - 1) / flat_wg) * flat_wg; };
      q.memset(out, 0, (size_t)H * sizeof(float));
      q.parallel_for(sycl::nd_range<1>(sycl::range<1>(round_wg(gate_total)), sycl::range<1>(flat_wg)),
                     [=](sycl::nd_item<1> it) {
        const size_t idx = it.get_global_linear_id();
        if (idx >= gate_total) return;
        const int a = (int)(idx / I), i = (int)(idx - (size_t)a * I);
        const uint32_t* wg = gq[a]; const uint32_t* wu = uq[a];
        const float* sg = gsc[a]; const float* su = usc[a];
        float gsum = 0.f, usum = 0.f;
        for (int g = 0; g < ngH; ++g) {
          int gd = 0, ud = 0;
          const int kp0 = g * packed_per_group;
          for (int kp = kp0; kp < kp0 + packed_per_group; ++kp) {
            const uint32_t pg = wg[(size_t)kp * I + i];
            const uint32_t pu = wu[(size_t)kp * I + i];
            const int8_t* xb = xq + (size_t)kp * 8;
#pragma unroll
            for (int bb = 0; bb < 8; ++bb) {
              const int xv = (int)xb[bb];
              gd += xv * ((int)((pg >> (bb * 4)) & 0xF) - 8);
              ud += xv * ((int)((pu >> (bb * 4)) & 0xF) - 8);
            }
          }
          gsum += (float)gd * xs[g] * sg[(size_t)g * I + i];
          usum += (float)ud * xs[g] * su[(size_t)g * I + i];
        }
        const float gg = fast_silu
                             ? (gsum > 20.f ? gsum
                                            : (gsum < -20.f ? 0.f : gsum / (1.f + sycl::native::exp(-gsum))))
                             : gsum / (1.f + sycl::exp(-gsum));
        act[(size_t)a * I + i] = gg * usum;
      });
      q.parallel_for(sycl::nd_range<1>(sycl::range<1>(round_wg(actq_total)), sycl::range<1>(flat_wg)),
                     [=](sycl::nd_item<1> it) {
        const size_t idx = it.get_global_linear_id();
        if (idx >= actq_total) return;
        const int a = (int)(idx / ngI), g = (int)(idx - (size_t)a * ngI);
        float amax = 0.f;
        const size_t base = (size_t)a * I + (size_t)g * gs;
        for (int t = 0; t < gs; ++t) amax = sycl::fmax(amax, sycl::fabs(act[base + t]));
        const float sact = amax > 0.f ? amax / 127.f : 0.f;
        as[(size_t)a * ngI + g] = sact;
        const float inv = sact > 0.f ? 1.f / sact : 0.f;
        for (int t = 0; t < gs; ++t) {
          const int qv = sycl::clamp((int)sycl::rint(act[base + t] * inv), -127, 127);
          aq[base + t] = (int8_t)qv;
        }
      });
      q.parallel_for(sycl::nd_range<1>(sycl::range<1>(round_wg(down_total)), sycl::range<1>(flat_wg)),
                     [=](sycl::nd_item<1> it) {
        const size_t idx = it.get_global_linear_id();
        if (idx >= down_total) return;
        const int a = (int)(idx / H), hh = (int)(idx - (size_t)a * H);
        const uint32_t* wd = dq[a];
        const float* sd = dsc[a];
        float acc = 0.f;
        for (int g = 0; g < ngI; ++g) {
          int dot = 0;
          const int kp0 = g * packed_per_group;
          for (int kp = kp0; kp < kp0 + packed_per_group; ++kp) {
            const uint32_t pd = wd[(size_t)kp * H + hh];
            const int8_t* ab = aq + (size_t)a * I + (size_t)kp * 8;
#pragma unroll
            for (int bb = 0; bb < 8; ++bb) {
              dot += (int)ab[bb] * ((int)((pd >> (bb * 4)) & 0xF) - 8);
            }
          }
          acc += (float)dot * as[(size_t)a * ngI + g] * sd[(size_t)g * H + hh];
        }
        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            ar(out[hh]);
        ar.fetch_add(rw[a] * acc);
      }).wait_and_throw();
      const uint64_t t_after_kernel = trace ? sycl_int4::now_us() : 0;
      std::memcpy(output, out, (size_t)H * sizeof(float));
      if (trace) {
        const uint64_t t_end = sycl_int4::now_us();
        s->calls++;
        s->active_sum += (uint64_t)na;
        s->total_us += t_end - t0;
        s->submit_us += t_before_submit - t0;
        s->stage_submit_us += t_after_stage_submit - t_stage_begin;
        s->stage_wait_us += t_after_stage_wait - t_after_stage_submit;
        s->kernel_wait_us += t_after_kernel - t_before_submit;
        s->copy_us += t_end - t_after_kernel;
        if ((s->calls % (uint64_t)trace_every) == 0) {
          const double inv = 1.0 / (double)s->calls;
          std::printf(
              "[SYCL_GPTQ_INT4 fused] mode=flat_i8 layer=%d calls=%llu avg_total=%.3fms avg_submit=%.3fms "
              "avg_stage_submit=%.3fms avg_stage_wait=%.3fms avg_kernel_wait=%.3fms avg_copy=%.3fms avg_active=%.2f "
              "H=%d I=%d topk=%d wg=%zu stage=%d\n",
              config_.layer_idx, (unsigned long long)s->calls, (double)s->total_us * inv / 1000.0,
              (double)s->submit_us * inv / 1000.0, (double)s->stage_submit_us * inv / 1000.0,
              (double)s->stage_wait_us * inv / 1000.0, (double)s->kernel_wait_us * inv / 1000.0,
              (double)s->copy_us * inv / 1000.0,
              (double)s->active_sum * inv, H, I, k, flat_wg, stage_active ? 1 : 0);
        }
      }
      return;
    }
    if (!force_bf16) {
      sycl_int4::quantize_bf16_i8_groups((const uint16_t*)input, H, gs, s->xq, s->xs);
      if (na == 0) { std::memset(output, 0, (size_t)H * sizeof(float)); return; }

      if (H == 2048 && I == 512 && gs == 128 && sycl_int4::env_flag("KT_SYCL_INT4_SPECIALIZE", true)) {
        constexpr int FH = 2048, FI = 512, FGS = 128;
        constexpr int FNGH = FH / FGS, FNGI = FI / FGS, FPACK = FGS / 8;
        constexpr int WG = 256;
        int8_t* xq = s->xq;
        float* xs = s->xs;
        float* out = s->out;
        float* rw = s->rw;
        uint32_t **gq = s->gq, **uq = s->uq, **dq = s->dq;
        float **gsc = s->gs, **usc = s->us, **dsc = s->ds;
        const int cache_slot_req = sycl_int4::env_int("KT_SYCL_INT4_DEVICE_CACHE_SLOTS", 0);
        const int cache_layer_filter = sycl_int4::env_int("KT_SYCL_INT4_DEVICE_CACHE_LAYER", -1);
        const bool cache_layer_match = cache_layer_filter < 0 || cache_layer_filter == config_.layer_idx;
        const bool cache_preload = sycl_int4::env_flag("KT_SYCL_INT4_DEVICE_CACHE_PRELOAD", false);
        const bool cache_enabled = cache_slot_req > 0 && !stage_active && na > 0 && cache_layer_match;
        std::vector<sycl::event> cache_events;
        uint64_t cache_hits_local = 0, cache_misses_local = 0;
        int cache_slots_active = 0;
        const uint64_t t_cache_begin = (trace && cache_enabled) ? sycl_int4::now_us() : 0;
        if (cache_enabled) {
          cache_slots_active = std::clamp(cache_slot_req, 1, config_.expert_num);
          const size_t gu_qw_elems = (size_t)(FH / 8) * FI;
          const size_t gu_sc_elems = (size_t)FNGH * FI;
          const size_t d_qw_elems = (size_t)(FI / 8) * FH;
          const size_t d_sc_elems = (size_t)FNGI * FH;
          if (s->cache_cap != cache_slots_active) {
            sycl_int4::usm_free(s->cache_gq);
            sycl_int4::usm_free(s->cache_uq);
            sycl_int4::usm_free(s->cache_dq);
            sycl_int4::usm_free(s->cache_gs);
            sycl_int4::usm_free(s->cache_us);
            sycl_int4::usm_free(s->cache_ds);
            s->cache_gq = sycl_int4::usm_alloc_device<uint32_t>((size_t)cache_slots_active * gu_qw_elems,
                                                                "cache gate q");
            s->cache_uq = sycl_int4::usm_alloc_device<uint32_t>((size_t)cache_slots_active * gu_qw_elems,
                                                                "cache up q");
            s->cache_dq = sycl_int4::usm_alloc_device<uint32_t>((size_t)cache_slots_active * d_qw_elems,
                                                                "cache down q");
            s->cache_gs = sycl_int4::usm_alloc_device<float>((size_t)cache_slots_active * gu_sc_elems,
                                                             "cache gate scales");
            s->cache_us = sycl_int4::usm_alloc_device<float>((size_t)cache_slots_active * gu_sc_elems,
                                                             "cache up scales");
            s->cache_ds = sycl_int4::usm_alloc_device<float>((size_t)cache_slots_active * d_sc_elems,
                                                             "cache down scales");
            s->cache_cap = cache_slots_active;
            s->cache_expert_ids.assign((size_t)cache_slots_active, -1);
            s->cache_lru.assign((size_t)cache_slots_active, 0);
            s->cache_tick = 0;
            s->cache_preloaded = false;
          }
          auto copy_expert_to_cache = [&](int slot, int expert_id) {
            uint32_t* dst_gq = s->cache_gq + (size_t)slot * gu_qw_elems;
            uint32_t* dst_uq = s->cache_uq + (size_t)slot * gu_qw_elems;
            uint32_t* dst_dq = s->cache_dq + (size_t)slot * d_qw_elems;
            float* dst_gs = s->cache_gs + (size_t)slot * gu_sc_elems;
            float* dst_us = s->cache_us + (size_t)slot * gu_sc_elems;
            float* dst_ds = s->cache_ds + (size_t)slot * d_sc_elems;
            cache_events.push_back(q.memcpy(dst_gq, gate_bb_[expert_id]->qw, gu_qw_elems * sizeof(uint32_t)));
            cache_events.push_back(q.memcpy(dst_uq, up_bb_[expert_id]->qw, gu_qw_elems * sizeof(uint32_t)));
            cache_events.push_back(q.memcpy(dst_dq, down_bb_[expert_id]->qw, d_qw_elems * sizeof(uint32_t)));
            cache_events.push_back(q.memcpy(dst_gs, gate_bb_[expert_id]->scales, gu_sc_elems * sizeof(float)));
            cache_events.push_back(q.memcpy(dst_us, up_bb_[expert_id]->scales, gu_sc_elems * sizeof(float)));
            cache_events.push_back(q.memcpy(dst_ds, down_bb_[expert_id]->scales, d_sc_elems * sizeof(float)));
            s->cache_expert_ids[(size_t)slot] = expert_id;
            s->cache_lru[(size_t)slot] = ++s->cache_tick;
          };
          if (cache_preload && !s->cache_preloaded) {
            int slot = 0;
            for (int expert_id = 0; expert_id < config_.expert_num && slot < s->cache_cap; ++expert_id) {
              if (config_.should_skip_expert(expert_id)) continue;
              copy_expert_to_cache(slot, expert_id);
              ++slot;
            }
            s->cache_preloaded = true;
          }
          for (int a = 0; a < na; ++a) {
            const int expert_id = active_expert_ids[(size_t)a];
            int slot = -1;
            for (int i = 0; i < s->cache_cap; ++i) {
              if (s->cache_expert_ids[(size_t)i] == expert_id) {
                slot = i;
                break;
              }
            }
            if (slot >= 0) {
              ++cache_hits_local;
            } else {
              ++cache_misses_local;
              int victim = -1;
              for (int i = 0; i < s->cache_cap; ++i) {
                if (s->cache_expert_ids[(size_t)i] < 0) {
                  victim = i;
                  break;
                }
              }
              if (victim < 0) {
                victim = 0;
                uint64_t best = s->cache_lru[0];
                for (int i = 1; i < s->cache_cap; ++i) {
                  if (s->cache_lru[(size_t)i] < best) {
                    best = s->cache_lru[(size_t)i];
                    victim = i;
                  }
                }
              }
              slot = victim;
              copy_expert_to_cache(slot, expert_id);
            }
            s->cache_lru[(size_t)slot] = ++s->cache_tick;
            s->gq[a] = s->cache_gq + (size_t)slot * gu_qw_elems;
            s->uq[a] = s->cache_uq + (size_t)slot * gu_qw_elems;
            s->dq[a] = s->cache_dq + (size_t)slot * d_qw_elems;
            s->gs[a] = s->cache_gs + (size_t)slot * gu_sc_elems;
            s->us[a] = s->cache_us + (size_t)slot * gu_sc_elems;
            s->ds[a] = s->cache_ds + (size_t)slot * d_sc_elems;
          }
        }
        const uint64_t t_after_cache_submit = (trace && cache_enabled) ? sycl_int4::now_us() : t_cache_begin;
        const uint64_t t_before_submit = trace ? sycl_int4::now_us() : 0;
        sycl::event ping_ev;
        sycl_int4::EventTiming ping_timing;
        if (pre_kernel_ping) {
          ping_ev = q.single_task([=]() {});
          ping_ev.wait_and_throw();
          ping_timing = profile_events ? sycl_int4::event_timing_us(ping_ev) : sycl_int4::EventTiming{};
        }
        std::vector<sycl::event> kernel_deps = stage_events;
        kernel_deps.insert(kernel_deps.end(), cache_events.begin(), cache_events.end());
        std::vector<sycl::event> control_events;
        if (device_scratch) {
          xq = s->xq_dev;
          xs = s->xs_dev;
          out = s->out_dev;
          rw = s->rw_dev;
          gq = s->gq_dev;
          uq = s->uq_dev;
          dq = s->dq_dev;
          gsc = s->gs_dev;
          usc = s->us_dev;
          dsc = s->ds_dev;
        }
        sycl::event memset_ev;
        if (host_zero && !device_scratch) {
          std::memset(out, 0, (size_t)FH * sizeof(float));
        } else {
          memset_ev = q.memset(out, 0, (size_t)FH * sizeof(float));
        }
        const uint64_t t_after_memset_submit = trace ? sycl_int4::now_us() : t_before_submit;
        if (device_scratch) {
          control_events.push_back(q.memcpy(s->xq_dev, s->xq, (size_t)FH * sizeof(int8_t)));
          control_events.push_back(q.memcpy(s->xs_dev, s->xs, (size_t)FNGH * sizeof(float)));
          control_events.push_back(q.memcpy(s->rw_dev, s->rw, (size_t)na * sizeof(float)));
          control_events.push_back(q.memcpy(s->gq_dev, s->gq, (size_t)na * sizeof(uint32_t*)));
          control_events.push_back(q.memcpy(s->uq_dev, s->uq, (size_t)na * sizeof(uint32_t*)));
          control_events.push_back(q.memcpy(s->dq_dev, s->dq, (size_t)na * sizeof(uint32_t*)));
          control_events.push_back(q.memcpy(s->gs_dev, s->gs, (size_t)na * sizeof(float*)));
          control_events.push_back(q.memcpy(s->us_dev, s->us, (size_t)na * sizeof(float*)));
          control_events.push_back(q.memcpy(s->ds_dev, s->ds, (size_t)na * sizeof(float*)));
          kernel_deps.insert(kernel_deps.end(), control_events.begin(), control_events.end());
        }
        const uint64_t t_after_control_submit = trace ? sycl_int4::now_us() : t_after_memset_submit;
        sycl::event shape_ev;
        sycl_int4::EventTiming shape_timing;
        if (shape_ping) {
          shape_ev = q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> actf(sycl::range<1>(FI), h);
            sycl::local_accessor<int8_t, 1> actq(sycl::range<1>(FI), h);
            sycl::local_accessor<float, 1> ascl(sycl::range<1>(FNGI), h);
            h.parallel_for(sycl::nd_range<1>((size_t)na * WG, WG), [=](sycl::nd_item<1> it) {
              const int lid = (int)it.get_local_id(0);
              if (lid < FI) {
                actf[lid] = 0.f;
                actq[lid] = 0;
              }
              if (lid < FNGI) ascl[lid] = 0.f;
              it.barrier(sycl::access::fence_space::local_space);
              if (it.get_group(0) == 0 && lid == 0) {
                out[0] = actf[0] + (float)actq[0] + ascl[0];
              }
            });
          });
          shape_ev.wait_and_throw();
          shape_timing = profile_events ? sycl_int4::event_timing_us(shape_ev) : sycl_int4::EventTiming{};
        }
        sycl::event weight_ev;
        sycl_int4::EventTiming weight_timing;
        if (weight_ping) {
          float* ping_sink = s->ping_sink;
          weight_ev = q.submit([&](sycl::handler& h) {
            if (!kernel_deps.empty()) h.depends_on(kernel_deps);
            h.parallel_for(sycl::nd_range<1>((size_t)na * WG, WG), [=](sycl::nd_item<1> it) {
              if (it.get_global_linear_id() != 0) return;
              float acc = 0.f;
              if (na > 0) {
                const uint32_t* wg0 = gq[0];
                const uint32_t* wu0 = uq[0];
                const uint32_t* wd0 = dq[0];
                const float* sg0 = gsc[0];
                const float* su0 = usc[0];
                const float* sd0 = dsc[0];
                acc += (float)((wg0[0] ^ wu0[0] ^ wd0[0]) & 0xffu);
                acc += sg0[0] + su0[0] + sd0[0] + xs[0] + rw[0] + (float)xq[0];
              }
              ping_sink[0] = acc;
            });
          });
          weight_ev.wait_and_throw();
          weight_timing = profile_events ? sycl_int4::event_timing_us(weight_ev) : sycl_int4::EventTiming{};
        }
        sycl::event kernel_ev = q.submit([&](sycl::handler& h) {
           if (!host_zero || device_scratch) h.depends_on(memset_ev);
           if (!kernel_deps.empty()) h.depends_on(kernel_deps);
		           sycl::local_accessor<float, 1> actf(sycl::range<1>(FI), h);
	           sycl::local_accessor<int8_t, 1> actq(sycl::range<1>(FI), h);
	           sycl::local_accessor<float, 1> ascl(sycl::range<1>(FNGI), h);
           h.parallel_for(sycl::nd_range<1>((size_t)na * WG, WG), [=](sycl::nd_item<1> it) {
             const int a = (int)it.get_group(0), lid = (int)it.get_local_id(0);
             const uint32_t* wg = gq[a]; const uint32_t* wu = uq[a];
             const float* sg = gsc[a]; const float* su = usc[a];
             for (int i = lid; i < FI; i += WG) {
               float gsum = 0.f, usum = 0.f;
#pragma unroll
               for (int g = 0; g < FNGH; ++g) {
                 int gd = 0, ud = 0;
                 const int kp0 = g * FPACK;
#pragma unroll
                 for (int kpi = 0; kpi < FPACK; ++kpi) {
                   const int kp = kp0 + kpi;
                   const uint32_t pg = wg[(size_t)kp * FI + i];
                   const uint32_t pu = wu[(size_t)kp * FI + i];
                   const int8_t* xb = xq + (size_t)kp * 8;
#pragma unroll
                   for (int bb = 0; bb < 8; ++bb) {
                     const int xv = (int)xb[bb];
                     gd += xv * ((int)((pg >> (bb * 4)) & 0xF) - 8);
                     ud += xv * ((int)((pu >> (bb * 4)) & 0xF) - 8);
                   }
                 }
                 gsum += (float)gd * xs[g] * sg[(size_t)g * FI + i];
                 usum += (float)ud * xs[g] * su[(size_t)g * FI + i];
               }
               const float gg = fast_silu
                                    ? (gsum > 20.f ? gsum
                                                   : (gsum < -20.f ? 0.f : gsum / (1.f + sycl::native::exp(-gsum))))
                                    : gsum / (1.f + sycl::exp(-gsum));
               actf[i] = gg * usum;
             }
             it.barrier(sycl::access::fence_space::local_space);
             for (int g = lid; g < FNGI; g += WG) {
               float amax = 0.f;
               const int base = g * FGS;
#pragma unroll
               for (int t = 0; t < FGS; ++t) amax = sycl::fmax(amax, sycl::fabs(actf[base + t]));
               const float sact = amax > 0.f ? amax / 127.f : 0.f;
               ascl[g] = sact;
               const float inv = sact > 0.f ? 1.f / sact : 0.f;
#pragma unroll
               for (int t = 0; t < FGS; ++t) {
                 const int qv = sycl::clamp((int)sycl::rint(actf[base + t] * inv), -127, 127);
                 actq[base + t] = (int8_t)qv;
               }
             }
             it.barrier(sycl::access::fence_space::local_space);
             const uint32_t* wd = dq[a]; const float* sd = dsc[a]; float rwa = rw[a];
             for (int hh = lid; hh < FH; hh += WG) {
               float acc = 0.f;
#pragma unroll
               for (int g = 0; g < FNGI; ++g) {
                 int dot = 0;
                 const int kp0 = g * FPACK;
#pragma unroll
                 for (int kpi = 0; kpi < FPACK; ++kpi) {
                   const int kp = kp0 + kpi;
                   const uint32_t pd = wd[(size_t)kp * FH + hh];
                   const int8_t* ab = &actq[kp * 8];
#pragma unroll
                   for (int bb = 0; bb < 8; ++bb) {
                     dot += (int)ab[bb] * ((int)((pd >> (bb * 4)) & 0xF) - 8);
                   }
                 }
                 acc += (float)dot * ascl[g] * sd[(size_t)g * FH + hh];
               }
               sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                   ar(out[hh]);
	               ar.fetch_add(rwa * acc);
	             }
	           });
	         });
        const uint64_t t_after_submit = trace ? sycl_int4::now_us() : t_before_submit;
        kernel_ev.wait_and_throw();
        const auto memset_timing = profile_events ? sycl_int4::event_timing_us(memset_ev) : sycl_int4::EventTiming{};
        const auto kernel_timing = profile_events ? sycl_int4::event_timing_us(kernel_ev) : sycl_int4::EventTiming{};
        uint64_t control_event_us = 0, control_device_us = 0;
        if (profile_events) {
          for (const auto& ev : control_events) {
            const auto timing = sycl_int4::event_timing_us(ev);
            control_event_us += timing.submit_end_us;
            control_device_us += timing.start_end_us;
          }
        }
        uint64_t cache_event_us = 0, cache_device_us = 0;
        if (profile_events) {
          for (const auto& ev : cache_events) {
            const auto timing = sycl_int4::event_timing_us(ev);
            cache_event_us += timing.submit_end_us;
            cache_device_us += timing.start_end_us;
          }
        }
        const uint64_t t_after_kernel = trace ? sycl_int4::now_us() : 0;
        if (device_scratch) q.memcpy(s->out, out, (size_t)FH * sizeof(float)).wait_and_throw();
        const uint64_t t_after_device_copy = trace ? sycl_int4::now_us() : t_after_kernel;
        std::memcpy(output, s->out, (size_t)FH * sizeof(float));
        if (trace) {
          const uint64_t t_end = sycl_int4::now_us();
          s->calls++;
          s->active_sum += (uint64_t)na;
          s->total_us += t_end - t0;
          s->submit_us += t_before_submit - t0;
          s->stage_submit_us += t_after_stage_submit - t_stage_begin;
          s->stage_wait_us += t_after_stage_wait - t_after_stage_submit;
          s->cache_hits += cache_hits_local;
          s->cache_misses += cache_misses_local;
          s->cache_submit_us += t_after_cache_submit - t_cache_begin;
          s->cache_event_us += cache_event_us;
          s->cache_device_us += cache_device_us;
          s->control_submit_us += t_after_control_submit - t_after_memset_submit;
          s->kernel_wait_us += t_after_kernel - t_before_submit;
          s->sycl_submit_us += t_after_submit - t_before_submit;
          s->sycl_memset_submit_us += t_after_memset_submit - t_before_submit;
          s->sycl_kernel_submit_us += t_after_submit - t_after_control_submit;
          s->sycl_wait_us += t_after_kernel - t_after_submit;
          s->memset_event_us += memset_timing.submit_end_us;
          s->control_event_us += control_event_us;
          s->control_device_us += control_device_us;
          s->ping_event_us += ping_timing.submit_end_us;
          s->ping_device_us += ping_timing.start_end_us;
          s->shape_event_us += shape_timing.submit_end_us;
          s->shape_device_us += shape_timing.start_end_us;
          s->weight_event_us += weight_timing.submit_end_us;
          s->weight_device_us += weight_timing.start_end_us;
          s->kernel_event_us += kernel_timing.submit_end_us;
          s->kernel_submit_start_us += kernel_timing.submit_start_us;
          s->device_kernel_us += kernel_timing.start_end_us;
          s->output_copy_us += t_after_device_copy - t_after_kernel;
          s->copy_us += t_end - t_after_kernel;
          if ((s->calls % (uint64_t)trace_every) == 0) {
            const double inv = 1.0 / (double)s->calls;
            std::printf(
                "[SYCL_GPTQ_INT4 fused] mode=quant_i8_fixed layer=%d calls=%llu avg_total=%.3fms avg_submit=%.3fms "
                "avg_stage_submit=%.3fms avg_stage_wait=%.3fms avg_control_submit=%.3fms avg_kernel_wait=%.3fms "
                "avg_sycl_submit=%.3fms avg_memset_submit=%.3fms avg_kernel_submit=%.3fms "
                "avg_sycl_wait=%.3fms avg_device_kernel=%.3fms "
                "avg_memset_evt=%.3fms avg_control_evt=%.3fms avg_control_device=%.3fms "
                "avg_cache_hit=%.2f avg_cache_miss=%.2f avg_cache_submit=%.3fms avg_cache_evt=%.3fms avg_cache_device=%.3fms "
                "avg_ping_evt=%.3fms avg_ping_device=%.3fms avg_shape_evt=%.3fms avg_shape_device=%.3fms "
                "avg_weight_evt=%.3fms avg_weight_device=%.3fms "
                "avg_kernel_evt=%.3fms avg_k_sstart=%.3fms "
                "avg_dev_to_shared=%.3fms avg_copy=%.3fms avg_active=%.2f H=%d I=%d topk=%d stage=%d host_zero=%d dev_scratch=%d fast_silu=%d cache_slots=%d cache_layer=%d cache_preload=%d\n",
                config_.layer_idx, (unsigned long long)s->calls, (double)s->total_us * inv / 1000.0,
                (double)s->submit_us * inv / 1000.0, (double)s->stage_submit_us * inv / 1000.0,
                (double)s->stage_wait_us * inv / 1000.0, (double)s->control_submit_us * inv / 1000.0,
                (double)s->kernel_wait_us * inv / 1000.0,
                (double)s->sycl_submit_us * inv / 1000.0, (double)s->sycl_memset_submit_us * inv / 1000.0,
                (double)s->sycl_kernel_submit_us * inv / 1000.0, (double)s->sycl_wait_us * inv / 1000.0,
                (double)s->device_kernel_us * inv / 1000.0, (double)s->memset_event_us * inv / 1000.0,
                (double)s->control_event_us * inv / 1000.0, (double)s->control_device_us * inv / 1000.0,
                (double)s->cache_hits * inv, (double)s->cache_misses * inv,
                (double)s->cache_submit_us * inv / 1000.0, (double)s->cache_event_us * inv / 1000.0,
                (double)s->cache_device_us * inv / 1000.0,
                (double)s->ping_event_us * inv / 1000.0, (double)s->ping_device_us * inv / 1000.0,
                (double)s->shape_event_us * inv / 1000.0, (double)s->shape_device_us * inv / 1000.0,
                (double)s->weight_event_us * inv / 1000.0, (double)s->weight_device_us * inv / 1000.0,
                (double)s->kernel_event_us * inv / 1000.0, (double)s->kernel_submit_start_us * inv / 1000.0,
                (double)s->output_copy_us * inv / 1000.0, (double)s->copy_us * inv / 1000.0,
                (double)s->active_sum * inv, H, I, k, stage_active ? 1 : 0, host_zero ? 1 : 0,
                device_scratch ? 1 : 0, fast_silu ? 1 : 0, cache_slots_active, cache_layer_filter,
                cache_preload ? 1 : 0);
          }
        }
        return;
      }

      int8_t* xq = s->xq;
      float* xs = s->xs;
      float* out = s->out;
      float* rw = s->rw;
      uint32_t **gq = s->gq, **uq = s->uq, **dq = s->dq;
      float **gsc = s->gs, **usc = s->us, **dsc = s->ds;
      const uint64_t t_before_submit = trace ? sycl_int4::now_us() : 0;
      sycl::event memset_ev;
      if (host_zero) {
        std::memset(out, 0, (size_t)H * sizeof(float));
      } else {
        memset_ev = q.memset(out, 0, (size_t)H * sizeof(float));
      }
      const uint64_t t_after_memset_submit = trace ? sycl_int4::now_us() : t_before_submit;
      constexpr int WG = 256;
      sycl::event kernel_ev = q.submit([&](sycl::handler& h) {
         if (!host_zero) h.depends_on(memset_ev);
         if (!stage_events.empty()) h.depends_on(stage_events);
	         sycl::local_accessor<float, 1> actf(sycl::range<1>(I), h);
	         sycl::local_accessor<int8_t, 1> actq(sycl::range<1>(I), h);
	         sycl::local_accessor<float, 1> ascl(sycl::range<1>(ngI), h);
         h.parallel_for(sycl::nd_range<1>((size_t)na * WG, WG), [=](sycl::nd_item<1> it) {
           const int a = (int)it.get_group(0), lid = (int)it.get_local_id(0);
           const uint32_t* wg = gq[a]; const uint32_t* wu = uq[a];
           const float* sg = gsc[a]; const float* su = usc[a];
           const int packed_per_group = gs / 8;
           for (int i = lid; i < I; i += WG) {
             float gsum = 0.f, usum = 0.f;
             for (int g = 0; g < ngH; ++g) {
               int gd = 0, ud = 0;
               const int kp0 = g * packed_per_group;
               for (int kp = kp0; kp < kp0 + packed_per_group; ++kp) {
                 const uint32_t pg = wg[(size_t)kp * I + i];
                 const uint32_t pu = wu[(size_t)kp * I + i];
                 const int8_t* xb = xq + (size_t)kp * 8;
#pragma unroll
                 for (int bb = 0; bb < 8; ++bb) {
                   const int xv = (int)xb[bb];
                   gd += xv * ((int)((pg >> (bb * 4)) & 0xF) - 8);
                   ud += xv * ((int)((pu >> (bb * 4)) & 0xF) - 8);
                 }
               }
               gsum += (float)gd * xs[g] * sg[(size_t)g * I + i];
               usum += (float)ud * xs[g] * su[(size_t)g * I + i];
             }
             float gg = fast_silu
                            ? (gsum > 20.f ? gsum
                                           : (gsum < -20.f ? 0.f : gsum / (1.f + sycl::native::exp(-gsum))))
                            : gsum / (1.f + sycl::exp(-gsum));
             actf[i] = gg * usum;
           }
           it.barrier(sycl::access::fence_space::local_space);
           for (int g = lid; g < ngI; g += WG) {
             float amax = 0.f;
             const int base = g * gs;
             for (int t = 0; t < gs; ++t) amax = sycl::fmax(amax, sycl::fabs(actf[base + t]));
             const float sact = amax > 0.f ? amax / 127.f : 0.f;
             ascl[g] = sact;
             const float inv = sact > 0.f ? 1.f / sact : 0.f;
             for (int t = 0; t < gs; ++t) {
               const int qv = sycl::clamp((int)sycl::rint(actf[base + t] * inv), -127, 127);
               actq[base + t] = (int8_t)qv;
             }
           }
           it.barrier(sycl::access::fence_space::local_space);
           const uint32_t* wd = dq[a]; const float* sd = dsc[a]; float rwa = rw[a];
           for (int hh = lid; hh < H; hh += WG) {
             float acc = 0.f;
             for (int g = 0; g < ngI; ++g) {
               int dot = 0;
               const int kp0 = g * packed_per_group;
               for (int kp = kp0; kp < kp0 + packed_per_group; ++kp) {
                 const uint32_t pd = wd[(size_t)kp * H + hh];
                 const int8_t* ab = &actq[kp * 8];
#pragma unroll
                 for (int bb = 0; bb < 8; ++bb) {
                   dot += (int)ab[bb] * ((int)((pd >> (bb * 4)) & 0xF) - 8);
                 }
               }
               acc += (float)dot * ascl[g] * sd[(size_t)g * H + hh];
             }
             sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                              sycl::access::address_space::global_space>
	             ar(out[hh]);
	             ar.fetch_add(rwa * acc);
		           }
		         });
		       });
      const uint64_t t_after_submit = trace ? sycl_int4::now_us() : t_before_submit;
      kernel_ev.wait_and_throw();
      const auto memset_timing = profile_events ? sycl_int4::event_timing_us(memset_ev) : sycl_int4::EventTiming{};
      const auto kernel_timing = profile_events ? sycl_int4::event_timing_us(kernel_ev) : sycl_int4::EventTiming{};
      const uint64_t t_after_kernel = trace ? sycl_int4::now_us() : 0;
      std::memcpy(output, out, (size_t)H * sizeof(float));
      if (trace) {
        const uint64_t t_end = sycl_int4::now_us();
        s->calls++;
        s->active_sum += (uint64_t)na;
        s->total_us += t_end - t0;
        s->submit_us += t_before_submit - t0;
        s->stage_submit_us += t_after_stage_submit - t_stage_begin;
        s->stage_wait_us += t_after_stage_wait - t_after_stage_submit;
        s->kernel_wait_us += t_after_kernel - t_before_submit;
        s->sycl_submit_us += t_after_submit - t_before_submit;
        s->sycl_memset_submit_us += t_after_memset_submit - t_before_submit;
        s->sycl_kernel_submit_us += t_after_submit - t_after_memset_submit;
        s->sycl_wait_us += t_after_kernel - t_after_submit;
        s->memset_event_us += memset_timing.submit_end_us;
        s->kernel_event_us += kernel_timing.submit_end_us;
        s->kernel_submit_start_us += kernel_timing.submit_start_us;
        s->device_kernel_us += kernel_timing.start_end_us;
        s->copy_us += t_end - t_after_kernel;
        if ((s->calls % (uint64_t)trace_every) == 0) {
          const double inv = 1.0 / (double)s->calls;
          std::printf(
              "[SYCL_GPTQ_INT4 fused] mode=quant_i8 layer=%d calls=%llu avg_total=%.3fms avg_submit=%.3fms "
              "avg_stage_submit=%.3fms avg_stage_wait=%.3fms avg_kernel_wait=%.3fms "
              "avg_sycl_submit=%.3fms avg_memset_submit=%.3fms avg_kernel_submit=%.3fms "
              "avg_sycl_wait=%.3fms avg_device_kernel=%.3fms "
              "avg_memset_evt=%.3fms avg_kernel_evt=%.3fms avg_k_sstart=%.3fms "
              "avg_copy=%.3fms avg_active=%.2f H=%d I=%d topk=%d stage=%d host_zero=%d\n",
              config_.layer_idx, (unsigned long long)s->calls, (double)s->total_us * inv / 1000.0,
              (double)s->submit_us * inv / 1000.0, (double)s->stage_submit_us * inv / 1000.0,
              (double)s->stage_wait_us * inv / 1000.0, (double)s->kernel_wait_us * inv / 1000.0,
              (double)s->sycl_submit_us * inv / 1000.0, (double)s->sycl_memset_submit_us * inv / 1000.0,
              (double)s->sycl_kernel_submit_us * inv / 1000.0, (double)s->sycl_wait_us * inv / 1000.0,
              (double)s->device_kernel_us * inv / 1000.0, (double)s->memset_event_us * inv / 1000.0,
              (double)s->kernel_event_us * inv / 1000.0, (double)s->kernel_submit_start_us * inv / 1000.0,
              (double)s->copy_us * inv / 1000.0,
              (double)s->active_sum * inv, H, I, k, stage_active ? 1 : 0, host_zero ? 1 : 0);
        }
      }
      return;
    }
    std::memcpy(s->x, input, (size_t)H * sizeof(uint16_t));
    if (na == 0) { std::memset(output, 0, (size_t)H * sizeof(float)); return; }

    uint16_t* x = s->x; float* out = s->out; float* rw = s->rw;
    uint32_t **gq = s->gq, **uq = s->uq, **dq = s->dq;
    float **gsc = s->gs, **usc = s->us, **dsc = s->ds;
    const uint64_t t_before_submit = trace ? sycl_int4::now_us() : 0;
    q.memset(out, 0, (size_t)H * sizeof(float));
    constexpr int WG = 256;
    q.submit([&](sycl::handler& h) {
       sycl::local_accessor<float, 1> act(sycl::range<1>(I), h);
       h.parallel_for(sycl::nd_range<1>((size_t)na * WG, WG), [=](sycl::nd_item<1> it) {
         const int a = (int)it.get_group(0), lid = (int)it.get_local_id(0);
         const uint32_t* wg = gq[a]; const uint32_t* wu = uq[a];
         const float* sg = gsc[a]; const float* su = usc[a];
         for (int i = lid; i < I; i += WG) {
           float gsum = 0.f, usum = 0.f;
           for (int g = 0; g < ngH; ++g) {
             int kb = g * gs; float gd = 0.f, ud = 0.f;
             for (int t = 0; t < gs; ++t) {
               int kk = kb + t;
               float xv = sycl_int4::bf16_to_fp32(x[kk]);
               int sh = (kk & 7) * 4;
               uint32_t pgk = wg[(size_t)(kk >> 3) * I + i];
               uint32_t puk = wu[(size_t)(kk >> 3) * I + i];
               gd += xv * (float)((int)((pgk >> sh) & 0xF) - 8);
               ud += xv * (float)((int)((puk >> sh) & 0xF) - 8);
             }
             gsum += gd * sg[(size_t)g * I + i];
             usum += ud * su[(size_t)g * I + i];
           }
           float gg = fast_silu
                          ? (gsum > 20.f ? gsum
                                         : (gsum < -20.f ? 0.f : gsum / (1.f + sycl::native::exp(-gsum))))
                          : gsum / (1.f + sycl::exp(-gsum));
           act[i] = gg * usum;
         }
         it.barrier(sycl::access::fence_space::local_space);
         const uint32_t* wd = dq[a]; const float* sd = dsc[a]; float rwa = rw[a];
         for (int hh = lid; hh < H; hh += WG) {
           float acc = 0.f;
           for (int g = 0; g < ngI; ++g) {
             int kb = g * gs; float dd = 0.f;
             for (int t = 0; t < gs; ++t) {
               int kk = kb + t;
               uint32_t pdk = wd[(size_t)(kk >> 3) * H + hh];
               dd += act[kk] * (float)((int)((pdk >> ((kk & 7) * 4)) & 0xF) - 8);
             }
             acc += dd * sd[(size_t)g * H + hh];
           }
           sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
               ar(out[hh]);
           ar.fetch_add(rwa * acc);
         }
       });
     }).wait_and_throw();
    const uint64_t t_after_kernel = trace ? sycl_int4::now_us() : 0;
    std::memcpy(output, out, (size_t)H * sizeof(float));
    if (trace) {
      const uint64_t t_end = sycl_int4::now_us();
      s->calls++;
      s->active_sum += (uint64_t)na;
      s->total_us += t_end - t0;
      s->submit_us += t_before_submit - t0;
      s->stage_submit_us += t_after_stage_submit - t_stage_begin;
      s->stage_wait_us += t_after_stage_wait - t_after_stage_submit;
      s->kernel_wait_us += t_after_kernel - t_before_submit;
      s->copy_us += t_end - t_after_kernel;
      if ((s->calls % (uint64_t)trace_every) == 0) {
        const double inv = 1.0 / (double)s->calls;
        std::printf(
            "[SYCL_GPTQ_INT4 fused] mode=bf16_exact layer=%d calls=%llu avg_total=%.3fms avg_submit=%.3fms "
            "avg_stage_submit=%.3fms avg_stage_wait=%.3fms avg_kernel_wait=%.3fms avg_copy=%.3fms avg_active=%.2f "
            "H=%d I=%d topk=%d stage=%d\n",
            config_.layer_idx, (unsigned long long)s->calls, (double)s->total_us * inv / 1000.0,
            (double)s->submit_us * inv / 1000.0, (double)s->stage_submit_us * inv / 1000.0,
            (double)s->stage_wait_us * inv / 1000.0, (double)s->kernel_wait_us * inv / 1000.0,
            (double)s->copy_us * inv / 1000.0,
            (double)s->active_sum * inv, H, I, k, stage_active ? 1 : 0);
      }
    }
  }

  void prefetch_weights_if_enabled() {
    if (!sycl_int4::env_flag("KT_SYCL_INT4_PREFETCH", false)) return;
    auto& q = sycl_int4::queue();
    size_t bytes = 0;
    int experts = 0;
    for (int e = 0; e < config_.expert_num; ++e) {
      if (config_.should_skip_expert(e)) continue;
      auto prefetch_b = [&](auto& bb) {
        if (!bb || bb->qw == nullptr || bb->scales == nullptr) return;
        if (bb->device_storage) return;
        q.prefetch(bb->qw, bb->qweight_bytes());
        q.prefetch(bb->scales, bb->scales_bytes());
        bytes += bb->qweight_bytes() + bb->scales_bytes();
      };
      prefetch_b(gate_bb_[e]);
      prefetch_b(up_bb_[e]);
      prefetch_b(down_bb_[e]);
      ++experts;
    }
    q.wait_and_throw();
    std::printf("[SYCL_GPTQ_INT4] layer=%d prefetched %.2f MiB across %d CPU/iGPU experts\n", config_.layer_idx,
                (double)bytes / (1024.0 * 1024.0), experts);
  }

  // Load GPTQ int4 weights from contiguous per-expert memory (qweight uint32 + scales float).
  void load_weights() {
    const int group_size = config_.quant_config.group_size;
    const uint64_t* p2l = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);
    if (config_.gate_scale == nullptr) throw std::runtime_error("SYCL GPTQ-Int4 MOE requires scale pointers.");

    const int gate_up_k = config_.hidden_size, gate_up_n = config_.intermediate_size;
    const size_t gu_qw = (size_t)(gate_up_k / 8) * gate_up_n;
    const size_t gu_sc = (size_t)(gate_up_k / group_size) * gate_up_n;
    int nth = T::recommended_nth(gate_up_n);
    pool->do_work_stealing_job(nth * config_.expert_num, nullptr,
        [this, nth, p2l, gu_qw, gu_sc](int task_id) {
          uint64_t e = task_id / nth; uint64_t logi = expert_map(p2l, e); int ith = task_id % nth;
          if (config_.should_skip_expert(logi)) return;
          gate_bb_[e]->from_mat((uint32_t*)config_.gate_proj + logi * gu_qw, (float*)config_.gate_scale + logi * gu_sc, ith, nth);
          up_bb_[e]->from_mat((uint32_t*)config_.up_proj + logi * gu_qw, (float*)config_.up_scale + logi * gu_sc, ith, nth);
        }, nullptr);

    const int down_k = config_.intermediate_size, down_n = config_.hidden_size;
    const size_t d_qw = (size_t)(down_k / 8) * down_n;
    const size_t d_sc = (size_t)(down_k / group_size) * down_n;
    nth = T::recommended_nth(down_n);
    pool->do_work_stealing_job(nth * config_.expert_num, nullptr,
        [this, nth, p2l, d_qw, d_sc](int task_id) {
          uint64_t e = task_id / nth; uint64_t logi = expert_map(p2l, e); int ith = task_id % nth;
          if (config_.should_skip_expert(logi)) return;
          down_bb_[e]->from_mat((uint32_t*)config_.down_proj + logi * d_qw, (float*)config_.down_scale + logi * d_sc, ith, nth);
        }, nullptr);
    prefetch_weights_if_enabled();
  }

  void write_weights_to_buffer(int, int, int, const GeneralMOEConfig&, const std::vector<uintptr_t>&,
                               const std::vector<uintptr_t>&, const std::vector<uintptr_t>&,
                               const std::vector<uintptr_t>&) const {
    throw std::runtime_error("SYCL GPTQ-Int4 MoE does not support write_weights_to_buffer yet.");
  }

};

// TP_MOE specialization: per-expert weight load + TP split (mirrors gptq_int4_avxvnni-moe.hpp).
template <typename K>
class TP_MOE<SYCL_GPTQ_INT4_MOE_TP<K>> : public TP_MOE<AVX2_MOE_BASE<K, SYCL_GPTQ_INT4_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AVX2_MOE_BASE<K, SYCL_GPTQ_INT4_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto pool = config.pool;
    const uint64_t* p2l = (const uint64_t*)config.physical_to_logical_map;
    const int group_size = config.quant_config.group_size;
    if (group_size == 0) throw std::runtime_error("GPTQ-Int4 requires group_size > 0");
    if (config.gate_projs.empty() && config.gate_proj == nullptr) throw std::runtime_error("no weight source");
    const bool per_expert = !config.gate_projs.empty();

    const int full_I = config.intermediate_size, full_H = config.hidden_size;
    const int gu_kp = full_H / 8, gu_ng = full_H / group_size;
    const size_t full_gu_qw = (size_t)gu_kp * full_I, full_gu_sc = (size_t)gu_ng * full_I;
    const int d_kp = full_I / 8, d_ng = full_I / group_size;
    const size_t full_d_qw = (size_t)d_kp * full_H, full_d_sc = (size_t)d_ng * full_H;

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      const int tp_I = tpc.intermediate_size;
      const size_t tp_gu_qw = (size_t)gu_kp * tp_I, tp_gu_sc = (size_t)gu_ng * tp_I;
      tpc.gate_proj = new uint32_t[tpc.expert_num * tp_gu_qw];
      tpc.up_proj = new uint32_t[tpc.expert_num * tp_gu_qw];
      tpc.gate_scale = new float[tpc.expert_num * tp_gu_sc];
      tpc.up_scale = new float[tpc.expert_num * tp_gu_sc];
      const int tp_d_kp = tp_I / 8, tp_d_ng = tp_I / group_size;
      const size_t tp_d_qw = (size_t)tp_d_kp * full_H, tp_d_sc = (size_t)tp_d_ng * full_H;
      tpc.down_proj = new uint32_t[tpc.expert_num * tp_d_qw];
      tpc.down_scale = new float[tpc.expert_num * tp_d_sc];

      const int gu_n_off = i * tp_I;             // column offset into intermediate
      const int d_kp_off = i * tp_d_kp;          // packed-row offset into intermediate (down K)
      const int d_ng_off = i * tp_d_ng;          // group offset into intermediate (down K)

      pool->get_subpool(i)->do_work_stealing_job(tpc.expert_num, nullptr, [&](int e_) {
        const size_t e = expert_map(p2l, e_);
        const uint32_t *g_qw, *u_qw, *d_qw_s; const float *g_sc, *u_sc, *d_sc_s;
        if (per_expert) {
          g_qw=(const uint32_t*)config.gate_projs[0][e]; u_qw=(const uint32_t*)config.up_projs[0][e]; d_qw_s=(const uint32_t*)config.down_projs[0][e];
          g_sc=(const float*)config.gate_scales[0][e]; u_sc=(const float*)config.up_scales[0][e]; d_sc_s=(const float*)config.down_scales[0][e];
        } else {
          g_qw=(const uint32_t*)config.gate_proj + e*full_gu_qw; u_qw=(const uint32_t*)config.up_proj + e*full_gu_qw; d_qw_s=(const uint32_t*)config.down_proj + e*full_d_qw;
          g_sc=(const float*)config.gate_scale + e*full_gu_sc; u_sc=(const float*)config.up_scale + e*full_gu_sc; d_sc_s=(const float*)config.down_scale + e*full_d_sc;
        }
        uint32_t* g_dst=(uint32_t*)tpc.gate_proj + e*tp_gu_qw; uint32_t* u_dst=(uint32_t*)tpc.up_proj + e*tp_gu_qw;
        float* gs_dst=(float*)tpc.gate_scale + e*tp_gu_sc; float* us_dst=(float*)tpc.up_scale + e*tp_gu_sc;
        // gate/up: slice N (intermediate) columns [gu_n_off, gu_n_off+tp_I)
        for (int kp = 0; kp < gu_kp; ++kp) {
          std::memcpy(g_dst + (size_t)kp*tp_I, g_qw + (size_t)kp*full_I + gu_n_off, (size_t)tp_I*sizeof(uint32_t));
          std::memcpy(u_dst + (size_t)kp*tp_I, u_qw + (size_t)kp*full_I + gu_n_off, (size_t)tp_I*sizeof(uint32_t));
        }
        for (int g = 0; g < gu_ng; ++g) {
          std::memcpy(gs_dst + (size_t)g*tp_I, g_sc + (size_t)g*full_I + gu_n_off, (size_t)tp_I*sizeof(float));
          std::memcpy(us_dst + (size_t)g*tp_I, u_sc + (size_t)g*full_I + gu_n_off, (size_t)tp_I*sizeof(float));
        }
        // down: slice K (intermediate) rows -> packed-row range [d_kp_off, +tp_d_kp), groups [d_ng_off,+tp_d_ng)
        uint32_t* d_dst=(uint32_t*)tpc.down_proj + e*tp_d_qw; float* ds_dst=(float*)tpc.down_scale + e*tp_d_sc;
        for (int kp = 0; kp < tp_d_kp; ++kp)
          std::memcpy(d_dst + (size_t)kp*full_H, d_qw_s + (size_t)(d_kp_off + kp)*full_H, (size_t)full_H*sizeof(uint32_t));
        for (int g = 0; g < tp_d_ng; ++g)
          std::memcpy(ds_dst + (size_t)g*full_H, d_sc_s + (size_t)(d_ng_off + g)*full_H, (size_t)full_H*sizeof(float));
      }, nullptr);
    });

    pool->dispense_backend()->do_numa_job([&, this](int i) { tps[i]->load_weights(); });
    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      delete[] (uint32_t*)tpc.gate_proj; delete[] (uint32_t*)tpc.up_proj; delete[] (uint32_t*)tpc.down_proj;
      delete[] (float*)tpc.gate_scale; delete[] (float*)tpc.up_scale; delete[] (float*)tpc.down_scale;
    });
    this->weights_loaded = true;
  }

  void write_weight_scale_to_buffer(int, int, const std::vector<uintptr_t>&, const std::vector<uintptr_t>&,
                                    const std::vector<uintptr_t>&, const std::vector<uintptr_t>&) {
    throw std::runtime_error("SYCL GPTQ-Int4 write_weight_scale_to_buffer not implemented");
  }
};

#endif  // CPUINFER_OPERATOR_SYCL_GPTQ_INT4_MOE_H
