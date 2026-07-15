// Focused decode benchmark for the GPTQ-Int4 down projection.
//
// It mirrors the production kernels in gptq_int4_sycl-moe.hpp and compares:
//   1. BF16 activations + packed INT4 weights, SG16, two output rows/work-group.
//   2. One batched BF16->Q8 quantization + eight asynchronous DP4A GEMVs.

// Build:
//   icpx -O3 -std=c++17 -fsycl down_q8_bench.cpp -o down_q8_bench

// Run:
//   ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./down_q8_bench [iters]


#include <sycl/ext/oneapi/dot_product.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace {

constexpr int kActive = 8;
constexpr int kN = 2048;
constexpr int kK = 512;
constexpr int kGroupSize = 128;
constexpr int kNumGroups = kK / kGroupSize;
constexpr int kKPack = kK / 8;
constexpr int kSubgroup = 16;
constexpr int kRowsPerWg = 2;

double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

uint16_t fp32_to_bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

inline float bf16_to_fp32(uint16_t value) {
  return sycl::bit_cast<float>(static_cast<uint32_t>(value) << 16);
}

inline int32_t unpack_i4x4_to_i8x4(uint32_t packed) {
  uint32_t x = packed & 0xffffu;
  x = (x | (x << 8)) & 0x00ff00ffu;
  x = (x | (x << 4)) & 0x0f0f0f0fu;
  const uint32_t neg = (~x) & 0x08080808u;
  x = (x & 0x07070707u) | neg | (neg << 1) | (neg << 2) | (neg << 3) | (neg << 4);
  return sycl::bit_cast<int32_t>(x);
}

struct Data {
  uint16_t* x = nullptr;       // [active, K]
  uint16_t** x_ptrs = nullptr; // [active]
  uint32_t* qw = nullptr;      // output-major [active, N, K/8]
  float* ws = nullptr;         // output-major [active, N, num_groups]
  sycl::half* ws_half = nullptr;
  int8_t* xq = nullptr;        // [active, K]
  float* xs = nullptr;         // [active, num_groups]
  float* out_bf16 = nullptr;   // [active, N]
  float* out_q8 = nullptr;     // [active, N]
  float* out_half = nullptr;   // [active, N]
};

Data allocate_data(sycl::queue& q) {
  Data d;
  d.x = sycl::malloc_shared<uint16_t>(static_cast<size_t>(kActive) * kK, q);
  d.x_ptrs = sycl::malloc_shared<uint16_t*>(kActive, q);
  d.qw = sycl::malloc_shared<uint32_t>(static_cast<size_t>(kActive) * kN * kKPack, q);
  d.ws = sycl::malloc_shared<float>(static_cast<size_t>(kActive) * kN * kNumGroups, q);
  d.ws_half = sycl::malloc_shared<sycl::half>(static_cast<size_t>(kActive) * kN * kNumGroups, q);
  d.xq = sycl::malloc_shared<int8_t>(static_cast<size_t>(kActive) * kK, q);
  d.xs = sycl::malloc_shared<float>(static_cast<size_t>(kActive) * kNumGroups, q);
  d.out_bf16 = sycl::malloc_shared<float>(static_cast<size_t>(kActive) * kN, q);
  d.out_q8 = sycl::malloc_shared<float>(static_cast<size_t>(kActive) * kN, q);
  d.out_half = sycl::malloc_shared<float>(static_cast<size_t>(kActive) * kN, q);
  if (!d.x || !d.x_ptrs || !d.qw || !d.ws || !d.ws_half || !d.xq || !d.xs || !d.out_bf16 ||
      !d.out_q8 || !d.out_half) {
    throw std::runtime_error("USM allocation failed");
  }
  for (int expert = 0; expert < kActive; ++expert) {
    d.x_ptrs[expert] = d.x + static_cast<size_t>(expert) * kK;
  }
  return d;
}

void free_data(sycl::queue& q, Data& d) {
  for (void* ptr : {static_cast<void*>(d.x), static_cast<void*>(d.x_ptrs), static_cast<void*>(d.qw),
                    static_cast<void*>(d.ws), static_cast<void*>(d.ws_half), static_cast<void*>(d.xq),
                    static_cast<void*>(d.xs), static_cast<void*>(d.out_bf16), static_cast<void*>(d.out_q8),
                    static_cast<void*>(d.out_half)}) {
    if (ptr) sycl::free(ptr, q);
  }
}

void initialize(Data& d) {
  std::mt19937 rng(0x5a17u);
  std::normal_distribution<float> activation(0.0f, 1.25f);
  std::uniform_int_distribution<int> weight(0, 15);
  std::uniform_real_distribution<float> scale(0.006f, 0.025f);

  for (int expert = 0; expert < kActive; ++expert) {
    for (int kk = 0; kk < kK; ++kk) {
      float value = activation(rng);
      if ((kk % 13) == 0) value = 0.0f;
      d.x[static_cast<size_t>(expert) * kK + kk] = fp32_to_bf16(value);
    }
    for (int ni = 0; ni < kN; ++ni) {
      uint32_t* row = d.qw + (static_cast<size_t>(expert) * kN + ni) * kKPack;
      for (int kp = 0; kp < kKPack; ++kp) {
        uint32_t packed = 0;
        for (int bb = 0; bb < 8; ++bb) packed |= static_cast<uint32_t>(weight(rng)) << (bb * 4);
        row[kp] = packed;
      }
      float* row_scales = d.ws + (static_cast<size_t>(expert) * kN + ni) * kNumGroups;
      sycl::half* row_half_scales =
          d.ws_half + (static_cast<size_t>(expert) * kN + ni) * kNumGroups;
      for (int g = 0; g < kNumGroups; ++g) {
        row_scales[g] = scale(rng);
        row_half_scales[g] = sycl::half(row_scales[g]);
      }
    }
  }
}

template <int RowsPerWg>
sycl::event submit_bf16_half_scale_expert(sycl::queue& q, const Data& d, int expert) {
  constexpr size_t local_size = static_cast<size_t>(kSubgroup) * RowsPerWg;
  constexpr size_t work_groups = (kN + RowsPerWg - 1) / RowsPerWg;
  const uint16_t* x = d.x + static_cast<size_t>(expert) * kK;
  const uint32_t* qw = d.qw + static_cast<size_t>(expert) * kN * kKPack;
  const sycl::half* ws = d.ws_half + static_cast<size_t>(expert) * kN * kNumGroups;
  float* out = d.out_half + static_cast<size_t>(expert) * kN;
  return q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(work_groups * local_size, local_size),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubgroup)]] {
      const auto subgroup = it.get_sub_group();
      const int row_in_wg = static_cast<int>(subgroup.get_group_linear_id());
      const int lane = static_cast<int>(subgroup.get_local_linear_id());
      const int ni = static_cast<int>(it.get_group(0)) * RowsPerWg + row_in_wg;
      if (ni >= kN) return;
      float acc = 0.0f;
      for (int g = 0; g < kNumGroups; ++g) {
        float partial = 0.0f;
        const int kp_base = g * (kGroupSize / 8);
        for (int kpi = lane; kpi < kGroupSize / 8; kpi += kSubgroup) {
          const int kp = kp_base + kpi;
          const uint32_t packed = qw[static_cast<size_t>(ni) * kKPack + kp];
          const uint16_t* xb = x + static_cast<size_t>(kp) * 8;
#pragma unroll
          for (int bb = 0; bb < 8; ++bb) {
            const int w = static_cast<int>((packed >> (bb * 4)) & 0x0fu) - 8;
            partial += bf16_to_fp32(xb[bb]) * static_cast<float>(w);
          }
        }
        const float sum = sycl::reduce_over_group(subgroup, partial, sycl::plus<float>());
        if (lane == 0) acc += sum * static_cast<float>(ws[static_cast<size_t>(ni) * kNumGroups + g]);
      }
      if (lane == 0) out[ni] = acc;
    });
  });
}

template <int Wg>
sycl::event submit_quantize_groups(sycl::queue& q, const Data& d) {
  return q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(static_cast<size_t>(kActive) * kNumGroups * Wg, Wg),
                   [=](sycl::nd_item<1> it) {
      const int group_id = static_cast<int>(it.get_group(0));
      const int expert = group_id / kNumGroups;
      const int group = group_id - expert * kNumGroups;
      const int lane = static_cast<int>(it.get_local_id(0));
      const uint16_t* src = d.x_ptrs[expert] + static_cast<size_t>(group) * kGroupSize;
      float amax = 0.0f;
      for (int t = lane; t < kGroupSize; t += Wg) {
        amax = sycl::fmax(amax, sycl::fabs(bf16_to_fp32(src[t])));
      }
      amax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
      const float quant_scale = amax > 0.0f ? amax / 127.0f : 0.0f;
      if (lane == 0) d.xs[static_cast<size_t>(expert) * kNumGroups + group] = quant_scale;
      const float inv = quant_scale > 0.0f ? 1.0f / quant_scale : 0.0f;
      for (int t = lane; t < kGroupSize; t += Wg) {
        int value = static_cast<int>(sycl::rint(bf16_to_fp32(src[t]) * inv));
        value = sycl::max(-127, sycl::min(127, value));
        d.xq[static_cast<size_t>(expert) * kK + static_cast<size_t>(group) * kGroupSize + t] =
            static_cast<int8_t>(value);
      }
    });
  });
}

// Use one work-group per expert and process its four quantization groups in
// sequence. This trades some parallelism for fewer tiny work-groups.
template <int Wg>
sycl::event submit_quantize_experts(sycl::queue& q, const Data& d) {
  return q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(static_cast<size_t>(kActive) * Wg, Wg),
                   [=](sycl::nd_item<1> it) {
      const int expert = static_cast<int>(it.get_group(0));
      const int lane = static_cast<int>(it.get_local_id(0));
      const uint16_t* expert_src = d.x_ptrs[expert];
      for (int group = 0; group < kNumGroups; ++group) {
        const uint16_t* src = expert_src + static_cast<size_t>(group) * kGroupSize;
        float amax = 0.0f;
        for (int t = lane; t < kGroupSize; t += Wg) {
          amax = sycl::fmax(amax, sycl::fabs(bf16_to_fp32(src[t])));
        }
        amax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
        const float quant_scale = amax > 0.0f ? amax / 127.0f : 0.0f;
        if (lane == 0) d.xs[static_cast<size_t>(expert) * kNumGroups + group] = quant_scale;
        const float inv = quant_scale > 0.0f ? 1.0f / quant_scale : 0.0f;
        for (int t = lane; t < kGroupSize; t += Wg) {
          int value = static_cast<int>(sycl::rint(bf16_to_fp32(src[t]) * inv));
          value = sycl::max(-127, sycl::min(127, value));
          d.xq[static_cast<size_t>(expert) * kK + static_cast<size_t>(group) * kGroupSize + t] =
              static_cast<int8_t>(value);
        }
      }
    });
  });
}

template <int RowsPerWg>
sycl::event submit_bf16_expert(sycl::queue& q, const Data& d, int expert) {
  constexpr size_t local_size = static_cast<size_t>(kSubgroup) * RowsPerWg;
  constexpr size_t work_groups = (kN + RowsPerWg - 1) / RowsPerWg;
  const uint16_t* x = d.x + static_cast<size_t>(expert) * kK;
  const uint32_t* qw = d.qw + static_cast<size_t>(expert) * kN * kKPack;
  const float* ws = d.ws + static_cast<size_t>(expert) * kN * kNumGroups;
  float* out = d.out_bf16 + static_cast<size_t>(expert) * kN;
  return q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(work_groups * local_size, local_size),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubgroup)]] {
      const auto subgroup = it.get_sub_group();
      const int row_in_wg = static_cast<int>(subgroup.get_group_linear_id());
      const int lane = static_cast<int>(subgroup.get_local_linear_id());
      const int ni = static_cast<int>(it.get_group(0)) * RowsPerWg + row_in_wg;
      if (ni >= kN) return;
      float acc = 0.0f;
      for (int g = 0; g < kNumGroups; ++g) {
        float partial = 0.0f;
        const int kp_base = g * (kGroupSize / 8);
        for (int kpi = lane; kpi < kGroupSize / 8; kpi += kSubgroup) {
          const int kp = kp_base + kpi;
          const uint32_t packed = qw[static_cast<size_t>(ni) * kKPack + kp];
          const uint16_t* xb = x + static_cast<size_t>(kp) * 8;
#pragma unroll
          for (int bb = 0; bb < 8; ++bb) {
            const int w = static_cast<int>((packed >> (bb * 4)) & 0x0fu) - 8;
            partial += bf16_to_fp32(xb[bb]) * static_cast<float>(w);
          }
        }
        const float sum = sycl::reduce_over_group(subgroup, partial, sycl::plus<float>());
        if (lane == 0) acc += sum * ws[static_cast<size_t>(ni) * kNumGroups + g];
      }
      if (lane == 0) out[ni] = acc;
    });
  });
}

// Mirrors the production kernel before shape specialization: all dimensions,
// grouping, and weight layout are captured as runtime values.
template <int RowsPerWg>
__attribute__((noinline)) sycl::event submit_bf16_expert_generic(
    sycl::queue& q, const Data& d, int expert, int n, int k, int group_size,
    bool output_major) {
  constexpr size_t local_size = static_cast<size_t>(kSubgroup) * RowsPerWg;
  const int num_groups = k / group_size;
  const int k_pack = k / 8;
  const int packed_per_group = group_size / 8;
  const size_t work_groups = (static_cast<size_t>(n) + RowsPerWg - 1) / RowsPerWg;
  const uint16_t* x = d.x + static_cast<size_t>(expert) * k;
  const uint32_t* qw = d.qw + static_cast<size_t>(expert) * n * k_pack;
  const float* ws = d.ws + static_cast<size_t>(expert) * n * num_groups;
  float* out = d.out_bf16 + static_cast<size_t>(expert) * n;
  return q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(work_groups * local_size, local_size),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubgroup)]] {
      const auto subgroup = it.get_sub_group();
      const int row_in_wg = static_cast<int>(subgroup.get_group_linear_id());
      const int lane = static_cast<int>(subgroup.get_local_linear_id());
      const int ni = static_cast<int>(it.get_group(0)) * RowsPerWg + row_in_wg;
      if (ni >= n) return;
      float acc = 0.0f;
      for (int g = 0; g < num_groups; ++g) {
        float partial = 0.0f;
        const int kp_base = g * packed_per_group;
        for (int kpi = lane; kpi < packed_per_group; kpi += kSubgroup) {
          const int kp = kp_base + kpi;
          const size_t qw_offset = output_major ? static_cast<size_t>(ni) * k_pack + kp
                                                : static_cast<size_t>(kp) * n + ni;
          const uint32_t packed = qw[qw_offset];
          const uint16_t* xb = x + static_cast<size_t>(kp) * 8;
#pragma unroll
          for (int bb = 0; bb < 8; ++bb) {
            const int w = static_cast<int>((packed >> (bb * 4)) & 0x0fu) - 8;
            partial += bf16_to_fp32(xb[bb]) * static_cast<float>(w);
          }
        }
        const float sum = sycl::reduce_over_group(subgroup, partial, sycl::plus<float>());
        const size_t scale_offset = output_major ? static_cast<size_t>(ni) * num_groups + g
                                                 : static_cast<size_t>(g) * n + ni;
        if (lane == 0) acc += sum * ws[scale_offset];
      }
      if (lane == 0) out[ni] = acc;
    });
  });
}

template <int RowsPerWg>
sycl::event submit_q8_expert(sycl::queue& q, const Data& d, int expert,
                             const sycl::event* dependency = nullptr) {
  constexpr size_t local_size = static_cast<size_t>(kSubgroup) * RowsPerWg;
  constexpr size_t work_groups = (kN + RowsPerWg - 1) / RowsPerWg;
  const int8_t* xq = d.xq + static_cast<size_t>(expert) * kK;
  const float* xs = d.xs + static_cast<size_t>(expert) * kNumGroups;
  const uint32_t* qw = d.qw + static_cast<size_t>(expert) * kN * kKPack;
  const float* ws = d.ws + static_cast<size_t>(expert) * kN * kNumGroups;
  float* out = d.out_q8 + static_cast<size_t>(expert) * kN;
  return q.submit([&](sycl::handler& h) {
    if (dependency != nullptr) h.depends_on(*dependency);
    h.parallel_for(sycl::nd_range<1>(work_groups * local_size, local_size),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubgroup)]] {
      const auto subgroup = it.get_sub_group();
      const int row_in_wg = static_cast<int>(subgroup.get_group_linear_id());
      const int lane = static_cast<int>(subgroup.get_local_linear_id());
      const int ni = static_cast<int>(it.get_group(0)) * RowsPerWg + row_in_wg;
      if (ni >= kN) return;
      float acc = 0.0f;
      for (int g = 0; g < kNumGroups; ++g) {
        int partial = 0;
        const int kp_base = g * (kGroupSize / 8);
        for (int kpi = lane; kpi < kGroupSize / 8; kpi += kSubgroup) {
          const int kp = kp_base + kpi;
          const uint32_t packed = qw[static_cast<size_t>(ni) * kKPack + kp];
          const int8_t* xb = xq + static_cast<size_t>(kp) * 8;
          const int32_t x0 = *reinterpret_cast<const int32_t*>(xb);
          const int32_t x1 = *reinterpret_cast<const int32_t*>(xb + 4);
          partial = sycl::ext::oneapi::dot_acc(x0, unpack_i4x4_to_i8x4(packed), partial);
          partial = sycl::ext::oneapi::dot_acc(x1, unpack_i4x4_to_i8x4(packed >> 16), partial);
        }
        const int sum = sycl::reduce_over_group(subgroup, partial, sycl::plus<int>());
        if (lane == 0) {
          acc += static_cast<float>(sum) * xs[g] * ws[static_cast<size_t>(ni) * kNumGroups + g];
        }
      }
      if (lane == 0) out[ni] = acc;
    });
  });
}

template <int RowsPerWg>
void run_bf16(sycl::queue& q, const Data& d) {
  std::vector<sycl::event> events;
  events.reserve(kActive);
  for (int expert = 0; expert < kActive; ++expert) {
    events.push_back(submit_bf16_expert<RowsPerWg>(q, d, expert));
  }
  sycl::event::wait_and_throw(events);
}

template <int RowsPerWg>
void run_bf16_generic(sycl::queue& q, const Data& d) {
  std::vector<sycl::event> events;
  events.reserve(kActive);
  for (int expert = 0; expert < kActive; ++expert) {
    events.push_back(submit_bf16_expert_generic<RowsPerWg>(
        q, d, expert, kN, kK, kGroupSize, true));
  }
  sycl::event::wait_and_throw(events);
}

template <int RowsPerWg>
void run_bf16_half_scale(sycl::queue& q, const Data& d) {
  std::vector<sycl::event> events;
  events.reserve(kActive);
  for (int expert = 0; expert < kActive; ++expert) {
    events.push_back(submit_bf16_half_scale_expert<RowsPerWg>(q, d, expert));
  }
  sycl::event::wait_and_throw(events);
}

template <int RowsPerWg, typename Quantizer>
void run_q8(sycl::queue& q, const Data& d, Quantizer&& quantizer) {
  const sycl::event quant_event = quantizer();
  std::vector<sycl::event> events;
  events.reserve(kActive);
  for (int expert = 0; expert < kActive; ++expert) {
    events.push_back(submit_q8_expert<RowsPerWg>(q, d, expert, &quant_event));
  }
  sycl::event::wait_and_throw(events);
}

template <int RowsPerWg>
void run_q8_prequantized(sycl::queue& q, const Data& d) {
  std::vector<sycl::event> events;
  events.reserve(kActive);
  for (int expert = 0; expert < kActive; ++expert) {
    events.push_back(submit_q8_expert<RowsPerWg>(q, d, expert));
  }
  sycl::event::wait_and_throw(events);
}

template <typename F>
std::vector<double> measure(F&& function, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) function();
  std::vector<double> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    const double start = now_ms();
    function();
    samples.push_back(now_ms() - start);
  }
  return samples;
}

void print_samples(const char* label, std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  const double median = samples[samples.size() / 2];
  const size_t p95_index = std::min(samples.size() - 1, static_cast<size_t>(samples.size() * 0.95));
  std::printf("%-24s best=%7.3f ms median=%7.3f ms mean=%7.3f ms p95=%7.3f ms\n", label,
              samples.front(), median, mean, samples[p95_index]);
}

}  // namespace

int main(int argc, char** argv) {
  const int iters = argc > 1 ? std::max(10, std::atoi(argv[1])) : 200;
  sycl::queue q{sycl::gpu_selector_v};
  std::printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  std::printf("Shape: active=%d N=%d K=%d group=%d SG=%d rows/WG=%d\n", kActive, kN, kK,
              kGroupSize, kSubgroup, kRowsPerWg);

  Data data = allocate_data(q);
  initialize(data);

  run_bf16<2>(q, data);
  run_q8<2>(q, data, [&] { return submit_quantize_groups<128>(q, data); });
  double max_abs = 0.0;
  double max_rel = 0.0;
  double sum_abs = 0.0;
  double sum_sq_error = 0.0;
  double sum_sq_ref = 0.0;
  const size_t count = static_cast<size_t>(kActive) * kN;
  for (size_t i = 0; i < count; ++i) {
    const double reference = data.out_bf16[i];
    const double error = std::abs(static_cast<double>(data.out_q8[i]) - reference);
    max_abs = std::max(max_abs, error);
    max_rel = std::max(max_rel, error / std::max(std::abs(reference), 1e-3));
    sum_abs += error;
    sum_sq_error += error * error;
    sum_sq_ref += reference * reference;
  }
  std::printf("Error: max_abs=%.6f max_rel=%.6f mean_abs=%.6f rel_l2=%.6f rmse=%.6f\n",
              max_abs, max_rel, sum_abs / count, std::sqrt(sum_sq_error / sum_sq_ref),
              std::sqrt(sum_sq_error / count));

  run_bf16_half_scale<2>(q, data);
  max_abs = 0.0;
  sum_abs = 0.0;
  sum_sq_error = 0.0;
  sum_sq_ref = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double reference = data.out_bf16[i];
    const double error = std::abs(static_cast<double>(data.out_half[i]) - reference);
    max_abs = std::max(max_abs, error);
    sum_abs += error;
    sum_sq_error += error * error;
    sum_sq_ref += reference * reference;
  }
  std::printf("FP16 scale error: max_abs=%.6f mean_abs=%.6f rel_l2=%.6f\n", max_abs,
              sum_abs / count, std::sqrt(sum_sq_error / sum_sq_ref));

  const auto quant_group16 = measure([&] { submit_quantize_groups<16>(q, data).wait_and_throw(); }, 20, iters);
  const auto quant_group32 = measure([&] { submit_quantize_groups<32>(q, data).wait_and_throw(); }, 20, iters);
  const auto quant_group64 = measure([&] { submit_quantize_groups<64>(q, data).wait_and_throw(); }, 20, iters);
  const auto quant_group128 = measure([&] { submit_quantize_groups<128>(q, data).wait_and_throw(); }, 20, iters);
  const auto quant_expert32 = measure([&] { submit_quantize_experts<32>(q, data).wait_and_throw(); }, 20, iters);
  const auto quant_expert64 = measure([&] { submit_quantize_experts<64>(q, data).wait_and_throw(); }, 20, iters);
  const auto quant_expert128 = measure([&] { submit_quantize_experts<128>(q, data).wait_and_throw(); }, 20, iters);
  const auto bf16_rows1 = measure([&] { run_bf16<1>(q, data); }, 20, iters);
  const auto bf16_rows2 = measure([&] { run_bf16<2>(q, data); }, 20, iters);
  const auto bf16_generic_rows2 = measure([&] { run_bf16_generic<2>(q, data); }, 20, iters);
  const auto bf16_half_rows2 = measure([&] { run_bf16_half_scale<2>(q, data); }, 20, iters);
  const auto bf16_rows4 = measure([&] { run_bf16<4>(q, data); }, 20, iters);
  const auto bf16_rows8 = measure([&] { run_bf16<8>(q, data); }, 20, iters);
  const auto bf16_rows16 = measure([&] { run_bf16<16>(q, data); }, 20, iters);
  const auto bf16_rows32 = measure([&] { run_bf16<32>(q, data); }, 20, iters);
  submit_quantize_groups<32>(q, data).wait_and_throw();
  const auto q8_rows1 = measure([&] { run_q8_prequantized<1>(q, data); }, 20, iters);
  const auto q8_rows2 = measure([&] { run_q8_prequantized<2>(q, data); }, 20, iters);
  const auto q8_rows4 = measure([&] { run_q8_prequantized<4>(q, data); }, 20, iters);
  const auto q8_rows8 = measure([&] { run_q8_prequantized<8>(q, data); }, 20, iters);
  const auto q8_full = measure(
      [&] { run_q8<2>(q, data, [&] { return submit_quantize_groups<32>(q, data); }); }, 20, iters);
  print_samples("Quant group WG16", quant_group16);
  print_samples("Quant group WG32", quant_group32);
  print_samples("Quant group WG64", quant_group64);
  print_samples("Quant group WG128", quant_group128);
  print_samples("Quant expert WG32", quant_expert32);
  print_samples("Quant expert WG64", quant_expert64);
  print_samples("Quant expert WG128", quant_expert128);
  print_samples("BF16 rows/WG=1", bf16_rows1);
  print_samples("BF16 rows/WG=2", bf16_rows2);
  print_samples("BF16 generic WG2", bf16_generic_rows2);
  print_samples("BF16 FP16-scale WG2", bf16_half_rows2);
  print_samples("BF16 rows/WG=4", bf16_rows4);
  print_samples("BF16 rows/WG=8", bf16_rows8);
  print_samples("BF16 rows/WG=16", bf16_rows16);
  print_samples("BF16 rows/WG=32", bf16_rows32);
  print_samples("Q8 DP4A rows/WG=1", q8_rows1);
  print_samples("Q8 DP4A rows/WG=2", q8_rows2);
  print_samples("Q8 DP4A rows/WG=4", q8_rows4);
  print_samples("Q8 DP4A rows/WG=8", q8_rows8);
  print_samples("Q8 WG32 + DP4A WG2", q8_full);

  free_data(q, data);
  return 0;
}
