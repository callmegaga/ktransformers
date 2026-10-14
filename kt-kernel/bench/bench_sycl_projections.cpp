// SPDX-License-Identifier: Apache-2.0
// Build: icpx -fsycl -O3 bench_sycl_projections.cpp -o bench_sycl_projections
// Set KT_SYCL_PROFILE=1 before startup for device timestamps.
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../operators/sycl/gptq_int4_vector-kernels.hpp"

using namespace sycl_xmx;

struct Variant {
  std::string name;
  int splits;
};

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  size_t n = samples.size();
  return n % 2 ? samples[n / 2] : (samples[n / 2 - 1] + samples[n / 2]) / 2;
}

template <bool GateUp, int Rows, int Splits = 1>
sycl::event launch_xmx(bool q8, const Tile* tiles, int count, const int* tokens, const uint16_t* input,
                       const int8_t* qi, const float* sa, const Weight& first, const Weight& second,
                       uint16_t* activated, float* down) {
  return q8 ? projection<Rows, true, GateUp, false, Splits>(tiles, count, tokens, input, qi, sa, first, &second,
                                                            activated, down, 0, 0)
            : projection<Rows, false, GateUp, false, Splits>(tiles, count, tokens, input, qi, sa, first, &second,
                                                             activated, down, 0, 0);
}

template <bool GateUp>
void benchmark(int experts, int active, int m, int repeat, bool rotate, bool fp16) {
  const int k = GateUp ? 2048 : 512, n = GateUp ? 512 : 2048, gs = 128;
  std::mt19937 random(20261014);
  Weight weight, up;
  weight.init(experts, n, k, gs);
  if constexpr (GateUp) up.init(experts, n, k, gs);
  std::vector<uint32_t> source(n * k / 8);
  std::vector<float> scale(n * k / gs);
  for (int e = 0; e < experts; ++e) {
    for (auto& word : source) word = random();
    for (auto& s : scale) s = 0.002f + (random() % 16) * 0.0001f;
    weight.from_mat(e, source.data(), scale.data());
    if constexpr (GateUp) {
      for (auto& word : source) word = random();
      up.from_mat(e, source.data(), scale.data());
    }
  }
  const int tile_rows = m <= 1 ? 1 : m <= 2 ? 2 : m <= 4 ? 4 : 8;
  const int nt = active * ((m + tile_rows - 1) / tile_rows);
  USM<Tile> tiles;
  tiles.reserve(nt);
  USM<int> tokens;
  tokens.reserve(active * m);
  USM<uint16_t> input, activated;
  input.reserve(active * m * k);
  activated.reserve(active * m * n);
  USM<int8_t> qi;
  qi.reserve(active * m * k);
  USM<float> sa, down;
  sa.reserve(active * m * k / gs);
  down.reserve(active * m * n);
  USM<uint32_t> evict;
  if (rotate) {
    evict.reserve(16 * 1024 * 1024);  // Cache displacement is outside the timed region.
    std::fill_n(evict.data, evict.capacity, 0u);
  }
  for (int r = 0; r < active * m; ++r) {
    tokens.data[r] = r;
    for (int j = 0; j < k; ++j) {
      int v = static_cast<int>(random() % 255) - 127;
      qi.data[r * k + j] = v;
      input.data[r * k + j] = sycl_int4::fp32_to_bf16(v * 0.003f);
    }
    for (int g = 0; g < k / gs; ++g) sa.data[r * (k / gs) + g] = 0.003f;
  }
  std::vector<Variant> variants = {{"xmx-int8", 1}};
  if (fp16) variants.push_back({"xmx-fp16", 1});
  for (int s : {1, 2, 4, 8}) {
    variants.push_back({"dp4a", s});
    if (fp16) variants.push_back({"vector", s});
    if (s > 1) {
      variants.push_back({"xmx-int8-split", s});
      if (fp16) variants.push_back({"xmx-fp16-split", s});
    }
  }
  std::vector<std::vector<double>> wall(variants.size()), device(variants.size());
  std::vector<int> order;
  for (int v = 0; v < static_cast<int>(variants.size()); ++v) order.push_back(v);
  for (int iteration = -4; iteration < repeat; ++iteration) {
    std::shuffle(order.begin(), order.end(), random);
    int base = rotate ? (std::max(0, iteration) * active) % experts : 0;
    int t = 0;
    for (int e = 0; e < active; ++e)
      for (int r = 0; r < m; r += tile_rows)
        tiles.data[t++] = {(base + e) % experts, e * m + r, std::min(tile_rows, m - r)};
    for (int v : order) {
      const auto& variant = variants[v];
      if (rotate) {
        auto* storage = evict.data;
        auto e =
            sycl_int4::queue().parallel_for(sycl::range<1>(evict.capacity), [=](sycl::id<1> id) { storage[id] += 1; });
        sycl_int4::wait_and_throw(e);
      }
      const auto start = std::chrono::steady_clock::now();
      sycl::event event;
      if (variant.name == "vector")
        event = vector_dispatch<false, GateUp>(variant.splits, tiles.data, nt, tile_rows, tokens.data, input.data,
                                               qi.data, sa.data, weight, &up, activated.data, down.data, 0, 0);
      else if (variant.name == "dp4a")
        event = vector_dispatch<true, GateUp>(variant.splits, tiles.data, nt, tile_rows, tokens.data, input.data,
                                              qi.data, sa.data, weight, &up, activated.data, down.data, 0, 0);
      else {
        const bool q8 = variant.name.find("int8") != std::string::npos;
#define KT_LAUNCH_XMX(R)                                                                                           \
  (variant.splits == 2   ? launch_xmx<GateUp, R, 2>(q8, tiles.data, nt, tokens.data, input.data, qi.data, sa.data, \
                                                    weight, up, activated.data, down.data)                         \
   : variant.splits == 4 ? launch_xmx<GateUp, R, 4>(q8, tiles.data, nt, tokens.data, input.data, qi.data, sa.data, \
                                                    weight, up, activated.data, down.data)                         \
   : variant.splits == 8 ? launch_xmx<GateUp, R, 8>(q8, tiles.data, nt, tokens.data, input.data, qi.data, sa.data, \
                                                    weight, up, activated.data, down.data)                         \
                         : launch_xmx<GateUp, R>(q8, tiles.data, nt, tokens.data, input.data, qi.data, sa.data,    \
                                                 weight, up, activated.data, down.data))
        switch (tile_rows) {
          case 1:
            event = KT_LAUNCH_XMX(1);
            break;
          case 2:
            event = KT_LAUNCH_XMX(2);
            break;
          case 4:
            event = KT_LAUNCH_XMX(4);
            break;
          default:
            event = KT_LAUNCH_XMX(8);
            break;
        }
#undef KT_LAUNCH_XMX
      }
      sycl_int4::wait_and_throw(event);
      double elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
      if (iteration >= 0) {
        wall[v].push_back(elapsed);
        if (sycl_int4::queue().has_property<sycl::property::queue::enable_profiling>()) {
          auto begin = event.get_profiling_info<sycl::info::event_profiling::command_start>();
          auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
          device[v].push_back((end - begin) / 1000.0);
        }
      }
    }
  }
  for (size_t v = 0; v < variants.size(); ++v)
    std::cout << (GateUp ? "gate_up" : "down") << ',' << active << ',' << m << ','
              << (rotate ? "rotate+evict64MiB" : "hot") << ',' << variants[v].name << ',' << variants[v].splits << ','
              << median(wall[v]) << ',' << (device[v].empty() ? -1 : median(device[v])) << '\n';
  std::cout.flush();
}

int main(int argc, char** argv) {
  try {
    int repeat = 30, active = 8;
    bool rotate = false, fp16 = true;
    std::vector<int> rows = {1, 2, 4, 8, 16, 32};
    for (int j = 1; j < argc; ++j) {
      std::string arg = argv[j];
      if (arg == "--rotate")
        rotate = true;
      else if (arg == "--int8-only")
        fp16 = false;
      else if (arg == "--repeat" && j + 1 < argc)
        repeat = std::stoi(argv[++j]);
      else if (arg == "--active" && j + 1 < argc)
        active = std::stoi(argv[++j]);
      else if (arg == "--rows" && j + 1 < argc)
        rows = {std::stoi(argv[++j])};
      else
        throw std::runtime_error("Unknown option: " + arg);
    }
    if (repeat <= 0 || active <= 0 || active > 256 || *std::min_element(rows.begin(), rows.end()) <= 0)
      throw std::runtime_error("Invalid benchmark size");
    std::cerr << "device=" << sycl_int4::queue().get_device().get_info<sycl::info::device::name>() << '\n';
    std::cout << "phase,active_experts,rows_per_expert,cache,kernel,k_splits,wall_us,device_us\n";
    for (int m : rows) {
      benchmark<true>(256, active, m, repeat, rotate, fp16);
      benchmark<false>(256, active, m, repeat, rotate, fp16);
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
