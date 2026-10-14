// SPDX-License-Identifier: Apache-2.0
#ifndef KT_SYCL_GPTQ_INT4_XMX_MOE_HPP
#define KT_SYCL_GPTQ_INT4_XMX_MOE_HPP

#include <cstdlib>
#include <limits>
#include <string>
#include <sycl/ext/oneapi/experimental/graph.hpp>

#include "../avx2/avx2_bf16_utils.hpp"
#include "../moe-tp.hpp"
#include "gptq_int4_xmx-kernels.hpp"

// Dedicated single-NUMA executor. No CPU expert-input pools or FP16 weight copy.
class SYCL_XMX_GPTQ_INT4_MOE_TP {
 public:
  using input_t = ggml_bf16_t;
  using output_t = float;
  GeneralMOEConfig config;
  explicit SYCL_XMX_GPTQ_INT4_MOE_TP(GeneralMOEConfig c, int part = 0) : config(c) {
    if (part != 0 || c.pool->config.subpool_count != 1)
      throw std::runtime_error("SYCL XMX currently requires one CPU threadpool/NUMA part");
    if (c.quant_config.bits != 4 || c.quant_config.zero_point)
      throw std::runtime_error("SYCL XMX requires symmetric GPTQ INT4 weights");
    int gs = c.quant_config.group_size;
    if (gs <= 0 || gs % 32 || c.hidden_size <= 0 || c.intermediate_size <= 0 || c.hidden_size % gs ||
        c.intermediate_size % gs || c.hidden_size % 16 || c.intermediate_size % 16 || c.expert_num <= 0 ||
        c.num_experts_per_tok <= 0 || c.max_possible_qlen() <= 0)
      throw std::runtime_error("SYCL XMX requires positive dimensions, N aligned to 16 and K/group_size aligned to 32");
    auto device = sycl_int4::queue().get_device();
    if (!device.has(sycl::aspect::ext_intel_esimd) || !device.has(sycl::aspect::ext_intel_matrix))
      throw std::runtime_error("SYCL XMX requires an Intel GPU with ESIMD and XMX");
    const char* mode = std::getenv("KT_SYCL_XMX_MODE");
    std::string value = mode ? mode : "fp16";
    if (value != "fp16" && value != "int8") throw std::runtime_error("KT_SYCL_XMX_MODE must be fp16 or int8");
    q8_ = value == "int8";
    const char* tile_rows = std::getenv("KT_SYCL_XMX_ROWS");
    std::string tile = tile_rows ? tile_rows : "8";
    if (tile != "1" && tile != "2" && tile != "4" && tile != "8")
      throw std::runtime_error("KT_SYCL_XMX_ROWS must be 1, 2, 4 or 8");
    tile_rows_ = std::stoi(tile);
    const char* graph = std::getenv("KT_SYCL_XMX_GRAPH");
    if (graph && std::string(graph) != "0" && std::string(graph) != "1")
      throw std::runtime_error("KT_SYCL_XMX_GRAPH must be 0 or 1");
    use_graph_ = graph && std::string(graph) == "1";
    const char* hq = std::getenv("KT_SYCL_XMX_HOST_QUANT");
    if (hq && std::string(hq) != "0" && std::string(hq) != "1")
      throw std::runtime_error("KT_SYCL_XMX_HOST_QUANT must be 0 or 1");
    host_quant_ = hq && std::string(hq) == "1";
    const char* fq = std::getenv("KT_SYCL_XMX_FUSED_QUANT");
    if (fq && std::string(fq) != "0" && std::string(fq) != "1" && std::string(fq) != "prefill")
      throw std::runtime_error("KT_SYCL_XMX_FUSED_QUANT must be 0, 1 or prefill");
    fused_quant_ = fq && std::string(fq) != "0";
    fused_decode_ = fq && std::string(fq) == "1";
    if (fused_quant_ && (!q8_ || gs != 128 || c.intermediate_size % 128))
      throw std::runtime_error("SYCL XMX fused quantization requires int8 mode and group_size=128");
    if (use_graph_ && !device.has(sycl::aspect::ext_oneapi_limited_graph))
      throw std::runtime_error("SYCL XMX command graph is unavailable on this device");
    const int max_rows = c.max_possible_qlen();
    if (max_rows > std::numeric_limits<int>::max() / c.num_experts_per_tok ||
        max_rows > std::numeric_limits<int>::max() / c.hidden_size)
      throw std::runtime_error("SYCL XMX workspace dimensions exceed 32-bit indexing");
    const int max_routed = max_rows * c.num_experts_per_tok;
    if (max_routed > std::numeric_limits<int>::max() / c.hidden_size ||
        max_routed > std::numeric_limits<int>::max() / c.intermediate_size)
      throw std::runtime_error("SYCL XMX routed workspace exceeds 32-bit indexing");
    if (use_graph_) graphs_.resize(static_cast<size_t>(c.num_experts_per_tok + 1) * (c.num_experts_per_tok + 1));
    counts_.resize(c.expert_num);
    offsets_.resize(c.expert_num + 1);
    cursors_.resize(c.expert_num);
    input_.reserve(static_cast<size_t>(max_rows) * c.hidden_size);
    activated_.reserve(static_cast<size_t>(max_routed) * c.intermediate_size);
    down_.reserve(static_cast<size_t>(max_routed) * c.hidden_size);
    output_.reserve(static_cast<size_t>(max_rows) * c.hidden_size);
    row_tokens_.reserve(max_routed);
    slot_rows_.reserve(max_routed);
    routing_.reserve(max_routed);
    tiles_.reserve(max_routed);
    if (q8_) {
      qi_.reserve(static_cast<size_t>(max_rows) * c.hidden_size);
      qm_.reserve(static_cast<size_t>(max_routed) * c.intermediate_size);
      si_.reserve(static_cast<size_t>(max_rows) * c.hidden_size / gs);
      sm_.reserve(static_cast<size_t>(max_routed) * c.intermediate_size / gs);
    }
    printf("Created SYCL_XMX_GPTQ_INT4_MOE mode=%s layer=%d\n", value.c_str(), c.layer_idx);
  }

  void load_weights(const GeneralMOEConfig& source) {
    const int e = config.expert_num, h = config.hidden_size, i = config.intermediate_size;
    const int gs = config.quant_config.group_size;
    gate_.init(e, i, h, gs);
    up_.init(e, i, h, gs);
    down_weight_.init(e, h, i, gs);
    const auto* map = static_cast<const uint64_t*>(source.physical_to_logical_map);
    loaded_.assign(e, false);
    for (int physical = 0; physical < e; ++physical) {
      uint64_t mapped = map ? map[physical] : static_cast<uint64_t>(physical);
      if (mapped >= static_cast<uint64_t>(e)) throw std::runtime_error("Invalid XMX physical_to_logical_map");
      int logical = static_cast<int>(mapped);
      if (source.should_skip_expert(logical)) continue;
      auto load = [&](sycl_xmx::Weight& w, void* p, void* s, const auto& ps, const auto& ss) {
        const uint32_t* qw =
            ps.empty() ? static_cast<const uint32_t*>(p) : static_cast<const uint32_t*>(ps.at(0).at(logical));
        const float* sc = ss.empty() ? static_cast<const float*>(s) : static_cast<const float*>(ss.at(0).at(logical));
        if (ps.empty() && qw) qw += static_cast<size_t>(logical) * w.n * w.k / 8;
        if (ss.empty() && sc) sc += static_cast<size_t>(logical) * w.n * (w.k / w.gs);
        w.from_mat(physical, qw, sc);
      };
      load(gate_, source.gate_proj, source.gate_scale, source.gate_projs, source.gate_scales);
      load(up_, source.up_proj, source.up_scale, source.up_projs, source.up_scales);
      load(down_weight_, source.down_proj, source.down_scale, source.down_projs, source.down_scales);
      loaded_[physical] = true;
    }
  }

  void forward(int rows, int topk, const int64_t* ids, const float* weights, const void* input, void* output) {
    if (rows < 0 || rows > config.max_possible_qlen() || topk < 0 || topk > config.num_experts_per_tok)
      throw std::runtime_error("XMX forward exceeds configured workspace");
    if (rows == 0) return;
    if (loaded_.empty()) throw std::runtime_error("XMX weights not loaded");
    const int h = config.hidden_size, i = config.intermediate_size;
    std::fill(counts_.begin(), counts_.end(), 0);
    std::fill(slot_rows_.data, slot_rows_.data + rows * topk, -1);
    for (int slot = 0; slot < rows * topk; ++slot) {
      int64_t expert = ids[slot];
      if (expert >= 0 && expert < config.expert_num && loaded_[expert]) ++counts_[expert];
    }
    offsets_[0] = 0;
    for (int e = 0; e < config.expert_num; ++e) {
      offsets_[e + 1] = offsets_[e] + counts_[e];
      cursors_[e] = offsets_[e];
    }
    for (int slot = 0; slot < rows * topk; ++slot) {
      int64_t expert = ids[slot];
      if (expert < 0 || expert >= config.expert_num || !loaded_[expert]) continue;
      const int routed = cursors_[expert]++;
      row_tokens_.data[routed] = slot / topk;
      slot_rows_.data[slot] = routed;
    }
    int routed = offsets_.back(), nt = 0, tile_rows = rows == 1 ? 1 : tile_rows_;
    for (int e = 0; e < config.expert_num; ++e) {
      for (int r = 0; r < counts_[e]; r += tile_rows)
        tiles_.data[nt++] = {e, offsets_[e] + r, std::min(tile_rows, counts_[e] - r)};
    }
    if (nt == 0) {
      std::fill_n(static_cast<float*>(output), static_cast<size_t>(rows) * h, 0.0f);
      return;
    }
    std::memcpy(input_.data, input, static_cast<size_t>(rows) * h * 2);
    std::memcpy(routing_.data, weights, static_cast<size_t>(rows) * topk * sizeof(float));
    if (q8_ && host_quant_ && rows == 1) {
      const int gs = config.quant_config.group_size;
      for (int g = 0; g < h / gs; ++g) {
        const auto* src = static_cast<const ggml_bf16_t*>(input) + g * gs;
        float maximum = 0;
        for (int j = 0; j < gs; j += 8) {
          __m256 v = avx2::load_bf16_to_fp32(src + j);
          v = _mm256_and_ps(v, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
          alignas(32) float values[8];
          _mm256_store_ps(values, v);
          for (float value : values) maximum = std::max(maximum, value);
        }
        const float scale = maximum / 127.0f;
        si_.data[g] = scale;
        const __m256 inv = _mm256_set1_ps(scale > 0 ? 1.0f / scale : 0.0f);
        for (int j = 0; j < gs; j += 8) {
          __m256 v = _mm256_mul_ps(avx2::load_bf16_to_fp32(src + j), inv);
          __m256i iv = _mm256_cvtps_epi32(_mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
          alignas(32) int values[8];
          _mm256_store_si256(reinterpret_cast<__m256i*>(values), iv);
          for (int r = 0; r < 8; ++r) qi_.data[g * gs + j + r] = static_cast<int8_t>(std::clamp(values[r], -127, 127));
        }
      }
    }
    try {
      switch (tile_rows) {
        case 1:
          run<1>(nt, rows, topk, routed);
          break;
        case 2:
          run<2>(nt, rows, topk, routed);
          break;
        case 4:
          run<4>(nt, rows, topk, routed);
          break;
        default:
          run<8>(nt, rows, topk, routed);
          break;
      }
    } catch (...) {
      // A failed submission may leave earlier stages in flight. Drain before
      // the caller reuses the shared workspace or destroys this executor.
      try {
        sycl_int4::queue().wait_and_throw();
      } catch (...) {
      }
      throw;
    }
    std::memcpy(output, output_.data, static_cast<size_t>(rows) * h * sizeof(float));
  }

 private:
  bool q8_ = false;
  int tile_rows_ = 8;
  bool use_graph_ = false;
  bool host_quant_ = false;
  bool fused_quant_ = false;
  bool fused_decode_ = false;
  using Graph =
      sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>;
  std::vector<bool> loaded_;
  std::vector<int> counts_, offsets_, cursors_;
  sycl_xmx::Weight gate_, up_, down_weight_;
  sycl_xmx::USM<uint16_t> input_, activated_;
  sycl_xmx::USM<float> down_, output_, routing_, si_, sm_;
  sycl_xmx::USM<int8_t> qi_, qm_;
  sycl_xmx::USM<int> row_tokens_, slot_rows_;
  sycl_xmx::USM<sycl_xmx::Tile> tiles_;
  // Executable graphs must be destroyed before the USM storage they reference.
  std::vector<std::unique_ptr<Graph>> graphs_;

  template <int Rows, bool Q8>
  sycl::event execute(int nt, int rows, int topk, int routed) {
    sycl::event qi;
    const bool device_quant = Q8 && !(host_quant_ && rows == 1);
    const bool use_fused_quant = fused_quant_ && (rows > 1 || fused_decode_);
    if constexpr (Q8)
      if (device_quant)
        qi = sycl_xmx::quantize(input_.data, qi_.data, si_.data, rows, config.hidden_size,
                                config.quant_config.group_size);
    sycl::event gu;
    if constexpr (Q8) {
      if (use_fused_quant)
        gu = sycl_xmx::projection<Rows, true, true, true>(
            tiles_.data, nt, row_tokens_.data, input_.data, qi_.data, si_.data, gate_, &up_, nullptr, nullptr,
            config.swiglu_limit, config.swiglu_alpha, device_quant ? &qi : nullptr, qm_.data, sm_.data);
    }
    if (!use_fused_quant)
      gu = sycl_xmx::projection<Rows, Q8, true>(tiles_.data, nt, row_tokens_.data, input_.data, qi_.data, si_.data,
                                                gate_, &up_, activated_.data, nullptr, config.swiglu_limit,
                                                config.swiglu_alpha, device_quant ? &qi : nullptr);
    sycl::event middle = gu;
    if constexpr (Q8)
      if (!use_fused_quant)
        middle = sycl_xmx::quantize(activated_.data, qm_.data, sm_.data, routed, config.intermediate_size,
                                    config.quant_config.group_size, &gu);
    auto d = sycl_xmx::projection<Rows, Q8, false>(tiles_.data, nt, nullptr, activated_.data, qm_.data, sm_.data,
                                                   down_weight_, nullptr, nullptr, down_.data, 0, 0, &middle);
    auto result =
        sycl_xmx::merge(slot_rows_.data, routing_.data, down_.data, output_.data, rows, topk, config.hidden_size, d);
    return result;
  }
  template <int Rows>
  sycl::event submit(int nt, int rows, int topk, int routed) {
    return q8_ ? execute<Rows, true>(nt, rows, topk, routed) : execute<Rows, false>(nt, rows, topk, routed);
  }
  template <int Rows>
  void run(int nt, int rows, int topk, int routed) {
    if constexpr (Rows == 1) {
      if (use_graph_ && rows == 1) {
        namespace gx = sycl::ext::oneapi::experimental;
        auto& cached = graphs_[topk * (config.num_experts_per_tok + 1) + nt];
        auto& q = sycl_int4::queue();
        if (!cached) {
          gx::command_graph graph(q, {gx::property::graph::assume_buffer_outlives_graph{}});
          graph.begin_recording(q);
          try {
            (void)submit<Rows>(nt, rows, topk, routed);
          } catch (...) {
            graph.end_recording();
            throw;
          }
          graph.end_recording();
          cached = std::make_unique<Graph>(graph.finalize());
        }
        auto result = q.ext_oneapi_graph(*cached);
        sycl_int4::wait_and_throw(result);
        return;
      }
    }
    auto result = submit<Rows>(nt, rows, topk, routed);
    sycl_int4::wait_and_throw(result);
  }
};

template <>
class TP_MOE<SYCL_XMX_GPTQ_INT4_MOE_TP> : public TP_MOE_Common<SYCL_XMX_GPTQ_INT4_MOE_TP> {
 public:
  using Base = TP_MOE_Common<SYCL_XMX_GPTQ_INT4_MOE_TP>;
  explicit TP_MOE(const GeneralMOEConfig& config) : Base(config) {
    const char* numa = std::getenv("KT_SYCL_XMX_NUMA_DISPATCH");
    if (numa && std::string(numa) != "0" && std::string(numa) != "1")
      throw std::runtime_error("KT_SYCL_XMX_NUMA_DISPATCH must be 0 or 1");
    use_numa_dispatch_ = numa && std::string(numa) == "1";
  }
  void load_weights() override {
    this->weights_loaded = false;
    this->tps[0]->load_weights(this->config);
    this->weights_loaded = true;
  }
  void forward(int rows, int topk, const int64_t* ids, const float* weights, const void* input, void* output,
               bool incremental = false) override {
    if (!this->weights_loaded) throw std::runtime_error("XMX weights not loaded");
    if (use_numa_dispatch_) {
      // Worker exceptions must be captured here: the generic NUMA distributor
      // does not propagate exceptions from its worker threads.
      std::exception_ptr failure;
      this->config.pool->dispense_backend()->do_numa_job([&](int part) {
        try {
          this->tps[part]->forward(rows, topk, ids, weights, input, this->local_output_numa[part]);
        } catch (...) {
          failure = std::current_exception();
        }
      });
      if (failure) std::rethrow_exception(failure);
    } else {
      this->tps[0]->forward(rows, topk, ids, weights, input, this->local_output_numa[0]);
    }
    merge_results(rows, output, incremental);
  }
  void forward_binding(intptr_t rows, int topk, intptr_t ids, intptr_t weights, intptr_t input, intptr_t output,
                       bool incremental) {
    if (!rows) throw std::runtime_error("XMX qlen pointer is null");
    forward(*reinterpret_cast<const int*>(rows), topk, reinterpret_cast<const int64_t*>(ids),
            reinterpret_cast<const float*>(weights), reinterpret_cast<const void*>(input),
            reinterpret_cast<void*>(output), incremental);
  }
  void merge_results(int rows, void* output) override { merge_results(rows, output, false); }
  void merge_results(int rows, void* output, bool incremental) override {
    auto* dst = static_cast<ggml_bf16_t*>(output);
    const float* src = this->local_output_numa[0];
    for (int j = 0; j < rows * this->config.hidden_size; j += 8) {
      __m256 value = _mm256_loadu_ps(src + j);
      if (incremental) value = _mm256_add_ps(value, avx2::load_bf16_to_fp32(dst + j));
      avx2::store_fp32_to_bf16(dst + j, value);
    }
  }

 private:
  bool use_numa_dispatch_ = false;
};
#endif
