/** CPU + Intel iGPU GPTQ INT4 MoE with one shared packed weight copy. */
#ifndef CPUINFER_OPERATOR_SYCL_GPTQ_INT4_CPU_IGPU_MOE_H
#define CPUINFER_OPERATOR_SYCL_GPTQ_INT4_CPU_IGPU_MOE_H

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "../avx2/gptq_int4_packed_avxvnni-moe.hpp"
#include "../cpu_igpu_load_scheduler.hpp"
#include "../cpu_load_monitor.hpp"
#include "gptq_int4_sycl-moe.hpp"

// Tag type required by the generic TP_MOE binding. The specialization below
// owns the CPU and SYCL implementations directly.
class CPU_IGPU_GPTQ_INT4_MOE_PART {
 public:
  using output_t = float;

  CPU_IGPU_GPTQ_INT4_MOE_PART(GeneralMOEConfig, int) {}
  void forward(int, int, const int64_t*, const float*, const void*, void*) {}
};

template <>
class TP_MOE<CPU_IGPU_GPTQ_INT4_MOE_PART> : public MoE_Interface {
  using CPUKernel = avxvnni_packed::GemmKernelAVXVNNI256PackedGPTQInt4;
  using SYCLKernel = sycl_int4::GemmKernelSYCLGPTQInt4;
  using CPUPart = AVXVNNI256_PACKED_GPTQ_INT4_MOE_TP<CPUKernel>;
  using SYCLPart = SYCL_GPTQ_INT4_MOE_TP<SYCLKernel>;

  struct IGPURun {
    int qlen = 0;
    int topk = 0;
    const int64_t* expert_ids = nullptr;
    const float* weights = nullptr;
    const void* input = nullptr;
    void* output = nullptr;
  };

  struct ForwardScratch {
    std::vector<int64_t> cpu_expert_ids;
    std::vector<int64_t> igpu_expert_ids;
    std::vector<float> cpu_output;
    std::vector<float> igpu_output;
  };

 public:
  GeneralMOEConfig config;

  explicit TP_MOE(const GeneralMOEConfig& input_config) : config(input_config) {
    validate_config();

    GeneralMOEConfig sycl_config = config;
    GeneralMOEConfig cpu_config = config;
    cpu_config.external_moe_weights = true;
    sycl_part_ = std::make_unique<SYCLPart>(sycl_config, 0);
    cpu_part_ = std::make_unique<CPUPart>(cpu_config, 0);
    target_map_.resize(config.expert_num, 0);

    decode_policy_.igpu_ratio = config.cpu_igpu_dynamic ? 0.0f : configured_ratio(true);
    decode_policy_.calls_since_switch = config.cpu_igpu_decode_min_dwell;
    if (config.cpu_igpu_dynamic) {
      load_monitor_ =
          CPULoadMonitor::acquire(config.pool, config.cpu_igpu_load_sample_ms, config.cpu_igpu_load_ewma_alpha);
      prefill_scheduler_ = cpu_igpu_scheduler::acquire_prefill_scheduler(config.pool, config.cpu_igpu_prefill_load_low,
                                                                         config.cpu_igpu_prefill_load_high,
                                                                         config.cpu_igpu_prefill_min_dwell);
      prefill_scheduler_->register_layer(config.layer_idx);
    }
  }

  ~TP_MOE() {
    {
      std::lock_guard<std::mutex> lock(igpu_mutex_);
      igpu_stop_ = true;
    }
    igpu_cv_.notify_one();
    if (igpu_worker_.joinable()) igpu_worker_.join();
  }

  TP_MOE(const TP_MOE&) = delete;
  TP_MOE& operator=(const TP_MOE&) = delete;

  void load_weights() {
    sycl_part_->mutable_config().physical_to_logical_map = config.physical_to_logical_map;
    sycl_part_->load_weights();

    for (int expert = 0; expert < config.expert_num; ++expert) {
      bind_cpu_weight(cpu_part_->gate_weight(expert), sycl_part_->gate_weight(expert));
      bind_cpu_weight(cpu_part_->up_weight(expert), sycl_part_->up_weight(expert));
      bind_cpu_weight(cpu_part_->down_weight(expert), sycl_part_->down_weight(expert));
    }
    weights_loaded_ = true;
  }

  void warm_up() {
    std::vector<ggml_bf16_t> input(config.hidden_size);
    std::vector<ggml_bf16_t> output(config.hidden_size);
    std::vector<int64_t> expert_ids(config.num_experts_per_tok);
    std::vector<float> weights(config.num_experts_per_tok, 1.0f / config.num_experts_per_tok);
    std::iota(expert_ids.begin(), expert_ids.end(), 0);
    forward(1, config.num_experts_per_tok, expert_ids.data(), weights.data(), input.data(), output.data(), false);
  }

  void forward(int qlen, int topk, const int64_t* expert_ids, const float* weights, const void* input, void* output,
               bool incremental = false) override {
    if (!weights_loaded_) throw std::runtime_error("CPU/iGPU GPTQ INT4 weights are not loaded");
    if (qlen <= 0 || qlen > config.max_len) throw std::runtime_error("CPU/iGPU GPTQ INT4 received invalid qlen");
    if (topk <= 0 || topk > config.num_experts_per_tok) {
      throw std::runtime_error("CPU/iGPU GPTQ INT4 received invalid topk");
    }

    const bool decode = cpu_igpu_scheduler::decode_from_qlen(qlen);
    const float igpu_ratio = choose_igpu_ratio(decode);
    auto& scratch = forward_scratch();
    const size_t output_elements = static_cast<size_t>(qlen) * config.hidden_size;

    if (igpu_ratio <= 0.0f) {
      scratch.cpu_output.resize(output_elements);
      cpu_part_->forward(qlen, topk, expert_ids, weights, input, scratch.cpu_output.data());
      merge_single_output(scratch.cpu_output, qlen, output, incremental);
      return;
    }
    if (igpu_ratio >= 1.0f) {
      scratch.igpu_output.resize(output_elements);
      sycl_part_->forward(qlen, topk, expert_ids, weights, input, scratch.igpu_output.data());
      merge_single_output(scratch.igpu_output, qlen, output, incremental);
      return;
    }

    const size_t routed_items = static_cast<size_t>(qlen) * topk;
    scratch.cpu_expert_ids.assign(routed_items, -1);
    scratch.igpu_expert_ids.assign(routed_items, -1);
    partition_experts(qlen, topk, expert_ids, igpu_ratio, scratch.cpu_expert_ids, scratch.igpu_expert_ids);
    scratch.cpu_output.resize(output_elements);
    scratch.igpu_output.resize(output_elements);

    if (active_cpu_rows_ == 0) std::fill(scratch.cpu_output.begin(), scratch.cpu_output.end(), 0.0f);
    if (active_igpu_rows_ == 0) std::fill(scratch.igpu_output.begin(), scratch.igpu_output.end(), 0.0f);

    // Decode can overlap the fused SYCL path with the VNNI worker pool.
    // Prefill routing still uses the CPU pool inside the SYCL implementation,
    // so mixed fixed-ratio Prefill remains serialized.
    if (decode && active_cpu_rows_ > 0 && active_igpu_rows_ > 0) {
      submit_igpu({qlen, topk, scratch.igpu_expert_ids.data(), weights, input, scratch.igpu_output.data()});
      cpu_part_->forward(qlen, topk, scratch.cpu_expert_ids.data(), weights, input, scratch.cpu_output.data());
      wait_igpu();
    } else {
      if (active_cpu_rows_ > 0) {
        cpu_part_->forward(qlen, topk, scratch.cpu_expert_ids.data(), weights, input, scratch.cpu_output.data());
      }
      if (active_igpu_rows_ > 0) {
        sycl_part_->forward(qlen, topk, scratch.igpu_expert_ids.data(), weights, input, scratch.igpu_output.data());
      }
    }
    merge_outputs(scratch.cpu_output, scratch.igpu_output, qlen, output, incremental);
  }

  void forward_binding(intptr_t qlen_pointer, int topk, intptr_t expert_ids, intptr_t weights, intptr_t input,
                       intptr_t output, bool incremental) {
    forward(*reinterpret_cast<int*>(qlen_pointer), topk, reinterpret_cast<const int64_t*>(expert_ids),
            reinterpret_cast<const float*>(weights), reinterpret_cast<const void*>(input),
            reinterpret_cast<void*>(output), incremental);
  }

 private:
  std::unique_ptr<SYCLPart> sycl_part_;
  std::unique_ptr<CPUPart> cpu_part_;
  bool weights_loaded_ = false;

  std::vector<uint8_t> target_map_;  // 0 = CPU, 1 = iGPU
  int active_cpu_rows_ = 0;
  int active_igpu_rows_ = 0;
  std::shared_ptr<CPULoadMonitor> load_monitor_;
  std::shared_ptr<cpu_igpu_scheduler::CoherentPrefillScheduler> prefill_scheduler_;
  cpu_igpu_scheduler::LoadHysteresisState decode_policy_;

  std::thread igpu_worker_;
  std::mutex igpu_mutex_;
  std::condition_variable igpu_cv_;
  bool igpu_stop_ = false;
  bool igpu_pending_ = false;
  bool igpu_done_ = true;
  IGPURun igpu_run_;
  std::exception_ptr igpu_error_;

  static ForwardScratch& forward_scratch() {
    static thread_local ForwardScratch scratch;
    return scratch;
  }

  void validate_config() const {
    if (config.pool == nullptr) throw std::runtime_error("CPU/iGPU GPTQ INT4 requires a worker pool");
    if (config.pool->config.subpool_count != 1) {
      throw std::runtime_error("CPU/iGPU GPTQ INT4 currently supports one NUMA subpool and TP=1");
    }
    if (config.max_len <= 0) throw std::runtime_error("CPU/iGPU GPTQ INT4 requires max_len > 0");
    const auto valid_ratio = [](float ratio) { return ratio >= 0.0f && ratio <= 1.0f; };
    const auto valid_phase_ratio = [&](float ratio) { return ratio < 0.0f || valid_ratio(ratio); };
    if (!valid_ratio(config.cpu_igpu_igpu_ratio) || !valid_phase_ratio(config.cpu_igpu_prefill_ratio) ||
        !valid_phase_ratio(config.cpu_igpu_decode_ratio)) {
      throw std::runtime_error("CPU/iGPU ratios must be between 0 and 1, or negative to inherit");
    }
    const auto valid_thresholds = [](float low, float high) { return low >= 0.0f && low < high && high <= 1.0f; };
    if (!valid_thresholds(config.cpu_igpu_decode_load_low, config.cpu_igpu_decode_load_high) ||
        !valid_thresholds(config.cpu_igpu_prefill_load_low, config.cpu_igpu_prefill_load_high) ||
        config.cpu_igpu_load_ewma_alpha <= 0.0f || config.cpu_igpu_load_ewma_alpha > 1.0f ||
        config.cpu_igpu_load_sample_ms < 10 || config.cpu_igpu_decode_min_dwell <= 0 ||
        config.cpu_igpu_prefill_min_dwell <= 0) {
      throw std::runtime_error("CPU/iGPU scheduler parameters are invalid");
    }
  }

  float choose_igpu_ratio(bool decode) {
    if (!config.cpu_igpu_dynamic) return configured_ratio(decode);
    const float load = load_monitor_->contention();
    if (!decode) return prefill_scheduler_->begin_forward(config.layer_idx, load);
    return cpu_igpu_scheduler::update_load_hysteresis(decode_policy_, load, config.cpu_igpu_decode_load_low,
                                                      config.cpu_igpu_decode_load_high,
                                                      config.cpu_igpu_decode_min_dwell);
  }

  float configured_ratio(bool decode) const {
    const float phase_ratio = decode ? config.cpu_igpu_decode_ratio : config.cpu_igpu_prefill_ratio;
    return phase_ratio < 0.0f ? config.cpu_igpu_igpu_ratio : phase_ratio;
  }

  static void bind_cpu_weight(const std::shared_ptr<typename CPUKernel::BufferB>& cpu_weight,
                              const std::shared_ptr<typename SYCLKernel::BufferB>& sycl_weight) {
    cpu_weight->bind_view(sycl_weight->qweight, sycl_weight->scales, sycl_weight->weight_sums);
  }

  void partition_experts(int qlen, int topk, const int64_t* expert_ids, float igpu_ratio,
                         std::vector<int64_t>& cpu_expert_ids, std::vector<int64_t>& igpu_expert_ids) {
    std::vector<int> row_counts(config.expert_num, 0);
    for (int item = 0; item < qlen * topk; ++item) {
      const int64_t expert = expert_ids[item];
      if (!config.should_skip_expert(expert)) ++row_counts[expert];
    }

    std::vector<int> active_experts;
    int total_rows = 0;
    for (int expert = 0; expert < config.expert_num; ++expert) {
      if (row_counts[expert] <= 0) continue;
      active_experts.push_back(expert);
      total_rows += row_counts[expert];
    }
    std::sort(active_experts.begin(), active_experts.end(), [&](int left, int right) {
      if (row_counts[left] != row_counts[right]) return row_counts[left] > row_counts[right];
      return left < right;
    });

    std::fill(target_map_.begin(), target_map_.end(), 0);
    const int target_igpu_rows = static_cast<int>(std::lround(total_rows * igpu_ratio));
    int assigned_igpu_rows = 0;
    for (int expert : active_experts) {
      const int current_distance = std::abs(target_igpu_rows - assigned_igpu_rows);
      const int next_distance = std::abs(target_igpu_rows - assigned_igpu_rows - row_counts[expert]);
      if (assigned_igpu_rows < target_igpu_rows && next_distance <= current_distance) {
        target_map_[expert] = 1;
        assigned_igpu_rows += row_counts[expert];
      }
    }

    active_cpu_rows_ = 0;
    active_igpu_rows_ = 0;
    for (int item = 0; item < qlen * topk; ++item) {
      const int64_t expert = expert_ids[item];
      if (config.should_skip_expert(expert)) continue;
      if (target_map_[expert]) {
        igpu_expert_ids[item] = expert;
        ++active_igpu_rows_;
      } else {
        cpu_expert_ids[item] = expert;
        ++active_cpu_rows_;
      }
    }
  }

  void merge_single_output(const std::vector<float>& partial_output, int qlen, void* output_pointer,
                           bool incremental) const {
    auto* output = static_cast<ggml_bf16_t*>(output_pointer);
    const size_t elements = static_cast<size_t>(qlen) * config.hidden_size;
    for (size_t index = 0; index < elements; ++index) {
      float value = partial_output[index];
      if (incremental) value += GGML_BF16_TO_FP32(output[index]);
      output[index] = GGML_FP32_TO_BF16(value);
    }
  }

  void merge_outputs(const std::vector<float>& cpu_output, const std::vector<float>& igpu_output, int qlen,
                     void* output_pointer, bool incremental) const {
    auto* output = static_cast<ggml_bf16_t*>(output_pointer);
    const size_t elements = static_cast<size_t>(qlen) * config.hidden_size;
    for (size_t index = 0; index < elements; ++index) {
      float value = cpu_output[index] + igpu_output[index];
      if (incremental) value += GGML_BF16_TO_FP32(output[index]);
      output[index] = GGML_FP32_TO_BF16(value);
    }
  }

  void submit_igpu(IGPURun run) {
    if (!igpu_worker_.joinable()) igpu_worker_ = std::thread(&TP_MOE::igpu_worker_loop, this);
    std::lock_guard<std::mutex> lock(igpu_mutex_);
    if (igpu_pending_ || !igpu_done_) throw std::runtime_error("CPU/iGPU GPTQ INT4 has an overlapping iGPU job");
    igpu_run_ = run;
    igpu_error_ = nullptr;
    igpu_done_ = false;
    igpu_pending_ = true;
    igpu_cv_.notify_one();
  }

  void wait_igpu() {
    std::unique_lock<std::mutex> lock(igpu_mutex_);
    igpu_cv_.wait(lock, [&] { return igpu_done_; });
    if (igpu_error_) std::rethrow_exception(igpu_error_);
  }

  void igpu_worker_loop() {
    while (true) {
      IGPURun run;
      {
        std::unique_lock<std::mutex> lock(igpu_mutex_);
        igpu_cv_.wait(lock, [&] { return igpu_stop_ || igpu_pending_; });
        if (igpu_stop_) return;
        run = igpu_run_;
        igpu_pending_ = false;
      }

      std::exception_ptr error;
      try {
        sycl_part_->forward(run.qlen, run.topk, run.expert_ids, run.weights, run.input, run.output);
      } catch (...) {
        error = std::current_exception();
      }

      {
        std::lock_guard<std::mutex> lock(igpu_mutex_);
        igpu_error_ = error;
        igpu_done_ = true;
      }
      igpu_cv_.notify_all();
    }
  }
};

#endif  // CPUINFER_OPERATOR_SYCL_GPTQ_INT4_CPU_IGPU_MOE_H
