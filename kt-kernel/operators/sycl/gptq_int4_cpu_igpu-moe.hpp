/** CPU + Intel iGPU GPTQ INT4 MoE with one shared packed weight copy. */
#ifndef CPUINFER_OPERATOR_SYCL_GPTQ_INT4_CPU_IGPU_MOE_H
#define CPUINFER_OPERATOR_SYCL_GPTQ_INT4_CPU_IGPU_MOE_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "../cpu_load_monitor.hpp"
#include "../cpu_igpu_service_scheduler.hpp"
#include "../avx2/gptq_int4_packed_avxvnni-moe.hpp"
#include "gptq_int4_sycl-moe.hpp"

// This tag satisfies the TP_MOE binding constraint. The specialization below
// owns the actual CPU and SYCL parts directly because the first implementation
// targets the already-supported single-NUMA, TP=1 endpoint configuration.
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

  struct PolicyState {
    float igpu_ratio = 0.0f;
    int calls_since_switch = 0;
    float cpu_ms_per_row = 0.0f;
    float igpu_ms_per_row = 0.0f;
    float cpu_sample_load = 0.0f;
    float igpu_sample_load = 0.0f;
    bool has_cpu_sample = false;
    bool has_igpu_sample = false;
    bool high_load_epoch = false;
    int cpu_samples = 0;
    int igpu_samples = 0;
    int switch_count = 0;
  };

 public:
  GeneralMOEConfig config;

  explicit TP_MOE(const GeneralMOEConfig& input_config) : config(input_config) {
    if (config.pool == nullptr) throw std::runtime_error("CPU/iGPU GPTQ INT4 requires a worker pool");
    if (config.pool->config.subpool_count != 1) {
      throw std::runtime_error("CPU/iGPU GPTQ INT4 currently supports one NUMA subpool and TP=1");
    }
    if (config.max_len <= 0) throw std::runtime_error("CPU/iGPU GPTQ INT4 requires max_len > 0");
    if (config.cpu_igpu_igpu_ratio < 0.0f || config.cpu_igpu_igpu_ratio > 1.0f) {
      throw std::runtime_error("cpu_igpu_igpu_ratio must be between 0 and 1");
    }
    const auto valid_phase_ratio = [](float ratio) { return ratio < 0.0f || ratio <= 1.0f; };
    if (!valid_phase_ratio(config.cpu_igpu_prefill_ratio) || !valid_phase_ratio(config.cpu_igpu_decode_ratio)) {
      throw std::runtime_error("CPU/iGPU phase ratios must be negative (inherit) or between 0 and 1");
    }
    validate_dynamic_config();

    GeneralMOEConfig sycl_config = config;
    GeneralMOEConfig cpu_config = config;
    cpu_config.external_moe_weights = true;
    sycl_part_ = std::make_unique<SYCLPart>(sycl_config, 0);
    cpu_part_ = std::make_unique<CPUPart>(cpu_config, 0);
    target_map_.resize(config.expert_num, 0);
    decode_policy_.igpu_ratio = config.cpu_igpu_dynamic ? 0.0f : configured_ratio(true);
    prefill_policy_.igpu_ratio = config.cpu_igpu_dynamic ? 0.0f : configured_ratio(false);
    decode_policy_.calls_since_switch = config.cpu_igpu_decode_min_dwell;
    prefill_policy_.calls_since_switch = config.cpu_igpu_prefill_min_dwell;
    if (config.cpu_igpu_dynamic) {
      load_monitor_ = CPULoadMonitor::acquire(config.pool->get_bound_cpu_ids(), config.cpu_igpu_load_sample_ms,
                                              config.cpu_igpu_load_ewma_alpha);
      decode_scheduler_ = cpu_igpu_scheduler::acquire(
          config.pool,
          {config.cpu_igpu_cost_ewma_alpha, config.cpu_igpu_decode_switch_margin,
           config.cpu_igpu_decode_cost_load_match_delta, config.cpu_igpu_decode_load_reprobe_delta,
           config.cpu_igpu_decode_load_high,
           config.cpu_igpu_decode_calibration_samples, config.cpu_igpu_decode_min_dwell,
           config.cpu_igpu_decode_load_reprobe_grace, config.cpu_igpu_decode_reprobe_samples,
           config.cpu_igpu_decode_reprobe_interval});
      decode_scheduler_->register_layer(config.layer_idx);
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
    cpu_part_->release_owned_weight_storage();
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

    auto& scratch = forward_scratch();
    const float load = load_monitor_ ? load_monitor_->contention() : 0.0f;
    const bool decode = qlen == 1;
    PolicyState& policy = decode ? decode_policy_ : prefill_policy_;
    if (config.cpu_igpu_dynamic && !decode) decode_scheduler_->notify_phase_boundary(config.layer_idx);
    current_igpu_ratio_ = config.cpu_igpu_dynamic && decode
                              ? decode_scheduler_->begin_forward(config.layer_idx)
                              : choose_igpu_ratio(policy, decode, load);
    record_execution(decode, current_igpu_ratio_.load(std::memory_order_relaxed));
    const size_t output_elements = static_cast<size_t>(qlen) * config.hidden_size;

    if (current_igpu_ratio_.load(std::memory_order_relaxed) <= 0.0f) {
      scratch.cpu_output.resize(output_elements);
      const auto start = std::chrono::steady_clock::now();
      cpu_part_->forward(qlen, topk, expert_ids, weights, input, scratch.cpu_output.data());
      const float duration_ms = elapsed_ms(start);
      record_service_cost(policy, decode, true, duration_ms, count_active_rows(qlen, topk, expert_ids), load);
      merge_single_output(scratch.cpu_output, qlen, output, incremental);
      return;
    }
    if (current_igpu_ratio_.load(std::memory_order_relaxed) >= 1.0f) {
      scratch.igpu_output.resize(output_elements);
      const auto start = std::chrono::steady_clock::now();
      sycl_part_->forward(qlen, topk, expert_ids, weights, input, scratch.igpu_output.data());
      const float duration_ms = elapsed_ms(start);
      record_service_cost(policy, decode, false, duration_ms, count_active_rows(qlen, topk, expert_ids), load);
      merge_single_output(scratch.igpu_output, qlen, output, incremental);
      return;
    }

    const size_t routed_items = static_cast<size_t>(qlen) * topk;
    scratch.cpu_expert_ids.assign(routed_items, -1);
    scratch.igpu_expert_ids.assign(routed_items, -1);
    partition_experts(qlen, topk, expert_ids, scratch.cpu_expert_ids, scratch.igpu_expert_ids);

    scratch.cpu_output.resize(output_elements);
    scratch.igpu_output.resize(output_elements);

    if (active_cpu_rows_ == 0) std::fill(scratch.cpu_output.begin(), scratch.cpu_output.end(), 0.0f);
    if (active_igpu_rows_ == 0) std::fill(scratch.igpu_output.begin(), scratch.igpu_output.end(), 0.0f);

    // Decode does not use the shared CPU worker pool on the fused SYCL path, so
    // it is safe to overlap with the VNNI worker pool. Prefill still uses that
    // pool for routing and remains serialized until its submit/wait split is
    // refactored.
    const bool parallel_decode = qlen == 1 && active_cpu_rows_ > 0 && active_igpu_rows_ > 0;
    if (parallel_decode) {
      submit_igpu({qlen, topk, scratch.igpu_expert_ids.data(), weights, input, scratch.igpu_output.data()});
      const auto cpu_start = std::chrono::steady_clock::now();
      cpu_part_->forward(qlen, topk, scratch.cpu_expert_ids.data(), weights, input, scratch.cpu_output.data());
      const float cpu_duration_ms = elapsed_ms(cpu_start);
      wait_igpu();
      record_service_cost(policy, decode, true, cpu_duration_ms, active_cpu_rows_, load);
      record_service_cost(policy, decode, false, igpu_duration_ms_, active_igpu_rows_, load);
    } else {
      if (active_cpu_rows_ > 0) {
        const auto start = std::chrono::steady_clock::now();
        cpu_part_->forward(qlen, topk, scratch.cpu_expert_ids.data(), weights, input, scratch.cpu_output.data());
        record_service_cost(policy, decode, true, elapsed_ms(start), active_cpu_rows_, load);
      }
      if (active_igpu_rows_ > 0) {
        const auto start = std::chrono::steady_clock::now();
        sycl_part_->forward(qlen, topk, scratch.igpu_expert_ids.data(), weights, input, scratch.igpu_output.data());
        record_service_cost(policy, decode, false, elapsed_ms(start), active_igpu_rows_, load);
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

  float scheduler_igpu_ratio() const { return current_igpu_ratio_.load(std::memory_order_relaxed); }
  float scheduler_cpu_load() const { return load_monitor_ ? load_monitor_->contention() : 0.0f; }
  std::vector<uint64_t> scheduler_execution_debug(bool decode) const {
    const auto& calls = decode ? decode_execution_calls_ : prefill_execution_calls_;
    const auto& ratio_units = decode ? decode_execution_ratio_units_ : prefill_execution_ratio_units_;
    return {calls.load(std::memory_order_relaxed), ratio_units.load(std::memory_order_relaxed)};
  }
  std::vector<float> scheduler_debug(bool decode) const {
    if (config.cpu_igpu_dynamic && decode) {
      const auto snapshot = decode_scheduler_->snapshot();
      return {snapshot.igpu_ratio,
              snapshot.cpu_ms_per_row,
              snapshot.igpu_ms_per_row,
              static_cast<float>(snapshot.cpu_samples),
              static_cast<float>(snapshot.igpu_samples),
              static_cast<float>(snapshot.switch_count),
              0.0f,
              snapshot.exploring ? 1.0f : 0.0f,
              snapshot.cpu_sample_load,
              snapshot.igpu_sample_load,
              snapshot.igpu_reference_load,
              static_cast<float>(snapshot.reprobe_reason)};
    }
    const PolicyState& state = decode ? decode_policy_ : prefill_policy_;
    return {state.igpu_ratio,
            state.cpu_ms_per_row,
            state.igpu_ms_per_row,
            static_cast<float>(state.cpu_samples),
            static_cast<float>(state.igpu_samples),
            static_cast<float>(state.switch_count),
            state.high_load_epoch ? 1.0f : 0.0f,
            0.0f,
            state.cpu_sample_load,
            state.igpu_sample_load};
  }

 private:
  std::unique_ptr<SYCLPart> sycl_part_;
  std::unique_ptr<CPUPart> cpu_part_;
  bool weights_loaded_ = false;

  std::vector<uint8_t> target_map_;  // 0 = CPU, 1 = iGPU
  int active_cpu_rows_ = 0;
  int active_igpu_rows_ = 0;
  std::shared_ptr<CPULoadMonitor> load_monitor_;
  std::shared_ptr<cpu_igpu_scheduler::ServiceCostScheduler> decode_scheduler_;
  PolicyState decode_policy_;
  PolicyState prefill_policy_;
  static constexpr uint64_t kExecutionRatioScale = 1000000;
  std::atomic<float> current_igpu_ratio_{0.0f};
  std::atomic<uint64_t> decode_execution_calls_{0};
  std::atomic<uint64_t> decode_execution_ratio_units_{0};
  std::atomic<uint64_t> prefill_execution_calls_{0};
  std::atomic<uint64_t> prefill_execution_ratio_units_{0};

  std::thread igpu_worker_;
  std::mutex igpu_mutex_;
  std::condition_variable igpu_cv_;
  bool igpu_stop_ = false;
  bool igpu_pending_ = false;
  bool igpu_done_ = true;
  IGPURun igpu_run_;
  std::exception_ptr igpu_error_;
  float igpu_duration_ms_ = 0.0f;

  static ForwardScratch& forward_scratch() {
    static thread_local ForwardScratch scratch;
    return scratch;
  }

  static float elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
  }

  void record_execution(bool decode, float igpu_ratio) {
    auto& calls = decode ? decode_execution_calls_ : prefill_execution_calls_;
    auto& ratio_units = decode ? decode_execution_ratio_units_ : prefill_execution_ratio_units_;
    calls.fetch_add(1, std::memory_order_relaxed);
    ratio_units.fetch_add(static_cast<uint64_t>(std::llround(igpu_ratio * kExecutionRatioScale)),
                          std::memory_order_relaxed);
  }

  void validate_dynamic_config() const {
    const auto valid_thresholds = [](float low, float high) { return low >= 0.0f && low < high && high <= 1.0f; };
    if (!valid_thresholds(config.cpu_igpu_decode_load_low, config.cpu_igpu_decode_load_high) ||
        !valid_thresholds(config.cpu_igpu_prefill_load_low, config.cpu_igpu_prefill_load_high)) {
      throw std::runtime_error("CPU/iGPU dynamic load thresholds must satisfy 0 <= low < high <= 1");
    }
    if (config.cpu_igpu_load_ewma_alpha <= 0.0f || config.cpu_igpu_load_ewma_alpha > 1.0f ||
        config.cpu_igpu_cost_ewma_alpha <= 0.0f || config.cpu_igpu_cost_ewma_alpha > 1.0f ||
        config.cpu_igpu_decode_switch_margin < 0.0f || config.cpu_igpu_decode_switch_margin >= 1.0f ||
        config.cpu_igpu_decode_cost_load_match_delta < 0.0f ||
        config.cpu_igpu_decode_cost_load_match_delta > 1.0f ||
        config.cpu_igpu_decode_load_reprobe_delta <= 0.0f ||
        config.cpu_igpu_decode_load_reprobe_delta > 1.0f ||
        config.cpu_igpu_load_sample_ms < 10 || config.cpu_igpu_decode_min_dwell <= 0 ||
        config.cpu_igpu_prefill_min_dwell <= 0 || config.cpu_igpu_decode_calibration_samples <= 0 ||
        config.cpu_igpu_decode_load_reprobe_grace < 0 || config.cpu_igpu_decode_reprobe_samples <= 0 ||
        config.cpu_igpu_decode_reprobe_interval < 0) {
      throw std::runtime_error("CPU/iGPU dynamic scheduler parameters are invalid");
    }
  }

  float choose_igpu_ratio(PolicyState& state, bool decode, float load) const {
    if (!config.cpu_igpu_dynamic) return configured_ratio(decode);
    ++state.calls_since_switch;
    const float low = decode ? config.cpu_igpu_decode_load_low : config.cpu_igpu_prefill_load_low;
    const float high = decode ? config.cpu_igpu_decode_load_high : config.cpu_igpu_prefill_load_high;
    const int min_dwell = decode ? config.cpu_igpu_decode_min_dwell : config.cpu_igpu_prefill_min_dwell;

    float desired = state.igpu_ratio;
    if (load <= low) {
      desired = 0.0f;
      state.high_load_epoch = false;
    } else {
      if (load >= high && !state.high_load_epoch) {
        state.high_load_epoch = true;
        state.has_cpu_sample = false;
        state.has_igpu_sample = false;
        state.cpu_samples = 0;
        state.igpu_samples = 0;
      }
      if (state.high_load_epoch) {
        if (decode) {
          // Decode has only a handful of routed rows, so short service-time
          // samples are dominated by scheduler quanta. The load hysteresis is
          // more stable and favors the iGPU once most bound CPUs are contested.
          desired = 1.0f;
        } else if (state.cpu_samples < 3) {
          desired = 0.0f;
        } else if (state.igpu_samples < 10) {
          // SYCL prefill needs several executions to leave its cold-start
          // regime. Do not compare against CPU service time before then.
          desired = 1.0f;
        } else {
          desired = state.cpu_ms_per_row < state.igpu_ms_per_row * 0.99f ? 0.0f : 1.0f;
        }
      }
    }

    if (desired != state.igpu_ratio && state.calls_since_switch >= min_dwell) {
      state.igpu_ratio = desired;
      state.calls_since_switch = 0;
      ++state.switch_count;
    }
    return state.igpu_ratio;
  }

  float configured_ratio(bool decode) const {
    const float phase_ratio = decode ? config.cpu_igpu_decode_ratio : config.cpu_igpu_prefill_ratio;
    return phase_ratio < 0.0f ? config.cpu_igpu_igpu_ratio : phase_ratio;
  }

  void record_service_cost(PolicyState& state, bool decode, bool cpu, float duration_ms, int rows, float load) const {
    if (config.cpu_igpu_dynamic && decode) {
      decode_scheduler_->record_service(cpu, duration_ms, rows, load);
      return;
    }
    if (rows <= 0) return;
    const float sample = duration_ms / rows;
    float& value = cpu ? state.cpu_ms_per_row : state.igpu_ms_per_row;
    bool& initialized = cpu ? state.has_cpu_sample : state.has_igpu_sample;
    float& sample_load = cpu ? state.cpu_sample_load : state.igpu_sample_load;
    int& sample_count = cpu ? state.cpu_samples : state.igpu_samples;
    const float alpha = config.cpu_igpu_load_ewma_alpha;
    value = initialized ? value + alpha * (sample - value) : sample;
    sample_load = initialized ? sample_load + alpha * (load - sample_load) : load;
    initialized = true;
    ++sample_count;
  }

  int count_active_rows(int qlen, int topk, const int64_t* expert_ids) const {
    int rows = 0;
    for (int item = 0; item < qlen * topk; ++item) {
      if (!config.should_skip_expert(expert_ids[item])) ++rows;
    }
    return rows;
  }

  static void bind_cpu_weight(const std::shared_ptr<typename CPUKernel::BufferB>& cpu_weight,
                              const std::shared_ptr<typename SYCLKernel::BufferB>& sycl_weight) {
    cpu_weight->bind_view(sycl_weight->qweight, sycl_weight->scales, sycl_weight->weight_sums);
  }

  void partition_experts(int qlen, int topk, const int64_t* expert_ids, std::vector<int64_t>& cpu_expert_ids,
                         std::vector<int64_t>& igpu_expert_ids) {
    std::vector<int> row_counts(config.expert_num, 0);
    for (int item = 0; item < qlen * topk; ++item) {
      const int64_t expert = expert_ids[item];
      if (!config.should_skip_expert(expert)) ++row_counts[expert];
    }

    std::vector<int> active_experts;
    int total_rows = 0;
    for (int expert = 0; expert < config.expert_num; ++expert) {
      if (row_counts[expert] > 0) {
        active_experts.push_back(expert);
        total_rows += row_counts[expert];
      }
    }
    std::sort(active_experts.begin(), active_experts.end(), [&](int left, int right) {
      if (row_counts[left] != row_counts[right]) return row_counts[left] > row_counts[right];
      return left < right;
    });

    std::fill(target_map_.begin(), target_map_.end(), 0);
    const int target_igpu_rows =
        static_cast<int>(std::lround(total_rows * current_igpu_ratio_.load(std::memory_order_relaxed)));
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
      const auto start = std::chrono::steady_clock::now();
      try {
        sycl_part_->forward(run.qlen, run.topk, run.expert_ids, run.weights, run.input, run.output);
      } catch (...) {
        error = std::current_exception();
      }

      {
        std::lock_guard<std::mutex> lock(igpu_mutex_);
        igpu_error_ = error;
        igpu_duration_ms_ = elapsed_ms(start);
        igpu_done_ = true;
      }
      igpu_cv_.notify_all();
    }
  }
};

#endif  // CPUINFER_OPERATOR_SYCL_GPTQ_INT4_CPU_IGPU_MOE_H
