/** Phase-level service-cost scheduler shared by CPU/iGPU MoE layers. */
#ifndef CPUINFER_OPERATOR_CPU_IGPU_SERVICE_SCHEDULER_HPP
#define CPUINFER_OPERATOR_CPU_IGPU_SERVICE_SCHEDULER_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace cpu_igpu_scheduler {

struct ServiceCostConfig {
  float ewma_alpha = 0.20f;
  float switch_margin = 0.10f;
  float cost_load_match_delta = 0.10f;
  float load_reprobe_delta = 0.25f;
  float load_reprobe_max = 0.20f;
  int calibration_samples = 32;
  int min_dwell = 4;
  int load_reprobe_grace = 64;
  int reprobe_samples = 32;
  int reprobe_interval = 4096;
};

struct ServiceCostSnapshot {
  float igpu_ratio = 0.0f;
  float cpu_ms_per_row = 0.0f;
  float igpu_ms_per_row = 0.0f;
  float cpu_sample_load = 0.0f;
  float igpu_sample_load = 0.0f;
  int cpu_samples = 0;
  int igpu_samples = 0;
  int switch_count = 0;
  bool exploring = true;
  float igpu_reference_load = 0.0f;
  int reprobe_reason = 0;
};

class ServiceCostScheduler {
 public:
  explicit ServiceCostScheduler(ServiceCostConfig config) : config_(config) {
    state_.calls_since_switch = config_.min_dwell;
  }

  void register_layer(int layer_idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    leader_layer_ = std::min(leader_layer_, layer_idx);
  }

  void notify_phase_boundary(int layer_idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer_idx != leader_layer_) return;
    finalize_round();
    state_.load_reprobe_grace_remaining = config_.load_reprobe_grace;
  }

  float begin_forward(int layer_idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer_idx != leader_layer_) return state_.igpu_ratio;

    finalize_round();
    choose_ratio();
    return state_.igpu_ratio;
  }

  void record_service(bool cpu, float duration_ms, int rows, float load) {
    if (rows <= 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    RoundAccumulator& round = state_.round;
    if (cpu) {
      round.cpu_duration_ms += duration_ms;
      round.cpu_rows += rows;
    } else {
      round.igpu_duration_ms += duration_ms;
      round.igpu_rows += rows;
    }
    round.load_sum += load;
    ++round.load_samples;
  }

  ServiceCostSnapshot snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {state_.igpu_ratio,
            state_.cpu.cost,
            state_.igpu.cost,
            state_.cpu.load,
            state_.igpu.load,
            state_.cpu.samples,
            state_.igpu.samples,
            state_.switch_count,
            is_exploring(),
            state_.igpu_reference_load,
            static_cast<int>(state_.reprobe_reason)};
  }

 private:
  struct ArmEstimate {
    float cost = 0.0f;
    float load = 0.0f;
    bool initialized = false;
    int samples = 0;
    int last_sample_round = 0;
  };

  struct RoundAccumulator {
    float cpu_duration_ms = 0.0f;
    float igpu_duration_ms = 0.0f;
    float load_sum = 0.0f;
    int cpu_rows = 0;
    int igpu_rows = 0;
    int load_samples = 0;

    void clear() { *this = {}; }
  };

  struct State {
    float igpu_ratio = 0.0f;
    ArmEstimate cpu;
    ArmEstimate igpu;
    RoundAccumulator round;
    int completed_rounds = 0;
    int calls_since_switch = 0;
    int switch_count = 0;
    bool probing_cpu = false;
    bool igpu_reference_load_valid = false;
    float igpu_reference_load = 0.0f;
    int load_reprobe_grace_remaining = 0;
    enum class ReprobeReason { kNone = 0, kLoadDrop = 1, kPeriodic = 2 } reprobe_reason = ReprobeReason::kNone;
  };

  ServiceCostConfig config_;
  mutable std::mutex mutex_;
  int leader_layer_ = std::numeric_limits<int>::max();
  State state_;

  void finalize_round() {
    RoundAccumulator& round = state_.round;
    if (round.cpu_rows == 0 && round.igpu_rows == 0) return;

    ++state_.completed_rounds;
    ++state_.calls_since_switch;
    const float load = round.load_samples > 0 ? round.load_sum / round.load_samples : 0.0f;
    if (round.cpu_rows > 0) {
      update_arm(state_.cpu, round.cpu_duration_ms / round.cpu_rows, load);
    }
    if (round.igpu_rows > 0) {
      update_arm(state_.igpu, round.igpu_duration_ms / round.igpu_rows, load);
      if (!state_.igpu_reference_load_valid) {
        state_.igpu_reference_load = state_.igpu.load;
        state_.igpu_reference_load_valid = true;
      } else if (!state_.probing_cpu) {
        state_.igpu_reference_load = std::max(state_.igpu_reference_load, state_.igpu.load);
      }
    }
    if (state_.load_reprobe_grace_remaining > 0) --state_.load_reprobe_grace_remaining;
    round.clear();
  }

  void update_arm(ArmEstimate& arm, float sample, float load) {
    if (arm.initialized) {
      arm.cost += config_.ewma_alpha * (sample - arm.cost);
      arm.load += config_.ewma_alpha * (load - arm.load);
    } else {
      arm.cost = sample;
      arm.load = load;
      arm.initialized = true;
    }
    ++arm.samples;
    arm.last_sample_round = state_.completed_rounds;
  }

  void choose_ratio() {
    if (state_.probing_cpu) {
      if (state_.cpu.samples < config_.reprobe_samples) {
        set_ratio(0.0f);
        return;
      }
      state_.probing_cpu = false;
      state_.reprobe_reason = State::ReprobeReason::kNone;
    } else if (state_.cpu.samples < config_.calibration_samples) {
      set_ratio(0.0f);
      return;
    } else if (state_.igpu.samples < config_.calibration_samples) {
      set_ratio(1.0f);
      return;
    } else if (state_.igpu_ratio >= 0.5f) {
      const auto reason = cpu_reprobe_reason();
      if (reason != State::ReprobeReason::kNone) {
        state_.reprobe_reason = reason;
        state_.cpu = {};
        state_.cpu.last_sample_round = state_.completed_rounds;
        state_.probing_cpu = true;
        state_.igpu_reference_load_valid = false;
        set_ratio(0.0f);
        return;
      }
    }

    float desired = state_.igpu_ratio;
    if (state_.cpu.cost * (1.0f + config_.switch_margin) < state_.igpu.cost) {
      if (state_.igpu_ratio < 0.5f || arm_loads_comparable()) desired = 0.0f;
    } else if (state_.igpu.cost * (1.0f + config_.switch_margin) < state_.cpu.cost) {
      desired = 1.0f;
    }
    set_ratio(desired);
  }

  State::ReprobeReason cpu_reprobe_reason() const {
    const bool stale = config_.reprobe_interval > 0 &&
                       state_.completed_rounds - state_.cpu.last_sample_round >= config_.reprobe_interval;
    const bool load_dropped = state_.load_reprobe_grace_remaining == 0 && state_.igpu_reference_load_valid &&
                              state_.igpu.load <= config_.load_reprobe_max &&
                              state_.igpu_reference_load - state_.igpu.load >= config_.load_reprobe_delta;
    if (stale) return State::ReprobeReason::kPeriodic;
    if (load_dropped) return State::ReprobeReason::kLoadDrop;
    return State::ReprobeReason::kNone;
  }

  bool arm_loads_comparable() const {
    return std::abs(state_.cpu.load - state_.igpu.load) <= config_.cost_load_match_delta;
  }

  void set_ratio(float desired) {
    if (desired == state_.igpu_ratio || state_.calls_since_switch < config_.min_dwell) return;
    state_.igpu_ratio = desired;
    state_.calls_since_switch = 0;
    ++state_.switch_count;
  }

  bool is_exploring() const {
    return state_.probing_cpu || state_.cpu.samples < config_.calibration_samples ||
           state_.igpu.samples < config_.calibration_samples;
  }
};

inline std::shared_ptr<ServiceCostScheduler> acquire(const void* key, const ServiceCostConfig& config) {
  static std::mutex registry_mutex;
  static std::unordered_map<const void*, std::weak_ptr<ServiceCostScheduler>> registry;
  std::lock_guard<std::mutex> lock(registry_mutex);
  if (auto existing = registry[key].lock()) return existing;
  auto scheduler = std::make_shared<ServiceCostScheduler>(config);
  registry[key] = scheduler;
  return scheduler;
}

}  // namespace cpu_igpu_scheduler

#endif  // CPUINFER_OPERATOR_CPU_IGPU_SERVICE_SCHEDULER_HPP
