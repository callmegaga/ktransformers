/** Low-overhead load-hysteresis policy for CPU/iGPU scheduling. */
#ifndef CPUINFER_OPERATOR_CPU_IGPU_LOAD_SCHEDULER_H
#define CPUINFER_OPERATOR_CPU_IGPU_LOAD_SCHEDULER_H

#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace cpu_igpu_scheduler {

struct LoadHysteresisState {
  float igpu_ratio = 0.0f;
  int calls_since_switch = 0;
};

inline bool decode_from_qlen(int qlen) { return qlen == 1; }

inline float update_load_hysteresis(LoadHysteresisState& state, float load, float low, float high, int min_dwell) {
  if (state.calls_since_switch < min_dwell) ++state.calls_since_switch;
  float desired = state.igpu_ratio;
  load = std::clamp(load, 0.0f, 1.0f);
  if (load <= low) {
    desired = 0.0f;
  } else if (load >= high) {
    desired = 1.0f;
  }

  if (desired != state.igpu_ratio && state.calls_since_switch >= min_dwell) {
    state.igpu_ratio = desired;
    state.calls_since_switch = 0;
  }
  return state.igpu_ratio;
}

// Prefill decisions are shared by every MoE layer that uses the same worker
// pool. Only the lowest registered layer samples load for a forward group;
// later layers reuse that decision and cannot react to load changed by earlier
// layers in the same group.
class CoherentPrefillScheduler {
 public:
  CoherentPrefillScheduler(float low, float high, int min_dwell) : low_(low), high_(high), min_dwell_(min_dwell) {
    if (low_ < 0.0f || low_ >= high_ || high_ > 1.0f || min_dwell_ <= 0) {
      throw std::invalid_argument("invalid coherent Prefill scheduler configuration");
    }
    state_.calls_since_switch = min_dwell_;
  }

  void register_layer(int layer_idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    leader_layer_ = std::min(leader_layer_, layer_idx);
  }

  float begin_forward(int layer_idx, float load) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer_idx == leader_layer_) {
      update_load_hysteresis(state_, load, low_, high_, min_dwell_);
    }
    return state_.igpu_ratio;
  }

 private:
  const float low_;
  const float high_;
  const int min_dwell_;
  std::mutex mutex_;
  LoadHysteresisState state_;
  int leader_layer_ = std::numeric_limits<int>::max();
};

inline std::shared_ptr<CoherentPrefillScheduler> acquire_prefill_scheduler(const void* worker_pool, float low,
                                                                           float high, int min_dwell) {
  static std::mutex registry_mutex;
  static std::unordered_map<const void*, std::weak_ptr<CoherentPrefillScheduler>> registry;
  std::lock_guard<std::mutex> lock(registry_mutex);
  if (auto existing = registry[worker_pool].lock()) return existing;
  auto scheduler = std::make_shared<CoherentPrefillScheduler>(low, high, min_dwell);
  registry[worker_pool] = scheduler;
  return scheduler;
}

}  // namespace cpu_igpu_scheduler

#endif  // CPUINFER_OPERATOR_CPU_IGPU_LOAD_SCHEDULER_H
