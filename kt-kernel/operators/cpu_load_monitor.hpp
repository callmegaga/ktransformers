/** Shared low-overhead external CPU load monitor for heterogeneous scheduling. */
#ifndef CPUINFER_OPERATOR_CPU_LOAD_MONITOR_H
#define CPUINFER_OPERATOR_CPU_LOAD_MONITOR_H

#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace cpu_load_monitor_detail {

inline float external_busy_fraction(uint64_t total_delta, uint64_t idle_delta, uint64_t process_delta,
                                    uint64_t nice_delta, bool low_priority_process) {
  if (total_delta <= idle_delta) return 0.0f;
  const uint64_t busy_delta = total_delta - idle_delta;
  const uint64_t excluded_delta = low_priority_process ? nice_delta : process_delta;
  const uint64_t external_delta = busy_delta > excluded_delta ? busy_delta - excluded_delta : 0;
  return std::clamp(static_cast<float>(external_delta) / total_delta, 0.0f, 1.0f);
}

}  // namespace cpu_load_monitor_detail

class CPULoadMonitor {
  struct Snapshot {
    uint64_t total_ticks = 0;
    uint64_t idle_ticks = 0;
    uint64_t nice_ticks = 0;
    uint64_t process_ticks = 0;
    bool valid = false;
  };

 public:
  static std::shared_ptr<CPULoadMonitor> acquire(const void* worker_pool, int interval_ms, float ewma_alpha) {
    static std::mutex registry_mutex;
    static std::unordered_map<const void*, std::weak_ptr<CPULoadMonitor>> registry;
    std::lock_guard<std::mutex> lock(registry_mutex);
    if (auto existing = registry[worker_pool].lock()) return existing;
    auto monitor = std::shared_ptr<CPULoadMonitor>(new CPULoadMonitor(interval_ms, ewma_alpha));
    registry[worker_pool] = monitor;
    return monitor;
  }

  ~CPULoadMonitor() {
    {
      std::lock_guard<std::mutex> lock(stop_mutex_);
      stop_ = true;
    }
    stop_cv_.notify_one();
    if (worker_.joinable()) worker_.join();
  }

  float contention() const { return contention_.load(std::memory_order_relaxed); }

 private:
  const int interval_ms_;
  const float ewma_alpha_;
  std::atomic<float> contention_{0.0f};
  std::thread worker_;
  std::mutex stop_mutex_;
  std::condition_variable stop_cv_;
  bool stop_ = false;
  bool low_priority_process_ = false;

  CPULoadMonitor(int interval_ms, float ewma_alpha)
      : interval_ms_(std::max(10, interval_ms)), ewma_alpha_(std::clamp(ewma_alpha, 0.01f, 1.0f)) {
    errno = 0;
    const int process_nice = getpriority(PRIO_PROCESS, 0);
    low_priority_process_ = errno == 0 && process_nice > 0;
    worker_ = std::thread(&CPULoadMonitor::run, this);
  }

  static bool read_process_ticks(uint64_t& ticks) {
    std::ifstream input("/proc/self/stat");
    std::string line;
    if (!std::getline(input, line)) return false;
    const size_t command_end = line.rfind(')');
    if (command_end == std::string::npos || command_end + 2 >= line.size()) return false;

    std::istringstream fields(line.substr(command_end + 2));
    char state;
    uint64_t ignored = 0;
    uint64_t user_ticks = 0;
    uint64_t system_ticks = 0;
    if (!(fields >> state)) return false;
    for (int field = 4; field <= 13; ++field) {
      if (!(fields >> ignored)) return false;
    }
    if (!(fields >> user_ticks >> system_ticks)) return false;
    ticks = user_ticks + system_ticks;
    return true;
  }

  static Snapshot read_snapshot() {
    Snapshot snapshot;
    std::ifstream input("/proc/stat");
    std::string line;
    while (std::getline(input, line)) {
      if (line.rfind("cpu ", 0) != 0) continue;
      std::istringstream fields(line);
      std::string name;
      fields >> name;
      std::vector<uint64_t> values;
      uint64_t value = 0;
      while (fields >> value) values.push_back(value);
      if (values.size() < 4) break;
      snapshot.total_ticks = std::accumulate(values.begin(), values.end(), uint64_t{0});
      snapshot.idle_ticks = values[3] + (values.size() > 4 ? values[4] : 0);
      snapshot.nice_ticks = values[1];
      break;
    }
    snapshot.valid = snapshot.total_ticks > 0 && read_process_ticks(snapshot.process_ticks);
    return snapshot;
  }

  void run() {
    Snapshot previous = read_snapshot();
    bool initialized = false;
    while (true) {
      {
        std::unique_lock<std::mutex> lock(stop_mutex_);
        if (stop_cv_.wait_for(lock, std::chrono::milliseconds(interval_ms_), [&] { return stop_; })) return;
      }

      const Snapshot current = read_snapshot();
      if (!previous.valid || !current.valid || current.total_ticks < previous.total_ticks ||
          current.idle_ticks < previous.idle_ticks || current.nice_ticks < previous.nice_ticks ||
          current.process_ticks < previous.process_ticks) {
        previous = current;
        continue;
      }

      const uint64_t total_delta = current.total_ticks - previous.total_ticks;
      if (total_delta > 0) {
        const float sample = cpu_load_monitor_detail::external_busy_fraction(
            total_delta, current.idle_ticks - previous.idle_ticks, current.process_ticks - previous.process_ticks,
            current.nice_ticks - previous.nice_ticks, low_priority_process_);
        const float old_value = contention_.load(std::memory_order_relaxed);
        contention_.store(initialized ? old_value + ewma_alpha_ * (sample - old_value) : sample,
                          std::memory_order_relaxed);
        initialized = true;
      }
      previous = current;
    }
  }
};

#endif  // CPUINFER_OPERATOR_CPU_LOAD_MONITOR_H
