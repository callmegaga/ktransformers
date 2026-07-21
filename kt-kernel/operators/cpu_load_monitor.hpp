/** Shared low-overhead CPU contention monitor for heterogeneous scheduling. */
#ifndef CPUINFER_OPERATOR_CPU_LOAD_MONITOR_H
#define CPUINFER_OPERATOR_CPU_LOAD_MONITOR_H

#include <algorithm>
#include <atomic>
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
#include <unordered_set>
#include <utility>
#include <vector>

class CPULoadMonitor {
  struct Snapshot {
    uint64_t total_ticks = 0;
    uint64_t idle_ticks = 0;
    uint64_t process_ticks = 0;
    uint64_t psi_some_us = 0;
    std::chrono::steady_clock::time_point timestamp;
    bool valid = false;
  };

 public:
  static std::shared_ptr<CPULoadMonitor> acquire(const std::vector<int>& cpu_ids, int interval_ms, float ewma_alpha) {
    static std::mutex instance_mutex;
    static std::weak_ptr<CPULoadMonitor> weak_instance;
    std::lock_guard<std::mutex> lock(instance_mutex);
    auto instance = weak_instance.lock();
    if (!instance) {
      instance = std::shared_ptr<CPULoadMonitor>(new CPULoadMonitor(cpu_ids, interval_ms, ewma_alpha));
      weak_instance = instance;
    }
    return instance;
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
  std::vector<int> cpu_ids_;
  std::unordered_set<int> cpu_id_set_;
  int interval_ms_;
  float ewma_alpha_;
  std::atomic<float> contention_{0.0f};
  std::thread worker_;
  std::mutex stop_mutex_;
  std::condition_variable stop_cv_;
  bool stop_ = false;

  CPULoadMonitor(std::vector<int> cpu_ids, int interval_ms, float ewma_alpha)
      : cpu_ids_(std::move(cpu_ids)),
        cpu_id_set_(cpu_ids_.begin(), cpu_ids_.end()),
        interval_ms_(std::max(10, interval_ms)),
        ewma_alpha_(std::clamp(ewma_alpha, 0.01f, 1.0f)) {
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

  static bool read_psi_some(uint64_t& total_us) {
    std::ifstream input("/proc/pressure/cpu");
    std::string line;
    while (std::getline(input, line)) {
      if (line.rfind("some ", 0) != 0) continue;
      const size_t position = line.find("total=");
      if (position == std::string::npos) return false;
      std::istringstream value(line.substr(position + 6));
      return static_cast<bool>(value >> total_us);
    }
    return false;
  }

  Snapshot read_snapshot() const {
    Snapshot snapshot;
    snapshot.timestamp = std::chrono::steady_clock::now();

    std::ifstream input("/proc/stat");
    std::string line;
    size_t matched_cpus = 0;
    while (std::getline(input, line)) {
      if (line.rfind("cpu", 0) != 0 || line.size() < 4 || line[3] == ' ') continue;
      std::istringstream fields(line);
      std::string cpu_name;
      fields >> cpu_name;
      int cpu_id = -1;
      try {
        cpu_id = std::stoi(cpu_name.substr(3));
      } catch (...) {
        continue;
      }
      if (!cpu_id_set_.contains(cpu_id)) continue;

      std::vector<uint64_t> values;
      uint64_t value = 0;
      while (fields >> value) values.push_back(value);
      if (values.size() < 4) continue;
      snapshot.total_ticks += std::accumulate(values.begin(), values.end(), uint64_t{0});
      snapshot.idle_ticks += values[3] + (values.size() > 4 ? values[4] : 0);
      ++matched_cpus;
    }

    snapshot.valid = matched_cpus == cpu_ids_.size() && read_process_ticks(snapshot.process_ticks) &&
                     read_psi_some(snapshot.psi_some_us);
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

      Snapshot current = read_snapshot();
      if (!previous.valid || !current.valid) {
        previous = current;
        continue;
      }

      const uint64_t total_delta = current.total_ticks - previous.total_ticks;
      const uint64_t idle_delta = current.idle_ticks - previous.idle_ticks;
      const uint64_t process_delta = current.process_ticks - previous.process_ticks;
      const auto wall_us =
          std::chrono::duration_cast<std::chrono::microseconds>(current.timestamp - previous.timestamp).count();
      if (total_delta > idle_delta && wall_us > 0) {
        const uint64_t system_busy_ticks = total_delta - idle_delta;
        const uint64_t external_busy_ticks = system_busy_ticks > process_delta ? system_busy_ticks - process_delta : 0;
        const float external_busy = std::clamp(static_cast<float>(external_busy_ticks) / total_delta, 0.0f, 1.0f);
        const uint64_t psi_delta = current.psi_some_us - previous.psi_some_us;
        const float psi_fraction = std::clamp(static_cast<float>(psi_delta) / wall_us, 0.0f, 1.0f);
        const float sample = std::max(external_busy, psi_fraction);
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
