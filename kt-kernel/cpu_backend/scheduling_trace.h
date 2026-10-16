// SPDX-License-Identifier: Apache-2.0
#pragma once

// Optional external diagnostics. The disabled path never reads a clock or
// wraps a task; the recorder is supplied by an explicit LD_PRELOAD experiment.
#include <cstdint>
#include <cstdlib>
#include <utility>
#ifdef __linux__
#include <dlfcn.h>
#include <sched.h>
#include <time.h>
#endif

namespace kt_scheduling_trace {
enum Kind : uint32_t { QueueEnqueue = 2, QueueRun = 3, QueueSync = 4, NumaSubmit = 5, NumaRun = 6 };
using RecordFn = void (*)(uint32_t, uint64_t, uintptr_t, uint64_t, uint64_t, int);
using NextIdFn = uint64_t (*)();
struct Recorder {
  RecordFn record = nullptr;
  NextIdFn next = nullptr;
};
inline const Recorder& recorder() {
  static const Recorder value = [] {
    Recorder result;
#ifdef __linux__
    const char* enabled = std::getenv("KT_SCHED_TRACE");
    if (enabled && enabled[0] == '1' && enabled[1] == '\0') {
      result.record = reinterpret_cast<RecordFn>(dlsym(RTLD_DEFAULT, "kt_sched_trace_event"));
      result.next = reinterpret_cast<NextIdFn>(dlsym(RTLD_DEFAULT, "kt_sched_trace_next_id"));
      if (!result.record || !result.next) result = {};
    }
#endif
    return result;
  }();
  return value;
}
inline uint64_t next_id() {
  const auto& r = recorder();
  return r.next ? r.next() : 0;
}
inline uint64_t now() {
#ifdef __linux__
  timespec time{};
  clock_gettime(CLOCK_REALTIME, &time);
  return uint64_t(time.tv_sec) * 1000000000 + time.tv_nsec;
#else
  return 0;
#endif
}
class Scope {
 public:
  Scope(Kind kind, const void* object, uint64_t id = 0)
      : fn_(recorder().record), kind_(kind), object_(reinterpret_cast<uintptr_t>(object)), id_(id) {
    if (fn_) {
#ifdef __linux__
      cpu_ = sched_getcpu();
#endif
      begin_ = now();
    }
  }
  ~Scope() {
    if (fn_) fn_(kind_, id_, object_, begin_, now(), cpu_);
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

 private:
  RecordFn fn_;
  Kind kind_;
  uintptr_t object_;
  uint64_t id_, begin_ = 0;
  int cpu_ = -1;
};
}  // namespace kt_scheduling_trace
