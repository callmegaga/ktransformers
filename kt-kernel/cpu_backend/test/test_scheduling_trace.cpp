// SPDX-License-Identifier: Apache-2.0
#include "../scheduling_trace.h"
#include "../task_queue.h"

#include <atomic>
// Debug helper targets may be built from a Release CMake tree. Do not compile
// their correctness assertions away under -DNDEBUG.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Event {
  uint32_t kind;
  uint64_t id;
  uintptr_t object;
  uint64_t begin, end;
};
std::mutex mutex;
std::vector<Event> events;
std::atomic<uint64_t> ids{1};
}  // namespace
extern "C" __attribute__((visibility("default"))) uint64_t kt_sched_trace_next_id() { return ids.fetch_add(1); }
extern "C" __attribute__((visibility("default"))) void kt_sched_trace_event(
    uint32_t kind, uint64_t id, uintptr_t object, uint64_t begin, uint64_t end, int) {
  std::lock_guard<std::mutex> lock(mutex);
  events.push_back({kind, id, object, begin, end});
}
int main(int argc, char** argv) {
  const bool enabled = argc > 1 && std::string(argv[1]) == "enabled";
  setenv("KT_SCHED_TRACE", enabled ? "1" : "0", 1);
  TaskQueue queue;
  std::atomic<int> count{0};
  for (int i = 0; i < 100; ++i) queue.enqueue([&] { count.fetch_add(1); });
  queue.sync(0);
  assert(count == 100);
  queue.enqueue([] { throw std::runtime_error("trace preserves errors"); });
  bool caught = false;
  try {
    queue.sync(0);
  } catch (const std::runtime_error& error) {
    caught = std::string(error.what()) == "trace preserves errors";
  }
  assert(caught);
  queue.enqueue([&] { count.fetch_add(1); });
  queue.sync(0);
  assert(count == 101);
  std::lock_guard<std::mutex> lock(mutex);
  if (!enabled) {
    assert(events.empty());
    assert(ids == 1);  // Disabled path must not call either external symbol.
    return 0;
  }
  std::map<uint64_t, Event> enqueues, runs;
  int syncs = 0;
  for (const auto& event : events) {
    assert(event.object == reinterpret_cast<uintptr_t>(&queue));
    assert(event.end >= event.begin);
    if (event.kind == kt_scheduling_trace::QueueEnqueue) {
      assert(enqueues.emplace(event.id, event).second);
    } else if (event.kind == kt_scheduling_trace::QueueRun) {
      assert(runs.emplace(event.id, event).second);
    } else if (event.kind == kt_scheduling_trace::QueueSync) {
      ++syncs;
    } else {
      assert(false);
    }
  }
  assert(enqueues.size() == 102 && runs.size() == 102 && syncs == 3);
  for (const auto& [id, event] : runs) {
    assert(enqueues.count(id));
    assert(event.begin >= enqueues.at(id).begin);
  }
}
