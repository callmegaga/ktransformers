/**
 * @Description :
 * @Author    : chenht2022
 * @Date     : 2024-07-17 12:25:51
 * @Version   : 1.0.0
 * @LastEditors : chenht2022
 * @LastEditTime : 2024-10-09 11:08:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "task_queue.h"
#include "scheduling_trace.h"

#include <pthread.h>
#include <sched.h>

#include <chrono>
#include <iostream>
#include <thread>

TaskQueue::TaskQueue() : done(false), pending(0) {
  Node* dummy = new Node();
  head.store(dummy, std::memory_order_relaxed);
  tail.store(dummy, std::memory_order_relaxed);
  workerThread = std::thread(&TaskQueue::worker, this);
#ifdef __linux__
  pthread_setname_np(workerThread.native_handle(), "kt_task_queue");
#endif
}

TaskQueue::~TaskQueue() {
  {
    std::lock_guard<std::mutex> lock(mtx);
    done.store(true, std::memory_order_release);
  }
  cv.notify_all();
  if (workerThread.joinable()) workerThread.join();

  Node* node = head.load(std::memory_order_relaxed);
  while (node) {
    Node* next = node->next.load(std::memory_order_relaxed);
    delete node;
    node = next;
  }
}

void TaskQueue::enqueue(std::function<void()> task) {
  const auto trace_id = kt_scheduling_trace::next_id();
  kt_scheduling_trace::Scope trace(kt_scheduling_trace::QueueEnqueue, this, trace_id);
  pending.fetch_add(1, std::memory_order_acq_rel);
  Node* node;
  if (trace_id) {
    node = new Node([task = std::move(task), trace_id, this] {
      kt_scheduling_trace::Scope trace(kt_scheduling_trace::QueueRun, this, trace_id);
      task();
    });
  } else {
    node = new Node(task);
  }
  Node* prev = tail.exchange(node, std::memory_order_acq_rel);
  prev->next.store(node, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(mtx);
  }
  cv.notify_one();
}

void TaskQueue::sync(size_t allow_n_pending) {
  kt_scheduling_trace::Scope trace(kt_scheduling_trace::QueueSync, this);
  std::exception_ptr task_exception;
  {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&] {
      return pending.load(std::memory_order_acquire) <= allow_n_pending
          || done.load(std::memory_order_acquire);
    });
    task_exception = first_exception;
    first_exception = nullptr;
  }
  if (task_exception) std::rethrow_exception(task_exception);
}

void TaskQueue::worker() {
  Node* curr = head.load(std::memory_order_relaxed);
  while (!done.load(std::memory_order_acquire)) {
    Node* next = curr->next.load(std::memory_order_acquire);
    if (next) {
      std::exception_ptr task_exception;
      if (next->task) {
        try {
          next->task();
        } catch (...) {
          task_exception = std::current_exception();
        }
      }
      delete curr;
      curr = next;
      head.store(curr, std::memory_order_release);
      {
        std::lock_guard<std::mutex> lock(mtx);
        if (task_exception && !first_exception) {
          first_exception = task_exception;
        }
        pending.fetch_sub(1, std::memory_order_acq_rel);
      }
      cv.notify_all();
    } else {
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, [&] {
        return curr->next.load(std::memory_order_acquire) != nullptr
            || done.load(std::memory_order_acquire);
      });
    }
  }
}

// Feature probe for the external diagnostic collector, not a dispatch profile.
extern "C" __attribute__((visibility("default"))) int kt_cpuinfer_scheduling_revision() { return 1; }
