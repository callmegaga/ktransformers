#!/usr/bin/env python
"""Generate pinned or freely scheduled CPU pressure for inference benchmarks."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import multiprocessing as mp
import os
import queue
import signal
import time


def parse_cpu_list(value: str) -> list[int]:
    cpus: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid CPU range: {part}")
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    if not cpus:
        raise ValueError("CPU list must not be empty")
    available = os.sched_getaffinity(0)
    invalid = sorted(set(cpus) - available)
    if invalid:
        raise ValueError(f"CPUs are outside this process affinity: {invalid}")
    return cpus


def configure_worker(cpu: int | None, requested_nice: int) -> tuple[list[int], int]:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    os.setpriority(os.PRIO_PROCESS, 0, requested_nice)
    effective_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if effective_nice != requested_nice:
        raise RuntimeError(f"requested nice {requested_nice}, but effective nice is {effective_nice}")
    return sorted(os.sched_getaffinity(0)), effective_nice


def prepare_worker(cpu: int | None, requested_nice: int, ready: mp.Queue) -> tuple[list[int], int] | None:
    try:
        return configure_worker(cpu, requested_nice)
    except BaseException as error:
        ready.put(
            {
                "status": "error",
                "pid": os.getpid(),
                "assigned_cpu": cpu,
                "requested_nice": requested_nice,
                "error": repr(error),
            }
        )
        return None


def compute_worker(
    cpu: int | None,
    worker_index: int,
    requested_nice: int,
    stop: mp.synchronize.Event,
    ready: mp.Queue,
) -> None:
    prepared = prepare_worker(cpu, requested_nice, ready)
    if prepared is None:
        return
    allowed_cpus, effective_nice = prepared
    payload = bytes((index * 131 + worker_index) & 0xFF for index in range(16 * 1024))
    ready.put(
        {
            "status": "ready",
            "pid": os.getpid(),
            "assigned_cpu": cpu,
            "allowed_cpus": allowed_cpus,
            "requested_nice": requested_nice,
            "effective_nice": effective_nice,
        }
    )
    while not stop.is_set():
        for _ in range(256):
            hashlib.sha256(payload).digest()


def memory_worker(
    cpu: int | None,
    memory_mib: int,
    requested_nice: int,
    stop: mp.synchronize.Event,
    ready: mp.Queue,
) -> None:
    prepared = prepare_worker(cpu, requested_nice, ready)
    if prepared is None:
        return
    allowed_cpus, effective_nice = prepared
    size = memory_mib * 1024 * 1024
    source = bytearray(size)
    destination = bytearray(size)
    source_address = ctypes.addressof(ctypes.c_char.from_buffer(source))
    destination_address = ctypes.addressof(ctypes.c_char.from_buffer(destination))
    ready.put(
        {
            "status": "ready",
            "pid": os.getpid(),
            "assigned_cpu": cpu,
            "allowed_cpus": allowed_cpus,
            "requested_nice": requested_nice,
            "effective_nice": effective_nice,
        }
    )
    while not stop.is_set():
        ctypes.memmove(destination_address, source_address, size)
        ctypes.memmove(source_address, destination_address, size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("compute", "memory"), required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--affinity", choices=("pinned", "free"), default="pinned")
    parser.add_argument("--cpus", help="Pinned CPU list such as 0-7 or 0,2,4,6")
    parser.add_argument("--nice", type=int, default=0, help="Worker nice value (-20 is highest priority)")
    parser.add_argument("--memory-mib", type=int, default=64, help="Bytes per source/destination buffer per worker")
    parser.add_argument("--duration", type=float, default=0.0, help="Run time in seconds; zero means until signalled")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.memory_mib <= 0 or args.duration < 0:
        raise ValueError("memory-mib must be positive and duration must be non-negative")
    if not -20 <= args.nice <= 19:
        raise ValueError("nice must be between -20 and 19")
    if args.affinity == "pinned":
        if not args.cpus:
            raise ValueError("--cpus is required when --affinity=pinned")
        cpus = parse_cpu_list(args.cpus)
    else:
        cpus = sorted(os.sched_getaffinity(0))

    context = mp.get_context("fork")
    stop = context.Event()
    ready: mp.Queue = context.Queue()
    processes = []
    started = time.monotonic()

    terminate_requested = False

    def request_stop(_signum=None, _frame=None) -> None:
        nonlocal terminate_requested
        terminate_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    for worker_index in range(args.workers):
        cpu = cpus[worker_index % len(cpus)] if args.affinity == "pinned" else None
        target = compute_worker if args.kind == "compute" else memory_worker
        target_args = (
            (cpu, worker_index, args.nice, stop, ready)
            if args.kind == "compute"
            else (cpu, args.memory_mib, args.nice, stop, ready)
        )
        process = context.Process(target=target, args=target_args)
        process.start()
        processes.append(process)

    workers = []
    try:
        for _ in processes:
            workers.append(ready.get(timeout=30.0))
    except queue.Empty as error:
        stop.set()
        raise RuntimeError("background workers did not become ready within 30 seconds") from error

    worker_errors = [worker for worker in workers if worker.get("status") != "ready"]
    if worker_errors:
        stop.set()
        for process in processes:
            process.join(timeout=5.0)
        details = "; ".join(worker.get("error", "unknown error") for worker in worker_errors)
        raise RuntimeError(f"background worker setup failed: {details}")

    print(
        json.dumps(
            {
                "status": "ready",
                "parent_pid": os.getpid(),
                "kind": args.kind,
                "workers": args.workers,
                "affinity": args.affinity,
                "cpus": cpus,
                "requested_nice": args.nice,
                "effective_nice_values": sorted({worker["effective_nice"] for worker in workers}),
                "memory_mib_per_buffer": args.memory_mib if args.kind == "memory" else None,
                "worker_processes": workers,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    deadline = started + args.duration if args.duration else None
    while not terminate_requested:
        if deadline is not None and time.monotonic() >= deadline:
            break
        time.sleep(0.2)

    stop.set()

    for process in processes:
        process.join(timeout=5.0)
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()


if __name__ == "__main__":
    main()
