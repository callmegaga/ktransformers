"""Standalone tests for the C++ CPU/iGPU service-cost controller."""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

OPERATORS_DIR = Path(__file__).resolve().parents[1] / "operators"


def test_service_cost_scheduler_state_machine(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the standalone scheduler test")

    source = tmp_path / "service_scheduler_test.cpp"
    executable = tmp_path / "service_scheduler_test"
    source.write_text(
        textwrap.dedent(r"""
            #include <cassert>
            #include <cmath>
            #include "cpu_igpu_service_scheduler.hpp"

            using cpu_igpu_scheduler::ServiceCostConfig;
            using cpu_igpu_scheduler::ServiceCostScheduler;

            static float run_round(ServiceCostScheduler& scheduler, float cpu_cost, float igpu_cost, float load) {
              float ratio = scheduler.begin_forward(0);
              for (int layer = 0; layer < 4; ++layer) {
                if (layer > 0) assert(scheduler.begin_forward(layer) == ratio);
                const bool cpu = ratio < 0.5f;
                scheduler.record_service(cpu, (cpu ? cpu_cost : igpu_cost) * 8.0f, 8, load);
              }
              return ratio;
            }

            static ServiceCostScheduler make_scheduler(float load_delta = 0.15f) {
              ServiceCostConfig config;
              config.ewma_alpha = 1.0f;
              config.switch_margin = 0.10f;
              config.cost_load_match_delta = 0.15f;
              config.load_reprobe_delta = load_delta;
              config.calibration_samples = 2;
              config.min_dwell = 1;
              config.load_reprobe_grace = 3;
              config.reprobe_samples = 3;
              config.reprobe_interval = 1000;
              return ServiceCostScheduler(config);
            }

            static void register_layers(ServiceCostScheduler& scheduler) {
              for (int layer = 3; layer >= 0; --layer) scheduler.register_layer(layer);
            }

            int main() {
              {
                auto scheduler = make_scheduler();
                register_layers(scheduler);
                assert(run_round(scheduler, 1.0f, 2.0f, 0.0f) == 0.0f);
                assert(run_round(scheduler, 1.0f, 2.0f, 0.0f) == 0.0f);
                assert(run_round(scheduler, 1.0f, 2.0f, 0.0f) == 1.0f);
                assert(run_round(scheduler, 1.0f, 2.0f, 0.0f) == 1.0f);
                assert(run_round(scheduler, 1.0f, 2.0f, 0.0f) == 0.0f);
                const auto snapshot = scheduler.snapshot();
                assert(!snapshot.exploring);
                assert(snapshot.switch_count == 2);

                // Active CPU service degradation is sufficient to select the
                // already-calibrated iGPU arm; load is not a device threshold.
                assert(run_round(scheduler, 4.0f, 2.0f, 0.5f) == 0.0f);
                assert(run_round(scheduler, 4.0f, 2.0f, 0.5f) == 1.0f);
              }

              {
                auto scheduler = make_scheduler();
                register_layers(scheduler);
                assert(run_round(scheduler, 4.0f, 1.0f, 0.8f) == 0.0f);
                assert(run_round(scheduler, 4.0f, 1.0f, 0.8f) == 0.0f);
                assert(run_round(scheduler, 4.0f, 1.0f, 0.8f) == 1.0f);
                assert(run_round(scheduler, 4.0f, 1.0f, 0.8f) == 1.0f);
                assert(run_round(scheduler, 4.0f, 1.0f, 0.8f) == 1.0f);

                // A same-arm load decrease triggers fresh CPU samples. The
                // new CPU cost wins after the configured probe window.
                assert(run_round(scheduler, 0.5f, 1.0f, 0.0f) == 1.0f);
                assert(run_round(scheduler, 0.5f, 1.0f, 0.0f) == 0.0f);
                assert(run_round(scheduler, 0.5f, 1.0f, 0.0f) == 0.0f);
                assert(scheduler.snapshot().exploring);
                assert(run_round(scheduler, 0.5f, 1.0f, 0.0f) == 0.0f);
                assert(run_round(scheduler, 0.5f, 1.0f, 0.0f) == 0.0f);
                const auto snapshot = scheduler.snapshot();
                assert(!snapshot.exploring);
                assert(snapshot.cpu_ms_per_row < snapshot.igpu_ms_per_row);
              }

              {
                auto scheduler = make_scheduler();
                register_layers(scheduler);
                assert(run_round(scheduler, 4.0f, 1.0f, 0.8f) == 0.0f);
                assert(run_round(scheduler, 4.0f, 1.0f, 0.8f) == 0.0f);
                assert(run_round(scheduler, 4.0f, 1.0f, 0.8f) == 1.0f);
                assert(run_round(scheduler, 4.0f, 1.0f, 0.8f) == 1.0f);

                // A large relative drop that remains above the absolute
                // probe ceiling is still high load and must not force CPU.
                assert(run_round(scheduler, 4.0f, 1.0f, 0.5f) == 1.0f);
                assert(run_round(scheduler, 4.0f, 1.0f, 0.5f) == 1.0f);
                assert(!scheduler.snapshot().exploring);
                assert(scheduler.snapshot().switch_count == 1);

                // Once the same-arm EWMA reaches genuinely low load, retain
                // the existing load-drop CPU reprobe behavior.
                assert(run_round(scheduler, 0.5f, 1.0f, 0.0f) == 1.0f);
                assert(run_round(scheduler, 0.5f, 1.0f, 0.0f) == 0.0f);
                const auto snapshot = scheduler.snapshot();
                assert(snapshot.exploring);
                assert(snapshot.reprobe_reason == 1);
                assert(std::abs(snapshot.igpu_reference_load - 0.8f) < 1e-5f);
              }

              {
                ServiceCostConfig config;
                config.ewma_alpha = 1.0f;
                config.calibration_samples = 1;
                config.min_dwell = 1;
                config.reprobe_samples = 1;
                config.reprobe_interval = 2;
                config.load_reprobe_delta = 0.25f;
                config.load_reprobe_max = 0.2f;
                ServiceCostScheduler scheduler(config);
                register_layers(scheduler);

                assert(run_round(scheduler, 4.0f, 1.0f, 0.8f) == 0.0f);
                assert(run_round(scheduler, 4.0f, 1.0f, 0.8f) == 1.0f);
                assert(run_round(scheduler, 4.0f, 1.0f, 0.8f) == 1.0f);
                assert(run_round(scheduler, 4.0f, 1.0f, 0.8f) == 0.0f);
                assert(scheduler.snapshot().exploring);
                assert(scheduler.snapshot().reprobe_reason == 2);
              }

              {
                auto scheduler = make_scheduler();
                register_layers(scheduler);
                assert(run_round(scheduler, 4.0f, 1.0f, 0.8f) == 0.0f);
                assert(run_round(scheduler, 4.0f, 1.0f, 0.8f) == 0.0f);
                assert(run_round(scheduler, 4.0f, 1.0f, 0.8f) == 1.0f);
                assert(run_round(scheduler, 4.0f, 1.0f, 0.8f) == 1.0f);
                assert(run_round(scheduler, 4.0f, 1.0f, 0.8f) == 1.0f);

                // A transient low reading at a prefill/decode boundary must
                // not trigger a CPU probe if the same-arm load recovers.
                scheduler.notify_phase_boundary(0);
                assert(run_round(scheduler, 0.5f, 1.0f, 0.0f) == 1.0f);
                assert(run_round(scheduler, 0.5f, 1.0f, 0.8f) == 1.0f);
                assert(run_round(scheduler, 0.5f, 1.0f, 0.8f) == 1.0f);
                assert(run_round(scheduler, 0.5f, 1.0f, 0.8f) == 1.0f);
                assert(!scheduler.snapshot().exploring);

                // A persistent drop remains visible after the grace period.
                scheduler.notify_phase_boundary(0);
                assert(run_round(scheduler, 0.5f, 1.0f, 0.0f) == 1.0f);
                assert(run_round(scheduler, 0.5f, 1.0f, 0.0f) == 1.0f);
                assert(run_round(scheduler, 0.5f, 1.0f, 0.0f) == 1.0f);
                assert(run_round(scheduler, 0.5f, 1.0f, 0.0f) == 0.0f);
                assert(scheduler.snapshot().exploring);
              }

              {
                auto scheduler = make_scheduler();
                register_layers(scheduler);
                assert(run_round(scheduler, 1.0f, 0.8f, 0.4f) == 0.0f);
                assert(run_round(scheduler, 1.0f, 0.8f, 0.4f) == 0.0f);
                assert(run_round(scheduler, 1.0f, 0.8f, 0.4f) == 1.0f);
                assert(run_round(scheduler, 1.0f, 0.8f, 0.4f) == 1.0f);

                // The inactive CPU estimate came from a materially lower
                // load. A slower active-iGPU sample must not select that stale
                // estimate without a load-triggered or periodic CPU reprobe.
                assert(run_round(scheduler, 1.0f, 1.2f, 0.8f) == 1.0f);
                assert(run_round(scheduler, 1.0f, 1.2f, 0.8f) == 1.0f);
                assert(scheduler.snapshot().switch_count == 1);
              }

              {
                ServiceCostConfig config;
                config.ewma_alpha = 0.20f;
                config.switch_margin = 0.10f;
                config.cost_load_match_delta = 0.10f;
                config.calibration_samples = 32;
                config.min_dwell = 1;
                config.load_reprobe_grace = 3;
                config.reprobe_samples = 3;
                config.reprobe_interval = 1000;
                ServiceCostScheduler scheduler(config);
                register_layers(scheduler);

                // Four cold rounds must not lock the controller onto iGPU.
                // The full initial window lets both EWMA estimates converge.
                for (int round = 0; round < 32; ++round) {
                  const float cpu_cost = round < 4 ? 2.0f : 0.04f;
                  assert(run_round(scheduler, cpu_cost, 0.09f, 0.0f) == 0.0f);
                }
                for (int round = 0; round < 32; ++round) {
                  const float igpu_cost = round < 4 ? 0.6f : 0.09f;
                  assert(run_round(scheduler, 0.04f, igpu_cost, 0.0f) == 1.0f);
                }
                assert(run_round(scheduler, 0.04f, 0.09f, 0.0f) == 0.0f);
                assert(!scheduler.snapshot().exploring);
              }

              {
                ServiceCostConfig config;
                int key = 0;
                auto first = cpu_igpu_scheduler::acquire(&key, config);
                auto second = cpu_igpu_scheduler::acquire(&key, config);
                assert(first == second);
              }
              return 0;
            }
            """),
        encoding="utf-8",
    )
    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pthread",
            "-I",
            str(OPERATORS_DIR),
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10.0, check=False)
    assert run_result.returncode == 0, run_result.stderr
