"""Standalone tests for the production CPU/iGPU load scheduler."""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

OPERATORS_DIR = Path(__file__).resolve().parents[1] / "operators"


def test_phase_mapping_hysteresis_and_coherent_prefill(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the standalone scheduler test")

    source = tmp_path / "load_scheduler_test.cpp"
    executable = tmp_path / "load_scheduler_test"
    source.write_text(
        textwrap.dedent(r"""
            #include <array>
            #include <cassert>
            #include "cpu_igpu_load_scheduler.hpp"

            using cpu_igpu_scheduler::LoadHysteresisState;
            using cpu_igpu_scheduler::CoherentPrefillScheduler;
            using cpu_igpu_scheduler::decode_from_qlen;
            using cpu_igpu_scheduler::update_load_hysteresis;

            static float initial_choice(float load, float low, float high) {
              LoadHysteresisState state;
              state.calls_since_switch = 2;
              return update_load_hysteresis(state, load, low, high, 2);
            }

            int main() {
              assert(decode_from_qlen(1));
              assert(!decode_from_qlen(2));
              assert(!decode_from_qlen(1024));

              // Production thresholds map representative normalized loads.
              const std::array<float, 6> loads = {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f};
              const std::array<float, 6> decode = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
              const std::array<float, 6> prefill = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f};
              for (int index = 0; index < 6; ++index) {
                assert(initial_choice(loads[index], 0.45f, 0.55f) == decode[index]);
                assert(initial_choice(loads[index], 0.65f, 0.75f) == prefill[index]);
              }

              LoadHysteresisState state;
              state.calls_since_switch = 3;
              assert(update_load_hysteresis(state, 0.60f, 0.45f, 0.55f, 3) == 1.0f);
              assert(update_load_hysteresis(state, 0.50f, 0.45f, 0.55f, 3) == 1.0f);
              assert(update_load_hysteresis(state, 0.40f, 0.45f, 0.55f, 3) == 1.0f);
              assert(update_load_hysteresis(state, 0.40f, 0.45f, 0.55f, 3) == 0.0f);

              CoherentPrefillScheduler coherent(0.65f, 0.75f, 1);
              coherent.register_layer(3);
              coherent.register_layer(1);
              coherent.register_layer(2);
              assert(coherent.begin_forward(1, 0.80f) == 1.0f);
              // Later layers reuse the leader's decision even if CPU execution
              // changes the sampled load during the same Prefill.
              assert(coherent.begin_forward(2, 0.20f) == 1.0f);
              assert(coherent.begin_forward(3, 0.20f) == 1.0f);
              assert(coherent.begin_forward(1, 0.60f) == 0.0f);
              assert(coherent.begin_forward(2, 0.90f) == 0.0f);
              return 0;
            }
            """),
        encoding="utf-8",
    )
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-I", str(OPERATORS_DIR), str(source), "-o", str(executable)],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
