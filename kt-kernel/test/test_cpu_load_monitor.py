"""Standalone tests for external CPU busy accounting."""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

OPERATORS_DIR = Path(__file__).resolve().parents[1] / "operators"


def test_external_busy_excludes_the_inference_engine(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the standalone load-monitor test")

    source = tmp_path / "cpu_load_monitor_test.cpp"
    executable = tmp_path / "cpu_load_monitor_test"
    source.write_text(
        textwrap.dedent(r"""
            #include <cassert>
            #include <cmath>
            #include "cpu_load_monitor.hpp"

            int main() {
              using cpu_load_monitor_detail::external_busy_fraction;
              // A normal-priority engine is excluded using its process ticks.
              assert(std::abs(external_busy_fraction(100, 20, 30, 0, false) - 0.50f) < 1e-6f);
              // A low-priority engine is part of the global nice bucket, so all
              // nice ticks are excluded from the external normal-priority load.
              assert(std::abs(external_busy_fraction(100, 20, 30, 20, true) - 0.60f) < 1e-6f);
              assert(external_busy_fraction(100, 100, 0, 0, false) == 0.0f);
              assert(external_busy_fraction(100, 20, 90, 0, false) == 0.0f);
              return 0;
            }
            """),
        encoding="utf-8",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-pthread",
            "-I",
            str(OPERATORS_DIR),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
