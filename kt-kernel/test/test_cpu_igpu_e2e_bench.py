"""Unit tests for the CPU-iGPU end-to-end experiment driver."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "bench" / "bench_cpu_igpu_e2e.py"
BACKGROUND_LOAD_SCRIPT = SCRIPT_PATH.with_name("cpu_background_load.py")
SPEC = importlib.util.spec_from_file_location("bench_cpu_igpu_e2e", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
e2e = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = e2e
SPEC.loader.exec_module(e2e)


def test_parse_experiment_matrix():
    assert e2e.parse_cpu_list("0-3,6") == [0, 1, 2, 3, 6]
    assert [load.label for load in e2e.parse_loads("none,compute:4,memory:2")] == [
        "none",
        "compute-4",
        "memory-2",
    ]
    assert [workload.label for workload in e2e.parse_workloads("1:8,1024x128")] == [
        "p1-o8",
        "p1024-o128",
    ]


def test_invalid_load_is_rejected():
    with pytest.raises(Exception, match="invalid load specification"):
        e2e.parse_loads("compute:0")


def test_request_schedule_is_deterministic():
    workloads = e2e.parse_workloads("1:8,128:16")
    first = e2e.make_request_schedule(workloads, repetitions=3, seed=42)
    second = e2e.make_request_schedule(workloads, repetitions=3, seed=42)
    assert first == second
    assert len(first) == 6


def test_summary_and_comparison_speedup_direction():
    samples = []
    for backend, decode_tps, ttft_ms in (
        ("vnni-only", 10.0, 100.0),
        ("vnni-sycl-dynamic", 20.0, 50.0),
    ):
        for repetition in range(3):
            samples.append(
                {
                    "status": "ok",
                    "backend": backend,
                    "load_affinity": "free",
                    "load_nice": -5,
                    "load": "compute-8",
                    "workload": "p1024-o128",
                    "prefill_tps": decode_tps,
                    "decode_tps": decode_tps,
                    "ttft_ms": ttft_ms,
                    "topt_ms": 1000.0 / decode_tps,
                    "e2e_ms": ttft_ms + 100.0,
                    "server_repetition": 0,
                    "repetition": repetition,
                    "output_sha256": (
                        f"hash-{repetition}" if backend == "vnni-only" or repetition < 2 else "different"
                    ),
                }
            )

    summary = e2e.summarize_samples(samples, bootstrap_samples=100, seed=7)
    comparisons = e2e.comparison_rows_with_samples(summary, samples, bootstrap_samples=100, seed=7)

    assert len(summary) == 2
    assert len(comparisons) == 1
    assert comparisons[0]["load_affinity"] == "free"
    assert comparisons[0]["load_nice"] == -5
    assert comparisons[0]["decode_tps_speedup"] == pytest.approx(2.0)
    assert comparisons[0]["ttft_ms_speedup"] == pytest.approx(2.0)
    assert comparisons[0]["decode_tps_speedup_ci95_low"] == pytest.approx(2.0)
    assert comparisons[0]["paired_n"] == 3
    assert comparisons[0]["output_match_count"] == 2
    assert comparisons[0]["output_match_rate"] == pytest.approx(2.0 / 3.0)


def test_sample_nonce_does_not_include_backend():
    nonce = e2e.sample_nonce(1, "compute-8", "p1024-o128", 2)
    assert nonce == "sample-1-compute-8-p1024-o128-2"
    assert "vnni" not in nonce


def test_markdown_report_contains_ci_speedup_and_output_agreement(tmp_path):
    summary = [
        {
            "backend": backend,
            "load": "none",
            "workload": "p1-o8",
            "n": 2,
            "decode_tps_mean": value,
            "decode_tps_ci95_low": value - 1,
            "decode_tps_ci95_high": value + 1,
        }
        for backend, value in (("vnni-only", 10.0), ("vnni-sycl-dynamic", 20.0))
    ]
    comparisons = [
        {
            "load": "none",
            "workload": "p1-o8",
            "decode_tps_speedup": 2.0,
            "decode_tps_speedup_ci95_low": 1.8,
            "decode_tps_speedup_ci95_high": 2.2,
            "output_match_count": 2,
            "output_pairs": 2,
            "output_match_rate": 1.0,
        }
    ]
    report_path = tmp_path / "report.md"
    e2e.write_markdown_report(
        report_path,
        {
            "status": "complete",
            "started_at": "start",
            "finished_at": "finish",
            "sample_count": 4,
            "successful_sample_count": 4,
        },
        summary,
        comparisons,
    )
    report = report_path.read_text(encoding="utf-8")
    assert "2.000x [1.800, 2.200]" in report
    assert "| pinned | 0 | none | p1-o8 | 2 | 2 | 1.0000 |" in report


def test_server_startup_failure_cleans_up(monkeypatch, tmp_path):
    class FakeProcess:
        pid = 1234
        stdout = io.BytesIO()

    fake_process = FakeProcess()
    terminated = []
    monkeypatch.setattr(e2e.subprocess, "Popen", lambda *args, **kwargs: fake_process)
    monkeypatch.setattr(
        e2e.ServerSession,
        "_wait_until_ready",
        lambda self: (_ for _ in ()).throw(RuntimeError("startup failed")),
    )
    monkeypatch.setattr(
        e2e,
        "terminate_process_group",
        lambda process, timeout: terminated.append((process, timeout)),
    )

    session = e2e.ServerSession(["server"], {}, tmp_path / "server.log", 30100, 1.0, False)
    with pytest.raises(RuntimeError, match="startup failed"):
        session.__enter__()

    assert terminated == [(fake_process, 20.0)]
    assert session.process is None
    assert session.log_file is None


def test_background_load_startup_failure_cleans_up(monkeypatch):
    class FakeProcess:
        stdout = io.StringIO("not-json\n")
        stderr = io.StringIO()

    fake_process = FakeProcess()
    terminated = []
    monkeypatch.setattr(e2e.subprocess, "Popen", lambda *args, **kwargs: fake_process)
    monkeypatch.setattr(e2e.select, "select", lambda *args, **kwargs: ([fake_process.stdout], [], []))
    monkeypatch.setattr(
        e2e,
        "terminate_process_group",
        lambda process, timeout: terminated.append((process, timeout)),
    )
    args = SimpleNamespace(load_affinity="pinned", load_nice=0, load_cpus=[0], memory_mib_per_buffer=1)
    controller = e2e.BackgroundLoad(args, e2e.LoadSpec("compute", 1))

    with pytest.raises(e2e.json.JSONDecodeError):
        controller.__enter__()

    assert terminated == [(fake_process, 10.0)]
    assert controller.process is None


def test_incomplete_paired_matrix_is_reported():
    args = SimpleNamespace(
        backends=list(e2e.BACKENDS.values()),
        loads=[e2e.LoadSpec("none", 0)],
        workloads=[e2e.WorkloadSpec(1, 8)],
        load_affinity="pinned",
        load_nice=0,
        server_repetitions=2,
        request_repetitions=1,
    )
    errors = e2e.experiment_completion_errors(
        args,
        {
            "server_runs": [{"name": "server", "status": "ok"}],
            "scenario_runs": [{"name": "scenario", "status": "ok"}],
        },
        [],
        [
            {
                "load_affinity": "pinned",
                "load": "none",
                "workload": "p1-o8",
                "paired_n": 1,
            }
        ],
    )
    assert errors == ["none/p1-o8: expected 2 paired samples, found 1"]


def test_free_load_affinity_observes_all_allowed_cpus(monkeypatch):
    monkeypatch.setattr(e2e.os, "sched_getaffinity", lambda _pid: {0, 2, 4, 6})
    free_args = SimpleNamespace(load_affinity="free", load_cpus=[0])
    pinned_args = SimpleNamespace(load_affinity="pinned", load_cpus=[0, 2])

    assert e2e.load_observation_cpus(free_args) == [0, 2, 4, 6]
    assert e2e.load_observation_cpus(pinned_args) == [0, 2]


def test_background_load_free_affinity_does_not_pass_cpu_binding(monkeypatch):
    metadata = {
        "status": "ready",
        "kind": "compute",
        "workers": 2,
        "affinity": "free",
        "cpus": [0, 1, 2, 3],
        "requested_nice": -5,
        "effective_nice_values": [-5],
    }

    class FakeProcess:
        stdout = io.StringIO(e2e.json.dumps(metadata) + "\n")
        stderr = io.StringIO()

    fake_process = FakeProcess()
    commands = []

    def fake_popen(command, **_kwargs):
        commands.append(command)
        return fake_process

    monkeypatch.setattr(e2e.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(e2e.select, "select", lambda *args, **kwargs: ([fake_process.stdout], [], []))
    monkeypatch.setattr(e2e, "terminate_process_group", lambda *args, **kwargs: None)
    monkeypatch.setattr(e2e.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})
    args = SimpleNamespace(
        load_affinity="free",
        load_nice=-5,
        load_cpus=[0, 1],
        memory_mib_per_buffer=1,
    )

    with e2e.BackgroundLoad(args, e2e.LoadSpec("compute", 2)) as actual:
        assert actual == metadata

    assert "--affinity" in commands[0]
    assert commands[0][commands[0].index("--affinity") + 1] == "free"
    assert commands[0][commands[0].index("--nice") + 1] == "-5"
    assert "--cpus" not in commands[0]


def test_summary_keeps_load_affinities_separate():
    samples = []
    for affinity in ("pinned", "free"):
        for backend in ("vnni-only", "vnni-sycl-dynamic"):
            samples.append(
                {
                    "status": "ok",
                    "backend": backend,
                    "load_affinity": affinity,
                    "load": "compute-8",
                    "workload": "p1-o8",
                    "decode_tps": 1.0,
                    "server_repetition": 0,
                    "repetition": 0,
                    "output_sha256": "same",
                }
            )

    summary = e2e.summarize_samples(samples, bootstrap_samples=10, seed=1)
    comparisons = e2e.comparison_rows_with_samples(summary, samples, bootstrap_samples=10, seed=1)

    assert len(summary) == 4
    assert {row["load_affinity"] for row in comparisons} == {"pinned", "free"}


def test_summary_keeps_load_priorities_separate():
    samples = []
    for load_nice in (0, -5):
        for backend in ("vnni-only", "vnni-sycl-dynamic"):
            samples.append(
                {
                    "status": "ok",
                    "backend": backend,
                    "load_affinity": "free",
                    "load_nice": load_nice,
                    "load": "compute-8",
                    "workload": "p1-o8",
                    "decode_tps": 1.0,
                    "server_repetition": 0,
                    "repetition": 0,
                    "output_sha256": "same",
                }
            )

    summary = e2e.summarize_samples(samples, bootstrap_samples=10, seed=1)
    comparisons = e2e.comparison_rows_with_samples(summary, samples, bootstrap_samples=10, seed=1)

    assert len(summary) == 4
    assert {row["load_nice"] for row in comparisons} == {0, -5}


def test_background_load_generator_applies_default_nice():
    result = subprocess.run(
        [
            sys.executable,
            str(BACKGROUND_LOAD_SCRIPT),
            "--kind",
            "compute",
            "--workers",
            "1",
            "--affinity",
            "free",
            "--nice",
            "0",
            "--duration",
            "0.05",
        ],
        capture_output=True,
        text=True,
        timeout=10.0,
        check=True,
    )
    metadata = json.loads(result.stdout.splitlines()[0])

    assert metadata["requested_nice"] == 0
    assert metadata["effective_nice_values"] == [0]
    assert all(worker["effective_nice"] == 0 for worker in metadata["worker_processes"])


def test_background_load_preflight_accepts_exact_nice(monkeypatch):
    metadata = {
        "status": "ready",
        "affinity": "free",
        "requested_nice": -5,
        "effective_nice_values": [-5],
    }
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(metadata) + "\n", stderr="")
    monkeypatch.setattr(e2e.subprocess, "run", lambda *args, **kwargs: completed)
    args = SimpleNamespace(
        loads=[e2e.LoadSpec("compute", 1)],
        load_affinity="free",
        load_nice=-5,
        load_cpus=[0],
        memory_mib_per_buffer=1,
    )

    assert e2e.preflight_background_load(args) == metadata


def test_background_load_preflight_rejects_wrong_effective_nice(monkeypatch):
    metadata = {
        "status": "ready",
        "affinity": "free",
        "requested_nice": -5,
        "effective_nice_values": [0],
    }
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(metadata) + "\n", stderr="")
    monkeypatch.setattr(e2e.subprocess, "run", lambda *args, **kwargs: completed)
    args = SimpleNamespace(
        loads=[e2e.LoadSpec("compute", 1)],
        load_affinity="free",
        load_nice=-5,
        load_cpus=[0],
        memory_mib_per_buffer=1,
    )

    with pytest.raises(RuntimeError, match="did not apply the requested nice value"):
        e2e.preflight_background_load(args)


def test_background_load_preflight_reports_permission_failure(monkeypatch):
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout="",
        stderr="PermissionError(13, 'Permission denied')\n",
    )
    monkeypatch.setattr(e2e.subprocess, "run", lambda *args, **kwargs: completed)
    args = SimpleNamespace(
        loads=[e2e.LoadSpec("compute", 1)],
        load_affinity="free",
        load_nice=-5,
        load_cpus=[0],
        memory_mib_per_buffer=1,
    )

    with pytest.raises(RuntimeError, match=r"preflight failed: PermissionError\(13"):
        e2e.preflight_background_load(args)


def test_background_load_rejects_wrong_effective_nice_and_cleans_up(monkeypatch):
    metadata = {
        "status": "ready",
        "kind": "compute",
        "workers": 1,
        "affinity": "free",
        "cpus": [0, 1],
        "requested_nice": -5,
        "effective_nice_values": [0],
    }

    class FakeProcess:
        stdout = io.StringIO(json.dumps(metadata) + "\n")
        stderr = io.StringIO()

    fake_process = FakeProcess()
    terminated = []
    monkeypatch.setattr(e2e.subprocess, "Popen", lambda *args, **kwargs: fake_process)
    monkeypatch.setattr(e2e.select, "select", lambda *args, **kwargs: ([fake_process.stdout], [], []))
    monkeypatch.setattr(
        e2e,
        "terminate_process_group",
        lambda process, timeout: terminated.append((process, timeout)),
    )
    monkeypatch.setattr(e2e.os, "sched_getaffinity", lambda _pid: {0, 1})
    args = SimpleNamespace(
        load_affinity="free",
        load_nice=-5,
        load_cpus=[0],
        memory_mib_per_buffer=1,
    )

    with pytest.raises(RuntimeError, match="did not apply the requested nice value"):
        with e2e.BackgroundLoad(args, e2e.LoadSpec("compute", 1)):
            pass

    assert terminated == [(fake_process, 10.0)]


def test_process_group_termination_does_not_compete_for_stdout(monkeypatch):
    class FakeProcess:
        pid = 4321

        def __init__(self):
            self.wait_timeouts = []

        def wait(self, timeout):
            self.wait_timeouts.append(timeout)
            return 0

        def communicate(self, timeout):
            raise AssertionError("communicate must not read the tee thread's stdout")

    process = FakeProcess()
    signals = []
    monkeypatch.setattr(
        e2e.os, "killpg", lambda process_group, sent_signal: signals.append((process_group, sent_signal))
    )

    e2e.terminate_process_group(process, timeout=7.0)

    assert process.wait_timeouts == [7.0]
    assert signals == [(4321, e2e.signal.SIGTERM)]


def test_json_compatible_serializes_matrix_types(tmp_path):
    value = {
        "path": tmp_path,
        "backend": e2e.BACKENDS["vnni-only"],
        "loads": e2e.parse_loads("none,compute:4"),
    }
    converted = e2e.json_compatible(value)
    assert converted["path"] == str(tmp_path)
    assert converted["backend"]["method"] == "GPTQ_INT4"
    assert converted["loads"][1] == {"kind": "compute", "workers": 4}
