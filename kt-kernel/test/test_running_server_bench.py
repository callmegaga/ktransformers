"""Unit tests for the running-server benchmark client."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "bench" / "bench_running_server.py"
SPEC = importlib.util.spec_from_file_location("bench_running_server", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


def test_parse_workloads():
    workloads = bench.parse_workloads("1:300,1024x128,1:300")
    assert [workload.label for workload in workloads] == ["p1-o300", "p1024-o128"]


def test_make_prompt_is_deterministic_and_varies_short_requests():
    assert bench.make_prompt(1024, "sample-1") == bench.make_prompt(1024, "sample-1")
    prompts = {bench.make_prompt(1, f"sample-20260718-p1-o300-{repetition}") for repetition in range(5)}
    assert len(prompts) == 5


def test_extract_usage_accepts_nested_sglang_metadata():
    assert bench.extract_usage({"choices": [{"meta_info": {"input_tokens": 12, "output_tokens": 7}}]}) == (12, 7)


def test_streaming_metrics_use_first_to_last_token_intervals(monkeypatch):
    events = [
        b'data: {"choices": [{"text": "a"}], "usage": {"prompt_tokens": 100, "completion_tokens": 1}}',
        b'data: {"choices": [{"text": "b"}], "usage": {"prompt_tokens": 100, "completion_tokens": 2}}',
        b'data: {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 2}}',
        b"data: [DONE]",
    ]

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        def iter_lines(self):
            return iter(events)

    monkeypatch.setattr(bench.requests, "post", lambda *args, **kwargs: FakeResponse())
    times = iter((10.0, 10.1, 10.4, 10.45))
    output_tokens = []
    result = bench.run_streaming_request(
        "http://localhost:30100",
        {},
        "model",
        "prompt",
        100,
        2,
        1,
        30.0,
        False,
        False,
        clock=lambda: next(times),
        on_output_token=output_tokens.append,
    )

    assert result["usage_source"] == "server"
    assert result["ttft_ms"] == pytest.approx(100.0)
    assert result["output_phase_ms"] == pytest.approx(300.0)
    assert result["ttlt_ms"] == pytest.approx(400.0)
    assert result["e2e_ms"] == pytest.approx(450.0)
    assert result["prefill_tps"] == pytest.approx(1000.0)
    assert result["decode_tps"] == pytest.approx(1.0 / 0.3)
    assert result["tpot_ms"] == pytest.approx(300.0)
    assert result["stream_tokens"] == 2
    assert output_tokens == [1, 2]


def test_streaming_token_progress_includes_empty_and_batched_text_events(monkeypatch):
    events = [
        b'data: {"choices": [{"text": "a"}], "usage": {"prompt_tokens": 10, "completion_tokens": 1}}',
        b'data: {"choices": [{"text": ""}], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}',
        b'data: {"choices": [{"text": "bc"}], "usage": {"prompt_tokens": 10, "completion_tokens": 4}}',
        b'data: {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4}}',
        b"data: [DONE]",
    ]

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        def iter_lines(self):
            return iter(events)

    payload = {}

    def fake_post(*_args, **kwargs):
        payload.update(kwargs["json"])
        return FakeResponse()

    monkeypatch.setattr(bench.requests, "post", fake_post)
    times = iter((1.0, 1.1, 1.4, 1.5))
    output_tokens = []

    result = bench.run_streaming_request(
        "http://localhost:30100",
        {},
        "model",
        "prompt",
        10,
        4,
        1,
        30.0,
        False,
        False,
        clock=lambda: next(times),
        on_output_token=output_tokens.append,
    )

    assert payload["stream_options"]["continuous_usage_stats"] is True
    assert result["stream_chunks"] == 2
    assert result["stream_tokens"] == 4
    assert output_tokens == [1, 2, 3, 4]


def test_streaming_request_requires_server_token_usage(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        def iter_lines(self):
            return iter(
                [
                    b'data: {"choices": [{"text": "a"}]}',
                    b"data: [DONE]",
                ]
            )

    monkeypatch.setattr(bench.requests, "post", lambda *args, **kwargs: FakeResponse())
    times = iter((1.0, 1.1, 1.2))
    with pytest.raises(RuntimeError, match="omitted prompt/completion token usage"):
        bench.run_streaming_request(
            "http://localhost:30100",
            {},
            "model",
            "prompt",
            10,
            1,
            1,
            30.0,
            False,
            False,
            clock=lambda: next(times),
        )


def test_summary_uses_only_successful_samples():
    samples = [
        {
            "status": "ok",
            "workload": "p1-o2",
            "target_prompt_tokens": 1,
            "requested_output_tokens": 2,
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "prefill_tps": 10.0,
            "decode_tps": 5.0,
            "ttft_ms": 100.0,
            "tpot_ms": 200.0,
            "output_phase_ms": 200.0,
            "ttlt_ms": 300.0,
            "e2e_ms": 310.0,
        },
        {
            "status": "error",
            "workload": "p1-o2",
            "target_prompt_tokens": 1,
            "requested_output_tokens": 2,
        },
    ]

    summary = bench.summarize_samples(samples, bootstrap_samples=10, seed=1)

    assert len(summary) == 1
    assert summary[0]["n"] == 1
    assert summary[0]["error_count"] == 1
    assert summary[0]["decode_tps_mean"] == pytest.approx(5.0)


def test_cpu_utilization_delta_separates_user_and_nice_ticks():
    before = {0: (100, 50, 30, 10, 10)}
    after = {0: (200, 70, 60, 40, 20)}

    result = bench.cpu_utilization_delta(before, after)

    assert result["cpu_busy_fraction"] == pytest.approx(0.8)
    assert result["cpu_user_fraction"] == pytest.approx(0.3)
    assert result["cpu_nice_fraction"] == pytest.approx(0.3)
    assert result["cpu_system_fraction"] == pytest.approx(0.1)
    assert result["cpu_busy_by_cpu"] == {"0": pytest.approx(0.8)}


def test_scheduler_event_summary_is_split_by_phase():
    events = [
        {
            "phase": "prefill",
            "igpu_ratio": 0.0,
            "cpu_load": 0.1,
            "switch_count": 1,
            "high_load_epoch": False,
            "exploration": False,
        },
        {
            "phase": "decode",
            "igpu_ratio": 1.0,
            "cpu_load": 0.8,
            "switch_count": 2,
            "high_load_epoch": True,
            "exploration": True,
        },
        {
            "phase": "decode",
            "igpu_ratio": 0.5,
            "cpu_load": 0.6,
            "switch_count": 3,
            "high_load_epoch": True,
            "exploration": False,
        },
    ]

    result = bench.summarize_scheduler_events(events)

    assert result["scheduler_event_count"] == 3
    assert result["scheduler_prefill_igpu_ratio"] == 0.0
    assert result["scheduler_decode_igpu_ratio"] == pytest.approx(0.75)
    assert result["scheduler_decode_cpu_load"] == pytest.approx(0.7)
    assert result["scheduler_decode_switch_count_delta"] == 1
    assert result["scheduler_decode_high_load_fraction"] == 1.0
    assert result["scheduler_decode_exploration_fraction"] == 0.5


def test_scheduler_event_summary_ignores_no_execution_and_weights_calls():
    events = [
        {
            "phase": "prefill",
            "igpu_ratio": 1.0,
            "execution_calls_delta": 0,
            "cpu_load": 0.4,
            "switch_count": 0,
        },
        {
            "phase": "prefill",
            "igpu_ratio": 0.0,
            "execution_calls_delta": 2,
            "cpu_load": 0.8,
            "switch_count": 0,
        },
        {
            "phase": "decode",
            "igpu_ratio": 0.0,
            "execution_calls_delta": 1,
            "cpu_load": 0.8,
            "switch_count": 1,
        },
        {
            "phase": "decode",
            "igpu_ratio": 1.0,
            "execution_calls_delta": 3,
            "cpu_load": 0.8,
            "switch_count": 1,
        },
    ]

    result = bench.summarize_scheduler_events(events)

    assert result["scheduler_prefill_execution_calls"] == 2
    assert result["scheduler_prefill_igpu_ratio"] == 0.0
    assert result["scheduler_decode_execution_calls"] == 4
    assert result["scheduler_decode_igpu_ratio"] == pytest.approx(0.75)


def test_scheduler_transition_summary_reports_probe_and_settle_delay():
    events = [
        {
            "phase": "decode",
            "monotonic_ns": 90_000_000,
            "execution_calls_delta": 10,
            "igpu_ratio": 1.0,
            "policy_igpu_ratio": 1.0,
            "exploration": False,
            "sequence": 1,
        },
        {
            "phase": "decode",
            "monotonic_ns": 110_000_000,
            "execution_calls_delta": 2,
            "igpu_ratio": 1.0,
            "policy_igpu_ratio": 1.0,
            "exploration": False,
            "sequence": 2,
        },
        {
            "phase": "decode",
            "monotonic_ns": 120_000_000,
            "execution_calls_delta": 1,
            "igpu_ratio": 0.0,
            "policy_igpu_ratio": 0.0,
            "exploration": True,
            "sequence": 3,
        },
        {
            "phase": "decode",
            "monotonic_ns": 130_000_000,
            "execution_calls_delta": 32,
            "igpu_ratio": 0.0,
            "policy_igpu_ratio": 0.0,
            "exploration": False,
            "sequence": 4,
        },
    ]

    result = bench.summarize_scheduler_transition(events, 100_000_000)

    assert result["transition_decode_calls_before_signal"] == 10
    assert result["transition_decode_calls_after_signal"] == 35
    assert result["transition_first_cpu_execution_delay_calls"] == 2
    assert result["transition_first_cpu_execution_delay_ms"] == 20.0
    assert result["transition_settled_cpu_delay_calls"] == 3
    assert result["transition_settled_cpu_delay_ms"] == 30.0
    assert result["transition_final_igpu_ratio"] == 0.0


def test_scheduler_transition_summary_supports_igpu_target():
    events = [
        {
            "phase": "decode",
            "monotonic_ns": 110_000_000,
            "execution_calls_delta": 2,
            "igpu_ratio": 0.0,
            "policy_igpu_ratio": 0.0,
            "exploration": False,
            "sequence": 1,
        },
        {
            "phase": "decode",
            "monotonic_ns": 120_000_000,
            "execution_calls_delta": 1,
            "igpu_ratio": 1.0,
            "policy_igpu_ratio": 1.0,
            "exploration": False,
            "sequence": 2,
        },
    ]

    result = bench.summarize_scheduler_transition(events, 100_000_000, target="igpu")

    assert result["transition_first_igpu_execution_delay_calls"] == 2
    assert result["transition_first_igpu_execution_delay_ms"] == 20.0
    assert result["transition_settled_igpu_delay_calls"] == 2
    assert result["transition_final_igpu_ratio"] == 1.0


def test_client_transition_summary_splits_stream_intervals():
    result = bench.summarize_client_transition_times(
        [0, 100_000_000, 200_000_000, 400_000_000, 600_000_000],
        transition_output_token=3,
    )

    assert result["transition_client_pre_tps"] == 10.0
    assert result["transition_client_post_tps"] == 5.0


def test_sample_summary_bootstraps_transition_metrics():
    samples = [
        {
            "status": "ok",
            "workload": "p1-o10",
            "target_prompt_tokens": 1,
            "requested_output_tokens": 10,
            "background_ready_delay_ms": 10.0,
            "transition_client_pre_tps": 20.0,
            "transition_client_post_tps": 5.0,
        },
        {
            "status": "ok",
            "workload": "p1-o10",
            "target_prompt_tokens": 1,
            "requested_output_tokens": 10,
            "background_ready_delay_ms": 20.0,
            "transition_client_pre_tps": 22.0,
            "transition_client_post_tps": 7.0,
        },
    ]

    summary = bench.summarize_samples(samples, bootstrap_samples=100, seed=1)[0]

    assert summary["background_ready_delay_ms_mean"] == 15.0
    assert summary["transition_client_pre_tps_mean"] == 21.0
    assert summary["transition_client_post_tps_mean"] == 6.0
    assert summary["transition_client_pre_tps_ci95_low"] <= 21.0
    assert summary["transition_client_pre_tps_ci95_high"] >= 21.0


def test_report_uses_aggregated_transition_metrics(tmp_path):
    manifest = {
        "status": "complete",
        "run_label": "static-transition",
        "server": {"base_url": "http://localhost", "model": "model"},
        "successful_sample_count": 5,
        "sample_count": 5,
        "telemetry": {},
        "background_transition": {
            "direction": "low-to-high",
            "workers": 20,
            "start_after_output_tokens": 150,
            "result": {
                "background_ready_delay_ms": 10.0,
                "background_stopped": True,
                "transition_client_pre_tps": 20.0,
                "transition_client_post_tps": 5.0,
            },
        },
    }
    summary = [
        {
            "workload": "p1024-o600",
            "n": 5,
            "background_ready_delay_ms_mean": 50.0,
            "background_ready_delay_ms_ci95_low": 40.0,
            "background_ready_delay_ms_ci95_high": 60.0,
            "transition_client_pre_tps_mean": 29.0,
            "transition_client_pre_tps_ci95_low": 28.0,
            "transition_client_pre_tps_ci95_high": 30.0,
            "transition_client_post_tps_mean": 7.0,
            "transition_client_post_tps_ci95_low": 6.0,
            "transition_client_post_tps_ci95_high": 8.0,
        }
    ]
    report_path = tmp_path / "report.md"

    bench.write_report(report_path, manifest, summary)

    report = report_path.read_text(encoding="utf-8")
    assert "Transition samples: 5" in report
    assert "Background ready delay: 50.00 [40.00, 60.00] ms" in report
    assert "Client pre-transition throughput: 29.00 [28.00, 30.00] token/s" in report
    assert "Client post-transition throughput: 7.00 [6.00, 8.00] token/s" in report


def test_background_transition_arguments_require_single_measured_request(tmp_path):
    common = [
        "--output-dir",
        str(tmp_path / "run"),
        "--scheduler-telemetry-file",
        str(tmp_path / "telemetry.jsonl"),
        "--stop-background-pid",
        "123",
        "--stop-background-after-output-tokens",
        "5",
        "--workloads",
        "1:10",
    ]

    with pytest.raises(SystemExit):
        bench.parse_args(common)

    args = bench.parse_args(common + ["--repetitions", "1"])
    assert args.stop_background_pid == 123
    assert args.stop_background_after_output_tokens == 5


def test_start_background_transition_arguments(tmp_path):
    args = bench.parse_args(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--scheduler-telemetry-file",
            str(tmp_path / "telemetry.jsonl"),
            "--start-compute-background-workers",
            "20",
            "--start-background-after-output-tokens",
            "5",
            "--workloads",
            "1:10",
            "--repetitions",
            "1",
        ]
    )

    assert args.start_compute_background_workers == 20
    assert args.start_background_after_output_tokens == 5


def test_static_transition_allows_missing_scheduler_telemetry(tmp_path):
    args = bench.parse_args(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--start-compute-background-workers",
            "20",
            "--start-background-after-output-tokens",
            "5",
            "--transition-static-baseline",
            "--workloads",
            "1:10",
            "--repetitions",
            "1",
        ]
    )

    assert args.scheduler_telemetry_file is None
    assert args.transition_static_baseline is True


def test_managed_static_transition_allows_multiple_repetitions(tmp_path):
    args = bench.parse_args(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--start-compute-background-workers",
            "20",
            "--start-background-after-output-tokens",
            "5",
            "--transition-static-baseline",
            "--workloads",
            "1:10",
            "--repetitions",
            "5",
        ]
    )

    assert args.repetitions == 5


def test_dynamic_start_transition_still_requires_one_repetition(tmp_path):
    with pytest.raises(SystemExit):
        bench.parse_args(
            [
                "--output-dir",
                str(tmp_path / "run"),
                "--scheduler-telemetry-file",
                str(tmp_path / "telemetry.jsonl"),
                "--start-compute-background-workers",
                "20",
                "--start-background-after-output-tokens",
                "5",
                "--workloads",
                "1:10",
                "--repetitions",
                "5",
            ]
        )


def test_scheduler_telemetry_tail_reads_only_new_events(tmp_path):
    source = tmp_path / "server-telemetry.jsonl"
    source.write_text('{"sequence":0}\n', encoding="utf-8")
    tail = bench.SchedulerTelemetryTail(source)
    offset = tail.mark()
    with source.open("a", encoding="utf-8") as output:
        output.write('{"sequence":1}\n')

    assert tail.read_since(offset) == [{"sequence": 1}]


def test_running_server_benchmark_writes_complete_artifacts(monkeypatch, tmp_path):
    output_dir = tmp_path / "manual-run"
    args = bench.parse_args(
        [
            "--output-dir",
            str(output_dir),
            "--workloads",
            "1:2",
            "--repetitions",
            "1",
            "--warmups",
            "0",
            "--bootstrap-samples",
            "10",
        ]
    )
    monkeypatch.setattr(bench, "wait_for_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(bench, "discover_model", lambda *args, **kwargs: "served-model")
    monkeypatch.setattr(
        bench,
        "run_streaming_request",
        lambda *args, **kwargs: {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "usage_source": "server",
            "stream_chunks": 2,
            "ttft_ms": 100.0,
            "output_phase_ms": 200.0,
            "ttlt_ms": 300.0,
            "e2e_ms": 310.0,
            "stream_tail_ms": 10.0,
            "prefill_tps": 10.0,
            "decode_tps": 5.0,
            "tpot_ms": 200.0,
            "output_sha256": "hash",
            "output_characters": 2,
        },
    )

    assert bench.run_benchmark(args) == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "complete"
    assert manifest["arguments"]["workloads"] == [{"target_prompt_tokens": 1, "output_tokens": 2}]
    assert (output_dir / "samples.jsonl").is_file()
    assert (output_dir / "summary.csv").is_file()
    assert (output_dir / "report.md").is_file()


def test_running_server_benchmark_associates_scheduler_events(monkeypatch, tmp_path):
    output_dir = tmp_path / "manual-telemetry-run"
    telemetry_source = tmp_path / "server-telemetry.jsonl"
    telemetry_source.write_text('{"sequence":0,"phase":"warmup"}\n', encoding="utf-8")
    args = bench.parse_args(
        [
            "--output-dir",
            str(output_dir),
            "--scheduler-telemetry-file",
            str(telemetry_source),
            "--disable-cpu-telemetry",
            "--workloads",
            "1:2",
            "--repetitions",
            "1",
            "--warmups",
            "0",
            "--bootstrap-samples",
            "10",
        ]
    )
    monkeypatch.setattr(bench, "wait_for_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(bench, "discover_model", lambda *args, **kwargs: "served-model")

    def fake_request(*_args, **_kwargs):
        event = {
            "sequence": 1,
            "phase": "decode",
            "igpu_ratio": 1.0,
            "cpu_load": 0.75,
            "switch_count": 2,
            "high_load_epoch": True,
            "exploration": True,
        }
        with telemetry_source.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event) + "\n")
        return {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "usage_source": "server",
            "stream_chunks": 2,
            "ttft_ms": 100.0,
            "output_phase_ms": 200.0,
            "ttlt_ms": 300.0,
            "e2e_ms": 310.0,
            "stream_tail_ms": 10.0,
            "prefill_tps": 10.0,
            "decode_tps": 5.0,
            "tpot_ms": 200.0,
            "output_sha256": "hash",
            "output_characters": 2,
        }

    monkeypatch.setattr(bench, "run_streaming_request", fake_request)

    assert bench.run_benchmark(args) == 0
    sample = json.loads((output_dir / "samples.jsonl").read_text(encoding="utf-8"))
    raw_events = [
        json.loads(line) for line in (output_dir / "scheduler-telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert sample["scheduler_event_count"] == 1
    assert sample["scheduler_decode_igpu_ratio"] == 1.0
    assert sample["scheduler_decode_cpu_load"] == 0.75
    assert sample["scheduler_decode_exploration_fraction"] == 1.0
    assert raw_events[0]["workload"] == "p1-o2"
    assert manifest["telemetry"]["scheduler_event_count"] == 1
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "| p1-o2 | NA | NA | NA | 1.0000 | 0.7500 | 1.0000 |" in report


def test_running_server_benchmark_rejects_stale_scheduler_telemetry(monkeypatch, tmp_path):
    output_dir = tmp_path / "stale-telemetry-run"
    telemetry_source = tmp_path / "server-telemetry.jsonl"
    telemetry_source.write_text('{"sequence":0,"phase":"old-run"}\n', encoding="utf-8")
    args = bench.parse_args(
        [
            "--output-dir",
            str(output_dir),
            "--scheduler-telemetry-file",
            str(telemetry_source),
            "--disable-cpu-telemetry",
            "--workloads",
            "1:2",
            "--repetitions",
            "1",
            "--warmups",
            "1",
            "--bootstrap-samples",
            "10",
        ]
    )
    monkeypatch.setattr(bench, "wait_for_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(bench, "discover_model", lambda *args, **kwargs: "served-model")
    monkeypatch.setattr(bench, "run_streaming_request", lambda *args, **kwargs: {})

    assert bench.run_benchmark(args) == 1
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "failed"
    assert "did not advance during warmup" in manifest["error"]
