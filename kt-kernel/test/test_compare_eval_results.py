import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kt_kernel.eval.compare import compare_runs
from kt_kernel.eval.result_schema import EvaluationRun, MetricResult, RunMeta


def make_run(label: str, value: float) -> EvaluationRun:
    return EvaluationRun(
        schema_version=1,
        run_meta=RunMeta(
            run_id=f"run-{label}",
            label=label,
            model_name="demo",
            base_url="http://127.0.0.1:30000/v1",
            created_at="2026-04-07T21:30:00+08:00",
            git_commit="4606bf1",
            tasks=["wikitext"],
        ),
        results={"wikitext": MetricResult("perplexity", value, 8, {})},
    )


def test_compare_runs_reports_direction_and_deltas():
    rows = compare_runs(make_run("baseline", 8.0), make_run("quant", 8.8))

    assert rows == [
        {
            "task": "wikitext",
            "metric": "perplexity",
            "direction": "lower_is_better",
            "baseline": 8.0,
            "candidate": 8.8,
            "abs_delta": 0.8,
            "rel_delta_pct": 10.0,
        }
    ]


def test_compare_runs_raises_on_missing_task():
    baseline = make_run("baseline", 8.0)
    candidate = make_run("quant", 8.8)
    candidate.results = {}

    try:
        compare_runs(baseline, candidate)
    except ValueError as exc:
        assert "missing task" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")


def load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "compare_eval_results.py"
    spec = importlib.util.spec_from_file_location("test_compare_eval_results_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compare_cli_writes_terminal_json_and_csv_outputs(capsys, tmp_path):
    module = load_script_module()
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output_json = tmp_path / "compare.json"
    output_csv = tmp_path / "compare.csv"

    baseline_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_meta": {
                    "run_id": "baseline",
                    "label": "baseline",
                    "model_name": "demo",
                    "base_url": "http://127.0.0.1:30000/v1",
                    "created_at": "2026-04-07T21:30:00+08:00",
                    "git_commit": "4606bf1",
                    "tasks": ["wikitext"],
                },
                "results": {
                    "wikitext": {
                        "metric": "perplexity",
                        "value": 8.0,
                        "samples": 8,
                        "notes": {},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_meta": {
                    "run_id": "candidate",
                    "label": "candidate",
                    "model_name": "demo",
                    "base_url": "http://127.0.0.1:30001/v1",
                    "created_at": "2026-04-07T21:35:00+08:00",
                    "git_commit": "4606bf1",
                    "tasks": ["wikitext"],
                },
                "results": {
                    "wikitext": {
                        "metric": "perplexity",
                        "value": 8.8,
                        "samples": 8,
                        "notes": {},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    rc = module.main(
        [
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ]
    )

    assert rc == 0
    stdout = capsys.readouterr().out
    assert "wikitext" in stdout
    assert "perplexity" in stdout

    rows = json.loads(output_json.read_text(encoding="utf-8"))
    assert rows[0]["task"] == "wikitext"
    csv_text = output_csv.read_text(encoding="utf-8")
    assert "task,metric,direction,baseline,candidate,abs_delta,rel_delta_pct" in csv_text
