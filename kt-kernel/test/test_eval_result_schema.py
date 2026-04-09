import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kt_kernel.eval.result_schema import EvaluationRun, MetricResult, RunMeta
from kt_kernel.eval.task_registry import TASK_SPECS, resolve_tasks


def test_resolve_core_task_group_contains_expected_tasks():
    tasks = resolve_tasks(task_group="core", tasks_arg=None)
    assert tasks == [
        "wikitext",
        "c4",
        "hellaswag",
        "arc_challenge",
        "winogrande",
        "piqa",
        "mmlu_pro",
        "gsm8k",
        "ifeval",
    ]


def test_evaluation_run_round_trip_json():
    run = EvaluationRun(
        schema_version=1,
        run_meta=RunMeta(
            run_id="run-1",
            label="baseline",
            model_name="demo-model",
            base_url="http://127.0.0.1:30000/v1",
            created_at="2026-04-07T21:30:00+08:00",
            git_commit="4606bf1",
            tasks=["wikitext"],
        ),
        results={
            "wikitext": MetricResult(
                metric="perplexity",
                value=8.5,
                samples=32,
                notes={"source": "fixture"},
            )
        },
    )

    payload = run.to_dict()
    restored = EvaluationRun.from_dict(payload)

    assert restored == run


def test_task_specs_have_expected_metric_directions():
    assert TASK_SPECS["wikitext"].direction == "lower_is_better"
    assert TASK_SPECS["gsm8k"].direction == "higher_is_better"
