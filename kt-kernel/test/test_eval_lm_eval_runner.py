import sys
import types

import kt_kernel.eval.lm_eval_runner as lm_eval_runner


def test_select_metric_skips_non_numeric_primary_fields():
    metric_name, metric_value = lm_eval_runner._select_metric(
        {
            "alias": "ifeval",
            "prompt_level_strict_acc,none": 1.0,
            "prompt_level_strict_acc_stderr,none": "N/A",
        },
        "ifeval",
    )

    assert metric_name == "prompt_level_strict_acc,none"
    assert metric_value == 1.0


def test_run_lm_eval_tasks_sets_context_budget_from_task_generation_budget(monkeypatch):
    calls = []

    def fake_simple_evaluate(**kwargs):
        calls.append(kwargs)
        return {
            "results": {
                "mmlu_pro": {
                    "exact_match,custom-extract": 0.5,
                    "exact_match_stderr,custom-extract": 0.1,
                }
            },
            "n-samples": {"mmlu_pro": {"original": 14, "effective": 2}},
        }

    monkeypatch.setitem(sys.modules, "lm_eval", types.SimpleNamespace(simple_evaluate=fake_simple_evaluate))

    results = lm_eval_runner.run_lm_eval_tasks(
        base_url="http://127.0.0.1:30000/v1",
        model_name="demo-model",
        tasks=["mmlu_pro"],
        limit=2,
    )

    assert results["mmlu_pro"]["value"] == 0.5
    assert calls[0]["model_args"]["max_gen_toks"] == 2048
    assert calls[0]["model_args"]["max_length"] == 18049
