import importlib.util
from pathlib import Path

import kt_kernel.eval.lm_eval_runner as lm_eval_runner
import kt_kernel.eval.ppl as ppl


def load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_single_endpoint.py"
    spec = importlib.util.spec_from_file_location("test_eval_single_endpoint_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_selected_tasks_dispatches_to_ppl_and_lm_eval(monkeypatch):
    module = load_script_module()

    monkeypatch.setattr(
        ppl,
        "run_ppl_tasks",
        lambda **kwargs: {
            "wikitext": {
                "metric": "perplexity",
                "value": 8.0,
                "samples": 8,
                "notes": {"token_count": 128},
            }
        },
        raising=False,
    )
    monkeypatch.setattr(
        lm_eval_runner,
        "run_lm_eval_tasks",
        lambda **kwargs: {
            "gsm8k": {
                "metric": "exact_match",
                "value": 0.61,
                "samples": 32,
                "notes": {"num_fewshot": 8},
            }
        },
        raising=False,
    )

    results = module.run_selected_tasks(
        base_url="http://127.0.0.1:30000/v1",
        model_name="demo-model",
        tasks=["wikitext", "gsm8k"],
    )

    assert results["wikitext"]["metric"] == "perplexity"
    assert results["gsm8k"]["metric"] == "exact_match"
