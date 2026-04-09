import importlib.util
import json
from pathlib import Path


from kt_kernel.eval.result_schema import EvaluationRun


def load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_single_endpoint.py"
    spec = importlib.util.spec_from_file_location("test_eval_single_endpoint_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_single_endpoint_runner_writes_json_artifact(monkeypatch, tmp_path):
    module = load_script_module()
    output_path = tmp_path / "run.json"

    monkeypatch.setattr(
        module,
        "run_selected_tasks",
        lambda **kwargs: {
            "wikitext": {
                "metric": "perplexity",
                "value": 8.0,
                "samples": 8,
                "notes": {},
            }
        },
    )
    monkeypatch.setattr(module, "get_git_commit", lambda: "4606bf1")

    rc = module.main(
        [
            "--base-url",
            "http://127.0.0.1:30000/v1",
            "--model-name",
            "demo-model",
            "--label",
            "baseline",
            "--tasks",
            "wikitext",
            "--output",
            str(output_path),
        ]
    )

    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    restored = EvaluationRun.from_dict(payload)
    assert restored.run_meta.label == "baseline"
    assert restored.results["wikitext"].metric == "perplexity"
