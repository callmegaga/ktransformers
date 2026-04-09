# VNNI Accuracy Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-stage evaluation workflow that benchmarks one OpenAI-compatible endpoint at a time, stores normalized JSON results, and compares baseline vs VNNI-quantized runs across the agreed core benchmark package.

**Architecture:** Add a small `kt_kernel.eval` package for task registry, schema handling, endpoint requests, perplexity evaluation, and `lm-eval-harness` integration. Expose that package through two scripts: one for single-endpoint evaluation and one for result comparison. Keep first-version tests fully offline with fixtures and mocked HTTP responses.

**Tech Stack:** Python, `pytest`, `requests`, `datasets`, `lm-eval-harness`, existing `kt-kernel` packaging and script conventions

---

### Task 1: Create The Evaluation Package Skeleton And Schema Layer

**Files:**
- Create: `kt-kernel/python/eval/__init__.py`
- Create: `kt-kernel/python/eval/result_schema.py`
- Create: `kt-kernel/python/eval/task_registry.py`
- Create: `kt-kernel/test/test_eval_result_schema.py`
- Modify: `kt-kernel/setup.py`

- [ ] **Step 1: Write the failing schema and registry tests**

```python
from kt_kernel.eval.result_schema import MetricResult, RunMeta, EvaluationRun
from kt_kernel.eval.task_registry import CORE_TASK_GROUPS, TASK_SPECS, resolve_tasks


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
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest -q kt-kernel/test/test_eval_result_schema.py`

Expected: FAIL with `ModuleNotFoundError: No module named 'kt_kernel.eval'` or missing symbol errors.

- [ ] **Step 3: Write the minimal package skeleton and schema implementation**

```python
# kt-kernel/python/eval/result_schema.py
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass(eq=True)
class MetricResult:
    metric: str
    value: float
    samples: int
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass(eq=True)
class RunMeta:
    run_id: str
    label: str
    model_name: str
    base_url: str
    created_at: str
    git_commit: str
    tasks: List[str]
    task_group: str | None = None


@dataclass(eq=True)
class EvaluationRun:
    schema_version: int
    run_meta: RunMeta
    results: Dict[str, MetricResult]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvaluationRun":
        run_meta = RunMeta(**payload["run_meta"])
        results = {
            name: MetricResult(**result) for name, result in payload["results"].items()
        }
        return cls(
            schema_version=payload["schema_version"],
            run_meta=run_meta,
            results=results,
        )
```

```python
# kt-kernel/python/eval/task_registry.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    kind: str
    metric: str
    direction: str
    fewshot: int


TASK_SPECS = {
    "wikitext": TaskSpec("wikitext", "ppl", "perplexity", "lower_is_better", 0),
    "c4": TaskSpec("c4", "ppl", "perplexity", "lower_is_better", 0),
    "hellaswag": TaskSpec("hellaswag", "lm_eval", "accuracy", "higher_is_better", 0),
    "arc_challenge": TaskSpec("arc_challenge", "lm_eval", "accuracy", "higher_is_better", 0),
    "winogrande": TaskSpec("winogrande", "lm_eval", "accuracy", "higher_is_better", 0),
    "piqa": TaskSpec("piqa", "lm_eval", "accuracy", "higher_is_better", 0),
    "mmlu_pro": TaskSpec("mmlu_pro", "lm_eval", "accuracy", "higher_is_better", 5),
    "gsm8k": TaskSpec("gsm8k", "lm_eval", "exact_match", "higher_is_better", 8),
    "ifeval": TaskSpec("ifeval", "lm_eval", "primary", "higher_is_better", 0),
}

CORE_TASK_GROUPS = {
    "core": [
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
}


def resolve_tasks(task_group: str | None, tasks_arg: str | None) -> list[str]:
    if tasks_arg:
        return [task.strip() for task in tasks_arg.split(",") if task.strip()]
    if task_group:
        return CORE_TASK_GROUPS[task_group]
    return CORE_TASK_GROUPS["core"]
```

```python
# kt-kernel/python/eval/__init__.py
from .result_schema import EvaluationRun, MetricResult, RunMeta
from .task_registry import CORE_TASK_GROUPS, TASK_SPECS, TaskSpec, resolve_tasks
```

```python
# kt-kernel/setup.py
    packages=[
        "kt_kernel",
        "kt_kernel.eval",
        "kt_kernel.utils",
        "kt_kernel.cli",
        "kt_kernel.cli.commands",
        "kt_kernel.cli.config",
        "kt_kernel.cli.utils",
    ],
    package_dir={
        "kt_kernel": "python",
        "kt_kernel.eval": "python/eval",
        "kt_kernel.utils": "python/utils",
        "kt_kernel.cli": "python/cli",
        "kt_kernel.cli.commands": "python/cli/commands",
        "kt_kernel.cli.config": "python/cli/config",
        "kt_kernel.cli.utils": "python/cli/utils",
    },
```

- [ ] **Step 4: Run the schema tests to verify they pass**

Run: `pytest -q kt-kernel/test/test_eval_result_schema.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kt-kernel/python/eval/__init__.py kt-kernel/python/eval/result_schema.py kt-kernel/python/eval/task_registry.py kt-kernel/test/test_eval_result_schema.py kt-kernel/setup.py
git commit -m "feat: add evaluation schema and task registry"
```

### Task 2: Add Result Comparison Logic And CLI

**Files:**
- Create: `kt-kernel/python/eval/compare.py`
- Create: `kt-kernel/scripts/compare_eval_results.py`
- Create: `kt-kernel/test/test_compare_eval_results.py`

- [ ] **Step 1: Write the failing comparison tests**

```python
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
```

- [ ] **Step 2: Run the comparison tests to verify they fail**

Run: `pytest -q kt-kernel/test/test_compare_eval_results.py`

Expected: FAIL with missing module or function errors.

- [ ] **Step 3: Implement comparison logic and the comparison CLI**

```python
# kt-kernel/python/eval/compare.py
from __future__ import annotations

from kt_kernel.eval.result_schema import EvaluationRun
from kt_kernel.eval.task_registry import TASK_SPECS


def compare_runs(baseline: EvaluationRun, candidate: EvaluationRun) -> list[dict]:
    rows = []
    for task in baseline.results:
        if task not in candidate.results:
            raise ValueError(f"Missing task in candidate results: {task}")
        base_result = baseline.results[task]
        cand_result = candidate.results[task]
        if base_result.metric != cand_result.metric:
            raise ValueError(f"Mismatched metric for task {task}")
        baseline_value = float(base_result.value)
        candidate_value = float(cand_result.value)
        abs_delta = candidate_value - baseline_value
        rel_delta_pct = 0.0 if baseline_value == 0 else abs_delta / baseline_value * 100.0
        rows.append(
            {
                "task": task,
                "metric": base_result.metric,
                "direction": TASK_SPECS[task].direction,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "abs_delta": abs_delta,
                "rel_delta_pct": rel_delta_pct,
            }
        )
    return rows
```

```python
# kt-kernel/scripts/compare_eval_results.py
#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kt_kernel.eval.compare import compare_runs
from kt_kernel.eval.result_schema import EvaluationRun


def load_run(path: str) -> EvaluationRun:
    with open(path, "r", encoding="utf-8") as f:
        return EvaluationRun.from_dict(json.load(f))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()

    rows = compare_runs(load_run(args.baseline), load_run(args.candidate))
    for row in rows:
        print(
            f"{row['task']}\t{row['metric']}\t{row['direction']}\t"
            f"{row['baseline']:.6f}\t{row['candidate']:.6f}\t"
            f"{row['abs_delta']:+.6f}\t{row['rel_delta_pct']:+.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the comparison tests to verify they pass**

Run: `pytest -q kt-kernel/test/test_compare_eval_results.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kt-kernel/python/eval/compare.py kt-kernel/scripts/compare_eval_results.py kt-kernel/test/test_compare_eval_results.py
git commit -m "feat: add evaluation result comparison tooling"
```

### Task 3: Add Endpoint Utilities And Perplexity Evaluation

**Files:**
- Create: `kt-kernel/python/eval/endpoint_runner.py`
- Create: `kt-kernel/python/eval/ppl.py`
- Create: `kt-kernel/test/test_eval_ppl.py`

- [ ] **Step 1: Write the failing endpoint and PPL tests**

```python
from kt_kernel.eval.ppl import compute_perplexity_from_logprobs, summarize_nll


def test_compute_perplexity_from_logprobs_matches_exp_mean_nll():
    logprobs = [-0.5, -1.5]
    ppl = compute_perplexity_from_logprobs(logprobs)
    assert round(ppl, 6) == round(((2.718281828459045) ** 1.0), 6)


def test_summarize_nll_accumulates_tokens_and_samples():
    summary = summarize_nll(
        [
            {"token_logprobs": [-1.0, -1.0]},
            {"token_logprobs": [-2.0]},
        ]
    )
    assert summary["samples"] == 2
    assert summary["token_count"] == 3
    assert summary["mean_nll"] == 4.0 / 3.0
```

- [ ] **Step 2: Run the PPL tests to verify they fail**

Run: `pytest -q kt-kernel/test/test_eval_ppl.py`

Expected: FAIL with missing module or function errors.

- [ ] **Step 3: Implement the minimal endpoint helper and PPL math**

```python
# kt-kernel/python/eval/endpoint_runner.py
from __future__ import annotations

import requests


class EndpointClient:
    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post_completions(self, payload: dict) -> dict:
        response = requests.post(
            self.base_url + "/completions",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
```

```python
# kt-kernel/python/eval/ppl.py
from __future__ import annotations

import math


def summarize_nll(records: list[dict]) -> dict:
    token_count = 0
    total_nll = 0.0
    for record in records:
        values = list(record["token_logprobs"])
        token_count += len(values)
        total_nll -= sum(values)
    mean_nll = total_nll / token_count
    return {
        "samples": len(records),
        "token_count": token_count,
        "mean_nll": mean_nll,
    }


def compute_perplexity_from_logprobs(logprobs: list[float]) -> float:
    mean_nll = -sum(logprobs) / len(logprobs)
    return math.exp(mean_nll)
```

- [ ] **Step 4: Run the PPL tests to verify they pass**

Run: `pytest -q kt-kernel/test/test_eval_ppl.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kt-kernel/python/eval/endpoint_runner.py kt-kernel/python/eval/ppl.py kt-kernel/test/test_eval_ppl.py
git commit -m "feat: add endpoint helpers and perplexity math"
```

### Task 4: Add `lm-eval-harness` Integration And Single-Endpoint Runner

**Files:**
- Create: `kt-kernel/python/eval/lm_eval_runner.py`
- Create: `kt-kernel/scripts/eval_single_endpoint.py`
- Create: `kt-kernel/test/test_eval_single_endpoint.py`

- [ ] **Step 1: Write the failing runner tests**

```python
import json
import importlib.util
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
            "wikitext": {"metric": "perplexity", "value": 8.0, "samples": 8, "notes": {}}
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
```

- [ ] **Step 2: Run the runner tests to verify they fail**

Run: `pytest -q kt-kernel/test/test_eval_single_endpoint.py`

Expected: FAIL with missing script/module errors.

- [ ] **Step 3: Implement `lm-eval` normalization and the single-endpoint CLI**

```python
# kt-kernel/python/eval/lm_eval_runner.py
from __future__ import annotations


def normalize_lm_eval_results(raw_results: dict, requested_tasks: list[str]) -> dict:
    normalized = {}
    for task in requested_tasks:
        metrics = raw_results["results"][task]
        metric_name, metric_value = next(iter(metrics.items()))
        normalized[task] = {
            "metric": metric_name,
            "value": float(metric_value),
            "samples": raw_results.get("n-samples", {}).get(task, 0),
            "notes": {},
        }
    return normalized
```

```python
# kt-kernel/scripts/eval_single_endpoint.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kt_kernel.eval.result_schema import EvaluationRun, MetricResult, RunMeta
from kt_kernel.eval.task_registry import resolve_tasks


def get_git_commit() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        .strip()
    )


def run_selected_tasks(**kwargs) -> dict:
    return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks")
    parser.add_argument("--task-group", default="core")
    args = parser.parse_args(argv)

    tasks = resolve_tasks(task_group=args.task_group, tasks_arg=args.tasks)
    raw_results = run_selected_tasks(
        base_url=args.base_url,
        model_name=args.model_name,
        tasks=tasks,
    )
    run = EvaluationRun(
        schema_version=1,
        run_meta=RunMeta(
            run_id=f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.label}",
            label=args.label,
            model_name=args.model_name,
            base_url=args.base_url,
            created_at=datetime.now().astimezone().isoformat(),
            git_commit=get_git_commit(),
            tasks=tasks,
            task_group=args.task_group,
        ),
        results={name: MetricResult(**result) for name, result in raw_results.items()},
    )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(run.to_dict(), f, indent=2, ensure_ascii=False)
        f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the runner tests to verify they pass**

Run: `pytest -q kt-kernel/test/test_eval_single_endpoint.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kt-kernel/python/eval/lm_eval_runner.py kt-kernel/scripts/eval_single_endpoint.py kt-kernel/test/test_eval_single_endpoint.py
git commit -m "feat: add single-endpoint evaluation runner"
```

### Task 5: Wire Real Task Execution And Add Final Offline Coverage

**Files:**
- Modify: `kt-kernel/python/eval/ppl.py`
- Modify: `kt-kernel/python/eval/lm_eval_runner.py`
- Modify: `kt-kernel/scripts/eval_single_endpoint.py`
- Create: `kt-kernel/test/test_eval_end_to_end_offline.py`

- [ ] **Step 1: Write the failing offline integration test**

```python
import json
import importlib.util
from pathlib import Path


def load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_single_endpoint.py"
    spec = importlib.util.spec_from_file_location("test_eval_single_endpoint_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_offline_end_to_end_with_stubbed_backends(monkeypatch, tmp_path):
    module = load_script_module()

    output_path = tmp_path / "artifact.json"

    monkeypatch.setattr(
        module,
        "run_selected_tasks",
        lambda **kwargs: {
            "wikitext": {
                "metric": "perplexity",
                "value": 8.0,
                "samples": 8,
                "notes": {"token_count": 128},
            },
            "gsm8k": {
                "metric": "exact_match",
                "value": 0.61,
                "samples": 32,
                "notes": {"num_fewshot": 8},
            },
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
            "wikitext,gsm8k",
            "--output",
            str(output_path),
        ]
    )

    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["results"]["wikitext"]["notes"]["token_count"] == 128
    assert payload["results"]["gsm8k"]["metric"] == "exact_match"
```

- [ ] **Step 2: Run the offline integration test to verify it fails**

Run: `pytest -q kt-kernel/test/test_eval_end_to_end_offline.py`

Expected: FAIL before the task-execution wiring is complete.

- [ ] **Step 3: Implement the real task dispatch in the runner**

```python
# kt-kernel/scripts/eval_single_endpoint.py
from kt_kernel.eval.task_registry import TASK_SPECS


def run_selected_tasks(base_url: str, model_name: str, tasks: list[str]) -> dict:
    ppl_tasks = [task for task in tasks if TASK_SPECS[task].kind == "ppl"]
    lm_eval_tasks = [task for task in tasks if TASK_SPECS[task].kind == "lm_eval"]
    results = {}

    if ppl_tasks:
        from kt_kernel.eval.ppl import run_ppl_tasks

        results.update(run_ppl_tasks(base_url=base_url, model_name=model_name, tasks=ppl_tasks))

    if lm_eval_tasks:
        from kt_kernel.eval.lm_eval_runner import run_lm_eval_tasks

        results.update(
            run_lm_eval_tasks(base_url=base_url, model_name=model_name, tasks=lm_eval_tasks)
        )

    return results
```

```python
# kt-kernel/python/eval/lm_eval_runner.py
def run_lm_eval_tasks(base_url: str, model_name: str, tasks: list[str]) -> dict:
    import lm_eval

    results = lm_eval.simple_evaluate(
        model="local-completions",
        model_args={
            "model": model_name,
            "base_url": base_url.rstrip("/") + "/completions",
            "num_concurrent": 1,
        },
        tasks=tasks,
        num_fewshot=0,
        batch_size="auto",
        gen_kwargs={"temperature": 0, "top_p": 1},
    )
    return normalize_lm_eval_results(results, tasks)
```

```python
# kt-kernel/python/eval/ppl.py
def run_ppl_tasks(base_url: str, model_name: str, tasks: list[str]) -> dict:
    from kt_kernel.eval.endpoint_runner import EndpointClient

    client = EndpointClient(base_url)
    results = {}
    for task in tasks:
        texts = load_text_samples(task)
        records = []
        for text in texts:
            response = client.post_completions(
                {
                    "model": model_name,
                    "prompt": text,
                    "max_tokens": 1,
                    "temperature": 0,
                    "echo": True,
                    "logprobs": 1,
                }
            )
            token_logprobs = extract_token_logprobs(response)
            records.append({"token_logprobs": token_logprobs})
        summary = summarize_nll(records)
        results[task] = {
            "metric": "perplexity",
            "value": math.exp(summary["mean_nll"]),
            "samples": summary["samples"],
            "notes": {"token_count": summary["token_count"]},
        }
    return results
```

- [ ] **Step 4: Run the targeted evaluation tests to verify they pass**

Run: `pytest -q kt-kernel/test/test_eval_result_schema.py kt-kernel/test/test_compare_eval_results.py kt-kernel/test/test_eval_ppl.py kt-kernel/test/test_eval_single_endpoint.py kt-kernel/test/test_eval_end_to_end_offline.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kt-kernel/python/eval/ppl.py kt-kernel/python/eval/lm_eval_runner.py kt-kernel/scripts/eval_single_endpoint.py kt-kernel/test/test_eval_end_to_end_offline.py
git commit -m "feat: wire endpoint evaluation flow"
```

### Task 6: Final Verification And Operator Smoke Checks

**Files:**
- Modify: `kt-kernel/scripts/eval_single_endpoint.py`
- Modify: `kt-kernel/scripts/compare_eval_results.py`

- [ ] **Step 1: Add a final fixture-based CLI smoke test if output formatting is still untested**

```python
def test_compare_cli_smoke(monkeypatch, capsys, tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "compare_eval_results.py"
    spec = importlib.util.spec_from_file_location("test_compare_eval_results_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
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
                "results": {"wikitext": {"metric": "perplexity", "value": 8.0, "samples": 8, "notes": {}}},
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
                "results": {"wikitext": {"metric": "perplexity", "value": 8.8, "samples": 8, "notes": {}}},
            }
        ),
        encoding="utf-8",
    )

    rc = module.main(["--baseline", str(baseline_path), "--candidate", str(candidate_path)])

    assert rc == 0
    stdout = capsys.readouterr().out
    assert "wikitext" in stdout
    assert "perplexity" in stdout
```

- [ ] **Step 2: Run the full local test set for this feature**

Run: `pytest -q kt-kernel/test/test_eval_result_schema.py kt-kernel/test/test_compare_eval_results.py kt-kernel/test/test_eval_ppl.py kt-kernel/test/test_eval_single_endpoint.py kt-kernel/test/test_eval_end_to_end_offline.py`

Expected: PASS with no live endpoint dependency.

- [ ] **Step 3: Run a manual smoke evaluation against one live endpoint**

Run:

```bash
python kt-kernel/scripts/eval_single_endpoint.py \
  --base-url http://127.0.0.1:30000/v1 \
  --model-name Qwen3.5-35B-A3B-FP8 \
  --label baseline \
  --tasks wikitext \
  --output /tmp/kt-eval-baseline.json
```

Expected: exit `0` and write one JSON artifact.

- [ ] **Step 4: Run a manual compare smoke check with two fixture or live result files**

Run:

```bash
python kt-kernel/scripts/compare_eval_results.py \
  --baseline /tmp/kt-eval-baseline.json \
  --candidate /tmp/kt-eval-avxvnni-int4.json
```

Expected: terminal table with task, metric, direction, baseline, candidate, and deltas.

- [ ] **Step 5: Commit**

```bash
git add kt-kernel/scripts/eval_single_endpoint.py kt-kernel/scripts/compare_eval_results.py
git commit -m "test: verify VNNI evaluation workflow"
```
