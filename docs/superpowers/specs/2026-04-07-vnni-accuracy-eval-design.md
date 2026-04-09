# VNNI Accuracy Evaluation Design

## Goal

Build a repeatable evaluation workflow that compares inference accuracy between an unquantized baseline model service and a VNNI-quantized model service without requiring both services to run at the same time.

The first version targets a core benchmark package:

- WikiText-2
- C4
- HellaSwag
- ARC-Challenge
- WinoGrande
- PIQA
- MMLU-Pro
- GSM8K
- IFEval

The workflow must support running one service endpoint at a time, saving standardized single-run result files, and producing a final comparison report from multiple saved runs.

## Scope

In scope:

- Single-endpoint evaluation against an OpenAI-compatible service
- Standardized JSON result artifacts for each run
- Final comparison script for baseline vs VNNI result files
- Perplexity evaluation for WikiText-2 and C4
- Task evaluation for the remaining benchmarks
- CLI flags for task selection, limits, reproducibility, and output paths

Out of scope for the first version:

- Starting or managing model servers
- Simultaneous dual-endpoint evaluation
- Distributed evaluation orchestration
- Leaderboard-grade tuning per model family
- Additional benchmarks such as C-Eval, CMMLU, BBH, GPQA, or HumanEval

## Approaches Considered

### Approach A: Fully custom evaluation framework

Implement dataset loading, prompt construction, metric calculation, and reporting for every benchmark inside this repository.

Pros:

- Full control over every detail
- Single internal implementation style

Cons:

- Highest implementation cost
- Repeats benchmark logic already maintained elsewhere
- Higher risk of benchmark-specific scoring mistakes

### Approach B: `lm-eval-harness` for benchmark tasks plus custom PPL runner

Use `lm-eval-harness` for benchmark tasks that map well to OpenAI-compatible completions endpoints, and implement WikiText-2/C4 perplexity evaluation locally.

Pros:

- Best balance of implementation speed and correctness
- Reuses common academic evaluation conventions
- Keeps perplexity evaluation directly under project control

Cons:

- Adds an external dependency on `lm-eval-harness`
- Requires a thin compatibility layer and result normalization

### Approach C: Everything through `lm-eval-harness`

Route both task evaluation and perplexity-style evaluation through one external framework.

Pros:

- Most uniform external interface

Cons:

- More sensitive to endpoint loglikelihood/logprob support details
- Harder to debug when the service API and harness expectations diverge

### Recommendation

Choose Approach B.

It gives a standard evaluation path for common academic tasks while keeping the most API-sensitive part, perplexity, in a small internal implementation that can be adapted to the exact behavior of the local OpenAI-compatible service.

## Architecture

The design has two user-facing scripts and a small evaluation support package.

### 1. Single-endpoint evaluator

Suggested path:

- `kt-kernel/scripts/eval_single_endpoint.py`

Responsibilities:

- Evaluate one running OpenAI-compatible endpoint
- Run a selected subset of the core benchmark package
- Save a normalized JSON result artifact
- Print a short terminal summary

This script assumes the target service is already running.

### 2. Result comparison tool

Suggested path:

- `kt-kernel/scripts/compare_eval_results.py`

Responsibilities:

- Load two or more prior result artifacts
- Align results by task and metric
- Compute absolute and relative deltas
- Emit a human-readable table and optional CSV/JSON exports

### 3. Internal support package

Suggested paths:

- `kt-kernel/python/eval/endpoint_runner.py`
- `kt-kernel/python/eval/ppl.py`
- `kt-kernel/python/eval/lm_eval_runner.py`
- `kt-kernel/python/eval/result_schema.py`
- `kt-kernel/python/eval/task_registry.py`

Responsibilities:

- `endpoint_runner.py`: HTTP client, retries, timeout handling, service capability checks
- `ppl.py`: WikiText-2/C4 token-level negative log-likelihood and perplexity logic
- `lm_eval_runner.py`: `lm-eval-harness` invocation and output normalization
- `result_schema.py`: result serialization, validation, and compatibility checks
- `task_registry.py`: canonical task definitions, metrics, default few-shot counts, and task groups

## Data Flow

### Single evaluation run

1. Parse CLI arguments.
2. Validate the target endpoint and fetch lightweight run metadata.
3. Resolve the requested tasks from explicit task names or a task group.
4. For each task:
   - Run perplexity tasks through the internal PPL pipeline.
   - Run benchmark tasks through `lm-eval-harness`.
5. Normalize all task outputs into one common result schema.
6. Write the final JSON artifact.
7. Print a summary table.

### Final comparison

1. Load baseline and candidate JSON result files.
2. Validate schema compatibility.
3. Align entries by `task` and `metric`.
4. Compute:
   - `baseline`
   - `candidate`
   - `abs_delta`
   - `rel_delta_pct`
   - `direction`
5. Emit terminal output and optional machine-readable exports.

## Core Benchmark Package

### Perplexity tasks

- `wikitext`
  - Dataset: WikiText-2 test split
  - Metric: `perplexity`
  - Default sample cap: configurable, initial default `256`

- `c4`
  - Dataset: C4 validation subset
  - Metric: `perplexity`
  - Default sample cap: configurable, initial default `128`

### Task evaluation through `lm-eval-harness`

- `hellaswag`
  - Metric: `accuracy`
  - Default few-shot: `0`

- `arc_challenge`
  - Metric: `accuracy`
  - Default few-shot: `0`

- `winogrande`
  - Metric: `accuracy`
  - Default few-shot: `0`

- `piqa`
  - Metric: `accuracy`
  - Default few-shot: `0`

- `mmlu_pro`
  - Metric: `accuracy`
  - Default few-shot: `5`

- `gsm8k`
  - Metric: `exact_match`
  - Default few-shot: `8`

- `ifeval`
  - Metric: use the primary metric reported by `lm-eval-harness` and persist its exact metric name in the result artifact
  - Default few-shot: `0`

## Result Schema

Each single-run artifact should use one normalized JSON schema.

Example:

```json
{
  "schema_version": 1,
  "run_meta": {
    "run_id": "2026-04-07-qwen35-avxvnni-int4",
    "label": "avxvnni_int4",
    "model_name": "Qwen3.5-35B-A3B-FP8",
    "base_url": "http://127.0.0.1:30000/v1",
    "created_at": "2026-04-07T21:30:00+08:00",
    "git_commit": "4606bf1",
    "task_group": "core",
    "tasks": [
      "wikitext",
      "c4",
      "hellaswag",
      "arc_challenge",
      "winogrande",
      "piqa",
      "mmlu_pro",
      "gsm8k",
      "ifeval"
    ]
  },
  "results": {
    "wikitext": {
      "metric": "perplexity",
      "value": 8.42,
      "samples": 245,
      "notes": {}
    },
    "gsm8k": {
      "metric": "exact_match",
      "value": 0.621,
      "samples": 1319,
      "notes": {
        "num_fewshot": 8
      }
    }
  }
}
```

Required metadata fields:

- `schema_version`
- `run_meta.run_id`
- `run_meta.label`
- `run_meta.model_name`
- `run_meta.base_url`
- `run_meta.created_at`
- `run_meta.git_commit`
- `run_meta.tasks`

Required result fields per task:

- `metric`
- `value`
- `samples`
- `notes`

## Comparison Output

The comparison script should align results on `task + metric` and produce:

- `task`
- `metric`
- `direction`
- `baseline`
- `candidate`
- `abs_delta`
- `rel_delta_pct`

Example:

```text
task           metric        direction      baseline   candidate   abs_delta   rel_delta_pct
wikitext       perplexity    lower_is_better 8.42      8.77        +0.35       +4.16%
mmlu_pro       accuracy      higher_is_better 0.512    0.503       -0.009      -1.76%
gsm8k          exact_match   higher_is_better 0.621    0.602       -0.019      -3.06%
```

The `direction` field is required so that positive or negative deltas are not misread across metrics with different optimization directions.

## CLI Design

### `eval_single_endpoint.py`

Required flags:

- `--base-url`
- `--model-name`
- `--label`
- `--output`

Optional flags:

- `--tasks`
- `--task-group`
- `--limit`
- `--max-samples-per-task`
- `--seed`
- `--timeout`
- `--force`
- `--verbose`

Example:

```bash
python kt-kernel/scripts/eval_single_endpoint.py \
  --base-url http://127.0.0.1:30000/v1 \
  --model-name Qwen3.5-35B-A3B-FP8 \
  --label baseline \
  --task-group core \
  --limit 200 \
  --output perf/eval/baseline.json
```

### `compare_eval_results.py`

Required flags:

- `--baseline`
- `--candidate`

Optional flags:

- `--output-json`
- `--output-csv`
- `--sort-by`
- `--fail-on-missing`
- `--name-baseline`
- `--name-candidate`

Example:

```bash
python kt-kernel/scripts/compare_eval_results.py \
  --baseline perf/eval/baseline.json \
  --candidate perf/eval/avxvnni_int4.json \
  --output-csv perf/eval/compare_baseline_vs_avxvnni_int4.csv
```

## Sampling and Reproducibility Defaults

To reduce noise in baseline vs quantized comparisons:

- Use `temperature=0` for choice and reasoning tasks
- Use `top_p=1`
- Disable stochastic sampling where supported
- Accept an explicit `--seed`
- Support `--limit` for quick smoke tests and shorter iteration loops

The initial default task group is:

- `wikitext`
- `c4`
- `hellaswag`
- `arc_challenge`
- `winogrande`
- `piqa`
- `mmlu_pro`
- `gsm8k`
- `ifeval`

## Error Handling

The single-endpoint evaluator should fail clearly and early for the following cases:

- Endpoint is unreachable
- Endpoint does not implement the expected OpenAI-compatible path
- Required dependencies such as `lm-eval-harness` are missing
- A task returns no primary metric
- A result artifact would be overwritten without `--force`

Task failures should be reported with task names and root causes. The design should allow either:

- fail-fast behavior for strict runs, or
- best-effort behavior with explicit task failure records

The first version should prefer fail-fast for simplicity.

The comparison script should detect and report:

- schema version mismatch
- missing tasks in either file
- mismatched metric names for the same task

## Testing Strategy

Testing must cover both happy paths and normalization behavior.

### Unit tests

Add tests for:

- task registry resolution
- result schema serialization/deserialization
- delta calculation and metric direction handling
- PPL aggregation math on small synthetic examples
- compare-script behavior when tasks are missing or mismatched

### Integration tests

Add lightweight tests with mocked HTTP responses for:

- endpoint health checks
- single-endpoint JSON artifact generation
- comparison table generation from two small fixture files

### Non-goals for first-version tests

- Running the full benchmark package in CI
- Downloading full benchmark datasets in per-commit tests
- Depending on live external endpoints in automated tests

## File Plan

Expected new files:

- `kt-kernel/scripts/eval_single_endpoint.py`
- `kt-kernel/scripts/compare_eval_results.py`
- `kt-kernel/python/eval/endpoint_runner.py`
- `kt-kernel/python/eval/ppl.py`
- `kt-kernel/python/eval/lm_eval_runner.py`
- `kt-kernel/python/eval/result_schema.py`
- `kt-kernel/python/eval/task_registry.py`
- `kt-kernel/test/...` for unit and integration coverage

## Open Questions Resolved

- Run both services simultaneously: no
- Single endpoint per evaluation run: yes
- Baseline vs quantized comparison done later from saved artifacts: yes
- First version benchmark scope: core package only
- First version orchestration: evaluate only, no server launch

## Success Criteria

The design is successful when:

- A user can evaluate one service endpoint at a time
- Each run produces one stable JSON artifact
- A later comparison command can summarize baseline vs VNNI deltas
- The first-version benchmark package covers perplexity, common-sense QA, knowledge QA, math reasoning, and instruction-following
- The implementation remains easy to extend with additional benchmark tasks later
