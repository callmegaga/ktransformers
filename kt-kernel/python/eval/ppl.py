from __future__ import annotations

import math

from kt_kernel.eval.endpoint_runner import EndpointClient

DATASET_SPECS = {
    "wikitext": {
        "loader": "hf_dataset",
        "path": "wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "test",
        "field": "text",
        "default_limit": 256,
    },
    "c4": {
        "loader": "hf_file_json",
        "repo_id": "allenai/c4",
        "repo_type": "dataset",
        "filename": "en/c4-validation.00000-of-00008.json.gz",
        "field": "text",
        "default_limit": 128,
    },
}


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


def load_text_samples(task: str, max_samples: int | None = None) -> list[str]:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    spec = DATASET_SPECS[task]
    limit = max_samples or spec["default_limit"]

    if spec["loader"] == "hf_dataset":
        dataset = load_dataset(spec["path"], spec["name"], split=spec["split"])
    elif spec["loader"] == "hf_file_json":
        local_path = hf_hub_download(
            repo_id=spec["repo_id"],
            repo_type=spec["repo_type"],
            filename=spec["filename"],
        )
        dataset = load_dataset("json", data_files=local_path, split="train")
    else:
        raise ValueError(f"Unsupported loader for task {task}: {spec['loader']}")

    texts = []
    for row in dataset:
        text = row[spec["field"]].strip()
        if not text:
            continue
        texts.append(text)
        if len(texts) >= limit:
            break
    return texts


def extract_input_token_logprobs(response: dict) -> list[float]:
    values = response["meta_info"].get("input_token_logprobs")
    if values is None:
        raise ValueError("Generate response does not include meta_info.input_token_logprobs")
    return [float(item[0]) for item in values if item and item[0] is not None]


def run_ppl_tasks(
    base_url: str,
    model_name: str,
    tasks: list[str],
    max_samples_per_task: int | None = None,
    timeout: int = 120,
) -> dict:
    client = EndpointClient(base_url, timeout=timeout)
    results = {}

    for task in tasks:
        texts = load_text_samples(task, max_samples=max_samples_per_task)
        records = []
        all_logprobs = []
        for text in texts:
            response = client.post_generate(
                {
                    "text": text,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 1,
                    },
                    "return_logprob": True,
                    "logprob_start_len": 0,
                }
            )
            token_logprobs = extract_input_token_logprobs(response)
            records.append({"token_logprobs": token_logprobs})
            all_logprobs.extend(token_logprobs)

        summary = summarize_nll(records)
        results[task] = {
            "metric": "perplexity",
            "value": compute_perplexity_from_logprobs(all_logprobs),
            "samples": summary["samples"],
            "notes": {
                "token_count": summary["token_count"],
                "mean_nll": summary["mean_nll"],
            },
        }

    return results
