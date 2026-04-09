import math
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kt_kernel.eval.ppl import (
    compute_perplexity_from_logprobs,
    extract_input_token_logprobs,
    load_text_samples,
    summarize_nll,
)


def test_compute_perplexity_from_logprobs_matches_exp_mean_nll():
    logprobs = [-0.5, -1.5]
    ppl = compute_perplexity_from_logprobs(logprobs)
    assert round(ppl, 6) == round(math.e**1.0, 6)


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


def test_extract_input_token_logprobs_from_generate_response():
    response = {
        "meta_info": {
            "input_token_logprobs": [
                [-1.0, 10, "Hello"],
                [-2.0, 11, " world"],
            ]
        }
    }

    assert extract_input_token_logprobs(response) == [-1.0, -2.0]


def test_load_text_samples_uses_fixed_c4_validation_shard(monkeypatch):
    calls = {"download": None, "dataset": None}

    def fake_hf_hub_download(**kwargs):
        calls["download"] = kwargs
        return "/tmp/c4-validation.00000-of-00008.json.gz"

    def fake_load_dataset(*args, **kwargs):
        calls["dataset"] = {"args": args, "kwargs": kwargs}
        return [
            {"text": " first sample "},
            {"text": ""},
            {"text": "second sample"},
        ]

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=fake_hf_hub_download),
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=fake_load_dataset),
    )

    texts = load_text_samples("c4", max_samples=2)

    assert texts == ["first sample", "second sample"]
    assert calls["download"] == {
        "repo_id": "allenai/c4",
        "repo_type": "dataset",
        "filename": "en/c4-validation.00000-of-00008.json.gz",
    }
    assert calls["dataset"] == {
        "args": ("json",),
        "kwargs": {
            "data_files": "/tmp/c4-validation.00000-of-00008.json.gz",
            "split": "train",
        },
    }
