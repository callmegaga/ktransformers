"""Trace aggregation tests that do not import torch or a GPU extension."""
import importlib.util
import json
from pathlib import Path

import pytest

source = Path(__file__).parents[1] / 'bench/summarize_sycl_pipeline.py'
spec = importlib.util.spec_from_file_location('sycl_pipeline_summary', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def trace(layer, phase, calls, duration):
    return '  [server] KT_SYCL_PIPELINE ' + json.dumps({'layer': layer, 'phase': phase,
        'totals': {'calls': calls, 'host_executor_us': duration}})


def test_uses_latest_cumulative_record_without_double_counting():
    text = '\n'.join([trace(0, 'decode', 100, 1000), trace(0, 'decode', 200, 4000),
                      trace(1, 'decode', 200, 6000), trace(0, 'prefill', 1, 500)])
    result = module.summarize(text)
    decode = result['phases']['decode']
    assert decode['layer_count'] == 2
    assert decode['totals']['calls'] == 400
    assert decode['totals']['host_executor_us'] == 10000
    assert decode['mean_per_layer_call']['host_executor_us'] == 25
    assert result['phases']['prefill']['totals']['calls'] == 1
    assert result['last_intervals']['decode']['totals']['calls'] == 100
    assert result['last_intervals']['decode']['mean_per_layer_call']['host_executor_us'] == 30


@pytest.mark.parametrize('text', ['ordinary log', 'KT_SYCL_PIPELINE {bad json', trace(0, 'decode', 0, 0)])
def test_rejects_missing_or_invalid_trace(text):
    with pytest.raises(ValueError):
        module.summarize(text)
