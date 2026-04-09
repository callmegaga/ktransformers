#!/usr/bin/env python
# coding=utf-8
"""Offline quantization roundtrip smoke tests for AVX-VNNI INT8/INT4."""

import os
import shutil
import tempfile
import pytest

try:
    import torch
    from kt_kernel import KTMoEWrapper
    from kt_kernel.utils import amx as amx_utils

    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    import_error = str(e)


@pytest.mark.cpu
@pytest.mark.parametrize("method", ["AVXVNNI_INT8", "AVXVNNI_INT4"])
def test_moe_avxvnni_offline_roundtrip(method):
    if not HAS_DEPS:
        pytest.skip(f"Dependencies not available: {import_error}")
    if not amx_utils._HOST_HAS_AVX_VNNI:
        pytest.skip("Host CPU does not support AVX-VNNI")
    if method == "AVXVNNI_INT8" and not amx_utils._HAS_AVXVNNI256_INT8_SUPPORT:
        pytest.skip("AVX-VNNI INT8 backend not compiled")
    if method == "AVXVNNI_INT4" and not amx_utils._HAS_AVXVNNI256_INT4_SUPPORT:
        pytest.skip("AVX-VNNI INT4 backend not compiled")

    expert_num = 8
    hidden_size = 128
    intermediate_size = 64
    num_experts_per_tok = 2
    qlen = 2

    physical_to_logical_map = torch.arange(expert_num, dtype=torch.int64)
    expert_ids = torch.stack([torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]).contiguous()
    weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
    input_data = torch.randn((qlen, hidden_size), dtype=torch.bfloat16).contiguous() / 16
    output = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
    bsz_tensor = torch.tensor([qlen], dtype=torch.int32).contiguous()

    gate_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16).contiguous() / 16
    up_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16).contiguous() / 16
    down_proj = torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.bfloat16).contiguous() / 16

    tmp = tempfile.mkdtemp(prefix="kt_vnni_roundtrip_")
    try:
        save_wrapper = KTMoEWrapper(
            layer_idx=0,
            num_experts=expert_num,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=intermediate_size,
            gpu_experts_mask=torch.zeros(expert_num, dtype=torch.bool),
            cpuinfer_threads=4,
            threadpool_count=1,
            weight_path=tmp,
            chunked_prefill_size=16,
            cpu_save=True,
            method=method,
        )
        save_wrapper.load_weights_from_tensors(gate_proj, up_proj, down_proj, physical_to_logical_map)

        load_wrapper = KTMoEWrapper(
            layer_idx=0,
            num_experts=expert_num,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=intermediate_size,
            gpu_experts_mask=torch.zeros(expert_num, dtype=torch.bool),
            cpuinfer_threads=4,
            threadpool_count=1,
            weight_path=tmp,
            chunked_prefill_size=16,
            method=method,
        )
        load_wrapper.load_weights(physical_to_logical_map)
        load_wrapper.moe.forward(
            bsz_tensor.data_ptr(),
            num_experts_per_tok,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_data.data_ptr(),
            output.data_ptr(),
            False,
        )

        assert output.shape == (qlen, hidden_size)
        assert torch.isfinite(output.float()).all()
        assert torch.count_nonzero(output).item() > 0
        assert os.path.exists(os.path.join(tmp, "_layer_0", "_numa_0"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
