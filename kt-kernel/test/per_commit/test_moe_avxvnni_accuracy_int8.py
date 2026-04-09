#!/usr/bin/env python
# coding=utf-8
"""AVX-VNNI INT8 MoE accuracy tests for KT-Kernel."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120, suite="default")

try:
    import torch
    import kt_kernel

    kt_kernel_ext = kt_kernel.kt_kernel_ext
    from kt_kernel.utils import amx as amx_utils

    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    import_error = str(e)

expert_num = 64
hidden_size = 256
intermediate_size = 128
max_len = 512
num_experts_per_tok = 4
qlen = 2
validation_iter = 2


def act_fn(x):
    return x / (1.0 + torch.exp(-x))


def mlp_torch(input_data, gate_proj, up_proj, down_proj):
    gate_buf = torch.mm(input_data, gate_proj.t())
    up_buf = torch.mm(input_data, up_proj.t())
    return torch.mm(act_fn(gate_buf) * up_buf, down_proj.t())


def moe_torch(input_data, expert_ids, weights, gate_proj, up_proj, down_proj):
    cnts = expert_ids.new_zeros((expert_ids.shape[0], expert_num))
    cnts.scatter_(1, expert_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = expert_ids.view(-1).argsort()
    sorted_tokens = input_data[idxs // expert_ids.shape[1]]

    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        outputs.append(mlp_torch(sorted_tokens[start_idx:end_idx], gate_proj[i], up_proj[i], down_proj[i]))
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    return new_x.view(*expert_ids.shape, -1).mul_(weights.unsqueeze(-1)).sum(dim=1)


@pytest.mark.cpu
def test_moe_avxvnni_int8_accuracy():
    if not HAS_DEPS:
        pytest.skip(f"Dependencies not available: {import_error}")
    if not amx_utils._HAS_AVXVNNI256_INT8_SUPPORT or not amx_utils._HOST_HAS_AVX_VNNI:
        pytest.skip("AVX-VNNI INT8 backend not available on this host")

    physical_to_logical_map = torch.arange(expert_num, dtype=torch.int64).contiguous()
    cpuinfer = kt_kernel_ext.CPUInfer(8)

    with torch.inference_mode():
        gate_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16).contiguous() / 16
        up_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16).contiguous() / 16
        down_proj = torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.bfloat16).contiguous() / 16

        config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
        config.max_len = max_len
        config.gate_proj = gate_proj.data_ptr()
        config.up_proj = up_proj.data_ptr()
        config.down_proj = down_proj.data_ptr()
        config.pool = cpuinfer.backend_

        moe = kt_kernel_ext.moe.AVXVNNI256Int8_MOE(config)
        cpuinfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
        cpuinfer.sync()

        for i in range(validation_iter):
            bsz_tensor = torch.tensor([qlen], dtype=torch.int64)
            expert_ids = torch.stack(
                [torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]
            ).contiguous()
            weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
            input_data = torch.randn((qlen, hidden_size), dtype=torch.bfloat16).contiguous() / 16
            output = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()

            cpuinfer.submit(
                moe.forward_task(
                    bsz_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids.data_ptr(),
                    weights.data_ptr(),
                    input_data.data_ptr(),
                    output.data_ptr(),
                    False,
                )
            )
            cpuinfer.sync()

            ref = moe_torch(
                input_data.float(), expert_ids, weights, gate_proj.float(), up_proj.float(), down_proj.float()
            )
            diff = torch.mean(torch.abs(output.float() - ref)) / torch.mean(torch.abs(ref)).clamp_min(1e-6)
            print(f"Iteration {i}, diff = {diff:.6f}")
            assert diff < 0.05, f"AVX-VNNI INT8 accuracy test failed: diff={diff:.6f} >= 0.05"
