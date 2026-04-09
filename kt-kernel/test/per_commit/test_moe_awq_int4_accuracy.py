#!/usr/bin/env python
# coding=utf-8
"""AWQ INT4 MoE accuracy tests for KT-Kernel x86 backends."""

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

expert_num = 8
hidden_size = 256
intermediate_size = 128
num_experts_per_tok = 2
max_len = 128
group_size = 32
validation_iter = 3


def awq_sym_int4_quantize(weight_bf16: "torch.Tensor"):
    """Quantize [N, K] BF16 weight to AWQ symmetric INT4 layout."""
    n, k = weight_bf16.shape
    assert k % 8 == 0
    assert k % group_size == 0

    weight_fp32 = weight_bf16.float()
    packed = torch.zeros((n, k // 8), dtype=torch.int32)
    scales = torch.zeros((n, k // group_size), dtype=torch.float32)

    for ni in range(n):
        for g in range(k // group_size):
            k_start = g * group_size
            k_end = k_start + group_size
            block = weight_fp32[ni, k_start:k_end]
            amax = block.abs().max().item()
            scale = amax / 7.0 if amax > 0 else 1.0
            scales[ni, g] = scale

            for kk in range(k_start, k_end, 8):
                word = 0
                for nib in range(8):
                    q = int(round(weight_fp32[ni, kk + nib].item() / scale))
                    q = max(-8, min(7, q))
                    raw = q & 0x0F
                    word |= raw << (nib * 4)
                if word >= 2**31:
                    word -= 2**32
                packed[ni, kk // 8] = word

    return packed, scales


def awq_sym_int4_dequantize(packed: "torch.Tensor", scales: "torch.Tensor", out_features: int, in_features: int):
    result = torch.zeros((out_features, in_features), dtype=torch.float32)
    for ni in range(out_features):
        for g in range(in_features // group_size):
            scale = scales[ni, g].item()
            k_start = g * group_size
            k_end = k_start + group_size
            for kk in range(k_start, k_end, 8):
                word = int(packed[ni, kk // 8].item())
                for nib in range(8):
                    raw = (word >> (nib * 4)) & 0x0F
                    q = raw - 16 if raw & 0x08 else raw
                    result[ni, kk + nib] = q * scale
    return result


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
def test_moe_awq_int4_accuracy():
    if not HAS_DEPS:
        pytest.skip(f"Dependencies not available: {import_error}")
    if not amx_utils._HAS_AVXVNNI256_AWQ_INT4_SUPPORT or not amx_utils._HOST_HAS_AVX_VNNI:
        pytest.skip("AVX-VNNI AWQ INT4 backend not available on this host")

    physical_to_logical_map = torch.arange(expert_num, dtype=torch.int64).contiguous()
    cpuinfer = kt_kernel_ext.CPUInfer(8)

    with torch.inference_mode():
        gate_bf16 = (torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32) / 10.0).to(
            torch.bfloat16
        )
        up_bf16 = (torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32) / 10.0).to(
            torch.bfloat16
        )
        down_bf16 = (torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.float32) / 10.0).to(
            torch.bfloat16
        )

        gate_qw_list, gate_scale_list = [], []
        up_qw_list, up_scale_list = [], []
        down_qw_list, down_scale_list = [], []

        for e in range(expert_num):
            qw, sc = awq_sym_int4_quantize(gate_bf16[e])
            gate_qw_list.append(qw)
            gate_scale_list.append(sc)

            qw, sc = awq_sym_int4_quantize(up_bf16[e])
            up_qw_list.append(qw)
            up_scale_list.append(sc)

            qw, sc = awq_sym_int4_quantize(down_bf16[e])
            down_qw_list.append(qw)
            down_scale_list.append(sc)

        gate_qw = torch.stack(gate_qw_list).contiguous()
        gate_scales = torch.stack(gate_scale_list).contiguous()
        up_qw = torch.stack(up_qw_list).contiguous()
        up_scales = torch.stack(up_scale_list).contiguous()
        down_qw = torch.stack(down_qw_list).contiguous()
        down_scales = torch.stack(down_scale_list).contiguous()

        gate_deq = torch.stack(
            [
                awq_sym_int4_dequantize(gate_qw_list[e], gate_scale_list[e], intermediate_size, hidden_size)
                for e in range(expert_num)
            ]
        )
        up_deq = torch.stack(
            [
                awq_sym_int4_dequantize(up_qw_list[e], up_scale_list[e], intermediate_size, hidden_size)
                for e in range(expert_num)
            ]
        )
        down_deq = torch.stack(
            [
                awq_sym_int4_dequantize(down_qw_list[e], down_scale_list[e], hidden_size, intermediate_size)
                for e in range(expert_num)
            ]
        )

        config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
        config.max_len = max_len
        config.gate_proj = gate_qw.data_ptr()
        config.up_proj = up_qw.data_ptr()
        config.down_proj = down_qw.data_ptr()
        config.gate_scale = gate_scales.data_ptr()
        config.up_scale = up_scales.data_ptr()
        config.down_scale = down_scales.data_ptr()
        config.quant_config.bits = 4
        config.quant_config.group_size = group_size
        config.quant_config.zero_point = False
        config.pool = cpuinfer.backend_

        moe = kt_kernel_ext.moe.AVXVNNI256AWQInt4_MOE(config)
        cpuinfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
        cpuinfer.sync()

        for i in range(validation_iter):
            qlen = 1 if i == 0 else 8
            bsz_tensor = torch.tensor([qlen], dtype=torch.int64)
            expert_ids = torch.stack(
                [torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]
            ).contiguous()
            weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
            input_data = (torch.randn((qlen, hidden_size), dtype=torch.float32) / 100.0).to(torch.bfloat16).contiguous()
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

            ref = moe_torch(input_data.float(), expert_ids, weights, gate_deq, up_deq, down_deq)
            diff = torch.mean(torch.abs(output.float() - ref)) / torch.mean(torch.abs(ref)).clamp_min(1e-6)
            print(f"Iteration {i}, diff = {diff:.6f}")
            assert diff < 0.35, f"AWQ INT4 accuracy test failed: diff={diff:.6f} >= 0.35"
