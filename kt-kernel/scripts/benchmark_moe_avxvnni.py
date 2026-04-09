#!/usr/bin/env python3
"""Lightweight AVX-VNNI vs AMX MoE benchmark and output-alignment tool."""

import json
import time

import torch

import kt_kernel
from kt_kernel.utils import amx as amx_utils

kt_kernel_ext = kt_kernel.kt_kernel_ext
MOE_MOD = kt_kernel_ext.moe


def act_fn(x):
    return x / (1.0 + torch.exp(-x))


def mlp_torch(input_tensor, gate_proj, up_proj, down_proj):
    gate_buf = torch.mm(input_tensor, gate_proj.t())
    up_buf = torch.mm(input_tensor, up_proj.t())
    return torch.mm(act_fn(gate_buf) * up_buf, down_proj.t())


def moe_torch(input_tensor, expert_ids, weights, gate_proj, up_proj, down_proj):
    expert_num = gate_proj.shape[0]
    cnts = expert_ids.new_zeros((expert_ids.shape[0], expert_num))
    cnts.scatter_(1, expert_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = expert_ids.view(-1).argsort()
    sorted_tokens = input_tensor[idxs // expert_ids.shape[1]]

    outputs = []
    start_idx = 0
    for expert_idx, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        outputs.append(
            mlp_torch(
                sorted_tokens[start_idx:end_idx], gate_proj[expert_idx], up_proj[expert_idx], down_proj[expert_idx]
            )
        )
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    return (
        new_x.view(*expert_ids.shape, -1)
        .type(weights.dtype)
        .mul_(weights.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )


def build_case(expert_num=16, hidden_size=1024, intermediate_size=512, num_experts_per_tok=4, qlen=16):
    gate = (
        (torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32) / 16)
        .to(torch.bfloat16)
        .contiguous()
    )
    up = (
        (torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32) / 16)
        .to(torch.bfloat16)
        .contiguous()
    )
    down = (
        (torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.float32) / 16)
        .to(torch.bfloat16)
        .contiguous()
    )
    expert_ids = (
        torch.stack([torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)])
        .to(torch.int64)
        .contiguous()
    )
    weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
    input_tensor = (torch.randn((qlen, hidden_size), dtype=torch.float32) / 16).to(torch.bfloat16).contiguous()
    return {
        "gate": gate,
        "up": up,
        "down": down,
        "expert_ids": expert_ids,
        "weights": weights,
        "input": input_tensor,
        "physical_to_logical": torch.arange(expert_num, dtype=torch.int64).contiguous(),
        "expert_num": expert_num,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_experts_per_tok": num_experts_per_tok,
        "qlen": qlen,
    }


def create_moe(moe_cls, case, cpu_threads=8):
    cpuinfer = kt_kernel_ext.CPUInfer(cpu_threads)
    config = kt_kernel_ext.moe.MOEConfig(
        case["expert_num"],
        case["num_experts_per_tok"],
        case["hidden_size"],
        case["intermediate_size"],
        0,
    )
    config.max_len = case["qlen"]
    config.pool = cpuinfer.backend_
    config.gate_proj = case["gate"].data_ptr()
    config.up_proj = case["up"].data_ptr()
    config.down_proj = case["down"].data_ptr()
    moe = moe_cls(config)
    cpuinfer.submit(moe.load_weights_task(case["physical_to_logical"].data_ptr()))
    cpuinfer.sync()
    return cpuinfer, moe


def run_backend(moe_cls, case, warmup=20, iterations=100, cpu_threads=8):
    cpuinfer, moe = create_moe(moe_cls, case, cpu_threads=cpu_threads)
    output = torch.empty((case["qlen"], case["hidden_size"]), dtype=torch.bfloat16).contiguous()
    bsz = torch.tensor([case["qlen"]], dtype=torch.int32).contiguous()

    for _ in range(warmup):
        cpuinfer.submit(
            moe.forward_task(
                bsz.data_ptr(),
                case["num_experts_per_tok"],
                case["expert_ids"].data_ptr(),
                case["weights"].data_ptr(),
                case["input"].data_ptr(),
                output.data_ptr(),
                False,
            )
        )
        cpuinfer.sync()

    start = time.perf_counter()
    for _ in range(iterations):
        cpuinfer.submit(
            moe.forward_task(
                bsz.data_ptr(),
                case["num_experts_per_tok"],
                case["expert_ids"].data_ptr(),
                case["weights"].data_ptr(),
                case["input"].data_ptr(),
                output.data_ptr(),
                False,
            )
        )
        cpuinfer.sync()
    end = time.perf_counter()

    return output.clone(), (end - start) * 1e6 / iterations


def maybe_compare(label, baseline_label, baseline_available, baseline_cls, vnni_available, vnni_cls, case):
    if not vnni_available:
        print(f"{label}: skip, AVX-VNNI backend not available")
        return None

    reference = moe_torch(case["input"], case["expert_ids"], case["weights"], case["gate"], case["up"], case["down"])
    vnni_output, vnni_us = run_backend(vnni_cls, case)
    vnni_diff = (
        torch.mean(torch.abs(vnni_output.float() - reference.float())) / torch.mean(torch.abs(reference.float()))
    ).item()

    result = {
        "vnni_latency_us": vnni_us,
        "vnni_vs_torch_diff": vnni_diff,
    }

    if baseline_available:
        baseline_output, baseline_us = run_backend(baseline_cls, case)
        baseline_diff = (
            torch.mean(torch.abs(baseline_output.float() - reference.float()))
            / torch.mean(torch.abs(reference.float()))
        ).item()
        cross_diff = (
            torch.mean(torch.abs(vnni_output.float() - baseline_output.float()))
            / torch.mean(torch.abs(baseline_output.float()))
        ).item()
        result.update(
            {
                f"{baseline_label}_latency_us": baseline_us,
                f"{baseline_label}_vs_torch_diff": baseline_diff,
                f"vnni_vs_{baseline_label}_diff": cross_diff,
                f"vnni_speedup_vs_{baseline_label}": (baseline_us / vnni_us) if vnni_us > 0 else None,
            }
        )

    print(f"{label}: {json.dumps(result, ensure_ascii=False)}")
    return result


def main():
    if not amx_utils._HOST_HAS_AVX_VNNI:
        print("Host CPU does not support AVX-VNNI")
        return 0

    case_decode = build_case(qlen=1)
    case_prefill = build_case(qlen=32)

    print("Decode benchmark:")
    maybe_compare(
        "INT8",
        "avx2",
        getattr(MOE_MOD, "Int8_KERNEL_MOE", None) is not None,
        getattr(MOE_MOD, "Int8_KERNEL_MOE", None),
        amx_utils._HAS_AVXVNNI256_INT8_SUPPORT,
        getattr(MOE_MOD, "AVXVNNI256Int8_MOE", None),
        case_decode,
    )
    maybe_compare(
        "INT4",
        "avx2",
        getattr(MOE_MOD, "Int4_KERNEL_MOE", None) is not None,
        getattr(MOE_MOD, "Int4_KERNEL_MOE", None),
        amx_utils._HAS_AVXVNNI256_INT4_SUPPORT,
        getattr(MOE_MOD, "AVXVNNI256Int4_MOE", None),
        case_decode,
    )

    print("Prefill benchmark:")
    maybe_compare(
        "INT8",
        "avx2",
        getattr(MOE_MOD, "Int8_KERNEL_MOE", None) is not None,
        getattr(MOE_MOD, "Int8_KERNEL_MOE", None),
        amx_utils._HAS_AVXVNNI256_INT8_SUPPORT,
        getattr(MOE_MOD, "AVXVNNI256Int8_MOE", None),
        case_prefill,
    )
    maybe_compare(
        "INT4",
        "avx2",
        getattr(MOE_MOD, "Int4_KERNEL_MOE", None) is not None,
        getattr(MOE_MOD, "Int4_KERNEL_MOE", None),
        amx_utils._HAS_AVXVNNI256_INT4_SUPPORT,
        getattr(MOE_MOD, "AVXVNNI256Int4_MOE", None),
        case_prefill,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
