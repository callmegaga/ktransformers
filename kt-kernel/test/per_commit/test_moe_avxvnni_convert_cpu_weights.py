#!/usr/bin/env python
# coding=utf-8
"""End-to-end AVX-VNNI offline conversion tests for convert_cpu_weights.py."""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

try:
    import torch
    from safetensors.torch import save_file
    from kt_kernel import KTMoEWrapper
    from kt_kernel.utils import amx as amx_utils
    from kt_kernel.utils.amx import AMXMoEWrapper

    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    import_error = str(e)


def _load_script_module(module_name: str, file_name: str):
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    spec = importlib.util.spec_from_file_location(module_name, scripts_dir / file_name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


if HAS_DEPS:
    try:
        CONVERT_CPU_WEIGHTS = _load_script_module("test_convert_cpu_weights", "convert_cpu_weights.py")
        MERGE_CPU_WEIGHTS = _load_script_module("test_merge_cpu_weights", "merge_cpu_weights.py")
        HAS_SCRIPT_DEPS = True
    except Exception as e:
        HAS_SCRIPT_DEPS = False
        script_import_error = str(e)
else:
    HAS_SCRIPT_DEPS = False
    script_import_error = ""


def act_fn(x):
    return x / (1.0 + torch.exp(-x))


def mlp_torch(input_tensor, gate_proj, up_proj, down_proj):
    gate_buf = torch.mm(input_tensor, gate_proj.t())
    up_buf = torch.mm(input_tensor, up_proj.t())
    intermediate = act_fn(gate_buf) * up_buf
    return torch.mm(intermediate, down_proj.t())


def moe_torch(input_tensor, expert_ids, weights, gate_proj, up_proj, down_proj):
    expert_num = gate_proj.shape[0]
    cnts = expert_ids.new_zeros((expert_ids.shape[0], expert_num))
    cnts.scatter_(1, expert_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = expert_ids.view(-1).argsort()
    sorted_tokens = input_tensor[idxs // expert_ids.shape[1]]

    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        expert_out = mlp_torch(sorted_tokens[start_idx:end_idx], gate_proj[i], up_proj[i], down_proj[i])
        outputs.append(expert_out)
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


def _require_avxvnni_backend(quant_method: str):
    if not HAS_DEPS:
        pytest.skip(f"Dependencies not available: {import_error}")
    if not HAS_SCRIPT_DEPS:
        pytest.skip(f"Script dependencies not available: {script_import_error}")
    if not amx_utils._HOST_HAS_AVX_VNNI:
        pytest.skip("Host CPU does not support AVX-VNNI")
    if quant_method == "avxvnni_int8" and not amx_utils._HAS_AVXVNNI256_INT8_SUPPORT:
        pytest.skip("AVX-VNNI INT8 backend not compiled")
    if quant_method == "avxvnni_int4" and not amx_utils._HAS_AVXVNNI256_INT4_SUPPORT:
        pytest.skip("AVX-VNNI INT4 backend not compiled")


def _backend_method(quant_method: str) -> str:
    return CONVERT_CPU_WEIGHTS.format_quant_backend_name(quant_method)


def _reset_safetensor_loader():
    AMXMoEWrapper._safetensor_loader_instance = None


def _write_config(model_dir: str, input_type: str, expert_num: int, hidden_size: int, intermediate_size: int):
    config = {
        "model_type": "deepseek_v3",
        "n_routed_experts": expert_num,
        "num_experts_per_tok": 2,
        "hidden_size": hidden_size,
        "moe_intermediate_size": intermediate_size,
    }
    if input_type == "fp8":
        config["quantization_config"] = {"fmt": "e4m3", "weight_block_size": [128, 128]}
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _make_source_weight(shape, input_type: str):
    base = (torch.randn(shape, dtype=torch.float32) / 16).to(torch.bfloat16).contiguous()
    if input_type == "bf16":
        return base, base, None
    if input_type == "fp16":
        stored = base.to(torch.float16).contiguous()
        return stored, stored.to(torch.bfloat16).contiguous(), None
    if input_type == "fp8":
        if not torch.cuda.is_available():
            pytest.skip("FP8 conversion path requires CUDA for Triton dequantization")
        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("Current PyTorch build does not expose float8_e4m3fn")
        stored = base.to(torch.float32).clamp(min=-240.0, max=240.0).to(torch.float8_e4m3fn).contiguous()
        ref = stored.to(torch.float32).to(torch.bfloat16).contiguous()
        scale_inv = torch.ones((1, 1), dtype=torch.float32).contiguous()
        return stored, ref, scale_inv
    raise ValueError(f"Unsupported input_type: {input_type}")


def _write_tiny_model(model_dir: str, input_type: str):
    expert_num = 4
    hidden_size = 128
    intermediate_size = 64
    _write_config(model_dir, input_type, expert_num, hidden_size, intermediate_size)

    tensors = {
        "model.norm.weight": torch.ones(hidden_size, dtype=torch.bfloat16),
    }
    gate_refs = []
    up_refs = []
    down_refs = []

    for expert_id in range(expert_num):
        gate_weight, gate_ref, gate_scale_inv = _make_source_weight((intermediate_size, hidden_size), input_type)
        up_weight, up_ref, up_scale_inv = _make_source_weight((intermediate_size, hidden_size), input_type)
        down_weight, down_ref, down_scale_inv = _make_source_weight((hidden_size, intermediate_size), input_type)

        base = f"model.layers.0.mlp.experts.{expert_id}"
        tensors[f"{base}.gate_proj.weight"] = gate_weight
        tensors[f"{base}.up_proj.weight"] = up_weight
        tensors[f"{base}.down_proj.weight"] = down_weight
        if gate_scale_inv is not None:
            tensors[f"{base}.gate_proj.weight_scale_inv"] = gate_scale_inv
            tensors[f"{base}.up_proj.weight_scale_inv"] = up_scale_inv
            tensors[f"{base}.down_proj.weight_scale_inv"] = down_scale_inv

        gate_refs.append(gate_ref)
        up_refs.append(up_ref)
        down_refs.append(down_ref)

    try:
        save_file(tensors, os.path.join(model_dir, "model.safetensors"))
    except Exception as e:
        if input_type == "fp8":
            pytest.skip(f"Saving FP8 safetensors is not supported in this environment: {e}")
        raise

    return {
        "expert_num": expert_num,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_experts_per_tok": 2,
        "gate_proj": torch.stack(gate_refs).contiguous(),
        "up_proj": torch.stack(up_refs).contiguous(),
        "down_proj": torch.stack(down_refs).contiguous(),
    }


def _run_converter_main(
    monkeypatch, input_path: str, output_path: str, input_type: str, quant_method: str, extra_args=None
):
    argv = [
        "convert_cpu_weights.py",
        "--input-path",
        input_path,
        "--input-type",
        input_type,
        "--output",
        output_path,
        "--quant-method",
        quant_method,
        "--cpuinfer-threads",
        "4",
        "--threadpool-count",
        "1",
    ]
    if extra_args:
        argv.extend(extra_args)
    monkeypatch.setattr(sys, "argv", argv)
    assert CONVERT_CPU_WEIGHTS.main() == 0


def _run_merge_main(monkeypatch, input_path: str, output_path: str, original_path: str):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_cpu_weights.py",
            "--input-path",
            input_path,
            "--output",
            output_path,
            "--original-path",
            original_path,
        ],
    )
    assert MERGE_CPU_WEIGHTS.main() == 0


def _run_inference(weight_path: str, backend_method: str, source_model: dict):
    expert_num = source_model["expert_num"]
    hidden_size = source_model["hidden_size"]
    intermediate_size = source_model["intermediate_size"]
    num_experts_per_tok = source_model["num_experts_per_tok"]

    physical_to_logical_map = torch.arange(expert_num, dtype=torch.int64)
    expert_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64).contiguous()
    routing_weights = torch.tensor([[0.6, 0.4], [0.55, 0.45]], dtype=torch.float32).contiguous()
    input_tensor = (torch.randn((2, hidden_size), dtype=torch.float32) / 16).to(torch.bfloat16).contiguous()
    output = torch.empty_like(input_tensor)
    bsz = torch.tensor([input_tensor.shape[0]], dtype=torch.int32)

    _reset_safetensor_loader()
    wrapper = KTMoEWrapper(
        layer_idx=0,
        num_experts=expert_num,
        num_experts_per_tok=num_experts_per_tok,
        hidden_size=hidden_size,
        moe_intermediate_size=intermediate_size,
        gpu_experts_mask=torch.zeros(expert_num, dtype=torch.bool),
        cpuinfer_threads=4,
        threadpool_count=1,
        weight_path=weight_path,
        chunked_prefill_size=16,
        method=backend_method,
    )
    wrapper.load_weights(physical_to_logical_map)
    wrapper.moe.forward(
        bsz.data_ptr(),
        num_experts_per_tok,
        expert_ids.data_ptr(),
        routing_weights.data_ptr(),
        input_tensor.data_ptr(),
        output.data_ptr(),
        False,
    )

    ref = moe_torch(
        input_tensor,
        expert_ids,
        routing_weights,
        source_model["gate_proj"],
        source_model["up_proj"],
        source_model["down_proj"],
    )
    diff = torch.mean(torch.abs(output.float() - ref.float())) / torch.mean(torch.abs(ref.float()))
    return output, diff.item()


@pytest.mark.cpu
@pytest.mark.parametrize("quant_method", ["avxvnni_int8", "avxvnni_int4"])
@pytest.mark.parametrize("input_type", ["bf16", "fp16"])
def test_convert_cpu_weights_avxvnni_merged_roundtrip(monkeypatch, quant_method, input_type):
    _require_avxvnni_backend(quant_method)

    src_dir = tempfile.mkdtemp(prefix=f"kt_src_{input_type}_{quant_method}_")
    out_dir = tempfile.mkdtemp(prefix=f"kt_out_{input_type}_{quant_method}_")
    try:
        source_model = _write_tiny_model(src_dir, input_type)
        _run_converter_main(monkeypatch, src_dir, out_dir, input_type, quant_method)

        assert os.path.exists(os.path.join(out_dir, "model.safetensors"))
        with open(os.path.join(out_dir, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        assert config["amx_quantization"]["method"] == quant_method
        assert config["amx_quantization"]["backend"] == _backend_method(quant_method)

        output, diff = _run_inference(out_dir, _backend_method(quant_method), source_model)
        assert output.shape == (2, source_model["hidden_size"])
        assert torch.isfinite(output.float()).all()

        threshold = 0.05 if quant_method.endswith("int8") else 0.25
        assert diff < threshold, f"{quant_method} merged roundtrip diff={diff:.6f} >= {threshold:.2f}"
    finally:
        _reset_safetensor_loader()
        shutil.rmtree(src_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


@pytest.mark.cpu
@pytest.mark.parametrize("quant_method", ["avxvnni_int8", "avxvnni_int4"])
def test_merge_cpu_weights_avxvnni_roundtrip(monkeypatch, quant_method):
    _require_avxvnni_backend(quant_method)

    src_dir = tempfile.mkdtemp(prefix=f"kt_merge_src_{quant_method}_")
    raw_dir = tempfile.mkdtemp(prefix=f"kt_merge_raw_{quant_method}_")
    merged_dir = tempfile.mkdtemp(prefix=f"kt_merge_out_{quant_method}_")
    try:
        source_model = _write_tiny_model(src_dir, "bf16")
        _run_converter_main(monkeypatch, src_dir, raw_dir, "bf16", quant_method, extra_args=["--no-merge-safetensor"])
        assert os.path.exists(os.path.join(raw_dir, "_layer_0", "_numa_0"))

        _run_merge_main(monkeypatch, raw_dir, merged_dir, src_dir)
        assert os.path.exists(os.path.join(merged_dir, "model-00001.safetensors")) or os.path.exists(
            os.path.join(merged_dir, "model.safetensors")
        )

        with open(os.path.join(merged_dir, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        assert config["amx_quantization"]["method"] == quant_method
        assert config["amx_quantization"]["backend"] == _backend_method(quant_method)

        output, diff = _run_inference(merged_dir, _backend_method(quant_method), source_model)
        assert output.shape == (2, source_model["hidden_size"])
        threshold = 0.05 if quant_method.endswith("int8") else 0.25
        assert diff < threshold, f"{quant_method} merge script diff={diff:.6f} >= {threshold:.2f}"
    finally:
        _reset_safetensor_loader()
        shutil.rmtree(src_dir, ignore_errors=True)
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(merged_dir, ignore_errors=True)


@pytest.mark.cpu
@pytest.mark.parametrize("quant_method", ["avxvnni_int8", "avxvnni_int4"])
def test_convert_cpu_weights_avxvnni_fp8_roundtrip(monkeypatch, quant_method):
    _require_avxvnni_backend(quant_method)
    if not torch.cuda.is_available():
        pytest.skip("FP8 conversion path requires CUDA")

    src_dir = tempfile.mkdtemp(prefix=f"kt_src_fp8_{quant_method}_")
    out_dir = tempfile.mkdtemp(prefix=f"kt_out_fp8_{quant_method}_")
    try:
        source_model = _write_tiny_model(src_dir, "fp8")
        _run_converter_main(monkeypatch, src_dir, out_dir, "fp8", quant_method)
        output, diff = _run_inference(out_dir, _backend_method(quant_method), source_model)
        assert output.shape == (2, source_model["hidden_size"])
        threshold = 0.08 if quant_method.endswith("int8") else 0.30
        assert diff < threshold, f"{quant_method} fp8 roundtrip diff={diff:.6f} >= {threshold:.2f}"
    finally:
        _reset_safetensor_loader()
        shutil.rmtree(src_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)
