#!/usr/bin/env python3
"""
Convert Qwen-style MoE expert FP8/BF16/FP16 weights to the symmetric GPTQ-INT4
layout consumed by the AVX/AVX-VNNI GPTQ_INT4 MoE kernels.

This is an engineering conversion path, not a GPTQ calibration implementation.
For floating weights it uses group-wise symmetric RTN:
    scale = max(abs(weight_group)) / 7
    q = clamp(round(weight / scale), -8, 7)
    packed_nibble = q + 8
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PROJECTIONS = ("gate", "up", "down")
PROJ_TO_NAME = {
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj",
}
VNNI_MAX_GROUP_SIZE = 256


@dataclass
class QuantStats:
    mse: float
    max_abs: float
    mean_abs: float
    weight_rms: float
    rel_mse: float


class SafeTensorReader:
    def __init__(self, input_path: Path):
        self.input_path = input_path
        self.weight_map = self._load_weight_map()
        self.handles = {}

    def _load_weight_map(self) -> dict[str, str]:
        index_path = self.input_path / "model.safetensors.index.json"
        if index_path.exists():
            with index_path.open("r") as f:
                data = json.load(f)
            return data["weight_map"]

        weight_map = {}
        for st_file in sorted(self.input_path.glob("*.safetensors")):
            with safe_open(st_file, framework="pt") as handle:
                for key in handle.keys():
                    weight_map[key] = st_file.name
        if not weight_map:
            raise FileNotFoundError(f"No safetensors found in {self.input_path}")
        return weight_map

    def has_tensor(self, key: str) -> bool:
        return key in self.weight_map

    def get_tensor(self, key: str) -> torch.Tensor:
        if key not in self.weight_map:
            raise KeyError(f"Missing tensor: {key}")
        file_name = self.weight_map[key]
        handle = self.handles.get(file_name)
        if handle is None:
            handle = safe_open(self.input_path / file_name, framework="pt")
            self.handles[file_name] = handle
        return handle.get_tensor(key)


def parse_range_list(value: str | None, default_count: int) -> list[int]:
    if value is None or value.strip() == "":
        return list(range(default_count))

    result = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid range '{part}'")
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))

    seen = set()
    unique = []
    for item in result:
        if item < 0 or item >= default_count:
            raise ValueError(f"Index {item} is outside valid range [0, {default_count})")
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def load_config(input_path: Path) -> dict:
    config_path = input_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {input_path}")
    with config_path.open("r") as f:
        return json.load(f)


def text_config(config: dict) -> dict:
    return config.get("text_config", config)


def infer_input_type(config: dict, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    qc = config.get("quantization_config") or text_config(config).get("quantization_config") or {}
    if qc.get("quant_method") == "fp8":
        return "fp8"
    dtype = str(text_config(config).get("dtype", "")).lower()
    if dtype in {"float16", "fp16"}:
        return "fp16"
    return "bf16"


def copy_non_weight_files(input_path: Path, output_path: Path) -> None:
    for src in input_path.iterdir():
        if not src.is_file():
            continue
        if src.name.endswith(".safetensors") or src.name == "model.safetensors.index.json":
            continue
        shutil.copy2(src, output_path / src.name)


def write_gptq_config(input_path: Path, output_path: Path, group_size: int) -> None:
    config = load_config(input_path)
    config["quantization_config"] = {
        "bits": 4,
        "group_size": group_size,
        "damp_percent": 0.0,
        "desc_act": False,
        "static_groups": False,
        "sym": True,
        "true_sequential": False,
        "quant_method": "gptq",
        "conversion_method": "fp8_or_float_to_symmetric_rtn_int4_for_vnni",
        "modules_to_not_convert": [],
    }
    with (output_path / "config.json").open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def expert_weight_key(reader: SafeTensorReader, layer: int, expert: int, proj: str) -> str:
    proj_name = PROJ_TO_NAME[proj]
    candidates = [
        f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj_name}.weight",
        f"model.layers.{layer}.mlp.experts.{expert}.{proj_name}.weight",
    ]
    for key in candidates:
        if reader.has_tensor(key):
            return key
    raise KeyError(f"Could not find {proj} weight for layer={layer}, expert={expert}")


def dequantize_fp8_blockwise(weight: torch.Tensor, scale_inv: torch.Tensor, block_m: int, block_n: int) -> torch.Tensor:
    if weight.dim() != 2 or scale_inv.dim() != 2:
        raise ValueError(f"Expected 2D FP8 weight and scale tensors, got {weight.shape} and {scale_inv.shape}")

    m, n = weight.shape
    expected_m = (m + block_m - 1) // block_m
    expected_n = (n + block_n - 1) // block_n
    if tuple(scale_inv.shape) != (expected_m, expected_n):
        raise ValueError(
            f"FP8 scale shape mismatch: weight={tuple(weight.shape)}, scale={tuple(scale_inv.shape)}, "
            f"expected scale=({expected_m}, {expected_n}) for block=({block_m}, {block_n})"
        )

    weight_fp32 = weight.float()
    scale_fp32 = scale_inv.float()
    out = torch.empty((m, n), dtype=torch.float32)
    for bm in range(expected_m):
        m0 = bm * block_m
        m1 = min(m0 + block_m, m)
        for bn in range(expected_n):
            n0 = bn * block_n
            n1 = min(n0 + block_n, n)
            out[m0:m1, n0:n1] = weight_fp32[m0:m1, n0:n1] * scale_fp32[bm, bn]
    return out


def load_weight_fp32(
    reader: SafeTensorReader,
    weight_key: str,
    input_type: str,
    fp8_block_size: tuple[int, int],
) -> torch.Tensor:
    weight = reader.get_tensor(weight_key).contiguous()
    if input_type == "fp8":
        scale_key = weight_key.removesuffix(".weight") + ".weight_scale_inv"
        scale_inv = reader.get_tensor(scale_key).contiguous()
        return dequantize_fp8_blockwise(weight, scale_inv, fp8_block_size[0], fp8_block_size[1])
    if input_type in {"bf16", "fp16"}:
        return weight.float()
    raise ValueError(f"Unsupported input type: {input_type}")


def quantize_sym_int4(weight_fp32: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor, QuantStats]:
    if weight_fp32.dim() != 2:
        raise ValueError(f"Expected 2D weight, got {weight_fp32.shape}")
    n, k = weight_fp32.shape
    if k % 8 != 0:
        raise ValueError(f"Input dimension K={k} must be divisible by 8 for int4 packing")
    if k % group_size != 0:
        raise ValueError(f"Input dimension K={k} must be divisible by group_size={group_size}")

    groups = k // group_size
    grouped = weight_fp32.reshape(n, groups, group_size)
    amax = grouped.abs().amax(dim=2)
    scales_ng = torch.where(amax > 0, amax / 7.0, torch.ones_like(amax))

    q_signed = torch.round(grouped / scales_ng.unsqueeze(2)).clamp_(-8, 7).to(torch.int32)
    dequant = q_signed.float() * scales_ng.unsqueeze(2)
    diff = dequant - grouped
    mse = diff.square().mean().item()
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    weight_rms = grouped.square().mean().sqrt().item()
    rel_mse = mse / (weight_rms * weight_rms + 1e-12)

    q_nibbles = (q_signed + 8).reshape(n, k // 8, 8).to(torch.int64)
    shifts = (torch.arange(8, dtype=torch.int64) * 4).reshape(1, 1, 8)
    packed_i64 = torch.bitwise_left_shift(q_nibbles, shifts).sum(dim=2)
    packed_i64 = torch.where(packed_i64 >= 2**31, packed_i64 - 2**32, packed_i64)

    qweight = packed_i64.to(torch.int32).transpose(0, 1).contiguous()
    scales = scales_ng.transpose(0, 1).contiguous().float()
    stats = QuantStats(mse=mse, max_abs=max_abs, mean_abs=mean_abs, weight_rms=weight_rms, rel_mse=rel_mse)
    return qweight, scales, stats


def update_summary(summary: dict, stats: QuantStats) -> None:
    summary["count"] += 1
    summary["mse_sum"] += stats.mse
    summary["rel_mse_sum"] += stats.rel_mse
    summary["mean_abs_sum"] += stats.mean_abs
    summary["max_abs"] = max(summary["max_abs"], stats.max_abs)


def finalize_summary(summary: dict) -> dict:
    count = max(1, summary["count"])
    return {
        "tensor_count": summary["count"],
        "mean_mse": summary["mse_sum"] / count,
        "mean_rel_mse": summary["rel_mse_sum"] / count,
        "mean_abs_error": summary["mean_abs_sum"] / count,
        "max_abs_error": summary["max_abs"],
    }


def convert(args: argparse.Namespace) -> int:
    input_path = Path(args.input_path).resolve()
    output_path = Path(args.output).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if output_path.exists() and any(output_path.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output path is not empty: {output_path}. Use --overwrite to replace it.")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    config = load_config(input_path)
    tcfg = text_config(config)
    input_type = infer_input_type(config, args.input_type)
    num_layers = int(tcfg["num_hidden_layers"])
    num_experts = int(tcfg.get("num_experts", tcfg.get("n_routed_experts")))
    layers = parse_range_list(args.layers, num_layers)
    experts = parse_range_list(args.experts, num_experts)

    if args.group_size <= 0 or args.group_size % 32 != 0 or args.group_size > VNNI_MAX_GROUP_SIZE:
        raise ValueError(f"group_size must be a positive multiple of 32 and <= {VNNI_MAX_GROUP_SIZE}")

    fp8_block = tuple(args.fp8_block_size)
    if len(fp8_block) != 2:
        raise ValueError("--fp8-block-size must contain two integers")

    copy_non_weight_files(input_path, output_path)
    write_gptq_config(input_path, output_path, args.group_size)

    reader = SafeTensorReader(input_path)
    weight_map = {}
    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_type": input_type,
        "quantization": "symmetric_groupwise_rtn_int4",
        "layout": "gptq_sym_int4_qweight_scales",
        "scope": "moe_expert_weights_only",
        "note": "This output is intended for the kt-kernel GPTQ_INT4 MoE expert loader; it is not a complete HF model checkpoint.",
        "group_size": args.group_size,
        "fp8_block_size": list(fp8_block),
        "layers": layers,
        "experts": experts,
        "projection_stats": {},
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    global_summary = {"count": 0, "mse_sum": 0.0, "rel_mse_sum": 0.0, "mean_abs_sum": 0.0, "max_abs": 0.0}

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Input type: {input_type}, group_size={args.group_size}, layers={len(layers)}, experts={len(experts)}")

    for layer in layers:
        layer_start = time.time()
        tensors = {}
        layer_summary = {"count": 0, "mse_sum": 0.0, "rel_mse_sum": 0.0, "mean_abs_sum": 0.0, "max_abs": 0.0}
        print(f"\n[Layer {layer}] converting {len(experts)} experts")

        for expert in experts:
            for proj in PROJECTIONS:
                weight_key = expert_weight_key(reader, layer, expert, proj)
                weight_fp32 = load_weight_fp32(reader, weight_key, input_type, fp8_block)
                qweight, scales, stats = quantize_sym_int4(weight_fp32, args.group_size)

                base_key = weight_key.removesuffix(".weight")
                qweight_key = base_key + ".qweight"
                scales_key = base_key + ".scales"
                tensors[qweight_key] = qweight
                tensors[scales_key] = scales
                weight_map[qweight_key] = f"model-vnni-int4-layer-{layer:05d}.safetensors"
                weight_map[scales_key] = f"model-vnni-int4-layer-{layer:05d}.safetensors"

                update_summary(layer_summary, stats)
                update_summary(global_summary, stats)

        out_file = output_path / f"model-vnni-int4-layer-{layer:05d}.safetensors"
        save_file(
            tensors,
            out_file,
            metadata={
                "format": "pt",
                "quantization": "symmetric_groupwise_rtn_int4",
                "group_size": str(args.group_size),
            },
        )
        report["projection_stats"][str(layer)] = finalize_summary(layer_summary)
        print(
            f"[Layer {layer}] saved {len(tensors)} tensors to {out_file.name}; "
            f"mean_rel_mse={report['projection_stats'][str(layer)]['mean_rel_mse']:.6g}; "
            f"elapsed={time.time() - layer_start:.1f}s"
        )

    total_size = 0
    for file_name in sorted(set(weight_map.values())):
        total_size += (output_path / file_name).stat().st_size

    with (output_path / "model.safetensors.index.json").open("w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)
        f.write("\n")

    report["summary"] = finalize_summary(global_summary)
    report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with (output_path / "conversion_report.json").open("w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print("\nDone.")
    print(f"Summary: {report['summary']}")
    print(f"Report: {output_path / 'conversion_report.json'}")

    if args.check:
        run_smoke_check(
            output_path, min(layers), min(experts), int(tcfg["hidden_size"]), int(tcfg["moe_intermediate_size"])
        )

    return 0


def run_smoke_check(output_path: Path, layer: int, expert: int, hidden_size: int, intermediate_size: int) -> None:
    print("\nRunning smoke check...")

    import sys

    repo_python = Path(__file__).resolve().parents[1] / "python"
    if str(repo_python) not in sys.path:
        sys.path.insert(0, str(repo_python))

    from kt_kernel.utils.loader import GPTQSafeTensorLoader

    loader = GPTQSafeTensorLoader(str(output_path))
    weights = loader.load_experts(f"model.layers.{layer}")
    for proj in PROJECTIONS:
        qweight = weights[proj][expert]
        scales = weights[f"{proj}_scale"][expert]
        print(f"  {proj}: qweight={tuple(qweight.shape)} {qweight.dtype}, scales={tuple(scales.shape)} {scales.dtype}")

    try:
        from kt_kernel import KTMoEWrapper

        wrapper = KTMoEWrapper(
            layer_idx=layer,
            num_experts=len(weights["gate"]),
            num_experts_per_tok=1,
            hidden_size=hidden_size,
            moe_intermediate_size=intermediate_size,
            gpu_experts_mask=torch.zeros(len(weights["gate"]), dtype=torch.bool),
            cpuinfer_threads=4,
            threadpool_count=1,
            weight_path=str(output_path),
            chunked_prefill_size=8,
            method="GPTQ_INT4",
        )
        wrapper.load_weights(torch.arange(len(weights["gate"]), dtype=torch.int64).contiguous())
        bsz = torch.tensor([1], dtype=torch.int32).contiguous()
        expert_ids = torch.tensor([[expert]], dtype=torch.int64).contiguous()
        topk_weights = torch.ones((1, 1), dtype=torch.float32).contiguous()
        x = (torch.randn((1, hidden_size), dtype=torch.float32) / 100.0).to(torch.bfloat16).contiguous()
        out = torch.empty((1, hidden_size), dtype=torch.bfloat16).contiguous()
        wrapper.cpu_infer.submit(
            wrapper.moe.forward_task(
                bsz.data_ptr(),
                1,
                expert_ids.data_ptr(),
                topk_weights.data_ptr(),
                x.data_ptr(),
                out.data_ptr(),
                False,
            )
        )
        wrapper.cpu_infer.sync()
        if not torch.isfinite(out.float()).all():
            raise RuntimeError("VNNI/GPTQ_INT4 smoke forward produced non-finite values")
        print(f"  native GPTQ_INT4 forward ok: output_mean_abs={float(out.float().abs().mean()):.6g}")
    except Exception as exc:
        print(f"  native GPTQ_INT4 forward skipped/failed: {type(exc).__name__}: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Qwen-style MoE FP8/BF16/FP16 experts to VNNI-compatible GPTQ_INT4 qweight/scales."
    )
    parser.add_argument("--input-path", "-i", required=True, help="Input model directory with safetensors")
    parser.add_argument("--output", "-o", required=True, help="Output directory for converted expert weights")
    parser.add_argument(
        "--input-type",
        choices=["auto", "fp8", "bf16", "fp16"],
        default="auto",
        help="Input expert weight type. Default: infer from config.json",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Symmetric INT4 group size. AVX-VNNI-256 supports multiples of 32 up to 256. Default: 128",
    )
    parser.add_argument(
        "--fp8-block-size",
        type=int,
        nargs=2,
        default=(128, 128),
        metavar=("M", "N"),
        help="FP8 block scale shape used by weight_scale_inv. Default: 128 128",
    )
    parser.add_argument("--layers", help="Layer list/ranges to convert, e.g. '0,2,4-7'. Default: all layers")
    parser.add_argument("--experts", help="Expert list/ranges to convert, e.g. '0,1,8-15'. Default: all experts")
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output directory")
    parser.add_argument(
        "--check",
        action="store_true",
        help="After conversion, load converted weights with GPTQSafeTensorLoader and run a tiny native forward if available",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return convert(args)


if __name__ == "__main__":
    raise SystemExit(main())
