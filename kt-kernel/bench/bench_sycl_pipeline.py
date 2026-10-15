#!/usr/bin/env python3
"""Diagnose a full XMX MoE chain; profiling results are NOT production timings.

Use --trace device with KT_SYCL_PROFILE=1 for deferred event queries. Host
clocks, device clocks and CUDA event clocks are never mixed. Variants run in
randomized, interleaved order on common weights/inputs/routes. CUDA mode
includes pinned staging, CPUInfer stream callbacks and both transfer directions.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import statistics
import time

import torch
from kt_kernel import kt_kernel_ext as ext
from bench_sycl_xmx_moe import packed_weights

VARIANTS = {
    'baseline': {},
    'prefill-gather-merge': {'KT_SYCL_XMX_PREFILL_N': '32', 'KT_SYCL_XMX_PREFILL_DOWN_ROWS': '16', 'KT_SYCL_XMX_PREFILL_LOAD': 'gather', 'KT_SYCL_XMX_PREFILL_MERGE': 'vector'},
    'prefill-vector-merge': {'KT_SYCL_XMX_PREFILL_N': '32', 'KT_SYCL_XMX_PREFILL_DOWN_ROWS': '16', 'KT_SYCL_XMX_PREFILL_MERGE': 'vector'},
    'prefill-gather': {'KT_SYCL_XMX_PREFILL_N': '32', 'KT_SYCL_XMX_PREFILL_DOWN_ROWS': '16', 'KT_SYCL_XMX_PREFILL_LOAD': 'gather'},
    'prefill-g4-gather': {'KT_SYCL_XMX_PREFILL_N': '32', 'KT_SYCL_XMX_PREFILL_ROWS': '4', 'KT_SYCL_XMX_PREFILL_DOWN_ROWS': '16', 'KT_SYCL_XMX_PREFILL_LOAD': 'gather'},
    'prefill-g16-gather': {'KT_SYCL_XMX_PREFILL_N': '32', 'KT_SYCL_XMX_PREFILL_ROWS': '16', 'KT_SYCL_XMX_PREFILL_DOWN_ROWS': '16', 'KT_SYCL_XMX_PREFILL_LOAD': 'gather'},
    'prefill-d8-gather': {'KT_SYCL_XMX_PREFILL_N': '32', 'KT_SYCL_XMX_PREFILL_DOWN_ROWS': '8', 'KT_SYCL_XMX_PREFILL_LOAD': 'gather'},
    'prefill-m16': {'KT_SYCL_XMX_PREFILL_ROWS': '16'},
    'prefill-g8d16n32': {'KT_SYCL_XMX_PREFILL_N': '32', 'KT_SYCL_XMX_PREFILL_DOWN_ROWS': '16'},
    'prefill-n32': {'KT_SYCL_XMX_PREFILL_N': '32'},
    'prefill-m16n32': {'KT_SYCL_XMX_PREFILL_N': '32', 'KT_SYCL_XMX_PREFILL_ROWS': '16'},
    'prefill-m32': {'KT_SYCL_XMX_PREFILL_ROWS': '32'},
    'split': {'KT_SYCL_XMX_DECODE_GATE_KERNEL': 'xmx-split', 'KT_SYCL_XMX_GATE_SPLITS': '2'},
    'fused-merge': {'KT_SYCL_XMX_DECODE_MERGE': 'fused'},
    'split-fused-merge': {'KT_SYCL_XMX_DECODE_MERGE': 'fused', 'KT_SYCL_XMX_DECODE_GATE_KERNEL': 'xmx-split'},
    'fused-expert': {'KT_SYCL_XMX_FUSED_EXPERT': '1'},
    'split-fused-expert': {'KT_SYCL_XMX_FUSED_EXPERT': '1', 'KT_SYCL_XMX_DECODE_GATE_KERNEL': 'xmx-split'},
    'split-fused-expert-host': {'KT_SYCL_XMX_FUSED_EXPERT': '1', 'KT_SYCL_XMX_DECODE_GATE_KERNEL': 'xmx-split', 'KT_SYCL_XMX_HOST_QUANT': '1'},
    'split-fused-expert-graph': {'KT_SYCL_XMX_FUSED_EXPERT': '1', 'KT_SYCL_XMX_DECODE_GATE_KERNEL': 'xmx-split', 'KT_SYCL_XMX_GRAPH': '1'},
    'split-fused-expert-host-graph': {'KT_SYCL_XMX_FUSED_EXPERT': '1', 'KT_SYCL_XMX_DECODE_GATE_KERNEL': 'xmx-split', 'KT_SYCL_XMX_HOST_QUANT': '1', 'KT_SYCL_XMX_GRAPH': '1'},
    'direct': {'KT_SYCL_XMX_NUMA_DISPATCH': '0'},
    'host-quant': {'KT_SYCL_XMX_HOST_QUANT': '1'},
    'graph': {'KT_SYCL_XMX_GRAPH': '1'},
    'fused-quant': {'KT_SYCL_XMX_FUSED_QUANT': '1'},
}


def quantiles(samples):
    ordered = sorted(samples)
    return {'median_us': statistics.median(samples), 'p10_us': ordered[int(.1 * len(ordered))],
            'p90_us': ordered[min(len(ordered) - 1, int(.9 * len(ordered)))], 'samples_us': samples}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants', default='baseline,split,host-quant,fused-quant')
    p.add_argument('--trace', choices=['0', 'host', 'device'], default='device')
    p.add_argument('--transport', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--rows', default='1,17,64')
    p.add_argument('--experts', type=int, default=256)
    p.add_argument('--active', type=int, default=8, help='Top-k; mixed rows spread routes across all experts')
    p.add_argument('--samples', type=int, default=100)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--rotate', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    names, rows_list = args.variants.split(','), [int(n) for n in args.rows.split(',')]
    if not names or not set(names) <= VARIANTS.keys() or len(set(names)) != len(names):
        p.error('Invalid or duplicate variants')
    if args.samples <= 0 or args.warmup < 0 or min(rows_list) <= 0 or not 0 < args.active <= args.experts:
        p.error('Invalid dimensions or repetitions')
    if args.trace == 'device' and any(VARIANTS[name].get('KT_SYCL_XMX_GRAPH') == '1' for name in names):
        p.error('Recorded graph nodes have no individual event profiling; use --trace host')
    if args.transport == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA transport requires a NVIDIA GPU and CUDA-enabled torch')
    torch.set_num_threads(2)
    os.environ['KT_SYCL_XMX_TRACE_EVERY'] = '0'  # Logging is not part of interleaved timing.
    os.environ['KT_SYCL_XMX_TRACE'] = args.trace
    os.environ['KT_SYCL_PROFILE'] = '1' if args.trace == 'device' else '0'
    gen = torch.Generator().manual_seed(20261014)
    proj = [packed_weights(args.experts, 512, 2048, 128, gen),
            packed_weights(args.experts, 512, 2048, 128, gen),
            packed_weights(args.experts, 2048, 512, 128, gen)]
    cpu = ext.CPUInfer(10)
    mask = torch.zeros(args.experts, dtype=torch.bool)
    mapping = torch.arange(args.experts, dtype=torch.int64)
    executors, configs, settings = {}, {}, {}
    defaults = {'KT_SYCL_XMX_PREFILL_MERGE': 'scalar', 'KT_SYCL_XMX_PREFILL_LOAD': 'block', 'KT_SYCL_XMX_MODE': 'int8', 'KT_SYCL_XMX_GATE_KERNEL': 'xmx',
                'KT_SYCL_XMX_DOWN_KERNEL': 'xmx', 'KT_SYCL_XMX_DECODE_GATE_KERNEL': 'inherit',
                'KT_SYCL_XMX_DECODE_DOWN_KERNEL': 'inherit', 'KT_SYCL_XMX_GATE_SPLITS': '2',
                'KT_SYCL_XMX_DOWN_SPLITS': '1', 'KT_SYCL_XMX_FUSED_QUANT': 'prefill',
                'KT_SYCL_XMX_GRAPH': '0', 'KT_SYCL_XMX_HOST_QUANT': '0',
                'KT_SYCL_XMX_NUMA_DISPATCH': '1', 'KT_SYCL_XMX_DECODE_MERGE': 'separate', 'KT_SYCL_XMX_FUSED_EXPERT': '0', 'KT_SYCL_XMX_PREFILL_ROWS': '8', 'KT_SYCL_XMX_PREFILL_N': '16', 'KT_SYCL_XMX_PREFILL_DOWN_ROWS': '8'}
    for name in names:
        settings[name] = defaults | VARIANTS[name]
        if 'KT_SYCL_XMX_PREFILL_DOWN_ROWS' not in VARIANTS[name]:
            settings[name]['KT_SYCL_XMX_PREFILL_DOWN_ROWS'] = settings[name]['KT_SYCL_XMX_PREFILL_ROWS']
        os.environ.update(settings[name])
        config = ext.moe.MOEConfig(args.experts, args.active, 2048, 512, mask.data_ptr())
        config.pool = cpu.backend_
        config.max_len = max(rows_list)
        config.quant_config.bits = 4
        config.quant_config.group_size = 128
        config.quant_config.zero_point = False
        for stage, (qw, sc) in zip(('gate', 'up', 'down'), proj):
            setattr(config, stage + '_projs', [[t.data_ptr() for t in qw]])
            setattr(config, stage + '_scales', [[t.data_ptr() for t in sc]])
        moe = ext.moe.SYCLXMXGPTQInt4_MOE(config)
        cpu.submit(moe.load_weights_task(mapping.data_ptr())); cpu.sync()
        executors[name], configs[name] = moe, config
    random_order = random.Random(20261014)
    records = []
    for rows in rows_list:
        pinned = args.transport == 'cuda'
        def allocate(shape, dtype):
            return torch.empty(shape, dtype=dtype, pin_memory=pinned)
        qlen = torch.tensor([rows], dtype=torch.int32)
        x = allocate((rows, 2048), torch.bfloat16)
        x.copy_(torch.randn(rows, 2048, generator=gen).bfloat16())
        ids = allocate((rows, args.active), torch.int64)
        routing = allocate((rows, args.active), torch.float32)
        routing.copy_(torch.softmax(torch.randn(rows, args.active, generator=gen), 1))
        out = allocate((rows, 2048), torch.bfloat16)
        base = (torch.arange(rows * args.active).reshape(rows, args.active) % args.experts).contiguous()
        if pinned:
            stream = torch.cuda.Stream()
            dx, dr, di, dy = x.cuda(), routing.cuda(), ids.cuda(), out.cuda()
            torch.cuda.synchronize()
        times = {name: [] for name in names}
        cuda_times = {name: [] for name in names}
        reference = None
        for iteration in range(-args.warmup, args.samples):
            ids.copy_((base + max(0, iteration) * args.active) % args.experts if args.rotate else base)
            if pinned:
                di.copy_(ids); torch.cuda.synchronize()  # Route generation outside timing.
            if iteration == 0:
                for moe in executors.values(): moe.reset_profile()
            order = list(names); random_order.shuffle(order)
            for name in order:
                moe = executors[name]
                if pinned:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                start = time.perf_counter_ns()
                if pinned:
                    with torch.cuda.stream(stream):
                        start_event.record(stream)
                        x.copy_(dx, non_blocking=True)
                        ids.copy_(di, non_blocking=True)
                        routing.copy_(dr, non_blocking=True)
                        cpu.submit_with_cuda_stream(stream.cuda_stream, moe.forward_task(
                            qlen.data_ptr(), args.active, ids.data_ptr(), routing.data_ptr(),
                            x.data_ptr(), out.data_ptr(), False))
                        cpu.sync_with_cuda_stream(stream.cuda_stream)
                        dy.copy_(out, non_blocking=True)
                        end_event.record(stream)
                    stream.synchronize()
                else:
                    cpu.submit(moe.forward_task(qlen.data_ptr(), args.active, ids.data_ptr(), routing.data_ptr(),
                                               x.data_ptr(), out.data_ptr(), False)); cpu.sync()
                elapsed = (time.perf_counter_ns() - start) / 1000
                if iteration >= 0:
                    times[name].append(elapsed)
                    if pinned: cuda_times[name].append(start_event.elapsed_time(end_event) * 1000)
                if pinned: torch.testing.assert_close(dy.cpu(), out, rtol=0, atol=0)
                if not bool(torch.isfinite(out).all()): raise RuntimeError('Nonfinite output: ' + name)
                # Same routes are compared within each iteration; reset every iteration.
                if name == order[0]: reference = out.clone()
                else:
                    relative_rmse = ((out.float() - reference.float()).square().mean().sqrt() /
                                     reference.float().square().mean().sqrt().clamp_min(1e-12))
                    if relative_rmse > .03:
                        raise AssertionError(f'Candidate mismatch {name}: relative RMSE {relative_rmse.item()}')
        for name, moe in executors.items():
            totals = moe.profile_snapshot()
            prefix = 'decode.' if rows == 1 else 'prefill.'
            calls = totals.get(prefix + 'calls', 0)
            record = {'variant': name, 'rows': rows, 'wall': quantiles(times[name]),
                      'profile_total': totals,
                      'trace_enabled': args.trace != '0',
                      'profile_mean': {k.removeprefix(prefix): v / calls for k, v in totals.items()
                                       if k.startswith(prefix)} if calls else {}}
            if pinned: record['cuda_chain'] = quantiles(cuda_times[name])
            records.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != 'profile_total'}), flush=True)
        gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'config': vars(args) | {'output': str(args.output)},
                                     'settings': settings, 'results': records}, indent=2))


if __name__ == '__main__': main()
