#!/usr/bin/env python3
"""Randomized, interleaved full-MoE decode comparison, including quantize/merge/CPUInfer.

Keeps several executors resident to randomize variant order on the same inputs.
Fixed routes are not guaranteed cache-resident across all executors; rotating
routes cover the expert set. Run without another GPU test or compilation.
"""
import argparse
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
    'xmx-int8': ('int8', 'xmx', 'xmx'),
    'xmx-fp16': ('fp16', 'xmx', 'xmx'),
    'dp4a': ('int8', 'dp4a', 'dp4a'),
    'vector': ('fp16', 'vector', 'vector'),
    'dp4a-xmx': ('int8', 'dp4a', 'xmx'),
    'vector-xmx': ('int8', 'vector', 'xmx'),
    'xmx-split-xmx': ('int8', 'xmx-split', 'xmx'),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants', default=','.join(VARIANTS))
    p.add_argument('--experts', type=int, default=256)
    p.add_argument('--active', default='1,2,4,8')
    p.add_argument('--samples', type=int, default=100)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--rotate', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    names = args.variants.split(',')
    active_list = [int(x) for x in args.active.split(',')]
    if not set(names) <= VARIANTS.keys() or args.samples <= 0 or min(active_list) <= 0 or max(active_list) > args.experts:
        p.error('Invalid variants or sizes')
    torch.set_num_threads(2)
    gen = torch.Generator().manual_seed(20261014)
    proj = [packed_weights(args.experts, 512, 2048, 128, gen),
            packed_weights(args.experts, 512, 2048, 128, gen),
            packed_weights(args.experts, 2048, 512, 128, gen)]
    cpu = ext.CPUInfer(10)
    mask = torch.zeros(args.experts, dtype=torch.bool)
    mapping = torch.arange(args.experts, dtype=torch.int64)
    # Each constructor captures its settings; later environment changes cannot alter it.
    executors = {}
    configs = {}
    os.environ.update(KT_SYCL_XMX_FUSED_QUANT='0', KT_SYCL_XMX_GRAPH='0', KT_SYCL_XMX_HOST_QUANT='0',
                      KT_SYCL_XMX_NUMA_DISPATCH='1', KT_SYCL_XMX_DECODE_GATE_KERNEL='inherit',
                      KT_SYCL_XMX_DECODE_DOWN_KERNEL='inherit')
    for name in names:
        mode, gate, down = VARIANTS[name]
        os.environ.update(KT_SYCL_XMX_MODE=mode, KT_SYCL_XMX_GATE_KERNEL=gate, KT_SYCL_XMX_DOWN_KERNEL=down)
        config = ext.moe.MOEConfig(args.experts, max(active_list), 2048, 512, mask.data_ptr())
        config.pool = cpu.backend_
        config.max_len = 1
        config.quant_config.bits = 4
        config.quant_config.group_size = 128
        config.quant_config.zero_point = False
        for stage, (qw, sc) in zip(('gate', 'up', 'down'), proj):
            setattr(config, stage + '_projs', [[t.data_ptr() for t in qw]])
            setattr(config, stage + '_scales', [[t.data_ptr() for t in sc]])
        moe = ext.moe.SYCLXMXGPTQInt4_MOE(config)
        cpu.submit(moe.load_weights_task(mapping.data_ptr())); cpu.sync()
        executors[name] = moe
        configs[name] = config
    qlen = torch.tensor([1], dtype=torch.int32)
    x = torch.randn(1, 2048, generator=gen).bfloat16()
    out = torch.empty_like(x)
    random_order = random.Random(20261014)
    records = []
    for active in active_list:
        samples = {name: [] for name in names}
        routes = [((torch.arange(active) + n * active) % args.experts).reshape(1, active).contiguous()
                  for n in range(max(1, args.experts // active) if args.rotate else 1)]
        routing = torch.full((1, active), 1.0 / active)
        for iteration in range(-args.warmup, args.samples):
            order = list(names); random_order.shuffle(order)
            ids = routes[max(iteration, 0) % len(routes)]
            for name in order:
                start = time.perf_counter_ns()
                cpu.submit(executors[name].forward_task(qlen.data_ptr(), active, ids.data_ptr(), routing.data_ptr(),
                                                       x.data_ptr(), out.data_ptr(), False))
                cpu.sync()
                elapsed = (time.perf_counter_ns() - start) / 1e6
                if iteration >= 0: samples[name].append(elapsed)
                if not bool(torch.isfinite(out).all()): raise RuntimeError('nonfinite output: ' + name)
        for name in names:
            times = samples[name]
            sorted_times = sorted(times)
            record = {'variant': name, 'active': active, 'median_ms': statistics.median(times),
                      'p10_ms': sorted_times[int(len(times) * .1)], 'p90_ms': sorted_times[int(len(times) * .9)],
                      'samples_ms': times}
            records.append(record); print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'config': vars(args) | {'output': str(args.output)},
        'environment': {key: os.getenv(key) for key in ('KT_SYCL_XMX_GATE_SPLITS', 'KT_SYCL_XMX_DOWN_SPLITS')},
        'results': records}, indent=2))


if __name__ == '__main__': main()
