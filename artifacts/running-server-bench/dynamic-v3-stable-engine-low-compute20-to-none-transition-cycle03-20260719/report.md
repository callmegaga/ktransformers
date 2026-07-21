# Running Server Benchmark

- Status: complete
- Run label: dynamic-v3-stable__engine-low__compute20-to-none__transition-cycle03
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 1 / 1

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 1 | 1046.0 | 600.0 | 148.75 [148.75, 148.75] | 24.56 [24.56, 24.56] | 7031.84 [7031.84, 7031.84] | 40.72 [40.72, 40.72] | 24394.03 | 31425.87 | 31425.94 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1024-o600 | 0.7096 | 0.3927 | 0.3026 | 0.0143 | 0.1334 |

## Scheduler Telemetry

Values are request means from the MoE layer or layers selected at server launch.

| Workload | Prefill iGPU ratio | Prefill CPU load | Prefill exploration | Decode iGPU ratio | Decode CPU load | Decode exploration |
|---|---:|---:|---:|---:|---:|---:|
| p1024-o600 | NA | 0.4988 | 0.0000 | 0.2600 | 0.2040 | 0.0533 |

## Background Load Transition

- Signalled PID: 788925
- Stop after output token: 150
- Background stopped: True
- Decode calls before signal: 150
- First CPU execution delay: 6 calls / 317.55 ms
- Settled CPU delay: 38 calls / 1419.13 ms
- Final iGPU execution ratio: 0.0000

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
- [Per-request scheduler events](scheduler-telemetry.jsonl)
