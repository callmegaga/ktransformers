# Running Server Benchmark

- Status: complete
- Run label: dynamic-v5-reprobe-ceiling__engine-low__compute20-to-none__transition-token-exact-cycle04
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 1 / 1

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 1 | 1046.0 | 600.0 | 146.60 [146.60, 146.60] | 24.07 [24.07, 24.07] | 7135.12 [7135.12, 7135.12] | 41.55 [41.55, 41.55] | 24886.86 | 32021.98 | 32022.05 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1024-o600 | 0.7112 | 0.4027 | 0.2928 | 0.0157 | 0.1325 |

## Scheduler Telemetry

Values are request means from the MoE layer or layers selected at server launch.

| Workload | Prefill iGPU ratio | Prefill CPU load | Prefill exploration | Decode iGPU ratio | Decode CPU load | Decode exploration |
|---|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 0.0000 | 0.8192 | 0.0000 | 0.2750 | 0.2012 | 0.0533 |

## Background Load Transition

- Direction: high-to-low
- Signalled PID: 908329
- Stop after output token: 150
- Background stopped: True
- Decode calls before signal: 150
- First CPU execution delay: 15 calls / 786.43 ms
- Settled CPU delay: 47 calls / 1872.48 ms
- Client pre-transition throughput: 16.08 token/s
- Client post-transition throughput: 28.81 token/s
- Final iGPU execution ratio: 0.0000

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
- [Per-request scheduler events](scheduler-telemetry.jsonl)
