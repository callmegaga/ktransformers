# Running Server Benchmark

- Status: complete
- Run label: dynamic-v5-reprobe-ceiling__engine-low__compute20-to-none__transition-token-exact-cycle03
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 1 / 1

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 1 | 1046.0 | 600.0 | 142.28 [142.28, 142.28] | 23.57 [23.57, 23.57] | 7351.56 [7351.56, 7351.56] | 42.43 [42.43, 42.43] | 25412.63 | 32764.20 | 32764.27 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1024-o600 | 0.7111 | 0.4052 | 0.2919 | 0.0140 | 0.1274 |

## Scheduler Telemetry

Values are request means from the MoE layer or layers selected at server launch.

| Workload | Prefill iGPU ratio | Prefill CPU load | Prefill exploration | Decode iGPU ratio | Decode CPU load | Decode exploration |
|---|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 0.0000 | 0.8579 | 0.0000 | 0.2717 | 0.2026 | 0.0533 |

## Background Load Transition

- Direction: high-to-low
- Signalled PID: 906355
- Stop after output token: 150
- Background stopped: True
- Decode calls before signal: 151
- First CPU execution delay: 12 calls / 629.74 ms
- Settled CPU delay: 44 calls / 1725.05 ms
- Client pre-transition throughput: 15.51 token/s
- Client post-transition throughput: 28.47 token/s
- Final iGPU execution ratio: 0.0000

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
- [Per-request scheduler events](scheduler-telemetry.jsonl)
