# Running Server Benchmark

- Status: complete
- Run label: dynamic-v5-reprobe-ceiling__engine-low__compute20-to-none__transition-token-exact-smoke
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 1 / 1

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 1 | 1046.0 | 600.0 | 146.48 [146.48, 146.48] | 22.90 [22.90, 22.90] | 7141.10 [7141.10, 7141.10] | 43.67 [43.67, 43.67] | 26160.11 | 33301.20 | 33301.27 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1024-o600 | 0.7065 | 0.4068 | 0.2856 | 0.0141 | 0.1495 |

## Scheduler Telemetry

Values are request means from the MoE layer or layers selected at server launch.

| Workload | Prefill iGPU ratio | Prefill CPU load | Prefill exploration | Decode iGPU ratio | Decode CPU load | Decode exploration |
|---|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 0.0000 | 0.8018 | 0.0000 | 0.2733 | 0.1874 | 0.0533 |

## Background Load Transition

- Direction: high-to-low
- Signalled PID: 901610
- Stop after output token: 150
- Background stopped: True
- Decode calls before signal: 150
- First CPU execution delay: 14 calls / 731.54 ms
- Settled CPU delay: 46 calls / 1849.65 ms
- Client pre-transition throughput: 14.46 token/s
- Client post-transition throughput: 28.38 token/s
- Final iGPU execution ratio: 0.0000

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
- [Per-request scheduler events](scheduler-telemetry.jsonl)
