# Running Server Benchmark

- Status: complete
- Run label: packed-cpu-fixed__engine-low__none-to-compute20__transition-smoke
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 1 / 1

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 1 | 1046.0 | 600.0 | 168.08 [168.08, 168.08] | 6.43 [6.43, 6.43] | 6223.40 [6223.40, 6223.40] | 155.50 [155.50, 155.50] | 93142.90 | 99366.31 | 99367.79 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1024-o600 | 0.9360 | 0.5534 | 0.3623 | 0.0203 | 0.3982 |

## Background Load Transition

- Direction: low-to-high
- Managed workers: 20
- Start after output token: 150
- Background ready delay: 46.76 ms
- Background stopped after request: True
- Decode calls before launch: NA
- First iGPU execution delay: NA calls / NA ms
- Settled iGPU delay: NA calls / NA ms
- Client pre-transition throughput: 29.82 token/s
- Client post-transition throughput: 5.11 token/s
- Final iGPU execution ratio: NA

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
