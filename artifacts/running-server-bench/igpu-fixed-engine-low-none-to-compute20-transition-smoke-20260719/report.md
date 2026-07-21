# Running Server Benchmark

- Status: complete
- Run label: igpu-fixed__engine-low__none-to-compute20__transition-smoke
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 1 / 1

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 1 | 1046.0 | 600.0 | 85.16 [85.16, 85.16] | 14.61 [14.61, 14.61] | 12282.51 [12282.51, 12282.51] | 68.47 [68.47, 68.47] | 41012.78 | 53295.29 | 53295.39 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1024-o600 | 0.6638 | 0.5623 | 0.0646 | 0.0369 | 0.0712 |

## Background Load Transition

- Direction: low-to-high
- Managed workers: 20
- Start after output token: 150
- Background ready delay: 89.43 ms
- Background stopped after request: True
- Decode calls before launch: NA
- First iGPU execution delay: NA calls / NA ms
- Settled iGPU delay: NA calls / NA ms
- Client pre-transition throughput: 17.09 token/s
- Client post-transition throughput: 13.93 token/s
- Final iGPU execution ratio: NA

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
