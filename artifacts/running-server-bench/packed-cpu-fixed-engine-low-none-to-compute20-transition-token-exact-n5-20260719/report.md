# Running Server Benchmark

- Status: complete
- Run label: packed-cpu-fixed__engine-low__none-to-compute20__transition-token-exact-n5
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 5 / 5

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 5 | 1046.0 | 600.0 | 169.43 [168.77, 169.89] | 8.79 [8.20, 9.44] | 6173.68 [6157.46, 6197.46] | 114.57 [105.00, 122.37] | 68630.17 | 74803.86 | 74804.21 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1024-o600 | 0.9141 | 0.5217 | 0.3751 | 0.0173 | 0.3694 |

## Background Load Transition

- Direction: low-to-high
- Transition samples: 5
- Managed workers: 20
- Start after output token: 150
- Background ready delay: 207.29 [45.51, 526.13] ms
- Background stopped after request: True
- Decode calls before launch: NA
- First iGPU execution delay: NA calls / NA ms
- Settled iGPU delay: NA calls / NA ms
- Client pre-transition throughput: 29.22 [29.14, 29.29] token/s
- Client post-transition throughput: 7.15 [6.62, 7.83] token/s
- Final iGPU execution ratio: NA

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
