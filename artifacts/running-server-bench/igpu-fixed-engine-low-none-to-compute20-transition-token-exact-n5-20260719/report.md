# Running Server Benchmark

- Status: complete
- Run label: igpu-fixed__engine-low__none-to-compute20__transition-token-exact-n5
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 5 / 5

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 5 | 1046.0 | 600.0 | 83.36 [79.92, 87.73] | 13.94 [13.48, 14.42] | 12584.78 [11995.63, 13095.97] | 71.84 [69.33, 74.27] | 43033.03 | 55617.81 | 55620.41 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1024-o600 | 0.6648 | 0.5698 | 0.0600 | 0.0350 | 0.0710 |

## Background Load Transition

- Direction: low-to-high
- Transition samples: 5
- Managed workers: 20
- Start after output token: 150
- Background ready delay: 77.61 [64.08, 89.31] ms
- Background stopped after request: True
- Decode calls before launch: NA
- First iGPU execution delay: NA calls / NA ms
- Settled iGPU delay: NA calls / NA ms
- Client pre-transition throughput: 16.31 [15.29, 17.10] token/s
- Client post-transition throughput: 13.33 [12.72, 13.88] token/s
- Final iGPU execution ratio: NA

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
