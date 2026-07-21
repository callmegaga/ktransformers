# Running Server Benchmark

- Status: complete
- Run label: dynamic-v5-reprobe-ceiling__engine-low__none-to-compute20__transition-token-exact-cycle02
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 1 / 1

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 1 | 1046.0 | 600.0 | 170.41 [170.41, 170.41] | 16.38 [16.38, 16.38] | 6138.22 [6138.22, 6138.22] | 61.04 [61.04, 61.04] | 36565.53 | 42703.74 | 42703.86 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1024-o600 | 0.8461 | 0.6797 | 0.1473 | 0.0191 | 0.0986 |

## Scheduler Telemetry

Values are request means from the MoE layer or layers selected at server launch.

| Workload | Prefill iGPU ratio | Prefill CPU load | Prefill exploration | Decode iGPU ratio | Decode CPU load | Decode exploration |
|---|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 0.0000 | 0.0012 | 0.0000 | 0.7433 | 0.6036 | 0.0000 |

## Background Load Transition

- Direction: low-to-high
- Managed workers: 20
- Start after output token: 150
- Background ready delay: 623.30 ms
- Background stopped after request: True
- Decode calls before launch: 151
- First iGPU execution delay: 3 calls / 794.86 ms
- Settled iGPU delay: 3 calls / 794.86 ms
- Client pre-transition throughput: 28.38 token/s
- Client post-transition throughput: 14.37 token/s
- Final iGPU execution ratio: 1.0000

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
- [Per-request scheduler events](scheduler-telemetry.jsonl)
