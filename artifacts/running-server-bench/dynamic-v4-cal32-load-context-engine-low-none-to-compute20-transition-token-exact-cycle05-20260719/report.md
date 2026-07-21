# Running Server Benchmark

- Status: complete
- Run label: dynamic-v4-cal32-load-context__engine-low__none-to-compute20__transition-token-exact-cycle05
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 1 / 1

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 1 | 1046.0 | 600.0 | 170.15 [170.15, 170.15] | 15.21 [15.21, 15.21] | 6147.53 [6147.53, 6147.53] | 65.74 [65.74, 65.74] | 39380.68 | 45528.22 | 45528.36 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1024-o600 | 0.8529 | 0.6251 | 0.2047 | 0.0231 | 0.1668 |

## Scheduler Telemetry

Values are request means from the MoE layer or layers selected at server launch.

| Workload | Prefill iGPU ratio | Prefill CPU load | Prefill exploration | Decode iGPU ratio | Decode CPU load | Decode exploration |
|---|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 0.0000 | 0.0001 | 0.0000 | 0.6917 | 0.5509 | 0.0533 |

## Background Load Transition

- Direction: low-to-high
- Managed workers: 20
- Start after output token: 150
- Background ready delay: 185.52 ms
- Background stopped after request: True
- Decode calls before launch: 150
- First iGPU execution delay: 3 calls / 821.71 ms
- Settled iGPU delay: 3 calls / 821.71 ms
- Client pre-transition throughput: 29.66 token/s
- Client post-transition throughput: 13.10 token/s
- Final iGPU execution ratio: 1.0000

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
- [Per-request scheduler events](scheduler-telemetry.jsonl)
