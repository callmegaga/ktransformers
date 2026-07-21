# Running Server Benchmark

- Status: complete
- Run label: dynamic-v4-cal32-load-context__engine-low__none-to-compute20__transition-token-exact-cycle03
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 1 / 1

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 1 | 1046.0 | 600.0 | 170.12 [170.12, 170.12] | 18.75 [18.75, 18.75] | 6148.78 [6148.78, 6148.78] | 53.32 [53.32, 53.32] | 31940.19 | 38088.96 | 38089.07 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1024-o600 | 0.8273 | 0.6474 | 0.1571 | 0.0228 | 0.0872 |

## Scheduler Telemetry

Values are request means from the MoE layer or layers selected at server launch.

| Workload | Prefill iGPU ratio | Prefill CPU load | Prefill exploration | Decode iGPU ratio | Decode CPU load | Decode exploration |
|---|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 0.0000 | 0.0001 | 0.0000 | 0.7467 | 0.5797 | 0.0000 |

## Background Load Transition

- Direction: low-to-high
- Managed workers: 20
- Start after output token: 150
- Background ready delay: 50.70 ms
- Background stopped after request: True
- Decode calls before launch: 150
- First iGPU execution delay: 2 calls / 178.31 ms
- Settled iGPU delay: 2 calls / 178.31 ms
- Client pre-transition throughput: 29.62 token/s
- Client post-transition throughput: 16.72 token/s
- Final iGPU execution ratio: 1.0000

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
- [Per-request scheduler events](scheduler-telemetry.jsonl)
