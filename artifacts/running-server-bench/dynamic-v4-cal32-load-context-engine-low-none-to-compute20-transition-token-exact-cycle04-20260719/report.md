# Running Server Benchmark

- Status: complete
- Run label: dynamic-v4-cal32-load-context__engine-low__none-to-compute20__transition-token-exact-cycle04
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 1 / 1

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 1 | 1046.0 | 600.0 | 170.21 [170.21, 170.21] | 19.09 [19.09, 19.09] | 6145.50 [6145.50, 6145.50] | 52.37 [52.37, 52.37] | 31369.82 | 37515.32 | 37515.45 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1024-o600 | 0.8283 | 0.6351 | 0.1698 | 0.0234 | 0.0952 |

## Scheduler Telemetry

Values are request means from the MoE layer or layers selected at server launch.

| Workload | Prefill iGPU ratio | Prefill CPU load | Prefill exploration | Decode iGPU ratio | Decode CPU load | Decode exploration |
|---|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 0.0000 | 0.0000 | 0.0000 | 0.7200 | 0.5785 | 0.0000 |

## Background Load Transition

- Direction: low-to-high
- Managed workers: 20
- Start after output token: 150
- Background ready delay: 63.95 ms
- Background stopped after request: True
- Decode calls before launch: 150
- First iGPU execution delay: 18 calls / 1346.08 ms
- Settled iGPU delay: 18 calls / 1346.08 ms
- Client pre-transition throughput: 29.67 token/s
- Client post-transition throughput: 17.08 token/s
- Final iGPU execution ratio: 1.0000

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
- [Per-request scheduler events](scheduler-telemetry.jsonl)
