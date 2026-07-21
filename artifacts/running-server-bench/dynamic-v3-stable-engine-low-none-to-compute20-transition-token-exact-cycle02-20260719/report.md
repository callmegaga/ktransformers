# Running Server Benchmark

- Status: complete
- Run label: dynamic-v3-stable__engine-low__none-to-compute20__transition-token-exact-cycle02
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 1 / 1

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 1 | 1046.0 | 600.0 | 169.58 [169.58, 169.58] | 18.25 [18.25, 18.25] | 6168.06 [6168.06, 6168.06] | 54.81 [54.81, 54.81] | 32828.89 | 38996.95 | 38997.08 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1024-o600 | 0.8306 | 0.6491 | 0.1594 | 0.0221 | 0.0926 |

## Scheduler Telemetry

Values are request means from the MoE layer or layers selected at server launch.

| Workload | Prefill iGPU ratio | Prefill CPU load | Prefill exploration | Decode iGPU ratio | Decode CPU load | Decode exploration |
|---|---:|---:|---:|---:|---:|---:|
| p1024-o600 | NA | 0.0000 | 0.0000 | 0.7450 | 0.5958 | 0.0000 |

## Background Load Transition

- Direction: low-to-high
- Managed workers: 20
- Start after output token: 150
- Background ready delay: 43.22 ms
- Background stopped after request: True
- Decode calls before launch: 150
- First iGPU execution delay: 3 calls / 840.06 ms
- Settled iGPU delay: 3 calls / 840.06 ms
- Client pre-transition throughput: 29.59 token/s
- Client post-transition throughput: 16.19 token/s
- Final iGPU execution ratio: 1.0000

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
- [Per-request scheduler events](scheduler-telemetry.jsonl)
