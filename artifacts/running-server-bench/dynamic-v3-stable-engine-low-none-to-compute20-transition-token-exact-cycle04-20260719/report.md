# Running Server Benchmark

- Status: complete
- Run label: dynamic-v3-stable__engine-low__none-to-compute20__transition-token-exact-cycle04
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 1 / 1

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 1 | 1046.0 | 600.0 | 170.11 [170.11, 170.11] | 18.77 [18.77, 18.77] | 6149.08 [6149.08, 6149.08] | 53.28 [53.28, 53.28] | 31915.01 | 38064.10 | 38064.22 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1024-o600 | 0.8269 | 0.6404 | 0.1641 | 0.0224 | 0.0914 |

## Scheduler Telemetry

Values are request means from the MoE layer or layers selected at server launch.

| Workload | Prefill iGPU ratio | Prefill CPU load | Prefill exploration | Decode iGPU ratio | Decode CPU load | Decode exploration |
|---|---:|---:|---:|---:|---:|---:|
| p1024-o600 | NA | 0.0001 | 0.0000 | 0.7450 | 0.5860 | 0.0000 |

## Background Load Transition

- Direction: low-to-high
- Managed workers: 20
- Start after output token: 150
- Background ready delay: 44.51 ms
- Background stopped after request: True
- Decode calls before launch: 150
- First iGPU execution delay: 3 calls / 886.60 ms
- Settled iGPU delay: 3 calls / 886.60 ms
- Client pre-transition throughput: 29.62 token/s
- Client post-transition throughput: 16.74 token/s
- Final iGPU execution ratio: 1.0000

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
- [Per-request scheduler events](scheduler-telemetry.jsonl)
