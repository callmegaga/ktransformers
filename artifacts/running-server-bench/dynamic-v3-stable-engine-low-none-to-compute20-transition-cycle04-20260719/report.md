# Running Server Benchmark

- Status: complete
- Run label: dynamic-v3-stable__engine-low__none-to-compute20__transition-cycle04
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 1 / 1

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 1 | 1046.0 | 600.0 | 169.68 [169.68, 169.68] | 18.66 [18.66, 18.66] | 6164.64 [6164.64, 6164.64] | 53.59 [53.59, 53.59] | 32099.11 | 38263.74 | 38264.15 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1024-o600 | 0.8237 | 0.6342 | 0.1673 | 0.0222 | 0.0984 |

## Scheduler Telemetry

Values are request means from the MoE layer or layers selected at server launch.

| Workload | Prefill iGPU ratio | Prefill CPU load | Prefill exploration | Decode iGPU ratio | Decode CPU load | Decode exploration |
|---|---:|---:|---:|---:|---:|---:|
| p1024-o600 | NA | 0.0000 | 0.0000 | 0.7383 | 0.5730 | 0.0000 |

## Background Load Transition

- Direction: low-to-high
- Managed workers: 20
- Start after output token: 150
- Background ready delay: 58.17 ms
- Background stopped after request: True
- Decode calls before launch: 150
- First iGPU execution delay: 7 calls / 1219.53 ms
- Settled iGPU delay: 7 calls / 1219.53 ms
- Final iGPU execution ratio: 1.0000

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
- [Per-request scheduler events](scheduler-telemetry.jsonl)
