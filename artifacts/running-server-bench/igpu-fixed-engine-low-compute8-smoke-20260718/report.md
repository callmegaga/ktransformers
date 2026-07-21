# Running Server Benchmark

- Status: complete
- Run label: igpu-fixed__engine-low__compute8__smoke
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 6 / 6

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1-o300 | 3 | 1.0 | 300.0 | 6.57 [4.83, 7.51] | 20.18 [19.96, 20.48] | 158.72 [131.92, 206.89] | 49.56 [48.82, 50.48] | 14817.11 | 14975.84 | 14975.90 |
| p1024-o300 | 3 | 1046.0 | 300.0 | 89.04 [88.92, 89.18] | 20.52 [20.26, 20.67] | 11747.95 [11729.10, 11763.46] | 48.74 [48.38, 49.36] | 14573.36 | 26321.32 | 26321.39 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1-o300 | 0.4863 | 0.4044 | 0.0572 | 0.0247 | 0.0007 |
| p1024-o300 | 0.5172 | 0.3964 | 0.0784 | 0.0423 | 0.0383 |

## Scheduler Telemetry

Values are request means from the representative MoE layer selected at server launch.

| Workload | Prefill iGPU ratio | Prefill CPU load | Decode iGPU ratio | Decode CPU load |
|---|---:|---:|---:|---:|
| p1-o300 | NA | NA | 1.0000 | 0.0000 |
| p1024-o300 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
- [Per-request scheduler events](scheduler-telemetry.jsonl)
