# Running Server Benchmark

- Status: complete
- Run label: dynamic__engine-low__compute20__smoke
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 6 / 6

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1-o300 | 3 | 1.0 | 300.0 | 8.03 [6.32, 10.63] | 14.40 [14.31, 14.50] | 133.81 [94.11, 172.57] | 69.44 [68.99, 70.32] | 20763.00 | 20896.81 | 20896.92 |
| p1024-o300 | 3 | 1046.0 | 300.0 | 88.74 [85.96, 90.46] | 14.80 [13.88, 15.27] | 11792.59 [11563.49, 12168.34] | 67.72 [65.47, 72.03] | 20246.82 | 32039.40 | 32041.11 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1-o300 | 0.9949 | 0.9323 | 0.0416 | 0.0209 | 0.1364 |
| p1024-o300 | 0.9872 | 0.8889 | 0.0673 | 0.0309 | 0.1682 |

## Scheduler Telemetry

Values are request means from the representative MoE layer selected at server launch.

| Workload | Prefill iGPU ratio | Prefill CPU load | Decode iGPU ratio | Decode CPU load |
|---|---:|---:|---:|---:|
| p1-o300 | NA | NA | 1.0000 | 0.8190 |
| p1024-o300 | 1.0000 | 0.7387 | 1.0000 | 0.8164 |

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
- [Per-request scheduler events](scheduler-telemetry.jsonl)
