# Running Server Benchmark

- Status: complete
- Run label: packed-cpu-fixed__engine-low__compute8__smoke
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 6 / 6

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1-o300 | 3 | 1.0 | 300.0 | 9.54 [8.66, 10.43] | 24.35 [24.25, 24.40] | 106.81 [95.92, 123.38] | 41.07 [40.97, 41.24] | 12278.97 | 12385.79 | 12385.85 |
| p1024-o300 | 3 | 1046.0 | 300.0 | 168.70 [166.62, 169.91] | 24.37 [24.32, 24.41] | 6200.82 [6156.09, 6277.87] | 41.04 [40.96, 41.11] | 12269.74 | 18470.56 | 18470.63 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1-o300 | 0.8211 | 0.4034 | 0.4145 | 0.0032 | 0.0011 |
| p1024-o300 | 0.8308 | 0.4026 | 0.4184 | 0.0098 | 0.0012 |

## Scheduler Telemetry

Values are request means from the representative MoE layer selected at server launch.

| Workload | Prefill iGPU ratio | Prefill CPU load | Decode iGPU ratio | Decode CPU load |
|---|---:|---:|---:|---:|
| p1-o300 | NA | NA | 0.0000 | 0.0000 |
| p1024-o300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
- [Per-request scheduler events](scheduler-telemetry.jsonl)
