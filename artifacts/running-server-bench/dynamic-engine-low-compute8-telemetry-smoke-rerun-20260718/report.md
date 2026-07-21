# Running Server Benchmark

- Status: complete
- Run label: dynamic__engine-low__compute8__telemetry-smoke-rerun
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 6 / 6

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1-o300 | 3 | 1.0 | 300.0 | 10.42 [10.37, 10.48] | 23.71 [23.40, 24.03] | 95.98 [95.45, 96.66] | 42.19 [41.62, 43.31] | 12614.97 | 12710.95 | 12711.02 |
| p1024-o300 | 3 | 1046.0 | 300.0 | 167.95 [166.19, 168.84] | 23.78 [23.46, 24.04] | 6228.49 [6195.05, 6293.92] | 42.07 [41.59, 42.62] | 12577.45 | 18805.94 | 18806.01 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1-o300 | 0.8290 | 0.4094 | 0.4141 | 0.0055 | 0.0057 |
| p1024-o300 | 0.8408 | 0.4099 | 0.4186 | 0.0124 | 0.0047 |

## Scheduler Telemetry

Values are request means from the representative MoE layer selected at server launch.

| Workload | Prefill iGPU ratio | Prefill CPU load | Decode iGPU ratio | Decode CPU load |
|---|---:|---:|---:|---:|
| p1-o300 | NA | NA | 0.0000 | 0.0048 |
| p1024-o300 | 0.0000 | 0.0194 | 0.0000 | 0.0047 |

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
- [Per-request scheduler events](scheduler-telemetry.jsonl)
