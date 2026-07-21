# Running Server Benchmark

- Status: complete
- Run label: igpu-fixed__engine-low__compute8__smoke
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 6 / 6

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1-o300 | 3 | 1.0 | 300.0 | 9.97 [9.76, 10.27] | 24.12 [23.97, 24.22] | 100.35 [97.35, 102.50] | 41.45 [41.29, 41.72] | 12394.55 | 12494.90 | 12494.97 |
| p1024-o300 | 3 | 1046.0 | 300.0 | 167.54 [164.57, 169.23] | 23.48 [23.11, 23.89] | 6244.18 [6180.83, 6355.79] | 42.59 [41.85, 43.27] | 12734.26 | 18978.44 | 18978.50 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1-o300 | 0.8254 | 0.4065 | 0.4146 | 0.0044 | 0.0026 |
| p1024-o300 | 0.8308 | 0.4038 | 0.4162 | 0.0108 | 0.0045 |

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
- [Per-request scheduler events](scheduler-telemetry.jsonl)
