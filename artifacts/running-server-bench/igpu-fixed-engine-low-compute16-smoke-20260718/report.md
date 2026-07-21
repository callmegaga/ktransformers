# Running Server Benchmark

- Status: complete
- Run label: igpu-fixed__engine-low__compute16__smoke
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 3 / 3

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1024-o300 | 3 | 1046.0 | 300.0 | 82.92 [82.00, 83.68] | 21.05 [20.69, 21.35] | 12615.15 [12499.77, 12755.85] | 47.51 [46.83, 48.34] | 14206.98 | 26822.12 | 26822.19 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1024-o300 | 0.8879 | 0.7516 | 0.0945 | 0.0418 | 0.0641 |

## Scheduler Telemetry

Values are request means from the representative MoE layer selected at server launch.

| Workload | Prefill iGPU ratio | Prefill CPU load | Decode iGPU ratio | Decode CPU load |
|---|---:|---:|---:|---:|
| p1024-o300 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
- [Per-request scheduler events](scheduler-telemetry.jsonl)
