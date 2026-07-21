# Running Server Benchmark

- Status: complete
- Run label: packed-cpu-fixed__engine-low__compute20__smoke
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 6 / 6

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1-o300 | 3 | 1.0 | 300.0 | 0.62 [0.57, 0.68] | 1.36 [1.35, 1.39] | 1627.06 [1477.14, 1740.31] | 733.28 [719.44, 740.35] | 219250.17 | 220877.22 | 220877.30 |
| p1024-o300 | 3 | 1046.0 | 300.0 | 82.73 [79.12, 86.45] | 1.37 [1.35, 1.38] | 12660.20 [12099.66, 13221.18] | 731.22 [725.12, 738.63] | 218636.09 | 231296.29 | 231296.83 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1-o300 | 0.9989 | 0.7567 | 0.2307 | 0.0114 | 0.4073 |
| p1024-o300 | 0.9987 | 0.7597 | 0.2278 | 0.0112 | 0.4057 |

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
