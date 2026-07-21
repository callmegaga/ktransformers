# Running Server Benchmark

- Status: complete
- Run label: igpu-fixed__engine-low__compute12__smoke
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 3 / 3

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1024-o300 | 3 | 1046.0 | 300.0 | 86.01 [82.12, 91.83] | 18.42 [18.36, 18.53] | 12189.20 [11390.46, 12737.14] | 54.29 [53.97, 54.46] | 16231.66 | 28420.86 | 28420.94 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1024-o300 | 0.7219 | 0.5917 | 0.0853 | 0.0448 | 0.0193 |

## Scheduler Telemetry

Values are request means from the MoE layer or layers selected at server launch.

| Workload | Prefill iGPU ratio | Prefill CPU load | Decode iGPU ratio | Decode CPU load |
|---|---:|---:|---:|---:|
| p1024-o300 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
- [Per-request scheduler events](scheduler-telemetry.jsonl)
