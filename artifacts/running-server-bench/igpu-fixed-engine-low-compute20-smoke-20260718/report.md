# Running Server Benchmark

- Status: complete
- Run label: igpu-fixed__engine-low__compute20__smoke
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 6 / 6

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1-o300 | 3 | 1.0 | 300.0 | 6.73 [2.71, 10.11] | 14.91 [14.28, 15.88] | 201.13 [98.89, 368.63] | 67.23 [62.97, 71.52] | 20102.41 | 20303.53 | 20304.71 |
| p1024-o300 | 3 | 1046.0 | 300.0 | 83.60 [74.76, 88.75] | 14.16 [13.19, 15.50] | 12587.11 [11786.05, 13991.78] | 70.93 [64.50, 75.81] | 21207.97 | 33795.08 | 33795.73 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1-o300 | 0.9956 | 0.9357 | 0.0400 | 0.0200 | 0.1394 |
| p1024-o300 | 0.9906 | 0.8943 | 0.0645 | 0.0317 | 0.1736 |

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
