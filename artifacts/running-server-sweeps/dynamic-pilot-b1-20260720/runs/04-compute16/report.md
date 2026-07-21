# Running Server Benchmark

- Status: complete
- Run label: dynamic__engine-low__compute16__pilot-b1
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 5 / 5

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 5 | 1046.4 | 600.0 | 98.02 [94.90, 100.57] | 19.66 [16.13, 21.46] | 10688.02 [10405.93, 11035.45] | 53.20 [46.60, 66.27] | 31867.51 | 42555.54 | 42555.60 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1024-o600 | 0.9229 | 0.7751 | 0.1249 | 0.0229 | 0.0854 |

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
