# Running Server Benchmark

- Status: complete
- Run label: dynamic__engine-low__compute8__full
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 25 / 25

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1-o300 | 5 | 1.0 | 300.0 | 9.58 [8.80, 10.12] | 23.98 [23.71, 24.16] | 105.13 [98.87, 114.53] | 41.71 [41.40, 42.20] | 12472.29 | 12577.42 | 12577.49 |
| p1024-o300 | 5 | 1046.0 | 300.0 | 164.51 [163.83, 165.02] | 24.08 [23.93, 24.21] | 6358.43 [6338.52, 6385.92] | 41.54 [41.30, 41.82] | 12419.69 | 18778.11 | 18778.18 |
| p2048-o300 | 5 | 2070.0 | 300.0 | 163.41 [162.66, 164.18] | 24.07 [23.96, 24.14] | 12667.71 [12609.09, 12725.97] | 41.55 [41.42, 41.74] | 12424.65 | 25092.36 | 25092.43 |
| p4096-o300 | 5 | 4118.0 | 300.0 | 163.88 [163.10, 164.65] | 23.83 [23.62, 24.05] | 25129.29 [25015.23, 25257.38] | 41.96 [41.59, 42.33] | 12546.94 | 37676.23 | 37676.29 |
| p8192-o300 | 5 | 8214.2 | 300.0 | 164.74 [164.17, 165.31] | 24.16 [24.13, 24.20] | 49861.58 [49701.07, 50046.53] | 41.39 [41.32, 41.44] | 12376.23 | 62237.81 | 62237.87 |

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
