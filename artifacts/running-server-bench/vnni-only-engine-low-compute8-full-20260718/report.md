# Running Server Benchmark

- Status: complete
- Run label: vnni-only__engine-low__compute8__full
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 25 / 25

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1-o300 | 5 | 1.0 | 300.0 | 7.96 [7.67, 8.27] | 23.11 [22.91, 23.28] | 125.90 [121.49, 130.36] | 43.28 [42.96, 43.66] | 12941.12 | 13067.02 | 13067.12 |
| p1024-o300 | 5 | 1046.0 | 300.0 | 212.20 [209.55, 214.40] | 22.66 [21.75, 23.36] | 4930.19 [4878.78, 4992.86] | 44.22 [42.65, 46.07] | 13220.54 | 18150.74 | 18150.83 |
| p2048-o300 | 5 | 2070.0 | 300.0 | 214.99 [214.24, 215.74] | 22.74 [22.10, 23.39] | 9628.56 [9594.91, 9660.92] | 44.03 [42.77, 45.43] | 13163.80 | 22792.36 | 22792.44 |
| p4096-o300 | 5 | 4118.0 | 300.0 | 216.24 [215.18, 217.24] | 22.51 [21.96, 23.08] | 19044.04 [18954.05, 19134.12] | 44.47 [43.42, 45.51] | 13295.67 | 32339.71 | 32339.79 |
| p8192-o300 | 5 | 8214.2 | 300.0 | 217.01 [215.86, 218.18] | 22.15 [21.62, 22.97] | 37853.71 [37651.89, 38052.85] | 45.20 [43.62, 46.27] | 13514.19 | 51367.89 | 51367.97 |

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
