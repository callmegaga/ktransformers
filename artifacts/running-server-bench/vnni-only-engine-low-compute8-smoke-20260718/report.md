# Running Server Benchmark

- Status: complete
- Run label: vnni-only__engine-low__compute8__smoke
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 6 / 6

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1-o300 | 3 | 1.0 | 300.0 | 6.48 [2.68, 8.52] | 20.19 [18.42, 22.54] | 204.03 [115.67, 372.92] | 50.16 [44.36, 58.09] | 14996.72 | 15200.75 | 15200.83 |
| p1024-o300 | 3 | 1046.0 | 300.0 | 205.88 [199.82, 215.62] | 22.24 [22.02, 22.38] | 5086.22 [4851.12, 5234.73] | 44.97 [44.69, 45.42] | 13446.49 | 18532.71 | 18532.78 |

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
