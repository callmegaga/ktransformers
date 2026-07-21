# Running Server Benchmark

- Status: complete
- Run label: dynamic__engine-low__compute8__smoke
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 6 / 6

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1-o300 | 3 | 1.0 | 300.0 | 10.26 [10.22, 10.35] | 24.03 [24.01, 24.05] | 97.45 [96.63, 97.87] | 41.62 [41.58, 41.65] | 12444.13 | 12541.58 | 12541.65 |
| p1024-o300 | 3 | 1046.0 | 300.0 | 169.55 [168.82, 170.10] | 24.06 [23.98, 24.11] | 6169.32 [6149.18, 6196.01] | 41.56 [41.47, 41.71] | 12426.06 | 18595.38 | 18595.45 |

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
