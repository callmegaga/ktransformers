# Running Server Benchmark

- Status: complete
- Run label: dynamic__no-load__smoke
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 4 / 4

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1-o32 | 2 | 1.0 | 32.0 | 11.68 [11.57, 11.79] | 29.91 [29.82, 30.00] | 85.60 [84.79, 86.40] | 33.44 [33.34, 33.53] | 1036.51 | 1122.10 | 1122.17 |
| p1024-o64 | 2 | 1045.0 | 64.0 | 166.95 [164.10, 169.79] | 29.48 [29.39, 29.57] | 6261.32 [6154.62, 6368.02] | 33.92 [33.81, 34.03] | 2137.06 | 8398.38 | 8398.45 |

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
