# Running Server Benchmark

- Status: complete
- Run label: vnni-only__no-load__full
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 25 / 25

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1-o300 | 5 | 1.0 | 300.0 | 7.16 [5.87, 7.97] | 21.72 [20.24, 23.03] | 145.22 [125.55, 177.34] | 46.35 [43.46, 49.82] | 13858.84 | 14004.06 | 14004.13 |
| p1024-o300 | 5 | 1046.0 | 300.0 | 215.32 [211.98, 218.59] | 24.21 [24.07, 24.31] | 4859.35 [4785.43, 4936.19] | 41.31 [41.14, 41.55] | 12352.27 | 17211.63 | 17211.71 |
| p2048-o300 | 5 | 2070.0 | 300.0 | 221.00 [218.54, 222.53] | 24.19 [24.10, 24.28] | 9367.63 [9302.19, 9473.52] | 41.34 [41.18, 41.50] | 12360.89 | 21728.52 | 21728.59 |
| p4096-o300 | 5 | 4118.0 | 300.0 | 217.40 [208.54, 222.52] | 23.10 [21.26, 24.07] | 18973.82 [18506.33, 19786.87] | 43.61 [41.55, 47.55] | 13038.15 | 32011.97 | 32012.05 |
| p8192-o300 | 5 | 8214.2 | 300.0 | 222.10 [220.55, 223.63] | 23.48 [23.33, 23.62] | 36987.16 [36727.25, 37245.94] | 42.58 [42.34, 42.87] | 12732.61 | 49719.77 | 49719.84 |

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
