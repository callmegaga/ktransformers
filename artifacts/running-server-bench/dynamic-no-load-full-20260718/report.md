# Running Server Benchmark

- Status: complete
- Run label: dynamic__no-load__full
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 25 / 25

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1-o300 | 5 | 1.0 | 300.0 | 11.33 [10.79, 11.63] | 29.58 [29.50, 29.66] | 88.48 [85.94, 93.04] | 33.81 [33.71, 33.90] | 10107.77 | 10196.25 | 10196.32 |
| p1024-o300 | 5 | 1046.0 | 300.0 | 169.60 [168.84, 170.04] | 29.55 [29.47, 29.63] | 6167.53 [6152.04, 6195.46] | 33.85 [33.75, 33.94] | 10120.11 | 16287.65 | 16287.71 |
| p2048-o300 | 5 | 2070.0 | 300.0 | 169.14 [168.71, 169.39] | 29.52 [29.44, 29.62] | 12238.38 [12219.55, 12269.14] | 33.87 [33.76, 33.96] | 10128.47 | 22366.85 | 22366.92 |
| p4096-o300 | 5 | 4118.0 | 300.0 | 168.34 [167.75, 168.83] | 29.58 [29.51, 29.64] | 24462.63 [24398.53, 24550.41] | 33.81 [33.74, 33.89] | 10108.65 | 34571.28 | 34571.34 |
| p8192-o300 | 5 | 8214.2 | 300.0 | 169.17 [168.98, 169.34] | 29.45 [29.27, 29.58] | 48556.54 [48508.12, 48604.96] | 33.96 [33.80, 34.15] | 10153.42 | 58709.96 | 58710.03 |

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
