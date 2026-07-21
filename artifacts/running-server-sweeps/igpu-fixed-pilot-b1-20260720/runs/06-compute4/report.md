# Running Server Benchmark

- Status: complete
- Run label: igpu-fixed__engine-low__compute4__pilot-b1
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 5 / 5

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 5 | 1046.4 | 600.0 | 99.32 [98.90, 99.74] | 17.58 [17.43, 17.75] | 10535.84 [10491.66, 10580.01] | 56.87 [56.35, 57.37] | 34067.24 | 44603.08 | 44603.17 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1024-o600 | 0.3026 | 0.2015 | 0.0638 | 0.0373 | 0.0278 |

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
