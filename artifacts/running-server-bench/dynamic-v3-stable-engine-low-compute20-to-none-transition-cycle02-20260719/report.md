# Running Server Benchmark

- Status: complete
- Run label: dynamic-v3-stable__engine-low__compute20-to-none__transition-cycle02
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 1 / 1

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 1 | 1046.0 | 600.0 | 145.75 [145.75, 145.75] | 24.58 [24.58, 24.58] | 7176.68 [7176.68, 7176.68] | 40.68 [40.68, 40.68] | 24366.23 | 31542.91 | 31542.99 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1024-o600 | 0.7039 | 0.3916 | 0.3000 | 0.0123 | 0.1446 |

## Scheduler Telemetry

Values are request means from the MoE layer or layers selected at server launch.

| Workload | Prefill iGPU ratio | Prefill CPU load | Prefill exploration | Decode iGPU ratio | Decode CPU load | Decode exploration |
|---|---:|---:|---:|---:|---:|---:|
| p1024-o600 | NA | 0.6595 | 0.0000 | 0.2633 | 0.1906 | 0.0533 |

## Background Load Transition

- Signalled PID: 780195
- Stop after output token: 150
- Background stopped: True
- Decode calls before signal: 150
- First CPU execution delay: 8 calls / 424.78 ms
- Settled CPU delay: 40 calls / 1529.76 ms
- Final iGPU execution ratio: 0.0000

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
- [Per-request scheduler events](scheduler-telemetry.jsonl)
