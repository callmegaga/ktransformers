# Running Server Benchmark

- Status: complete
- Run label: dynamic-v3-stable__engine-low__compute20-to-none__transition-cycle04
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 1 / 1

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 1 | 1046.0 | 600.0 | 152.28 [152.28, 152.28] | 25.03 [25.03, 25.03] | 6868.73 [6868.73, 6868.73] | 39.96 [39.96, 39.96] | 23933.36 | 30802.10 | 30802.16 |

## CPU Telemetry

Fractions cover each complete request window. `user` includes normal-priority work; `nice` includes the low-priority inference scope.

| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1024-o600 | 0.7022 | 0.3788 | 0.3096 | 0.0137 | 0.1301 |

## Scheduler Telemetry

Values are request means from the MoE layer or layers selected at server launch.

| Workload | Prefill iGPU ratio | Prefill CPU load | Prefill exploration | Decode iGPU ratio | Decode CPU load | Decode exploration |
|---|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 0.0000 | 0.3760 | 0.0000 | 0.2667 | 0.2027 | 0.0533 |

## Background Load Transition

- Signalled PID: 790816
- Stop after output token: 150
- Background stopped: True
- Decode calls before signal: 151
- First CPU execution delay: 9 calls / 489.12 ms
- Settled CPU delay: 41 calls / 1601.18 ms
- Final iGPU execution ratio: 0.0000

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
- [Per-request scheduler events](scheduler-telemetry.jsonl)
