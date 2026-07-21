# Running Server Benchmark

- Status: failed
- Run label: dynamic-v4-load-context__engine-low__none-to-compute20__transition-token-exact-smoke
- Server: http://127.0.0.1:30100
- Model: Qwen3.5-35B-A3B-GPTQ-Int4
- Successful samples: 0 / 0

Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT use the intervals between the first and last output token. TTLT is time to last token; E2E additionally includes the final stream completion overhead.

| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | Output phase ms | TTLT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NA | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA |

## Background Load Transition

- Direction: low-to-high
- Managed workers: 20
- Start after output token: 150
- Background ready delay: NA ms
- Background stopped after request: False
- Decode calls before launch: NA
- First iGPU execution delay: NA calls / NA ms
- Settled iGPU delay: NA calls / NA ms
- Client pre-transition throughput: NA token/s
- Client post-transition throughput: NA token/s
- Final iGPU execution ratio: NA

## Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Summary CSV](summary.csv)
- [Per-request scheduler events](scheduler-telemetry.jsonl)
