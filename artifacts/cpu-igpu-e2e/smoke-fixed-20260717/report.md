# CPU-iGPU MoE End-to-End Experiment

This report is generated from the raw request samples. Values are arithmetic means; brackets contain bootstrap 95% confidence intervals.

- Status: complete_with_errors
- Started: 2026-07-17T16:43:01.344974+08:00
- Finished: 2026-07-17T16:49:23.185468+08:00
- Successful samples: 4 / 4

## Result Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Backend summaries](summary.csv)
- [Paired comparisons](comparisons.csv)

## Performance

Speedup is dynamic/VNNI for throughput and VNNI/dynamic for latency, so values above 1.0 always favor dynamic scheduling. Speedup intervals use paired bootstrap resampling.

| Load | Workload | Metric | VNNI mean [95% CI] | Dynamic mean [95% CI] | Speedup [95% CI] |
|---|---|---|---:|---:|---:|

## Output Agreement

Output agreement is exact SHA-256 equality for byte-identical prompts and deterministic decoding parameters.

| Load | Workload | Matched | Paired outputs | Match rate |
|---|---|---:|---:|---:|
| NA | NA | 0 | 0 | NA |
