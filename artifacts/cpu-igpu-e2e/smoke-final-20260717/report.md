# CPU-iGPU MoE End-to-End Experiment

This report is generated from the raw request samples. Values are arithmetic means; brackets contain bootstrap 95% confidence intervals.

- Status: complete
- Started: 2026-07-17T17:09:07.533225+08:00
- Finished: 2026-07-17T17:13:42.325834+08:00
- Successful samples: 8 / 8

## Result Files

- [Manifest](manifest.json)
- [Request samples](samples.jsonl)
- [Backend summaries](summary.csv)
- [Paired comparisons](comparisons.csv)

## Performance

Speedup is dynamic/VNNI for throughput and VNNI/dynamic for latency, so values above 1.0 always favor dynamic scheduling. Speedup intervals use paired bootstrap resampling.

| Load | Workload | Metric | VNNI mean [95% CI] | Dynamic mean [95% CI] | Speedup [95% CI] |
|---|---|---|---:|---:|---:|
| compute-8 | p1-o8 | prefill_tps | 10.422 [10.422, 10.422] | 8.462 [8.462, 8.462] | 0.812x [0.812, 0.812] |
| compute-8 | p1-o8 | decode_tps | 2.115 [2.115, 2.115] | 17.491 [17.491, 17.491] | 8.272x [8.272, 8.272] |
| compute-8 | p1-o8 | ttft_ms | 1343.332 [1343.332, 1343.332] | 1654.548 [1654.548, 1654.548] | 0.812x [0.812, 0.812] |
| compute-8 | p1-o8 | topt_ms | 472.918 [472.918, 472.918] | 57.172 [57.172, 57.172] | 8.272x [8.272, 8.272] |
| compute-8 | p1-o8 | e2e_ms | 4653.838 [4653.838, 4653.838] | 2054.981 [2054.981, 2054.981] | 2.265x [2.265, 2.265] |
| compute-8 | p1024-o8 | prefill_tps | 95.654 [95.654, 95.654] | 108.053 [108.053, 108.053] | 1.130x [1.130, 1.130] |
| compute-8 | p1024-o8 | decode_tps | 1.944 [1.944, 1.944] | 18.832 [18.832, 18.832] | 9.687x [9.687, 9.687] |
| compute-8 | p1024-o8 | ttft_ms | 10872.485 [10872.485, 10872.485] | 9624.933 [9624.933, 9624.933] | 1.130x [1.130, 1.130] |
| compute-8 | p1024-o8 | topt_ms | 514.393 [514.393, 514.393] | 53.100 [53.100, 53.100] | 9.687x [9.687, 9.687] |
| compute-8 | p1024-o8 | e2e_ms | 14473.316 [14473.316, 14473.316] | 9996.712 [9996.712, 9996.712] | 1.448x [1.448, 1.448] |
| none | p1-o8 | prefill_tps | 37.169 [37.169, 37.169] | 49.876 [49.876, 49.876] | 1.342x [1.342, 1.342] |
| none | p1-o8 | decode_tps | 22.589 [22.589, 22.589] | 28.206 [28.206, 28.206] | 1.249x [1.249, 1.249] |
| none | p1-o8 | ttft_ms | 295.945 [295.945, 295.945] | 220.545 [220.545, 220.545] | 1.342x [1.342, 1.342] |
| none | p1-o8 | topt_ms | 44.269 [44.269, 44.269] | 35.453 [35.453, 35.453] | 1.249x [1.249, 1.249] |
| none | p1-o8 | e2e_ms | 605.906 [605.906, 605.906] | 468.795 [468.795, 468.795] | 1.292x [1.292, 1.292] |
| none | p1024-o8 | prefill_tps | 208.847 [208.847, 208.847] | 167.875 [167.875, 167.875] | 0.804x [0.804, 0.804] |
| none | p1024-o8 | decode_tps | 21.025 [21.025, 21.025] | 28.025 [28.025, 28.025] | 1.333x [1.333, 1.333] |
| none | p1024-o8 | ttft_ms | 4965.369 [4965.369, 4965.369] | 6177.220 [6177.220, 6177.220] | 0.804x [0.804, 0.804] |
| none | p1024-o8 | topt_ms | 47.562 [47.562, 47.562] | 35.682 [35.682, 35.682] | 1.333x [1.333, 1.333] |
| none | p1024-o8 | e2e_ms | 5298.400 [5298.400, 5298.400] | 6427.071 [6427.071, 6427.071] | 0.824x [0.824, 0.824] |

## Output Agreement

Output agreement is exact SHA-256 equality for byte-identical prompts and deterministic decoding parameters.

| Load | Workload | Matched | Paired outputs | Match rate |
|---|---|---:|---:|---:|
| compute-8 | p1-o8 | 1 | 1 | 1.0000 |
| compute-8 | p1024-o8 | 1 | 1 | 1.0000 |
| none | p1-o8 | 1 | 1 | 1.0000 |
| none | p1024-o8 | 1 | 1 | 1.0000 |
