# Dynamic Load Transition Cycles

- Cycles: 5
- High-load rate excludes the first prefill/decode boundary interval.
- Confidence intervals are percentile bootstrap intervals of the cycle mean.

## Cycle Results

| Cycle | Decode tok/s | First CPU calls | First CPU ms | Settle calls | Settle ms | High iGPU calls/s | Low CPU calls/s | Exploration calls | Switches |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24.501 | 8 | 395.02 | 40 | 1494.77 | 16.807 | 29.074 | 32 | 1 |
| 2 | 24.583 | 8 | 424.78 | 40 | 1529.76 | 17.166 | 29.036 | 32 | 1 |
| 3 | 24.555 | 6 | 317.55 | 38 | 1419.13 | 16.940 | 29.029 | 32 | 1 |
| 4 | 25.028 | 9 | 489.12 | 41 | 1601.18 | 18.116 | 28.923 | 32 | 1 |
| 5 | 25.882 | 7 | 372.11 | 39 | 1477.26 | 19.944 | 29.019 | 32 | 1 |

## Aggregate Results

| Metric | N | Mean | Sample stdev | Bootstrap 95% CI | Unit |
|---|---:|---:|---:|---:|---|
| prefill_tps | 5 | 150.7472 | 3.5307 | [147.6571, 153.2694] | token/s |
| decode_tps | 5 | 24.9098 | 0.5827 | [24.5392, 25.4348] | token/s |
| ttft_ms | 5 | 6941.8412 | 163.9399 | [6825.0568, 7063.5493] | ms |
| tpot_ms | 5 | 40.1620 | 0.9187 | [39.3364, 40.7513] | ms/token |
| e2e_ms | 5 | 30998.9580 | 676.5875 | [30408.8383, 31452.3739] | ms |
| first_cpu_delay_calls | 5 | 7.6000 | 1.1402 | [6.6000, 8.4000] | calls |
| first_cpu_delay_ms | 5 | 399.7159 | 63.5536 | [350.2858, 451.4805] | ms |
| settle_cpu_delay_calls | 5 | 39.6000 | 1.1402 | [38.8000, 40.4000] | calls |
| settle_cpu_delay_ms | 5 | 1504.4196 | 67.2861 | [1454.0069, 1558.6166] | ms |
| high_load_igpu_calls_per_s | 5 | 17.7944 | 1.3061 | [16.9318, 18.9506] | calls/s |
| low_load_cpu_calls_per_s | 5 | 29.0164 | 0.0560 | [28.9658, 29.0539] | calls/s |
| segment_speedup | 5 | 1.6373 | 0.1143 | [1.5383, 1.7157] | x |

## Files

- [Cycle data](cycles.csv)
- [Aggregate statistics](summary.csv)
- [Manifest](manifest.json)
