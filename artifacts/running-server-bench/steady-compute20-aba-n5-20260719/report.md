# Blocked Strategy Comparison

- Strategy means give equal weight to each block.
- Confidence intervals resample blocks and then requests within selected blocks.
- Positive effect percentages favor the candidate.

## Block Means

| Strategy | Block | N | Prefill tok/s | Decode tok/s | TTFT ms | TPOT ms | E2E ms | CPU busy | CPU PSI some |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| igpu-fixed | 1 | 5 | 71.317 | 13.130 | 14688.42 | 76.26 | 60366.79 | 0.9902 | 0.1395 |
| igpu-fixed | 2 | 5 | 83.259 | 13.494 | 12570.17 | 74.24 | 57041.82 | 0.9910 | 0.1375 |
| dynamic-v5 | 1 | 5 | 145.412 | 15.396 | 7205.67 | 65.04 | 46166.17 | 0.9899 | 0.1726 |

## Strategy Statistics

| Strategy | Metric | Blocks | Samples | Mean | Block stdev | Hierarchical bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---:|
| igpu-fixed | prefill_tps | 2 | 10 | 77.2881 | 8.4441 | [70.1755, 84.0615] |
| igpu-fixed | decode_tps | 2 | 10 | 13.3121 | 0.2573 | [12.9257, 13.7526] |
| igpu-fixed | ttft_ms | 2 | 10 | 13629.2934 | 1497.8334 | [12454.0680, 14925.5510] |
| igpu-fixed | tpot_ms | 2 | 10 | 75.2502 | 1.4243 | [72.9027, 77.3869] |
| igpu-fixed | e2e_ms | 2 | 10 | 58704.3069 | 2351.1064 | [56299.7108, 61099.5907] |
| dynamic-v5 | prefill_tps | 1 | 5 | 145.4116 | 0.0000 | [140.1985, 150.6246] |
| dynamic-v5 | decode_tps | 1 | 5 | 15.3957 | 0.0000 | [14.8666, 15.8076] |
| dynamic-v5 | ttft_ms | 1 | 5 | 7205.6671 | 0.0000 | [6947.3982, 7463.9360] |
| dynamic-v5 | tpot_ms | 1 | 5 | 65.0425 | 0.0000 | [63.2721, 67.3876] |
| dynamic-v5 | e2e_ms | 1 | 5 | 46166.1728 | 0.0000 | [45118.3474, 47707.9086] |

## Candidate Effects

| Baseline | Metric | Effect | Hierarchical bootstrap 95% CI |
|---|---|---:|---:|
| igpu-fixed | prefill_tps improvement | +88.14% | [+70.11%, +109.63%] |
| igpu-fixed | decode_tps improvement | +15.65% | [+10.28%, +20.55%] |
| igpu-fixed | ttft_ms reduction | +47.13% | [+41.24%, +52.33%] |
| igpu-fixed | tpot_ms reduction | +13.57% | [+9.28%, +17.10%] |
| igpu-fixed | e2e_ms reduction | +21.36% | [+17.09%, +25.21%] |

## Files

- [Normalized samples](samples.csv)
- [Block means](blocks.csv)
- [Strategy statistics](summary.csv)
- [Candidate effects](effects.csv)
- [Manifest](manifest.json)
