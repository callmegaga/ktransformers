# Blocked Strategy Comparison

- Strategy means give equal weight to each block.
- Confidence intervals resample blocks and then requests within selected blocks.
- Positive effect percentages favor the candidate.

## Block Means

| Strategy | Block | N | Prefill tok/s | Decode tok/s | TTFT ms | TPOT ms | E2E ms | CPU busy | CPU PSI some |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| igpu-fixed | 1 | 5 | 71.317 | 13.130 | 14688.42 | 76.26 | 60366.79 | 0.9902 | 0.1395 |
| igpu-fixed | 2 | 5 | 83.259 | 13.494 | 12570.17 | 74.24 | 57041.82 | 0.9910 | 0.1375 |
| dynamic-v5-telemetry | 1 | 5 | 145.412 | 15.396 | 7205.67 | 65.04 | 46166.17 | 0.9899 | 0.1726 |
| phase-fixed | 1 | 5 | 143.787 | 16.347 | 7276.41 | 61.43 | 44076.00 | 0.9908 | 0.1672 |

## Strategy Statistics

| Strategy | Metric | Blocks | Samples | Mean | Block stdev | Hierarchical bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---:|
| igpu-fixed | prefill_tps | 2 | 10 | 77.2881 | 8.4441 | [70.1755, 84.0615] |
| igpu-fixed | decode_tps | 2 | 10 | 13.3121 | 0.2573 | [12.9257, 13.7526] |
| igpu-fixed | ttft_ms | 2 | 10 | 13629.2934 | 1497.8334 | [12454.0680, 14925.5510] |
| igpu-fixed | tpot_ms | 2 | 10 | 75.2502 | 1.4243 | [72.9027, 77.3869] |
| igpu-fixed | e2e_ms | 2 | 10 | 58704.3069 | 2351.1064 | [56299.7108, 61099.5907] |
| dynamic-v5-telemetry | prefill_tps | 1 | 5 | 145.4116 | 0.0000 | [140.1985, 150.6246] |
| dynamic-v5-telemetry | decode_tps | 1 | 5 | 15.3957 | 0.0000 | [14.8502, 15.7950] |
| dynamic-v5-telemetry | ttft_ms | 1 | 5 | 7205.6671 | 0.0000 | [6947.3982, 7463.9360] |
| dynamic-v5-telemetry | tpot_ms | 1 | 5 | 65.0425 | 0.0000 | [63.2721, 67.4561] |
| dynamic-v5-telemetry | e2e_ms | 1 | 5 | 46166.1728 | 0.0000 | [45118.3474, 47707.9086] |
| phase-fixed | prefill_tps | 1 | 5 | 143.7871 | 0.0000 | [141.8368, 145.7373] |
| phase-fixed | decode_tps | 1 | 5 | 16.3472 | 0.0000 | [15.3958, 17.2986] |
| phase-fixed | ttft_ms | 1 | 5 | 7276.4071 | 0.0000 | [7177.6346, 7375.1796] |
| phase-fixed | tpot_ms | 1 | 5 | 61.4349 | 0.0000 | [57.8708, 64.9991] |
| phase-fixed | e2e_ms | 1 | 5 | 44076.0019 | 0.0000 | [42008.5047, 46143.4990] |

## Candidate Effects

| Baseline | Metric | Effect | Hierarchical bootstrap 95% CI |
|---|---|---:|---:|
| igpu-fixed | prefill_tps improvement | +86.04% | [+70.53%, +105.26%] |
| igpu-fixed | decode_tps improvement | +22.80% | [+14.95%, +30.77%] |
| igpu-fixed | ttft_ms reduction | +46.61% | [+41.35%, +51.35%] |
| igpu-fixed | tpot_ms reduction | +18.36% | [+12.99%, +23.54%] |
| igpu-fixed | e2e_ms reduction | +24.92% | [+20.03%, +29.55%] |
| dynamic-v5-telemetry | prefill_tps improvement | -1.12% | [-4.85%, +2.77%] |
| dynamic-v5-telemetry | decode_tps improvement | +6.18% | [-0.51%, +13.35%] |
| dynamic-v5-telemetry | ttft_ms reduction | -0.98% | [-4.96%, +2.79%] |
| dynamic-v5-telemetry | tpot_ms reduction | +5.55% | [-0.56%, +11.68%] |
| dynamic-v5-telemetry | e2e_ms reduction | +4.53% | [-0.64%, +9.71%] |

## Files

- [Normalized samples](samples.csv)
- [Block means](blocks.csv)
- [Strategy statistics](summary.csv)
- [Candidate effects](effects.csv)
- [Manifest](manifest.json)
