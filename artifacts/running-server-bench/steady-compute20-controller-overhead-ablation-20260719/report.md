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
| dynamic-v5 | 1 | 5 | 142.090 | 16.310 | 7363.72 | 61.56 | 44239.99 | 0.9927 | 0.1700 |

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
| dynamic-v5 | prefill_tps | 1 | 5 | 142.0897 | 0.0000 | [140.0007, 144.2845] |
| dynamic-v5 | decode_tps | 1 | 5 | 16.3104 | 0.0000 | [15.4139, 17.2069] |
| dynamic-v5 | ttft_ms | 1 | 5 | 7363.7205 | 0.0000 | [7251.4873, 7471.5682] |
| dynamic-v5 | tpot_ms | 1 | 5 | 61.5630 | 0.0000 | [58.1375, 65.0490] |
| dynamic-v5 | e2e_ms | 1 | 5 | 44239.9925 | 0.0000 | [42107.0268, 46440.6006] |

## Candidate Effects

| Baseline | Metric | Effect | Hierarchical bootstrap 95% CI |
|---|---|---:|---:|
| igpu-fixed | prefill_tps improvement | +83.84% | [+68.44%, +103.00%] |
| igpu-fixed | decode_tps improvement | +22.52% | [+14.73%, +30.24%] |
| igpu-fixed | ttft_ms reduction | +45.97% | [+40.61%, +50.80%] |
| igpu-fixed | tpot_ms reduction | +18.19% | [+12.83%, +23.13%] |
| igpu-fixed | e2e_ms reduction | +24.64% | [+19.56%, +29.36%] |
| dynamic-v5-telemetry | prefill_tps improvement | -2.28% | [-6.01%, +1.63%] |
| dynamic-v5-telemetry | decode_tps improvement | +5.94% | [-0.84%, +12.76%] |
| dynamic-v5-telemetry | ttft_ms reduction | -2.19% | [-6.25%, +1.66%] |
| dynamic-v5-telemetry | tpot_ms reduction | +5.35% | [-0.95%, +11.20%] |
| dynamic-v5-telemetry | e2e_ms reduction | +4.17% | [-1.39%, +9.62%] |
| phase-fixed | prefill_tps improvement | -1.18% | [-3.11%, +0.91%] |
| phase-fixed | decode_tps improvement | -0.22% | [-8.03%, +8.10%] |
| phase-fixed | ttft_ms reduction | -1.20% | [-3.23%, +0.87%] |
| phase-fixed | tpot_ms reduction | -0.21% | [-8.78%, +7.40%] |
| phase-fixed | e2e_ms reduction | -0.37% | [-7.38%, +6.00%] |

## Files

- [Normalized samples](samples.csv)
- [Block means](blocks.csv)
- [Strategy statistics](summary.csv)
- [Candidate effects](effects.csv)
- [Manifest](manifest.json)
