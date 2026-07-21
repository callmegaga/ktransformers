# Steady-Load Sweep Comparison

- Strategy means give equal weight to each sweep block.
- Confidence intervals resample sweep blocks and then requests.
- Positive effect percentages favor the candidate.
- Static oracle means the better observed fixed baseline, not a theoretical oracle.

## Block Means

| Workers | Strategy | Block | N | Prefill tok/s | Decode tok/s | E2E ms | CPU busy | PSI some |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | dynamic | 1 | 5 | 167.040 | 29.483 | 26585.61 | 0.4306 | 0.0003 |
| 0 | packed-cpu-fixed | 1 | 5 | 166.015 | 29.629 | 26526.82 | 0.4356 | 0.0003 |
| 0 | igpu-fixed | 1 | 5 | 97.446 | 16.617 | 46919.35 | 0.1156 | 0.0025 |
| 4 | dynamic | 1 | 5 | 169.075 | 24.156 | 30987.00 | 0.6279 | 0.0002 |
| 4 | packed-cpu-fixed | 1 | 5 | 170.258 | 23.924 | 31184.98 | 0.6250 | 0.0002 |
| 4 | igpu-fixed | 1 | 5 | 99.321 | 17.585 | 44603.17 | 0.3026 | 0.0278 |
| 8 | dynamic | 1 | 5 | 164.833 | 22.408 | 33200.76 | 0.6607 | 0.0027 |
| 8 | packed-cpu-fixed | 1 | 5 | 167.419 | 24.045 | 31161.96 | 0.8252 | 0.0004 |
| 8 | igpu-fixed | 1 | 5 | 99.831 | 18.202 | 43395.04 | 0.4934 | 0.0368 |
| 12 | dynamic | 1 | 5 | 129.135 | 21.059 | 36548.20 | 0.7532 | 0.0162 |
| 12 | packed-cpu-fixed | 1 | 5 | 125.010 | 1.883 | 328101.68 | 0.9992 | 0.0578 |
| 12 | igpu-fixed | 1 | 5 | 103.274 | 18.673 | 42210.39 | 0.6989 | 0.0206 |
| 16 | dynamic | 1 | 5 | 98.017 | 19.662 | 42555.60 | 0.9229 | 0.0854 |
| 16 | packed-cpu-fixed | 1 | 5 | 90.888 | 1.441 | 427315.89 | 0.9993 | 0.2492 |
| 16 | igpu-fixed | 1 | 5 | 101.681 | 18.895 | 41995.49 | 0.8947 | 0.0238 |
| 20 | dynamic | 1 | 5 | 79.549 | 13.447 | 58981.14 | 0.9962 | 0.2030 |
| 20 | packed-cpu-fixed | 1 | 5 | 71.577 | 1.395 | 444200.31 | 0.9990 | 0.4204 |
| 20 | igpu-fixed | 1 | 5 | 96.625 | 13.539 | 55087.10 | 0.9947 | 0.1294 |

## Candidate Effects

| Workers | Baseline | Metric | Effect | Hierarchical bootstrap 95% CI |
|---:|---|---|---:|---:|
| 0 | packed-cpu-fixed | prefill_tps improvement | +0.62% | [-2.51%, +4.04%] |
| 0 | packed-cpu-fixed | decode_tps improvement | -0.49% | [-1.53%, +0.52%] |
| 0 | packed-cpu-fixed | e2e_ms reduction | -0.22% | [-1.61%, +1.12%] |
| 0 | igpu-fixed | prefill_tps improvement | +71.42% | [+61.45%, +88.76%] |
| 0 | igpu-fixed | decode_tps improvement | +77.43% | [+71.89%, +82.11%] |
| 0 | igpu-fixed | e2e_ms reduction | +43.34% | [+41.23%, +45.49%] |
| 4 | packed-cpu-fixed | prefill_tps improvement | -0.70% | [-0.84%, -0.56%] |
| 4 | packed-cpu-fixed | decode_tps improvement | +0.97% | [+0.29%, +1.65%] |
| 4 | packed-cpu-fixed | e2e_ms reduction | +0.63% | [+0.10%, +1.17%] |
| 4 | igpu-fixed | prefill_tps improvement | +70.23% | [+69.47%, +71.00%] |
| 4 | igpu-fixed | decode_tps improvement | +37.37% | [+36.01%, +38.63%] |
| 4 | igpu-fixed | e2e_ms reduction | +30.53% | [+30.01%, +31.00%] |
| 8 | packed-cpu-fixed | prefill_tps improvement | -1.54% | [-2.20%, -0.88%] |
| 8 | packed-cpu-fixed | decode_tps improvement | -6.81% | [-12.01%, -1.49%] |
| 8 | packed-cpu-fixed | e2e_ms reduction | -6.54% | [-11.25%, -1.70%] |
| 8 | igpu-fixed | prefill_tps improvement | +65.11% | [+63.98%, +66.23%] |
| 8 | igpu-fixed | decode_tps improvement | +23.11% | [+15.99%, +30.58%] |
| 8 | igpu-fixed | e2e_ms reduction | +23.49% | [+20.00%, +27.11%] |
| 12 | packed-cpu-fixed | prefill_tps improvement | +3.30% | [-0.00%, +7.17%] |
| 12 | packed-cpu-fixed | decode_tps improvement | +1018.47% | [+949.27%, +1086.01%] |
| 12 | packed-cpu-fixed | e2e_ms reduction | +88.86% | [+88.11%, +89.45%] |
| 12 | igpu-fixed | prefill_tps improvement | +25.04% | [+23.95%, +26.14%] |
| 12 | igpu-fixed | decode_tps improvement | +12.78% | [+12.28%, +13.31%] |
| 12 | igpu-fixed | e2e_ms reduction | +13.41% | [+13.05%, +13.83%] |
| 16 | packed-cpu-fixed | prefill_tps improvement | +7.84% | [+4.00%, +11.46%] |
| 16 | packed-cpu-fixed | decode_tps improvement | +1264.35% | [+1015.34%, +1406.71%] |
| 16 | packed-cpu-fixed | e2e_ms reduction | +90.04% | [+88.19%, +91.08%] |
| 16 | igpu-fixed | prefill_tps improvement | -3.60% | [-6.89%, -0.65%] |
| 16 | igpu-fixed | decode_tps improvement | +4.06% | [-14.65%, +13.65%] |
| 16 | igpu-fixed | e2e_ms reduction | -1.33% | [-19.78%, +8.53%] |
| 20 | packed-cpu-fixed | prefill_tps improvement | +11.14% | [+3.60%, +19.68%] |
| 20 | packed-cpu-fixed | decode_tps improvement | +863.97% | [+723.02%, +952.35%] |
| 20 | packed-cpu-fixed | e2e_ms reduction | +86.72% | [+84.96%, +87.84%] |
| 20 | igpu-fixed | prefill_tps improvement | -17.67% | [-22.79%, -11.79%] |
| 20 | igpu-fixed | decode_tps improvement | -0.68% | [-15.10%, +8.27%] |
| 20 | igpu-fixed | e2e_ms reduction | -7.07% | [-21.27%, +1.81%] |

## Static Oracle Attainment

| Workers | Metric | Candidate | Static oracle | Oracle strategy | Attainment |
|---:|---|---:|---:|---|---:|
| 0 | prefill_tps | 167.040 | 166.015 | packed-cpu-fixed | 100.62% |
| 0 | decode_tps | 29.483 | 29.629 | packed-cpu-fixed | 99.51% |
| 0 | e2e_ms | 26585.611 | 26526.824 | packed-cpu-fixed | 99.78% |
| 4 | prefill_tps | 169.075 | 170.258 | packed-cpu-fixed | 99.30% |
| 4 | decode_tps | 24.156 | 23.924 | packed-cpu-fixed | 100.97% |
| 4 | e2e_ms | 30986.998 | 31184.983 | packed-cpu-fixed | 100.64% |
| 8 | prefill_tps | 164.833 | 167.419 | packed-cpu-fixed | 98.46% |
| 8 | decode_tps | 22.408 | 24.045 | packed-cpu-fixed | 93.19% |
| 8 | e2e_ms | 33200.765 | 31161.961 | packed-cpu-fixed | 93.86% |
| 12 | prefill_tps | 129.135 | 125.010 | packed-cpu-fixed | 103.30% |
| 12 | decode_tps | 21.059 | 18.673 | igpu-fixed | 112.78% |
| 12 | e2e_ms | 36548.200 | 42210.388 | igpu-fixed | 115.49% |
| 16 | prefill_tps | 98.017 | 101.681 | igpu-fixed | 96.40% |
| 16 | decode_tps | 19.662 | 18.895 | igpu-fixed | 104.06% |
| 16 | e2e_ms | 42555.603 | 41995.487 | igpu-fixed | 98.68% |
| 20 | prefill_tps | 79.549 | 96.625 | igpu-fixed | 82.33% |
| 20 | decode_tps | 13.447 | 13.539 | igpu-fixed | 99.32% |
| 20 | e2e_ms | 58981.136 | 55087.098 | igpu-fixed | 93.40% |

## Files

- [Normalized samples](samples.csv)
- [Block means](blocks.csv)
- [Strategy statistics](summary.csv)
- [Candidate effects](effects.csv)
- [Static oracle attainment](oracle.csv)
- [Manifest](manifest.json)
