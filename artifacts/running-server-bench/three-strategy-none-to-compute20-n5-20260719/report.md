# Load Transition Strategy Comparison

- Candidate: dynamic-v5
- Confidence intervals use independent percentile bootstrap resampling of strategy means.
- Positive effect percentages favor the candidate.

## Strategy Results

| Strategy | N | Prefill tok/s | Decode tok/s | TTFT ms | TPOT ms/token | E2E ms | Client pre tok/s | Client post tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| packed-cpu-fixed | 5 | 169.431 [168.770, 169.876] | 8.794 [8.198, 9.525] | 6173.682 [6157.458, 6197.884] | 114.575 [106.711, 122.329] | 74804.214 [69196.256, 79495.657] | 29.216 [29.145, 29.288] | 7.146 [6.621, 7.791] |
| igpu-fixed | 5 | 83.356 [79.922, 87.513] | 13.941 [13.471, 14.430] | 12584.783 [11995.632, 13095.967] | 71.841 [69.334, 74.334] | 55620.408 [54365.933, 57159.088] | 16.312 [15.219, 17.119] | 13.331 [12.664, 13.882] |
| dynamic-v5 | 5 | 168.486 [165.165, 170.258] | 17.225 [16.663, 17.738] | 6210.721 [6143.629, 6336.814] | 58.130 [56.389, 60.053] | 41030.645 [39926.830, 42176.752] | 28.955 [28.607, 29.264] | 15.190 [14.638, 15.701] |

## Candidate Effects

| Baseline | Metric | Effect | Bootstrap 95% CI |
|---|---|---:|---:|
| packed-cpu-fixed | prefill_tps improvement | -0.56% | [-2.57%, +0.69%] |
| packed-cpu-fixed | decode_tps improvement | +95.89% | [+79.54%, +112.18%] |
| packed-cpu-fixed | ttft_ms reduction | -0.60% | [-2.73%, +0.67%] |
| packed-cpu-fixed | tpot_ms reduction | +49.26% | [+44.76%, +53.00%] |
| packed-cpu-fixed | e2e_ms reduction | +45.15% | [+40.61%, +48.87%] |
| packed-cpu-fixed | client_pre_transition_tps improvement | -0.89% | [-2.03%, +0.18%] |
| packed-cpu-fixed | client_post_transition_tps improvement | +112.58% | [+94.30%, +131.75%] |
| igpu-fixed | prefill_tps improvement | +102.13% | [+91.79%, +111.65%] |
| igpu-fixed | decode_tps improvement | +23.56% | [+17.73%, +29.47%] |
| igpu-fixed | ttft_ms reduction | +50.65% | [+47.92%, +52.76%] |
| igpu-fixed | tpot_ms reduction | +19.09% | [+15.10%, +22.77%] |
| igpu-fixed | e2e_ms reduction | +26.23% | [+23.41%, +28.97%] |
| igpu-fixed | client_pre_transition_tps improvement | +77.51% | [+68.88%, +90.18%] |
| igpu-fixed | client_post_transition_tps improvement | +13.94% | [+7.69%, +21.03%] |

## Files

- [Normalized samples](samples.csv)
- [Strategy statistics](summary.csv)
- [Candidate effects](effects.csv)
- [Manifest](manifest.json)
