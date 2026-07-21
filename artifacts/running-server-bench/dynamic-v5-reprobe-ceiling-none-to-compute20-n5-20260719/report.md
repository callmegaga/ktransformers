# Dynamic Load Transition Cycles

- Cycles: 5
- Direction: low-to-high
- Target: igpu
- Pre-transition rate excludes the first prefill/decode boundary interval.
- Confidence intervals are percentile bootstrap intervals of the cycle mean.

## Cycle Results

| Cycle | Decode tok/s | First target calls | First target ms | Settle calls | Settle ms | Pre calls/s | Post calls/s | Client pre tok/s | Client post tok/s | Exact token timestamps | Ready ms | Exploration calls | Switches |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| 1 | 16.650 | 15 | 1941.23 | 15 | 1941.23 | 29.018 | 15.060 | 29.025 | 14.590 | yes | 84.23 | 0 | 1 |
| 2 | 16.382 | 3 | 794.86 | 3 | 794.86 | 28.615 | 14.640 | 28.380 | 14.370 | yes | 623.30 | 0 | 1 |
| 3 | 17.523 | 8 | 1022.15 | 8 | 1022.15 | 29.100 | 15.782 | 29.102 | 15.483 | yes | 69.70 | 0 | 1 |
| 4 | 17.500 | 29 | 2286.48 | 29 | 2286.48 | 28.790 | 15.752 | 28.795 | 15.488 | yes | 47.41 | 0 | 1 |
| 5 | 18.073 | 13 | 884.40 | 13 | 884.40 | 29.471 | 16.054 | 29.474 | 16.021 | yes | 51.60 | 0 | 1 |

## Aggregate Results

| Metric | N | Mean | Sample stdev | Bootstrap 95% CI | Unit |
|---|---:|---:|---:|---:|---|
| prefill_tps | 5 | 168.4857 | 3.7070 | [165.1720, 170.2578] | token/s |
| decode_tps | 5 | 17.2253 | 0.6937 | [16.6634, 17.7381] | token/s |
| ttft_ms | 5 | 6210.7209 | 140.7752 | [6143.6295, 6336.8137] | ms |
| tpot_ms | 5 | 58.1299 | 2.3532 | [56.3891, 59.8708] | ms/token |
| e2e_ms | 5 | 41030.6452 | 1475.8154 | [39926.8295, 42134.4609] | ms |
| first_target_delay_calls | 5 | 13.6000 | 9.7877 | [6.4000, 22.0000] | calls |
| first_target_delay_ms | 5 | 1385.8211 | 680.5462 | [876.1327, 1937.0109] | ms |
| settle_target_delay_calls | 5 | 13.6000 | 9.7877 | [6.4000, 22.0000] | calls |
| settle_target_delay_ms | 5 | 1385.8211 | 680.5462 | [876.1327, 1937.0109] | ms |
| pre_transition_calls_per_s | 5 | 28.9988 | 0.3257 | [28.7471, 29.2482] | calls/s |
| post_transition_calls_per_s | 5 | 15.4577 | 0.5863 | [14.9522, 15.8850] | calls/s |
| post_vs_pre_rate | 5 | 0.5330 | 0.0164 | [0.5197, 0.5452] | x |
| background_ready_ms | 5 | 175.2481 | 250.8989 | [53.5465, 400.3048] | ms |
| ready_to_first_target_calls | 5 | 11.4000 | 10.0648 | [4.4000, 20.0000] | calls |
| ready_to_first_target_ms | 5 | 1210.5731 | 831.3471 | [592.2325, 1828.9136] | ms |
| ready_to_settle_target_calls | 5 | 11.4000 | 10.0648 | [3.8000, 20.0000] | calls |
| ready_to_settle_target_ms | 5 | 1210.5731 | 831.3471 | [592.2325, 1881.3982] | ms |
| client_pre_transition_tps | 5 | 28.9552 | 0.4040 | [28.6378, 29.2640] | token/s |
| client_post_transition_tps | 5 | 15.1904 | 0.6887 | [14.6377, 15.7002] | token/s |
| client_post_vs_pre_tps | 5 | 0.5245 | 0.0187 | [0.5100, 0.5390] | x |

## Files

- [Cycle data](cycles.csv)
- [Aggregate statistics](summary.csv)
- [Manifest](manifest.json)
