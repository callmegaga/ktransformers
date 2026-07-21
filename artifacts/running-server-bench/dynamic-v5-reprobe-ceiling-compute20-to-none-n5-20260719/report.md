# Dynamic Load Transition Cycles

- Cycles: 5
- Direction: high-to-low
- Target: cpu
- Pre-transition rate excludes the first prefill/decode boundary interval.
- Confidence intervals are percentile bootstrap intervals of the cycle mean.

## Cycle Results

| Cycle | Decode tok/s | First target calls | First target ms | Settle calls | Settle ms | Pre calls/s | Post calls/s | Client pre tok/s | Client post tok/s | Exact token timestamps | Ready ms | Exploration calls | Switches |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| 1 | 22.897 | 14 | 731.54 | 46 | 1849.65 | 14.490 | 28.849 | 14.462 | 28.378 | yes | NA | 32 | 1 |
| 2 | 24.195 | 15 | 741.81 | 47 | 1823.91 | 16.174 | 29.341 | 16.198 | 28.924 | yes | NA | 32 | 1 |
| 3 | 23.571 | 12 | 629.74 | 44 | 1725.05 | 15.514 | 28.770 | 15.512 | 28.468 | yes | NA | 32 | 1 |
| 4 | 24.069 | 15 | 786.43 | 47 | 1872.48 | 16.049 | 29.316 | 16.076 | 28.812 | yes | NA | 32 | 1 |
| 5 | 23.049 | 14 | 739.72 | 46 | 1858.90 | 14.787 | 28.867 | 14.709 | 28.375 | yes | NA | 32 | 1 |

## Aggregate Results

| Metric | N | Mean | Sample stdev | Bootstrap 95% CI | Unit |
|---|---:|---:|---:|---:|---|
| prefill_tps | 5 | 146.9586 | 3.2497 | [144.3234, 149.3988] | token/s |
| decode_tps | 5 | 23.5563 | 0.5838 | [23.0926, 24.0200] | token/s |
| ttft_ms | 5 | 7120.4452 | 158.1027 | [7003.0461, 7251.0473] | ms |
| tpot_ms | 5 | 42.4724 | 1.0536 | [41.6360, 43.3089] | ms/token |
| e2e_ms | 5 | 32561.4902 | 622.8783 | [32074.8667, 33034.8230] | ms |
| first_target_delay_calls | 5 | 14.0000 | 1.2247 | [13.0000, 14.8000] | calls |
| first_target_delay_ms | 5 | 725.8464 | 57.8523 | [673.7271, 766.5292] | ms |
| settle_target_delay_calls | 5 | 46.0000 | 1.2247 | [45.0000, 46.8000] | calls |
| settle_target_delay_ms | 5 | 1825.9975 | 59.1564 | [1774.3066, 1862.4827] | ms |
| pre_transition_calls_per_s | 5 | 15.4029 | 0.7479 | [14.8136, 15.9922] | calls/s |
| post_transition_calls_per_s | 5 | 29.0287 | 0.2765 | [28.8210, 29.2379] | calls/s |
| post_vs_pre_rate | 5 | 1.8877 | 0.0792 | [1.8272, 1.9503] | x |
| client_pre_transition_tps | 5 | 15.3916 | 0.7846 | [14.7711, 16.0121] | token/s |
| client_post_transition_tps | 5 | 28.5914 | 0.2583 | [28.3948, 28.7922] | token/s |
| client_post_vs_pre_tps | 5 | 1.8609 | 0.0805 | [1.7982, 1.9235] | x |

## Files

- [Cycle data](cycles.csv)
- [Aggregate statistics](summary.csv)
- [Manifest](manifest.json)
