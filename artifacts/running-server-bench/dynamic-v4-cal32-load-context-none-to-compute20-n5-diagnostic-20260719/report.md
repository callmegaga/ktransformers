# Dynamic Load Transition Cycles

- Cycles: 5
- Direction: low-to-high
- Target: igpu
- Pre-transition rate excludes the first prefill/decode boundary interval.
- Confidence intervals are percentile bootstrap intervals of the cycle mean.

## Cycle Results

| Cycle | Decode tok/s | First target calls | First target ms | Settle calls | Settle ms | Pre calls/s | Post calls/s | Client pre tok/s | Client post tok/s | Exact token timestamps | Ready ms | Exploration calls | Switches |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| 1 | 17.716 | 49 | 4240.99 | 49 | 4240.99 | 29.829 | 16.309 | 29.833 | 15.616 | yes | 46.33 | 0 | 1 |
| 2 | 18.615 | 3 | 453.25 | 3 | 453.25 | 29.787 | 16.720 | 29.789 | 16.559 | yes | 46.82 | 0 | 1 |
| 3 | 18.754 | 2 | 178.31 | 2 | 178.31 | 29.621 | 16.752 | 29.621 | 16.722 | yes | 50.70 | 0 | 1 |
| 4 | 19.095 | 18 | 1346.08 | 18 | 1346.08 | 29.673 | 17.276 | 29.673 | 17.079 | yes | 63.95 | 0 | 1 |
| 5 | 15.211 | 3 | 821.71 | 3 | 821.71 | 29.662 | 13.318 | 29.661 | 13.098 | yes | 185.52 | 32 | 3 |

## Aggregate Results

| Metric | N | Mean | Sample stdev | Bootstrap 95% CI | Unit |
|---|---:|---:|---:|---:|---|
| prefill_tps | 5 | 169.6930 | 1.1929 | [168.6061, 170.3145] | token/s |
| decode_tps | 5 | 17.8780 | 1.5758 | [16.4202, 18.8625] | token/s |
| ttft_ms | 5 | 6164.3204 | 43.7350 | [6141.8323, 6203.9094] | ms |
| tpot_ms | 5 | 56.3207 | 5.4812 | [53.0211, 61.2098] | ms/token |
| e2e_ms | 5 | 39900.5080 | 3285.4656 | [37904.9786, 42830.8651] | ms |
| first_target_delay_calls | 5 | 15.0000 | 20.1370 | [2.6000, 33.4050] | calls |
| first_target_delay_ms | 5 | 1408.0680 | 1642.9458 | [416.9656, 2873.2787] | ms |
| settle_target_delay_calls | 5 | 15.0000 | 20.1370 | [2.6000, 33.4000] | calls |
| settle_target_delay_ms | 5 | 1408.0680 | 1642.9458 | [435.6715, 2904.4590] | ms |
| pre_transition_calls_per_s | 5 | 29.7144 | 0.0889 | [29.6477, 29.7873] | calls/s |
| post_transition_calls_per_s | 5 | 16.0752 | 1.5790 | [14.6789, 16.9716] | calls/s |
| post_vs_pre_rate | 5 | 0.5410 | 0.0529 | [0.4939, 0.5718] | x |
| background_ready_ms | 5 | 78.6642 | 60.1590 | [47.3991, 133.3686] | ms |
| ready_to_first_target_calls | 5 | 13.0000 | 20.1370 | [0.6000, 31.4000] | calls |
| ready_to_first_target_ms | 5 | 1329.4037 | 1657.4576 | [340.8535, 2798.7430] | ms |
| ready_to_settle_target_calls | 5 | 13.0000 | 20.1370 | [0.6000, 31.4000] | calls |
| ready_to_settle_target_ms | 5 | 1329.4037 | 1657.4576 | [340.8535, 2798.7430] | ms |
| client_pre_transition_tps | 5 | 29.7155 | 0.0908 | [29.6475, 29.7900] | token/s |
| client_post_transition_tps | 5 | 15.8146 | 1.6122 | [14.3975, 16.8322] | token/s |
| client_post_vs_pre_tps | 5 | 0.5322 | 0.0543 | [0.4847, 0.5672] | x |

## Files

- [Cycle data](cycles.csv)
- [Aggregate statistics](summary.csv)
- [Manifest](manifest.json)
