# Dynamic Load Transition Cycles

- Cycles: 5
- Direction: low-to-high
- Target: igpu
- Pre-transition rate excludes the first prefill/decode boundary interval.
- Confidence intervals are percentile bootstrap intervals of the cycle mean.

## Cycle Results

| Cycle | Decode tok/s | First target calls | First target ms | Settle calls | Settle ms | Pre calls/s | Post calls/s | Ready ms | Exploration calls | Switches |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18.411 | 14 | 1174.47 | 14 | 1174.47 | 28.963 | 16.637 | 59.10 | 0 | 1 |
| 2 | 18.040 | 3 | 735.50 | 3 | 735.50 | 28.970 | 16.352 | 48.43 | 0 | 1 |
| 3 | 19.240 | 8 | 615.52 | 8 | 615.52 | 28.893 | 17.438 | 52.19 | 0 | 1 |
| 4 | 18.661 | 7 | 1219.53 | 7 | 1219.53 | 28.916 | 17.220 | 58.17 | 0 | 1 |
| 5 | 18.535 | 35 | 3248.19 | 35 | 3248.19 | 28.904 | 17.364 | 2385.62 | 0 | 1 |

## Aggregate Results

| Metric | N | Mean | Sample stdev | Bootstrap 95% CI | Unit |
|---|---:|---:|---:|---:|---|
| prefill_tps | 5 | 169.4564 | 0.2251 | [169.2814, 169.6314] | token/s |
| decode_tps | 5 | 18.5775 | 0.4368 | [18.2386, 18.9579] | token/s |
| ttft_ms | 5 | 6172.6871 | 8.1996 | [6166.3130, 6179.0611] | ms |
| tpot_ms | 5 | 53.8521 | 1.2557 | [52.7661, 54.7668] | ms/token |
| e2e_ms | 5 | 38430.2970 | 758.2653 | [37775.2468, 39025.0104] | ms |
| first_target_delay_calls | 5 | 13.4000 | 12.7004 | [5.6000, 24.4000] | calls |
| first_target_delay_ms | 5 | 1398.6432 | 1067.2619 | [760.3178, 2339.9207] | ms |
| settle_target_delay_calls | 5 | 13.4000 | 12.7004 | [5.6000, 24.4000] | calls |
| settle_target_delay_ms | 5 | 1398.6432 | 1067.2619 | [760.3178, 2339.9207] | ms |
| pre_transition_calls_per_s | 5 | 28.9294 | 0.0351 | [28.9022, 28.9566] | calls/s |
| post_transition_calls_per_s | 5 | 17.0022 | 0.4809 | [16.6113, 17.3649] | calls/s |
| post_vs_pre_rate | 5 | 0.5877 | 0.0173 | [0.5737, 0.6008] | x |
| background_ready_ms | 5 | 520.7036 | 1042.5305 | [51.4382, 1453.4463] | ms |
| ready_to_first_target_calls | 5 | 5.0000 | 4.5277 | [1.8000, 8.6000] | calls |
| ready_to_first_target_ms | 5 | 877.9397 | 260.9345 | [672.6739, 1083.2054] | ms |
| ready_to_settle_target_calls | 5 | 5.0000 | 4.5277 | [1.8000, 8.6000] | calls |
| ready_to_settle_target_ms | 5 | 877.9397 | 260.9345 | [672.6739, 1083.2054] | ms |

## Files

- [Cycle data](cycles.csv)
- [Aggregate statistics](summary.csv)
- [Manifest](manifest.json)
