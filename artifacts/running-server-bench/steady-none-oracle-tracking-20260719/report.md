# Blocked Strategy Comparison

- Strategy means give equal weight to each block.
- Confidence intervals resample blocks and then requests within selected blocks.
- Positive effect percentages favor the candidate.

## Block Means

| Strategy | Block | N | Prefill tok/s | Decode tok/s | TTFT ms | TPOT ms | E2E ms | CPU busy | CPU PSI some |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dynamic-v5 | 1 | 5 | 169.547 | 29.686 | 6169.40 | 33.69 | 26347.19 | 0.4302 | 0.0002 |
| phase-fixed | 1 | 5 | 168.288 | 19.435 | 6215.99 | 51.52 | 37076.60 | 0.1452 | 0.0012 |
| packed-cpu-fixed | 1 | 5 | 169.820 | 29.858 | 6159.55 | 33.49 | 26221.21 | 0.4301 | 0.0002 |

## Strategy Statistics

| Strategy | Metric | Blocks | Samples | Mean | Block stdev | Hierarchical bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---:|
| dynamic-v5 | prefill_tps | 1 | 5 | 169.5466 | 0.0000 | [169.4314, 169.6617] |
| dynamic-v5 | decode_tps | 1 | 5 | 29.6862 | 0.0000 | [29.6654, 29.7124] |
| dynamic-v5 | ttft_ms | 1 | 5 | 6169.4006 | 0.0000 | [6165.2114, 6173.5898] |
| dynamic-v5 | tpot_ms | 1 | 5 | 33.6857 | 0.0000 | [33.6539, 33.7098] |
| dynamic-v5 | e2e_ms | 1 | 5 | 26347.1906 | 0.0000 | [26332.3058, 26360.4582] |
| phase-fixed | prefill_tps | 1 | 5 | 168.2883 | 0.0000 | [166.8058, 169.2335] |
| phase-fixed | decode_tps | 1 | 5 | 19.4347 | 0.0000 | [18.8223, 20.0472] |
| phase-fixed | ttft_ms | 1 | 5 | 6215.9853 | 0.0000 | [6181.3762, 6271.3794] |
| phase-fixed | tpot_ms | 1 | 5 | 51.5201 | 0.0000 | [49.9020, 53.1381] |
| phase-fixed | e2e_ms | 1 | 5 | 37076.5969 | 0.0000 | [36138.1698, 38015.0240] |
| packed-cpu-fixed | prefill_tps | 1 | 5 | 169.8195 | 0.0000 | [169.2695, 170.2335] |
| packed-cpu-fixed | decode_tps | 1 | 5 | 29.8582 | 0.0000 | [29.8091, 29.9235] |
| packed-cpu-fixed | ttft_ms | 1 | 5 | 6159.5473 | 0.0000 | [6144.5172, 6179.5748] |
| packed-cpu-fixed | tpot_ms | 1 | 5 | 33.4918 | 0.0000 | [33.4188, 33.5468] |
| packed-cpu-fixed | e2e_ms | 1 | 5 | 26221.2080 | 0.0000 | [26194.1381, 26246.0789] |

## Candidate Effects

| Baseline | Metric | Effect | Hierarchical bootstrap 95% CI |
|---|---|---:|---:|
| phase-fixed | prefill_tps improvement | +0.75% | [+0.19%, +1.64%] |
| phase-fixed | decode_tps improvement | +52.75% | [+48.05%, +57.64%] |
| phase-fixed | ttft_ms reduction | +0.75% | [+0.19%, +1.61%] |
| phase-fixed | tpot_ms reduction | +34.62% | [+32.50%, +36.59%] |
| phase-fixed | e2e_ms reduction | +28.94% | [+27.09%, +30.67%] |
| packed-cpu-fixed | prefill_tps improvement | -0.16% | [-0.42%, +0.17%] |
| packed-cpu-fixed | decode_tps improvement | -0.58% | [-0.81%, -0.38%] |
| packed-cpu-fixed | ttft_ms reduction | -0.16% | [-0.42%, +0.17%] |
| packed-cpu-fixed | tpot_ms reduction | -0.58% | [-0.82%, -0.39%] |
| packed-cpu-fixed | e2e_ms reduction | -0.48% | [-0.60%, -0.37%] |

## Files

- [Normalized samples](samples.csv)
- [Block means](blocks.csv)
- [Strategy statistics](summary.csv)
- [Candidate effects](effects.csv)
- [Manifest](manifest.json)
