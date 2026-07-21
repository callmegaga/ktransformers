# Running-Server CPU Load Sweep

- Status: complete
- Protocol: cpu-igpu-steady-v1
- Backend: dynamic
- Block: pilot-b1
- Load order: [0, 12, 20, 16, 8, 4]
- Practical-equivalence margin: 2.00%

| Order | Load | Status | Prefill tok/s | Decode tok/s | TTFT ms | TPOT ms |
|---:|---|---|---:|---:|---:|---:|
| 1 | none | complete | 167.04049827872296 | 29.483116572724793 | 6267.048739800157 | 33.920682414023176 |
| 2 | compute12 | complete | 129.13455601293353 | 21.059196882298814 | 8103.9490358001785 | 47.486118949916076 |
| 3 | compute20 | complete | 79.54864115365658 | 13.446601881098621 | 13223.02586079968 | 76.39041633789682 |
| 4 | compute16 | complete | 98.01727773377299 | 19.662427282256992 | 10688.020462199893 | 53.201193295158475 |
| 5 | compute8 | complete | 164.8328255575002 | 22.408265652973064 | 6348.58843819984 | 44.82822539165301 |
| 6 | compute4 | complete | 169.07481115240262 | 24.155561337453832 | 6188.987566400101 | 41.398894168280535 |

## Files

- [Manifest](manifest.json)
- [Combined summary](summary.csv)
- Per-load artifacts are under `runs/`.
