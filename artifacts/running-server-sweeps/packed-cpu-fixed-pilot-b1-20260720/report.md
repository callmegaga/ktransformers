# Running-Server CPU Load Sweep

- Status: complete
- Protocol: cpu-igpu-steady-v1
- Backend: packed-cpu-fixed
- Block: pilot-b1
- Load order: [0, 12, 20, 16, 8, 4]
- Practical-equivalence margin: 2.00%

| Order | Load | Status | Prefill tok/s | Decode tok/s | TTFT ms | TPOT ms |
|---:|---|---|---:|---:|---:|---:|
| 1 | none | complete | 166.01471774754057 | 29.629302716642496 | 6309.248894799748 | 33.75209108013338 |
| 2 | compute12 | complete | 125.01027880068793 | 1.8828498751165708 | 8383.077747999778 | 533.7535745268781 |
| 3 | compute20 | complete | 71.57684757000109 | 1.394914375337943 | 14644.211899799484 | 717.1218721061772 |
| 4 | compute16 | complete | 90.88763400188571 | 1.4411560342999805 | 11518.445719199735 | 694.1525167569287 |
| 5 | compute8 | complete | 167.41933420106972 | 24.04498735019251 | 6250.200744399626 | 41.58880147078404 |
| 6 | compute4 | complete | 170.25833699668698 | 23.923805817027745 | 6145.956381199721 | 41.80126505141892 |

## Files

- [Manifest](manifest.json)
- [Combined summary](summary.csv)
- Per-load artifacts are under `runs/`.
