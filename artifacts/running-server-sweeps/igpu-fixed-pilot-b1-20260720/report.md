# Running-Server CPU Load Sweep

- Status: complete
- Protocol: cpu-igpu-steady-v1
- Backend: igpu-fixed
- Block: pilot-b1
- Load order: [0, 12, 20, 16, 8, 4]
- Practical-equivalence margin: 2.00%

| Order | Load | Status | Prefill tok/s | Decode tok/s | TTFT ms | TPOT ms |
|---:|---|---|---:|---:|---:|---:|
| 1 | none | complete | 97.4461492907607 | 16.616502527110395 | 10834.213583200011 | 60.24213516861435 |
| 2 | compute12 | complete | 103.27422804200873 | 18.673355890729244 | 10132.279999599972 | 53.55265204974965 |
| 3 | compute20 | complete | 96.6254683935728 | 13.539037262679852 | 10835.3701258 | 73.8747734637729 |
| 4 | compute16 | complete | 101.68062606420173 | 18.894988658006405 | 10293.655000599938 | 52.924456360267165 |
| 5 | compute8 | complete | 99.83063215312049 | 18.201701596978964 | 10481.797729799973 | 54.94686626611018 |
| 6 | compute4 | complete | 99.32054971196399 | 17.58477368465146 | 10535.835582200025 | 56.87352182804672 |

## Files

- [Manifest](manifest.json)
- [Combined summary](summary.csv)
- Per-load artifacts are under `runs/`.
