
## Part 1 - Adaptive multi-band wavelet (COD10K)

| Config                          | Protocol                                                 | S-alpha | F-w-beta | E-phi  | MAE    | bound-F |
|---------------------------------|----------------------------------------------------------|---------|----------|--------|--------|---------|
| K=2 adaptive kband              | 384px K=2 learnable hf=hh edge=kband bw=adaptive r=2     | 0.8016  | 0.6588   | 0.8730 | 0.0402 | 0.3772  |
| K=3 ablation: fixed filters     | 384px K=3 fixed-filters hf=hh edge=kband bw=adaptive r=2 | 0.7902  | 0.6463   | 0.8625 | 0.0436 | 0.3774  |
| K=3 ablation: no band weighting | 384px K=3 learnable hf=hh edge=kband r=2                 | 0.8021  | 0.6578   | 0.8707 | 0.0409 | 0.3763  |
| K=3 adaptive kband              | 384px K=3 learnable hf=hh edge=kband bw=adaptive r=2     | 0.8029  | 0.6605   | 0.8698 | 0.0404 | 0.3789  |
| K=4 adaptive kband              | 384px K=4 learnable hf=hh edge=kband bw=adaptive r=2     | 0.7975  | 0.6515   | 0.8671 | 0.0425 | 0.3802  |
| K=6 adaptive kband              | 384px K=6 learnable hf=hh edge=kband bw=adaptive r=2     | 0.7915  | 0.6531   | 0.8613 | 0.0406 | 0.3358  |

<!-- differences
  K=3 ablation: fixed filters: d(S-alpha) = -0.0114 vs K=2 -- real
  K=3 ablation: no band weighting: d(S-alpha) = +0.0005 vs K=2 -- within noise
  K=3 adaptive kband: d(S-alpha) = +0.0013 vs K=2 -- within noise
  K=4 adaptive kband: d(S-alpha) = -0.0041 vs K=2 -- within noise
  K=6 adaptive kband: d(S-alpha) = -0.0101 vs K=2 -- real
-->

Report mean E-phi. A gain of ~0.01 S-alpha is a real improvement;
0.002-0.003 is run-to-run noise and needs multiple seeds to claim.
