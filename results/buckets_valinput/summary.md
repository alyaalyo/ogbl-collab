# Test bucket analysis

Buckets use the training-only graph. Each run uses its own global 50th-negative
threshold and strict `>`; Hits@50 is shown as percent, mean +/- sample std across seeds.
An empty bucket is N/A. A single seed has no estimated standard deviation.

## Available inputs

| Method | Seed(s) | Status |
|---|---|---|
| aa_valinput | 0 | available |
| cn_valinput | 0 | available |
| sage_valinput | 0 | missing scores (metrics only) |

## Hits@50 by bucket

| Partition | Bucket | Size | Share (%) | aa_valinput | cn_valinput |
|---|---|---:|---:|---:|---:|
| all | all | 46329 | 100.00 | 63.06 | 52.82 |
| common_neighbors | 0 | 21061 | 45.46 | 22.48 | 14.98 |
| common_neighbors | 1 | 5949 | 12.84 | 86.77 | 33.59 |
| common_neighbors | 2-4 | 9466 | 20.43 | 100.00 | 100.00 |
| common_neighbors | >=5 | 9853 | 21.27 | 100.00 | 100.00 |
| min_degree | 0 | 0 | 0.00 | N/A | N/A |
| min_degree | 1-5 | 17489 | 37.75 | 52.87 | 38.59 |
| min_degree | >5 | 28840 | 62.25 | 69.24 | 61.45 |
| pair_history | repeat | 16965 | 36.62 | 97.73 | 91.49 |
| pair_history | new | 29364 | 63.38 | 43.03 | 30.48 |

## Hit intersections (counts, per seed)

| Left | Right | Partition | Bucket | Both | Left only | Right only | Neither |
|---|---|---|---|---:|---:|---:|---:|
| aa_valinput seed0 | cn_valinput seed0 | all | all | 24472 | 4743 | 0 | 17114 |
| aa_valinput seed0 | cn_valinput seed0 | common_neighbors | 0 | 3155 | 1579 | 0 | 16327 |
| aa_valinput seed0 | cn_valinput seed0 | common_neighbors | 1 | 1998 | 3164 | 0 | 787 |
| aa_valinput seed0 | cn_valinput seed0 | common_neighbors | 2-4 | 9466 | 0 | 0 | 0 |
| aa_valinput seed0 | cn_valinput seed0 | common_neighbors | >=5 | 9853 | 0 | 0 | 0 |
| aa_valinput seed0 | cn_valinput seed0 | min_degree | 0 | 0 | 0 | 0 | 0 |
| aa_valinput seed0 | cn_valinput seed0 | min_degree | 1-5 | 6749 | 2498 | 0 | 8242 |
| aa_valinput seed0 | cn_valinput seed0 | min_degree | >5 | 17723 | 2245 | 0 | 8872 |
| aa_valinput seed0 | cn_valinput seed0 | pair_history | repeat | 15521 | 1059 | 0 | 385 |
| aa_valinput seed0 | cn_valinput seed0 | pair_history | new | 8951 | 3684 | 0 | 16729 |

Missing score files cannot be reconstructed from aggregate metrics. All results remain
preliminary until the planned final seeds and cross-method inputs are available.
