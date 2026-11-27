# Test bucket analysis

Buckets use the training-only graph. Each run uses its own global 50th-negative
threshold and strict `>`; Hits@50 is shown as percent, mean +/- sample std across seeds.
An empty bucket is N/A. A single seed has no estimated standard deviation.

## Available inputs

| Method | Seed(s) | Status |
|---|---|---|
| aa | 0 | available |
| cn | 0 | available |
| cn_aa | 0 | available |
| mlp | 0 | missing scores (metrics only) |
| node2vec | - | missing scores |
| node2vec_e1 | 0 | missing scores (metrics only) |
| node2vec_e3 | 0 | missing scores (metrics only) |
| sage | 0 | missing scores (metrics only) |

## Hits@50 by bucket

| Partition | Bucket | Size | Share (%) | aa | cn | cn_aa |
|---|---|---:|---:|---:|---:|---:|
| all | all | 46329 | 100.00 | 52.40 | 41.70 | 52.40 |
| common_neighbors | 0 | 21061 | 45.46 | 0.00 | 0.00 | 0.00 |
| common_neighbors | 1 | 5949 | 12.84 | 83.34 | 0.00 | 83.34 |
| common_neighbors | 2-4 | 9466 | 20.43 | 100.00 | 100.00 | 100.00 |
| common_neighbors | >=5 | 9853 | 21.27 | 100.00 | 100.00 | 100.00 |
| min_degree | 0 | 0 | 0.00 | N/A | N/A | N/A |
| min_degree | 1-5 | 17489 | 37.75 | 41.13 | 26.36 | 41.13 |
| min_degree | >5 | 28840 | 62.25 | 59.23 | 51.00 | 59.23 |
| pair_history | repeat | 16965 | 36.62 | 97.07 | 89.03 | 97.07 |
| pair_history | new | 29364 | 63.38 | 26.59 | 14.35 | 26.59 |

## Hit intersections (counts, per seed)

| Left | Right | Partition | Bucket | Both | Left only | Right only | Neither |
|---|---|---|---|---:|---:|---:|---:|
| aa seed0 | cn seed0 | all | all | 19319 | 4958 | 0 | 22052 |
| aa seed0 | cn seed0 | common_neighbors | 0 | 0 | 0 | 0 | 21061 |
| aa seed0 | cn seed0 | common_neighbors | 1 | 0 | 4958 | 0 | 991 |
| aa seed0 | cn seed0 | common_neighbors | 2-4 | 9466 | 0 | 0 | 0 |
| aa seed0 | cn seed0 | common_neighbors | >=5 | 9853 | 0 | 0 | 0 |
| aa seed0 | cn seed0 | min_degree | 0 | 0 | 0 | 0 | 0 |
| aa seed0 | cn seed0 | min_degree | 1-5 | 4610 | 2584 | 0 | 10295 |
| aa seed0 | cn seed0 | min_degree | >5 | 14709 | 2374 | 0 | 11757 |
| aa seed0 | cn seed0 | pair_history | repeat | 15104 | 1364 | 0 | 497 |
| aa seed0 | cn seed0 | pair_history | new | 4215 | 3594 | 0 | 21555 |
| aa seed0 | cn_aa seed0 | all | all | 24277 | 0 | 0 | 22052 |
| aa seed0 | cn_aa seed0 | common_neighbors | 0 | 0 | 0 | 0 | 21061 |
| aa seed0 | cn_aa seed0 | common_neighbors | 1 | 4958 | 0 | 0 | 991 |
| aa seed0 | cn_aa seed0 | common_neighbors | 2-4 | 9466 | 0 | 0 | 0 |
| aa seed0 | cn_aa seed0 | common_neighbors | >=5 | 9853 | 0 | 0 | 0 |
| aa seed0 | cn_aa seed0 | min_degree | 0 | 0 | 0 | 0 | 0 |
| aa seed0 | cn_aa seed0 | min_degree | 1-5 | 7194 | 0 | 0 | 10295 |
| aa seed0 | cn_aa seed0 | min_degree | >5 | 17083 | 0 | 0 | 11757 |
| aa seed0 | cn_aa seed0 | pair_history | repeat | 16468 | 0 | 0 | 497 |
| aa seed0 | cn_aa seed0 | pair_history | new | 7809 | 0 | 0 | 21555 |

Missing score files cannot be reconstructed from aggregate metrics. All results remain
preliminary until the planned final seeds and cross-method inputs are available.
