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
| mlp_local | 0 | available |
| node2vec | - | missing scores |
| node2vec_e1 | 0 | missing scores (metrics only) |
| node2vec_e3 | 0 | missing scores (metrics only) |
| node2vec_local | 0 | available |
| sage | 0 | missing scores (metrics only) |

## Hits@50 by bucket

| Partition | Bucket | Size | Share (%) | aa | cn | cn_aa | mlp_local | node2vec_local |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | all | 46329 | 100.00 | 52.40 | 41.70 | 52.40 | 12.62 | 49.21 |
| common_neighbors | 0 | 21061 | 45.46 | 0.00 | 0.00 | 0.00 | 2.40 | 7.16 |
| common_neighbors | 1 | 5949 | 12.84 | 83.34 | 0.00 | 83.34 | 7.75 | 50.90 |
| common_neighbors | 2-4 | 9466 | 20.43 | 100.00 | 100.00 | 100.00 | 15.81 | 89.26 |
| common_neighbors | >=5 | 9853 | 21.27 | 100.00 | 100.00 | 100.00 | 34.34 | 99.58 |
| min_degree | 0 | 0 | 0.00 | N/A | N/A | N/A | N/A | N/A |
| min_degree | 1-5 | 17489 | 37.75 | 41.13 | 26.36 | 41.13 | 7.39 | 33.06 |
| min_degree | >5 | 28840 | 62.25 | 59.23 | 51.00 | 59.23 | 15.79 | 59.00 |
| pair_history | repeat | 16965 | 36.62 | 97.07 | 89.03 | 97.07 | 26.96 | 95.49 |
| pair_history | new | 29364 | 63.38 | 26.59 | 14.35 | 26.59 | 4.34 | 22.47 |

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
| aa seed0 | mlp_local seed0 | all | all | 5270 | 19007 | 577 | 21475 |
| aa seed0 | mlp_local seed0 | common_neighbors | 0 | 0 | 0 | 505 | 20556 |
| aa seed0 | mlp_local seed0 | common_neighbors | 1 | 389 | 4569 | 72 | 919 |
| aa seed0 | mlp_local seed0 | common_neighbors | 2-4 | 1497 | 7969 | 0 | 0 |
| aa seed0 | mlp_local seed0 | common_neighbors | >=5 | 3384 | 6469 | 0 | 0 |
| aa seed0 | mlp_local seed0 | min_degree | 0 | 0 | 0 | 0 | 0 |
| aa seed0 | mlp_local seed0 | min_degree | 1-5 | 1144 | 6050 | 148 | 10147 |
| aa seed0 | mlp_local seed0 | min_degree | >5 | 4126 | 12957 | 429 | 11328 |
| aa seed0 | mlp_local seed0 | pair_history | repeat | 4518 | 11950 | 55 | 442 |
| aa seed0 | mlp_local seed0 | pair_history | new | 752 | 7057 | 522 | 21033 |
| aa seed0 | node2vec_local seed0 | all | all | 20908 | 3369 | 1889 | 20163 |
| aa seed0 | node2vec_local seed0 | common_neighbors | 0 | 0 | 0 | 1508 | 19553 |
| aa seed0 | node2vec_local seed0 | common_neighbors | 1 | 2647 | 2311 | 381 | 610 |
| aa seed0 | node2vec_local seed0 | common_neighbors | 2-4 | 8449 | 1017 | 0 | 0 |
| aa seed0 | node2vec_local seed0 | common_neighbors | >=5 | 9812 | 41 | 0 | 0 |
| aa seed0 | node2vec_local seed0 | min_degree | 0 | 0 | 0 | 0 | 0 |
| aa seed0 | node2vec_local seed0 | min_degree | 1-5 | 5397 | 1797 | 384 | 9911 |
| aa seed0 | node2vec_local seed0 | min_degree | >5 | 15511 | 1572 | 1505 | 10252 |
| aa seed0 | node2vec_local seed0 | pair_history | repeat | 15907 | 561 | 293 | 204 |
| aa seed0 | node2vec_local seed0 | pair_history | new | 5001 | 2808 | 1596 | 19959 |

Missing score files cannot be reconstructed from aggregate metrics. All results remain
preliminary until the planned final seeds and cross-method inputs are available.
