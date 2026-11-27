# Role 1 analysis progress (27 Nov 2026)

The CN/AA midway PR is merged. This follow-up adds CN with AA tie-breaking,
test-only scoring with 2018 edges, and a working bucket-analysis pipeline.
The full dataset was scored locally on CPU. Independent seed-0 MLP and node2vec
runs subsequently supplied scores for the learned-model comparisons below.

## Implemented and run

- `collab.heuristics --tie_break`: CN, AA and CN+AA on the training graph.
- `collab.heuristics --valinput`: CN/AA on training + validation edges, test
  positives/negatives only. The graph has 999,644 unique undirected edges,
  compared with 967,632 in training alone.
- `collab.buckets`: the three proposal partitions, global-threshold Hits@50,
  per-seed tables, mean/sample std, and AA hit intersections with every
  available method/seed. Deterministic AA seed 0 is compared with each learned
  seed independently; score magnitudes are never averaged across seeds.
- Both graph variants have CSV/Markdown reports, manifests with pair-order and
  source-score SHA-256 hashes, and git-ignored per-pair membership artifacts.

CN+AA scores are exact float32 ranks of `(CN, AA)` tuples. Within each split,
all positive/negative queries are ranked together, without using their labels.
This preserves lexicographic order and equal-score ties through saving/loading.
It avoids choosing a fragile epsilon; the rank values are not probabilities
and must not be compared across splits or independently ranked query batches.

## Observed results

Official test Hits@K in percent; one deterministic run per heuristic.

| Method | Input graph | Hits@20 | Hits@50 | Hits@100 |
|---|---|---:|---:|---:|
| CN | training | 41.70 | 41.70 | 54.54 |
| AA | training | 44.64 | 52.40 | 54.54 |
| CN+AA | training | 44.65 | 52.40 | 54.54 |
| CN | training + 2018 | 52.82 | 52.82 | 65.60 |
| AA | training + 2018 | 56.06 | 63.06 | 65.60 |

See [baseline bucket tables](../results/buckets/summary.md),
[valinput bucket tables](../results/buckets_valinput/summary.md), and
[all recorded model metrics](../results/summary.md).

The following conclusions apply to the observed heuristic scores:

1. **45.46% of test positives have no common neighbour in the training graph**
   (21,061 of 46,329). All three training-only heuristics miss every such pair.
2. **CN's integer ties explain its Hits@50 deficit in this run.** Its global
   threshold is 1, so all 5,949 positives with exactly one common neighbour miss.
   AA recovers 4,958 of them (83.34%) and loses none of CN's hits. All positives
   with two or more common neighbours are hits for both methods.
3. **AA and CN+AA have identical test Hits@50 hit sets**, not merely equal
   aggregate values: both hit 24,277 positives. This supports a tie-related
   explanation of the CN-to-AA gain here; it does not establish universal
   equivalence of AA and CN+AA. Their Hits@20 values differ slightly.
4. **Repeat pairs are much easier than new pairs.** AA hits 97.07% of repeat
   pairs (16,965 positives) but 26.59% of new pairs (29,364 positives). Report
   this distinction alongside the overall metric.
5. **The degree-0 bucket is empty.** It is N/A, not 0% quality, and this dataset
   split supplies no observed degree-0 evidence for a text-only cold-start claim.
6. **Adding 2018 edges changes the available structural evidence.** AA rises
   to 63.06% overall. It now hits 22.48% of pairs that had zero common neighbours
   in training. Those pairs can acquire common neighbours in the expanded
   graph, so this does not contradict the training-only limitation.

## Analysis conventions and validation

All cohorts, including the secondary valinput tables, are defined using the
original training graph. This keeps their sizes and memberships fixed when
comparing graph variants. The labels in the valinput report therefore describe
the training history, not the updated neighbourhoods.

For each method and seed, Hits@50 uses strict `>` against that run's 50th-highest
score among all 100,000 test negatives. The threshold is never recomputed within
a bucket. Empty groups are represented by null/blank values (N/A in Markdown).
Mean and sample standard deviation are computed from per-seed bucket metrics;
a single seed has no estimated standard deviation.

The test suite covers bucket boundaries and orientation-invariant repeat
pairs, global-threshold ties, empty buckets, weighted-total identities,
sample standard deviation, intersection identities, exact rank serialization,
test-only valinput, malformed/incomplete scores, stale result records, the
GraphSAGE reproduction gate, and generated output files. Existing heuristic,
GraphSAGE and shared-code tests also pass.

Full-data consistency checks verify the exact CN+AA hit mask against direct
lexicographic comparison for every validation/test positive; partition sizes
sum to 46,329, bucket hits sum to overall hits, intersection index sets form a
disjoint cover, baseline CN/AA metrics are unchanged, and valinput cohorts match
the training-only cohorts exactly.

## Local learned-model comparison

The CPU reproduction queue completed both independent seed-0 runs. MLP selected
epoch 122 and stopped at 142 (validation/test Hits@50: 16.67%/12.62%);
node2vec selected decoder epoch 10 and stopped at 30 (59.15%/49.21%).
Their results use `mlp_local` and `node2vec_local` names and do not overwrite the
team's original aggregate metrics. See [run details](local-training.md).

| Test cohort | Positives | AA Hits@50 (%) | MLP local (%) | Node2vec local (%) |
|---|---:|---:|---:|---:|
| All | 46,329 | 52.40 | 12.62 | 49.21 |
| No common neighbours | 21,061 | 0.00 | 2.40 | 7.16 |
| Exactly one common neighbour | 5,949 | 83.34 | 7.75 | 50.90 |
| Minimum endpoint degree 1-5 | 17,489 | 41.13 | 7.39 | 33.06 |
| Repeat pair | 16,965 | 97.07 | 26.96 | 95.49 |
| New pair | 29,364 | 26.59 | 4.34 | 22.47 |

AA leads node2vec by 3.19 percentage points overall in these runs, but their
hit sets differ: 20,908 positives are hits for both, 3,369 for AA only and
1,889 for node2vec only. Of node2vec's exclusive hits, 1,508 have no common
neighbour and 381 have exactly one. Thus node2vec recovers some positives
missed by local common-neighbour heuristics while losing more elsewhere.
This observed complementarity is not an evaluated ensemble result.

MLP has 577 hits outside AA's hit set, including 505 with no common neighbour.
That shows the text-only model recovers some pairs missed by AA in this
run, despite low overall quality. It is not evidence about isolated-node cold
start: the degree-0 cohort remains empty. Repeat pairs remain substantially
easier than new pairs for both local models.

All eight local score arrays were checked for shape, finite float32 values,
agreement with the official evaluator and exact reproduction from saved weights
in official pair order. Validation histories agree with the selected epochs and
early stopping. Bucket partitions and overlap artifacts pass the full-data
consistency audit recorded in [local_reproduction_audit.json](../results/local_reproduction_audit.json).
These comparisons use one seed each and do not estimate run-to-run variance.

## Remaining inputs and final-report limits

The merged repository records GraphSAGE seed 0 at 48.13% test Hits@50 and its
valinput variant at 54.69%, both within the proposal's reproduction ranges.
The original teammate runs still have only aggregate metrics locally.
Independent MLP/node2vec scores are now available; GraphSAGE scores are not.
A separate [local GraphSAGE run](local-sage.md) is now launched to obtain those
scores. Its metrics and reproduction outcome are pending.

Accordingly, these outputs contain local MLP/node2vec bucket values but no
AA/GraphSAGE disagreements. Missing methods are explicitly listed in each
report. A PDF with aggregate metrics cannot determine which pairs were hits.

To finish the main cross-method deliverable, obtain:

- final MLP/node2vec seed files and records (local seed 0 is now available;
  the original teammate runs remain separate);
- `sage_seed{k}_test_{pos,neg}.npy` and `sage_valinput_seed{k}_test_{pos,neg}.npy`;
- final seed lists and corresponding result records/configurations (ten seeds,
  or the agreed fallback), plus the teammates' report sections.

Use the exact method names from their result records. Arrays must be finite
float32, in official OGB pair order, with 46,329 positives and 100,000 negatives.
Copy them into `scores/` and rerun the two `collab.buckets` commands, or specify
`--scores_dir PATH`. The report generator checks recorded metrics when present
and enforces the GraphSAGE reproduction criterion on the available scores.
Hashes record supplied file identity; they do not prove the supplier used the
right pair order. That order remains part of the shared score-file contract.

The final six-page team report and the main AA/GraphSAGE interpretation still
depend on those inputs. Also retain role 3's documented limitations about
single-seed variance, patience changes and test evaluation during tuning.
