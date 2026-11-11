# Midway check (12 Nov 2026): role 1

Scope for this milestone: CN and Adamic-Adar on the official validation and test
pairs, preliminary results, reproducible code, and blockers. Run completed on
11 Nov 2026, on a Windows CPU with Python 3.11.18.

## Done

- Reused the shared loader, official OGB evaluator, score-file convention and
  result summarizer. No changes to the training graph or learned models.
- Implemented CN and AA in `collab/heuristics.py`. Batched sparse row
  intersections compute both methods together, without a dense all-pairs matrix
  or materialising all of `A**2`. AA uses natural logarithms and float64
  accumulation before saving float32 scores.
- Saved all eight `scores/{cn,aa}_seed0_{split}.npy` arrays, in official OGB
  order. Scores and the downloaded dataset are generated, git-ignored files;
  rerun the documented command to reproduce them on another machine.
- Committed metrics and environment/timing metadata in `results/cn.jsonl` and
  `results/aa.jsonl`, and regenerated `results/summary.md`.
- Added PowerShell instructions to the README, a `make midway-role1` target,
  and five independent unit tests. The `make heuristics` target now runs the
  implementation and propagates failures instead of hiding them as a placeholder.
- Made summary-file encoding explicitly UTF-8 for Windows.

## Data and protocol audit

The downloaded official splits were checked against the shared tensors:

| Item | Observed |
|---|---:|
| Authors | 235,868 |
| Raw training records, years <= 2017 | 1,179,052 |
| Unique undirected training edges | 967,632 |
| Directed entries of symmetric training adjacency | 1,935,264 |
| Validation positives, year 2018 | 60,084 |
| Test positives, year 2019 | 46,329 |
| Negatives per evaluation split | 100,000 |
| Degree-0 authors in the training graph | 0 |

The adjacency is binary and loop-free. Historical pairs are not removed from
the evaluation positives. All evaluation pairs retain their original order.
The official negatives include one self-pair in each split (the same author,
ID 112151); it is retained and scored, not filtered. Its neighbours all have
degree above 1, so AA is finite. Zero weights for degree-0/1 common-neighbour
vertices avoid division by zero; such vertices cannot contribute to queries
with distinct endpoints.

Validation and test both use only the training graph. No 2018 edges are added
for this milestone, no hyperparameters are selected using test, and there is
no random seed variation for these deterministic methods.

## Preliminary results

Official Hits@K multiplied by 100 (percent); one deterministic run per method.

| Method | Val @20 | Val @50 | Val @100 | Test @20 | Test @50 | Test @100 |
|---|---:|---:|---:|---:|---:|---:|
| CN | 52.36 | 52.36 | 64.89 | 41.70 | 41.70 | 54.54 |
| AA | 55.48 | 63.01 | 64.89 | 44.64 | 52.40 | 54.54 |

Both methods together scored all 306,413 pairs in **0.136 seconds** at batch
size 4096. This is scoring time only: the first-run download, preprocessing,
data loading and adjacency preparation took **129.022 seconds** separately;
evaluation and writing files are not included in the scoring timer. These are
one-run local measurements, not a benchmark against the learned models.

The global negative threshold is computed separately for each split and K.
In these official splits the observed thresholds coincide between val/test:

| Method | Threshold @20 | Threshold @50 | Threshold @100 |
|---|---:|---:|---:|
| CN | 1 | 1 | 0 |
| AA | 0.34597626 | 0.20544282 | 0 |

The evaluator uses strict `score > threshold`. This explains why CN @20 and
@50 are equal: a positive pair with exactly one common neighbour misses both.
At @100, both methods hit exactly the pairs with a common neighbour. AA's test
Hits@50 exceeds CN by 10.70 percentage points. This preliminary comparison is
consistent with AA providing finer ranking than integer CN; CN with an AA
tie-break remains a separate follow-up experiment.

## Verification

- Existing four checks in `python -m collab.selftest`: passed.
- Five tests in `python -m unittest -v collab.test_heuristics`: passed, covering
  manual CN/AA values, random-graph set references, batching and pair order,
  duplicate/asymmetric edges, isolated vertices, empty input, diagonal queries,
  shared graph helpers, float32 score round-trips and strict OGB thresholds.
- Full-data audit: split years and sizes, canonical training edges, evaluation
  order, finite float32 score arrays and saved-metric round-trips passed.
- Independently checked 2,046 saved scores against Python neighbour-set
  calculations (random queries, highest-scoring queries and official self-pairs).
- Dependency compatibility check: passed. Key versions are torch 2.5.1,
  torch-geometric 2.6.1, ogb 1.3.6, NumPy 1.26.4, SciPy 1.17.1 and pandas 3.0.6.

## Blockers and next work

There are no remaining blockers for the role 1 midway CN/AA deliverable.

After midway:

1. Implement CN with AA tie-breaking, preserving its ordering in saved scores.
2. Implement and validate the three bucket partitions and per-bucket Hits@50
   at each method's global negative threshold. The degree-0 bucket is empty in
   this graph and must be reported as empty, not as zero quality.
3. Obtain the per-pair score files from roles 2 and 3. The repository currently
   contains MLP/node2vec aggregate metrics only, and no GraphSAGE results; these
   cannot support cross-method bucket analysis or AA/GraphSAGE intersections.
4. Before interpreting GraphSAGE, role 3 must meet the reproduction criterion
   from the proposal. Then compare all final seeds and aggregate bucket results.
5. Re-score CN/AA on training + 2018 edges for the separate test-only
   `val as input` comparison; integrate GraphSAGE's corresponding scores.
6. Write the analysis section and assemble the final six-page team report.

Coordination note for role 2: the current node2vec grid still invokes the shared
training function that scores test for every candidate. A validation-only mode
is needed before further tuning to enforce the proposal's final-configuration
test rule; that learned-model change is outside this midway implementation.
