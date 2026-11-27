# Link Prediction Three Ways on ogbl-collab

Course project (Data and Knowledge Representation, B24-DS-01). We compare three
families of link predictors on [ogbl-collab](https://ogb.stanford.edu/docs/linkprop/#ogbl-collab)
under one protocol and then ask *which test pairs each family gets right*:

1. **Heuristics**: Common Neighbours and Adamic–Adar (no training).
2. **Shallow embeddings**: node2vec vectors, frozen, plus a learned decoder.
3. **GNN**: GraphSAGE (3 layers, 256 units) trained end-to-end.

Plus the mandatory **features-only MLP** baseline (decoder on text features, no graph).

Metric: official OGB Hits@50 (Hits@20 and Hits@100 as robustness checks). Model
selection on 2018 validation edges only; test is scored once per final config.

## Quick start

```bash
make env      # pinned virtualenv in .venv (needs uv)
make check    # toy-data tests of the shared code, < 1 min
make midway   # features-only MLP + node2vec (p=q=1, 3 epochs), seed 0, then the results table
make midway-role1  # CN + AA on val/test, then the results table (CPU only)
make role1-analysis  # CN/AA/CN+AA, test-only valinput, and bucket analysis of available scores
make sage-probe  # role 3: time one GraphSAGE epoch (3 epochs), writes results/sage_timing.jsonl
make sage-tune   # role 3: lr in {1e-3, 1e-2} on seed 0, validation only
make all      # full pipeline: all methods, all seeds, node2vec grid, summary table
```

The dataset (~340 MB unpacked) downloads itself into `data/` on first use.
`DEVICE=cpu|mps|cuda` and `SEEDS=0-9` (or `0,3,5`) override the defaults, e.g.
`make mlp SEEDS=0-4 DEVICE=cuda`.

### Colab

```python
!git clone https://github.com/alyaalyo/ogbl-collab.git
%cd ogbl-collab
!pip install -q -r requirements.txt
# keep checkpoints, embeddings and scores on Drive so a disconnect loses nothing
from google.colab import drive; drive.mount('/content/drive')
import os
for k, d in {"COLLAB_CKPT": "checkpoints", "COLLAB_SCORES": "scores", "COLLAB_EMB": "embeddings"}.items():
    os.environ[k] = f"/content/drive/MyDrive/ogbl-collab/{d}"
!make mlp SEEDS=0-9 PY=python DEVICE=cuda
```

Training resumes from the last checkpoint (every 10 epochs) when `resume=True`
in `TrainConfig`.

### Role 1 midway on Windows (PowerShell)

With [uv](https://docs.astral.sh/uv/) available, create the same Python 3.11
environment as the other roles. No GPU or Make is needed:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-lock.txt
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
.\.venv\Scripts\python.exe -m collab.selftest
.\.venv\Scripts\python.exe -m unittest -v collab.test_heuristics
.\.venv\Scripts\python.exe -u -m collab.heuristics
.\.venv\Scripts\python.exe -m collab.summarize
```

CN and AA use the training-only graph, with the shared loader and official OGB
evaluator. Scores for all four official splits are written to
`scores/{cn,aa}_seed0_{split}.npy`; metrics and run metadata go to
`results/{cn,aa}.jsonl`. The methods are deterministic, so one run is sufficient.
The scorer intersects sparse adjacency rows for batches of requested pairs;
`--batch_size 1024` reduces temporary memory if necessary. It never builds a
dense all-pairs matrix or the full `A**2`. CN ties at the K-th negative count as
misses, exactly as in the evaluator.

### Role 1 analysis

On Windows, after setting up the environment above:

```powershell
.\.venv\Scripts\python.exe -u -m collab.heuristics --tie_break
.\.venv\Scripts\python.exe -u -m collab.heuristics --valinput
.\.venv\Scripts\python.exe -m collab.buckets
.\.venv\Scripts\python.exe -m collab.buckets --valinput
.\.venv\Scripts\python.exe -m collab.summarize
```

`cn_aa` breaks CN ties by AA. Its scores are exact float32 ordinal ranks of
distinct `(CN, AA)` tuples, computed jointly over positive and negative queries
within each evaluation split; labels are not used. Equal tuples still tie.
Rank magnitudes cannot be compared across splits or candidate sets.

`--valinput` adds 2018 edges for **test scoring only** and saves `cn_valinput`
and `aa_valinput`; it does not evaluate validation positives that are now in
the input graph. The default run still uses only training edges.

Bucket reports are written to `results/buckets/` and `results/buckets_valinput/`:
`summary.md`, per-seed and aggregated CSVs, overlap counts and an input manifest.
All bucket definitions use the original training graph, keeping cohorts fixed
between variants. Every method/seed uses its own threshold from all 100,000
test negatives. Empty buckets are N/A; standard deviations require at least two
seeds. Exact test-row memberships and AA disagreement indices are generated in
`scores/buckets*/` as `.npz` files; `test_buckets.npz` maps row indices to pairs.

Copy teammates' score files into `scores/`, or pass `--scores_dir PATH`.
`--methods aa sage` restricts a report to those names. Missing scores are listed
explicitly, while incomplete/invalid arrays and disagreements with existing
result records fail the run. GraphSAGE analysis requires the available scores'
mean test Hits@50 to meet the proposal's reproduction criterion (48.10 +/- 2,
or 54.63 +/- 2 for valinput). A single reproduced seed remains preliminary.
The arrays must follow the official OGB order: the manifest fingerprints that
pair order and the supplied files, but aggregate metrics alone cannot detect a
permutation within a score array.

See [role 1 analysis notes](docs/analysis-role1.md) for observed results and
remaining inputs needed for the final team comparison.

### Generate missing learned-model scores locally

`python -m collab.local_pipeline` runs/resumes the fixed midway seed-0 MLP and
node2vec configurations sequentially on CPU and rebuilds the bucket reports
after each. New methods are named `mlp_local` and `node2vec_local`, preserving
the team's existing results. The queue writes progress to `logs/`, resumes
decoder and skip-gram checkpoints, and does no tuning or automatic publication.
See [local training instructions](docs/local-training.md) for settings, logs,
outputs and the CPU timing estimate. These are independent single-seed runs;
they do not provide GraphSAGE scores or the final ten-seed comparison.

For the missing GraphSAGE scores, `python -m collab.sage_local` runs/resumes an
independent seed-0 CPU reproduction with sparse adjacency, per-epoch checkpoints
and automatic bucket reports after the reproduction checks pass. See
[local GraphSAGE instructions](docs/local-sage.md) for the fixed configuration,
estimated runtime, artifacts and gate behavior.

## Repository layout

```
collab/
  data.py       ogbl-collab -> shared tensors: input graph A (<=2017, unweighted,
                deduplicated), A + 2018 edges ("val as input"), official eval pairs
  evaluate.py   official OGB Hits@K, shared score-file format (save/load)
  decoder.py    shared decoder MLP(z_u * z_v), BCE loss, uniform negative sampler,
                training loop with epoch cap, early stopping, checkpoints
  mlp.py        features-only baseline
  node2vec.py   biased random walks + skip-gram, then the shared decoder on frozen vectors
  sage.py       GraphSAGE encoder (3 x SAGEConv, 256, mean) trained through the shared loop;
                --probe (timing), --tune (lr), "val as input" re-scoring without retraining
  test_sage.py  toy-graph checks for sage.py
  heuristics.py deterministic CN and AA, batched sparse scoring on the training graph
  buckets.py    global-threshold bucket metrics, seed aggregation, AA hit intersections
  test_buckets.py protocol tests for bucket analysis, tie-breaking, and valinput
  test_heuristics.py independent reference and score/evaluator checks on toy graphs
  summarize.py  results/*.jsonl -> results/summary.md (mean ± std over seeds)
  selftest.py   fast checks on toy graphs
results/        one JSON line per run (committed); summary.md
docs/           midway notes
scores/ checkpoints/ embeddings/ data/ logs/    generated, git-ignored
```

## Shared conventions

**Input graph.** All methods read the same matrix A: the 1,179,052 training
records (years <= 2017) collapse to **967,632 unique undirected unweighted
edges**. Learned methods also use these 967,632 pairs as training positives.

**Score files.** Every method writes
`scores/{method}_seed{k}_{split}.npy`, `split ∈ {valid_pos, valid_neg, test_pos, test_neg}`:
float32, one score per pair, in the official OGB order of that split. Learned
methods store decoder logits (monotone in the probability, so Hits@K is the same,
but no ties from sigmoid saturation). Deterministic methods use `seed0`. The
"val as input" variant appends `_valinput` to the method name. The bucket
analysis only reads these files.

**Results.** `collab.utils.append_result(method, record)` appends a JSON line
to `results/{method}.jsonl` with `method`, `seed`, `valid`, `test` (dicts with
`hits@20/50/100` as fractions) and optional timing; `make summary` turns them
into the table.

**Decoder (for every learned method).** `s(u, v) = MLP(z_u ⊙ z_v)`, 3 linear
layers, 256 hidden, ReLU, logits out (same shape as the OGB `LinkPredictor`).
Loss: BCE on each batch of positives against the same number of uniformly random
node pairs. Defaults: batch 65,536 positives, epoch cap 400, early stopping after
20 validation checks without a Hits@50 gain, selection on validation Hits@50.
An encoder is any `nn.Module` whose `forward()` with no arguments returns
`[num_nodes, d]`; if it has trainable parameters the loop trains it end-to-end,
otherwise its output is computed once per epoch:

```python
from collab.data import load_collab
from collab.decoder import LinkDecoder, TrainConfig, train_link_predictor

data = load_collab()
res = train_link_predictor(MyEncoder(...), LinkDecoder(in_dim=256), data.train_pos,
                           data.num_nodes, data.eval_pairs,
                           TrainConfig(lr=1e-3, device="cuda", ckpt_path="/content/drive/.../sage_seed0.pt"))
res.valid, res.test, res.scores, res.seconds_per_epoch
```

## Results

See [`results/summary.md`](results/summary.md) (regenerated by `make summary`)
and the midway status and blockers for [role 1](docs/midway-role1.md),
[role 2](docs/midway.md) and [role 3](docs/midway-role3.md).

## Team

| Role | Scope |
|---|---|
| 1 | data, CN/AA heuristics, bucket analysis, report assembly |
| 2 | shared decoder, node2vec, features-only MLP, repository and environment |
| 3 | GraphSAGE |
