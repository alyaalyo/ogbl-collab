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
misses, exactly as in the evaluator. CN with an AA tie-break, `val as input`, and
bucket analysis are follow-up work after the midway CN/AA baseline.

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
  heuristics.py deterministic CN and AA, batched sparse scoring on the training graph
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
and the midway status and blockers for [role 1](docs/midway-role1.md)
and [role 2](docs/midway.md).

## Team

| Role | Scope |
|---|---|
| 1 | data, CN/AA heuristics, bucket analysis, report assembly |
| 2 | shared decoder, node2vec, features-only MLP, repository and environment |
| 3 | GraphSAGE |
