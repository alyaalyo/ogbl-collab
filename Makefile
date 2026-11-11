# One-command entry points. `make all` = full pipeline from a clean checkout.
PY      ?= .venv/bin/python
RUN      = PYTHONWARNINGS=ignore $(PY) -u -m
SEEDS   ?= 0-9
DEVICE  ?= auto
# node2vec (p, q) picked by `make node2vec-grid` on validation Hits@50
N2V_P   ?= 1
N2V_Q   ?= 1

.PHONY: all env check midway midway-role1 mlp node2vec-grid node2vec heuristics sage summary clean

all: env check mlp node2vec-grid node2vec heuristics sage summary

env:            ## pinned virtualenv (needs uv: https://docs.astral.sh/uv/)
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements-lock.txt

check:          ## fast toy-data tests of the shared code
	$(RUN) collab.selftest
	$(PY) -m unittest collab.test_heuristics

midway:         ## midway-check runs: features-only MLP + node2vec (p=q=1, 3 skip-gram epochs), seed 0
	$(RUN) collab.mlp --seeds 0 --device $(DEVICE)
	$(RUN) collab.node2vec --p 1 --q 1 --n2v_epochs 3 --seeds 0 --device $(DEVICE)
	$(RUN) collab.summarize

mlp:            ## features-only MLP, all seeds
	$(RUN) collab.mlp --seeds $(SEEDS) --device $(DEVICE)

node2vec-grid:  ## 9 (p, q) configs, seed 0, select on validation Hits@50
	$(RUN) collab.node2vec --grid --seeds 0 --device $(DEVICE)

node2vec:       ## chosen (p, q), all seeds
	$(RUN) collab.node2vec --p $(N2V_P) --q $(N2V_Q) --seeds $(SEEDS) --device $(DEVICE)

heuristics:     ## role 1 midway: deterministic CN and AA on the training-only graph
	$(RUN) collab.heuristics

midway-role1: heuristics summary  ## role 1 preliminary results, no learned-model training

sage:           ## role 3 (GraphSAGE); placeholder until collab/sage.py lands
	@test -f collab/sage.py && $(RUN) collab.sage --seeds $(SEEDS) --device $(DEVICE) || echo "collab/sage.py not in repo yet (role 3)"

summary:
	$(RUN) collab.summarize

clean:          ## drop generated artefacts (keeps data/ and results/)
	rm -rf scores checkpoints embeddings logs
