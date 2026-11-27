"""Time three full-size node2vec training batches; never score validation/test.

    python -m collab.benchmark_node2vec

Probe embeddings are discarded. The extrapolation excludes decoder training,
includes first-step warm-up costs, and is not a measured full-epoch duration.
"""
import json
import math

import torch

from .data import load_collab
from .node2vec import train_node2vec
from .utils import ROOT


def main():
    data = load_collab()
    _, seconds = train_node2vec(data.edge_index, data.num_nodes, p=1, q=1, epochs=1,
                                batch_nodes=256, max_steps=3, device="cpu", seed=0)
    steps_per_epoch = math.ceil(data.num_nodes / 256)
    report = dict(device="cpu", torch_threads=torch.get_num_threads(), measured_steps=3,
                  measured_seconds=seconds, batch_nodes=256, steps_per_epoch=steps_per_epoch,
                  estimated_three_epochs_hours=seconds / 3 * steps_per_epoch * 3 / 3600,
                  note="Short training-only timing probe; no validation/test scoring. "
                       "Excludes decoder training; includes warm-up and concurrent workload effects.")
    path = ROOT / "logs" / "node2vec_cpu_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
