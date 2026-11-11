import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("COLLAB_DATA", ROOT / "data"))
SCORES_DIR = Path(os.environ.get("COLLAB_SCORES", ROOT / "scores"))
RESULTS_DIR = Path(os.environ.get("COLLAB_RESULTS", ROOT / "results"))
CKPT_DIR = Path(os.environ.get("COLLAB_CKPT", ROOT / "checkpoints"))
EMB_DIR = Path(os.environ.get("COLLAB_EMB", ROOT / "embeddings"))


def set_seed(seed: int) -> None:
    """Fix every RNG we use. cuDNN determinism is not forced (too slow for SAGE)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def append_result(name: str, record: dict) -> Path:
    """One JSON line per run in results/{name}.jsonl (committed to git)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.jsonl"
    record = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **record}
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return path


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
