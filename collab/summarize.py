"""Collect results/*.jsonl into one table: mean ± std over seeds (x100, as on the OGB leaderboard).

    python -m collab.summarize           # prints and writes results/summary.md
"""
import json

import numpy as np

from .utils import RESULTS_DIR

KS = ("hits@20", "hits@50", "hits@100")


def load_runs():
    runs = {}
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        if path.stem.endswith("_grid"):
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            runs[(r["method"], r.get("seed", 0))] = r   # later lines override reruns of a seed
    return runs


def fmt(vals):
    vals = 100 * np.asarray(vals)
    return f"{vals.mean():.2f}" if len(vals) == 1 else f"{vals.mean():.2f} ± {vals.std(ddof=1):.2f}"


def main():
    runs = load_runs()
    methods = sorted({m for m, _ in runs})
    head = "| method | seeds | " + " | ".join(f"val {k}" for k in KS) + " | " + " | ".join(f"test {k}" for k in KS) + " | s/epoch |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for m in methods:
        rs = [r for (mm, _), r in sorted(runs.items()) if mm == m]
        spe = [r["seconds_per_epoch"] for r in rs if r.get("seconds_per_epoch")]
        cells = [fmt([r["valid"][k] for r in rs]) for k in KS] + [fmt([r["test"][k] for r in rs]) for k in KS]
        lines.append(f"| {m} | {len(rs)} | " + " | ".join(cells) + f" | {np.mean(spe):.1f} |" if spe else
                     f"| {m} | {len(rs)} | " + " | ".join(cells) + " | – |")
    table = "\n".join(lines)
    print(table)
    (RESULTS_DIR / "summary.md").write_text("Hits@K × 100, mean ± std over seeds (sample std).\n\n" + table + "\n")


if __name__ == "__main__":
    main()
