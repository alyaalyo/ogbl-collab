"""Analyse official test positives at each method's GLOBAL negative threshold.

    python -m collab.buckets
    python -m collab.buckets --valinput
    python -m collab.buckets --methods aa sage --scores_dir /path/to/scores

Bucket definitions always use the training-only graph, including for valinput,
so the secondary experiment compares the same cohorts. Missing methods are
reported explicitly. No models are trained and no scores are inferred from
aggregate metrics. Per-pair memberships stay in the git-ignored scores folder.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .data import load_collab
from .evaluate import hits, score_path
from .heuristics import HeuristicScorer
from .utils import RESULTS_DIR, SCORES_DIR

BUCKETS = {"all": ("all",), "common_neighbors": ("0", "1", "2-4", ">=5"),
           "min_degree": ("0", "1-5", ">5"), "pair_history": ("repeat", "new")}
PER_SEED_FIELDS = ("method", "seed", "partition", "bucket", "size", "share", "hits", "hits50", "threshold")
OVERLAP_FIELDS = ("left", "left_seed", "right", "right_seed", "partition", "bucket", "size",
                  "both", "left_only", "right_only", "neither")


def make_buckets(edge_index, num_nodes, test_pairs):
    scorer = HeuristicScorer(edge_index, num_nodes)
    pairs = np.asarray(test_pairs)
    cn = scorer.score(pairs)["cn"].astype(np.int64)
    degree = np.minimum(scorer.degree[pairs[:, 0]], scorer.degree[pairs[:, 1]])
    repeat = np.asarray(scorer.adjacency[pairs[:, 0], pairs[:, 1]]).ravel() != 0
    labels = {
        "all": np.full(len(pairs), "all"),
        "common_neighbors": np.select([cn == 0, cn == 1, cn <= 4], ["0", "1", "2-4"], default=">=5"),
        "min_degree": np.select([degree == 0, degree <= 5], ["0", "1-5"], default=">5"),
        "pair_history": np.where(repeat, "repeat", "new"),
    }
    metadata = {"row_index": np.arange(len(pairs)), "pairs": pairs, "common_neighbors": cn,
                "min_degree": degree, "repeat": repeat}
    return labels, metadata


def masks(labels):
    for partition, buckets in BUCKETS.items():
        for bucket in buckets:
            yield partition, bucket, labels[partition] == bucket


def hit_mask(pos, neg, k=50):
    """Strict OGB rule, using ALL negatives, not negatives within a bucket."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if pos.ndim != 1 or neg.ndim != 1 or not np.isfinite(pos).all() or not np.isfinite(neg).all():
        raise ValueError("Expected finite, one-dimensional score arrays")
    if k <= 0 or len(neg) < k:
        raise ValueError("Need at least K negatives to define the global threshold")
    threshold = float(np.partition(neg, -k)[-k])
    return pos > threshold, threshold


def bucket_rows(method, seed, labels, hit, threshold):
    rows = []
    for partition, bucket, mask in masks(labels):
        size, count = int(mask.sum()), int(hit[mask].sum())
        rows.append(dict(method=method, seed=seed, partition=partition, bucket=bucket,
                         size=size, share=size / len(hit), hits=count,
                         hits50=count / size if size else None, threshold=threshold))
    return rows


def aggregate(rows):
    groups = {}
    for row in rows:
        key = row["method"], row["partition"], row["bucket"]
        groups.setdefault(key, []).append(row)
    result = []
    for (method, partition, bucket), group in groups.items():
        values = [r["hits50"] for r in group if r["hits50"] is not None]
        result.append(dict(method=method, partition=partition, bucket=bucket, size=group[0]["size"],
                           share=group[0]["share"], seeds=len(group),
                           mean=float(np.mean(values)) if values else None,
                           std=float(np.std(values, ddof=1)) if len(values) > 1 else None))
    return result


def overlap_masks(left, right):
    return {"both": left & right, "left_only": left & ~right,
            "right_only": ~left & right, "neither": ~left & ~right}


def load_test_scores(method, seed, root, expected_sizes):
    result = {}
    for kind in ("pos", "neg"):
        path = score_path(method, seed, f"test_{kind}", root)
        values = np.load(path, allow_pickle=False)
        if values.shape != (expected_sizes[kind],) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"Invalid {path.name}: expected {expected_sizes[kind]} finite float32 scores")
        result[kind] = values
    return result


def discover_runs(root):
    runs = set()
    for path in Path(root).glob("*_seed*_test_*.npy"):
        match = re.fullmatch(r"(.+)_seed(\d+)_test_(pos|neg)\.npy", path.name)
        if match:
            runs.add((match[1], int(match[2])))
    return runs


def result_records(root):
    records = {}
    for path in Path(root).glob("*.jsonl"):
        if path.stem.endswith(("_grid", "_timing")):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if "test" in record:
                    records[record["method"], record.get("seed", 0)] = record
    return records


def check_record(method, seed, metrics, records):
    record = records.get((method, seed))
    if record:
        for metric, value in metrics.items():
            if metric in record["test"] and not np.isclose(value, record["test"][metric], rtol=0, atol=1e-7):
                raise ValueError(f"{method} seed {seed}: saved scores disagree with recorded {metric}")


def check_sage_reproduction(method, values):
    reference = {"sage": 48.10, "sage_valinput": 54.63}.get(method)
    if reference is not None and abs(100 * np.mean(values) - reference) > 2:
        raise ValueError(f"{method}: mean test Hits@50 is outside {reference} +/- 2; analysis withheld")


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def report_markdown(summary, overlaps, status, labels):
    lines = ["# Test bucket analysis", "",
             "Buckets use the training-only graph. Each run uses its own global 50th-negative",
             "threshold and strict `>`; Hits@50 is shown as percent, mean +/- sample std across seeds.",
             "An empty bucket is N/A. A single seed has no estimated standard deviation.", "",
             "## Available inputs", "", "| Method | Seed(s) | Status |", "|---|---|---|"]
    for entry in status:
        lines.append(f"| {entry['method']} | {entry['seeds']} | {entry['status']} |")
    methods = list(dict.fromkeys(row["method"] for row in summary))
    lookup = {(r["method"], r["partition"], r["bucket"]): r for r in summary}
    lines += ["", "## Hits@50 by bucket", "", "| Partition | Bucket | Size | Share (%) | " + " | ".join(methods) + " |",
              "|---|---|---:|---:|" + "---:|" * len(methods)]
    for partition, bucket, mask in masks(labels):
        cells = []
        for method in methods:
            row = lookup[method, partition, bucket]
            value = "N/A" if row["mean"] is None else f"{100 * row['mean']:.2f}"
            if row["std"] is not None:
                value += f" +/- {100 * row['std']:.2f}"
            cells.append(value)
        lines.append(f"| {partition} | {bucket} | {mask.sum()} | {100 * mask.mean():.2f} | " + " | ".join(cells) + " |")
    lines += ["", "## Hit intersections (counts, per seed)", "",
              "| Left | Right | Partition | Bucket | Both | Left only | Right only | Neither |",
              "|---|---|---|---|---:|---:|---:|---:|"]
    for row in overlaps:
        lines.append(f"| {row['left']} seed{row['left_seed']} | {row['right']} seed{row['right_seed']} | "
                     f"{row['partition']} | {row['bucket']} | {row['both']} | {row['left_only']} | "
                     f"{row['right_only']} | {row['neither']} |")
    lines += ["", "Missing score files cannot be reconstructed from aggregate metrics. All results remain",
              "preliminary until the planned final seeds and cross-method inputs are available.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores_dir", type=Path, default=SCORES_DIR)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--valinput", action="store_true")
    args = parser.parse_args(argv)
    suffix = "_valinput" if args.valinput else ""
    output = args.output_dir or args.results_dir / ("buckets" + suffix)
    records, discovered = result_records(args.results_dir), discover_runs(args.scores_dir)
    expected = {m + suffix for m in (("cn", "aa", "sage") if args.valinput else
                                    ("cn", "aa", "cn_aa", "mlp", "node2vec", "sage"))}
    methods = sorted(set(args.methods) if args.methods else expected | {
        m for m, _ in discovered | records.keys() if m.endswith("_valinput") == args.valinput})
    if any(m.endswith("_valinput") != args.valinput for m in methods):
        parser.error("Use --valinput for _valinput methods; do not mix graph variants")
    data = load_collab()
    labels, metadata = make_buckets(data.edge_index, data.num_nodes, data.eval_pairs["test_pos"])
    sizes = {kind: len(data.eval_pairs[f"test_{kind}"]) for kind in ("pos", "neg")}
    loaded, status, sources = {}, [], []
    for method in methods:
        seeds = sorted(s for m, s in discovered | records.keys() if m == method)
        if not seeds:
            status.append(dict(method=method, seeds="-", status="missing scores"))
        for seed in seeds:
            paths = [score_path(method, seed, f"test_{kind}", args.scores_dir) for kind in ("pos", "neg")]
            if not any(path.exists() for path in paths):
                status.append(dict(method=method, seeds=str(seed), status="missing scores (metrics only)"))
                continue
            values = load_test_scores(method, seed, args.scores_dir, sizes)
            metrics = hits(values["pos"], values["neg"])
            check_record(method, seed, metrics, records)
            loaded[method, seed] = hit_mask(values["pos"], values["neg"])
            status.append(dict(method=method, seeds=str(seed), status="available"))
            sources.extend({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths)
        values = [hit.mean() for (m, _), (hit, _) in loaded.items() if m == method]
        if values:
            check_sage_reproduction(method, values)
    if not loaded:
        raise ValueError("No complete test-score runs are available")
    rows = [row for (method, seed), (hit, threshold) in loaded.items()
            for row in bucket_rows(method, seed, labels, hit, threshold)]
    summary = aggregate(rows)
    overlap_rows, artifacts = [], {}
    # Deterministic AA seed0 versus each other available method/seed; same graph variant.
    left = ("aa" + suffix, 0)
    if left in loaded:
        for right in loaded:
            if right == left:
                continue
            quadrants = overlap_masks(loaded[left][0], loaded[right][0])
            artifacts[f"{left[0]}_seed0__{right[0]}_seed{right[1]}.npz"] = {
                name: np.flatnonzero(mask) for name, mask in quadrants.items()}
            for partition, bucket, mask in masks(labels):
                overlap_rows.append(dict(left=left[0], left_seed=0, right=right[0], right_seed=right[1],
                                         partition=partition, bucket=bucket, size=int(mask.sum()),
                                         **{name: int((mask & quadrant).sum()) for name, quadrant in quadrants.items()}))
    output.mkdir(parents=True, exist_ok=True)
    details = args.scores_dir / ("buckets" + suffix)
    details.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(details / "test_buckets.npz", **metadata)
    for name, values in artifacts.items():
        np.savez_compressed(details / name, **values)
    write_csv(output / "per_seed.csv", rows, PER_SEED_FIELDS)
    write_csv(output / "aggregate.csv", summary, ("method", "partition", "bucket", "size", "share", "seeds", "mean", "std"))
    write_csv(output / "overlaps.csv", overlap_rows, OVERLAP_FIELDS)
    manifest = dict(graph_variant="train_plus_valid" if args.valinput else "train_only",
                    bucket_graph="train_only", k=50, threshold_rule="strict > 50th highest global negative",
                    train_pairs_sha256=digest(data.train_pos),
                    test_pos_sha256=digest(data.eval_pairs["test_pos"]),
                    test_neg_sha256=digest(data.eval_pairs["test_neg"]), status=status, sources=sources,
                    detail_files=["test_buckets.npz", *artifacts])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "summary.md").write_text(report_markdown(summary, overlap_rows, status, labels), encoding="utf-8")
    print(f"Wrote {output}: {len(loaded)} runs, {sizes['pos']} test positives")
    for entry in status:
        print(f"  {entry['method']} seed {entry['seeds']}: {entry['status']}")


if __name__ == "__main__":
    main()
