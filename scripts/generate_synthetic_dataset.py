#!/usr/bin/env python3
"""
Generate balanced synthetic datasets for adaptive sorting selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SampleMeta:
    sample_id: str
    n: int
    distribution_tag: str
    structure_tag: str
    size_bucket: str
    seed: int
    path: str


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_sizes(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def generate_base_array(
    rng: np.random.Generator, n: int, distribution: str, low: int, high: int
) -> np.ndarray:
    if distribution == "uniform":
        arr = rng.integers(low, high, size=n, dtype=np.int64)
    elif distribution == "normal":
        mean = (low + high) / 2.0
        std = (high - low) / 6.0
        arr = rng.normal(loc=mean, scale=std, size=n)
        arr = np.clip(arr, low, high - 1).astype(np.int64)
    elif distribution == "lognormal":
        arr = rng.lognormal(mean=2.5, sigma=1.0, size=n)
        arr = scale_to_int_range(arr, low, high)
    elif distribution == "exponential":
        arr = rng.exponential(scale=1.0, size=n)
        arr = scale_to_int_range(arr, low, high)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    return arr.astype(np.int32, copy=False)


def scale_to_int_range(arr: np.ndarray, low: int, high: int) -> np.ndarray:
    arr = arr.astype(np.float64, copy=False)
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v - min_v < 1e-12:
        return np.full_like(arr, low, dtype=np.int64)
    scaled = (arr - min_v) / (max_v - min_v)
    scaled = scaled * (high - low - 1) + low
    return scaled.astype(np.int64)


def apply_structure(
    rng: np.random.Generator, arr: np.ndarray, structure: str
) -> np.ndarray:
    n = arr.size
    if n == 0:
        return arr

    out = np.array(arr, copy=True)

    if structure == "random":
        rng.shuffle(out)
        return out

    if structure == "nearly_sorted":
        out.sort()
        swaps = max(1, int(0.02 * n))
        i_idx = rng.integers(0, n, size=swaps)
        j_idx = rng.integers(0, n, size=swaps)
        out[i_idx], out[j_idx] = out[j_idx], out[i_idx]
        return out

    if structure == "reverse_sorted":
        out.sort()
        return out[::-1]

    if structure == "few_unique":
        uniq = max(2, min(16, n // 100))
        unique_values = rng.choice(out, size=uniq, replace=False)
        return rng.choice(unique_values, size=n, replace=True).astype(np.int32)

    if structure == "many_duplicates":
        uniq = max(2, min(64, n // 40))
        unique_values = rng.choice(out, size=uniq, replace=False)
        probs = np.linspace(2.0, 0.2, uniq)
        probs = probs / probs.sum()
        return rng.choice(unique_values, size=n, replace=True, p=probs).astype(np.int32)

    if structure == "runs":
        rng.shuffle(out)
        chunks: list[np.ndarray] = []
        i = 0
        while i < n:
            run_len = int(rng.integers(32, 256))
            run_len = min(run_len, n - i)
            chunk = np.sort(out[i : i + run_len])
            if rng.random() < 0.5:
                chunk = chunk[::-1]
            chunks.append(chunk)
            i += run_len
        rng.shuffle(chunks)
        return np.concatenate(chunks).astype(np.int32, copy=False)

    raise ValueError(f"Unsupported structure: {structure}")


def split_group(
    indices: list[int], train_ratio: float, val_ratio: float, rng: np.random.Generator
) -> tuple[list[int], list[int], list[int]]:
    idx = np.array(indices, dtype=np.int64)
    rng.shuffle(idx)
    n = idx.size

    if n == 1:
        return idx.tolist(), [], []
    if n == 2:
        return [int(idx[0])], [], [int(idx[1])]

    n_train = int(np.floor(n * train_ratio))
    n_val = int(np.floor(n * val_ratio))
    n_test = n - n_train - n_val

    if n_train < 1:
        n_train = 1
    if n_val < 1:
        n_val = 1
    n_test = n - n_train - n_val
    if n_test < 1:
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1
        n_test = 1

    train = idx[:n_train].tolist()
    val = idx[n_train : n_train + n_val].tolist()
    test = idx[n_train + n_val :].tolist()
    return train, val, test


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic dataset bank.")
    parser.add_argument("--out-dir", default="data/synthetic")
    parser.add_argument(
        "--sizes",
        default="1000,2000,5000,8000,10000,15000,20000,30000,40000,50000",
    )
    parser.add_argument(
        "--distributions", default="uniform,normal,lognormal,exponential"
    )
    parser.add_argument(
        "--structures",
        default="random,nearly_sorted,reverse_sorted,few_unique,many_duplicates,runs",
    )
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low", type=int, default=0)
    parser.add_argument("--high", type=int, default=30_000_000)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    sizes = parse_sizes(args.sizes)
    distributions = parse_csv_list(args.distributions)
    structures = parse_csv_list(args.structures)
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Invalid split ratios. Need train_ratio + val_ratio < 1.")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    split_dir = out_dir / "splits"
    raw_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    master_rng = np.random.default_rng(args.seed)
    metas: list[SampleMeta] = []

    counter = 0
    total = len(sizes) * len(distributions) * len(structures) * args.repeats
    print(f"Generating {total} samples...")

    for n in sizes:
        for distribution in distributions:
            for structure in structures:
                for rep in range(args.repeats):
                    sample_seed = int(master_rng.integers(0, 2**31 - 1))
                    rng = np.random.default_rng(sample_seed)

                    base = generate_base_array(
                        rng=rng,
                        n=n,
                        distribution=distribution,
                        low=args.low,
                        high=args.high,
                    )
                    values = apply_structure(rng=rng, arr=base, structure=structure)

                    sample_id = f"s{counter:06d}"
                    sample_path = raw_dir / f"{sample_id}.npy"
                    np.save(sample_path, values)

                    metas.append(
                        SampleMeta(
                            sample_id=sample_id,
                            n=n,
                            distribution_tag=distribution,
                            structure_tag=structure,
                            size_bucket=str(n),
                            seed=sample_seed,
                            path=str(sample_path.as_posix()),
                        )
                    )
                    counter += 1

    index_rows = [
        {
            "sample_id": m.sample_id,
            "n": m.n,
            "distribution_tag": m.distribution_tag,
            "structure_tag": m.structure_tag,
            "size_bucket": m.size_bucket,
            "seed": m.seed,
            "path": m.path,
        }
        for m in metas
    ]
    write_csv(
        out_dir / "index.csv",
        index_rows,
        fieldnames=[
            "sample_id",
            "n",
            "distribution_tag",
            "structure_tag",
            "size_bucket",
            "seed",
            "path",
        ],
    )

    # Stratified splits by (size, distribution, structure)
    group_to_indices: dict[tuple[str, str, str], list[int]] = {}
    for i, m in enumerate(metas):
        key = (m.size_bucket, m.distribution_tag, m.structure_tag)
        group_to_indices.setdefault(key, []).append(i)

    split_rng = np.random.default_rng(args.seed + 1)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for key in sorted(group_to_indices):
        tr, va, te = split_group(
            group_to_indices[key], args.train_ratio, args.val_ratio, split_rng
        )
        train_idx.extend(tr)
        val_idx.extend(va)
        test_idx.extend(te)

    def to_rows(indices: list[int]) -> list[dict]:
        return [
            {
                "sample_id": metas[i].sample_id,
                "path": metas[i].path,
                "n": metas[i].n,
                "distribution_tag": metas[i].distribution_tag,
                "structure_tag": metas[i].structure_tag,
                "size_bucket": metas[i].size_bucket,
            }
            for i in indices
        ]

    split_fields = ["sample_id", "path", "n", "distribution_tag", "structure_tag", "size_bucket"]
    write_csv(split_dir / "train_ids.csv", to_rows(train_idx), fieldnames=split_fields)
    write_csv(split_dir / "val_ids.csv", to_rows(val_idx), fieldnames=split_fields)
    write_csv(split_dir / "test_ids.csv", to_rows(test_idx), fieldnames=split_fields)

    config = {
        "seed": args.seed,
        "sizes": sizes,
        "distributions": distributions,
        "structures": structures,
        "repeats": args.repeats,
        "low": args.low,
        "high": args.high,
        "split": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        },
        "total_samples": len(metas),
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "test_count": len(test_idx),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print("Done.")
    print(f"Total: {len(metas)}")
    print(f"Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print(f"Index: {out_dir / 'index.csv'}")
    print(f"Splits: {split_dir}")


if __name__ == "__main__":
    main()
