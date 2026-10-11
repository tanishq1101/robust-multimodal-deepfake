"""Identity-wise train/val/test splits.

Splitting by `group` (person identity / source video) instead of by file prevents the
model from seeing the same face or voice in both training and testing.
Rows that already carry an official split (ASVspoof, Celeb-DF test list) keep it.
Unassigned relatives of official test samples are excluded, preserving the exact test list.

Usage:
  python -m deepfake.data.splits dfdc
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from deepfake.config import get_path, load_config


def assign_splits(df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int) -> pd.DataFrame:
    if not (0 <= val_ratio < 1 and 0 <= test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError("Split ratios must be non-negative and sum to less than 1")
    df = df.copy()
    df["split"] = df["split"].fillna("").astype(str)
    if not df["split"].isin(["", "train", "val", "test", "excluded"]).all():
        raise ValueError("Unknown split in manifest")
    if df["group"].isna().any() or (df["group"].astype(str) == "").any():
        raise ValueError("Every sample needs a non-empty split group")
    fixed = df[df["split"].isin(["train", "val", "test"])]
    if (fixed.groupby("group")["split"].nunique() > 1).any():
        raise ValueError("Official assignments contain group leakage across splits")
    assigned = fixed.drop_duplicates("group").set_index("group")["split"]
    inherited = df["group"].map(assigned)
    inherited = inherited.replace("test", "excluded")
    df.loc[(df["split"] == "") & inherited.notna(), "split"] = inherited
    todo = df["split"] == ""
    if not todo.any():
        return df
    # If the dataset already has an official test set, only carve out validation.
    has_test = (df["split"] == "test").any()
    t_ratio = 0.0 if has_test else test_ratio

    groups = np.array(sorted(df.loc[todo, "group"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_test = int(round(n * t_ratio))
    n_val = max(1, int(round(n * val_ratio))) if n > 2 and val_ratio > 0 else 0
    mapping = {g: "test" for g in groups[:n_test]}
    mapping.update({g: "val" for g in groups[n_test:n_test + n_val]})
    mapping.update({g: "train" for g in groups[n_test + n_val:]})
    df.loc[todo, "split"] = df.loc[todo, "group"].astype(str).map(mapping)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("split")["label"].agg(total="count", fake="sum").assign(
        real=lambda t: t.total - t.fake
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    args = p.parse_args(argv)
    cfg = load_config()
    path = get_path(cfg, "manifests", f"{args.dataset}.csv")
    df = assign_splits(pd.read_csv(path, keep_default_na=False), cfg["splits"]["val_ratio"],
                       cfg["splits"]["test_ratio"], cfg["seed"])
    df.to_csv(path, index=False)
    print(summarize(df))


if __name__ == "__main__":
    main()
