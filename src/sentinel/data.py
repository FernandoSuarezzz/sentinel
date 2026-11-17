"""Data pipeline: load public datasets + synthetic data, clean, deduplicate, split.

Run:  python -m sentinel.data --out artifacts/data/dataset.parquet

Output: one parquet file with columns
    text, label (0 safe / 1 injection), source, lang, family, split
where split is one of  train | val | test | ood_test.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from sentinel import synth

LABEL_SAFE, LABEL_INJECTION = 0, 1
COLUMNS = ["text", "label", "source", "lang", "family", "split"]

# When the same text appears in several splits we keep the copy in the *most protected*
# split, so evaluation sets are never shrunk to the benefit of the training set.
_SPLIT_PRIORITY = {"ood_test": 0, "test": 1, "val": 2, "train": 3}


@dataclass(frozen=True)
class Source:
    """How to read one public Hugging Face dataset into our schema."""

    hf_id: str
    text_col: str
    label_col: str
    positive: object  # value of label_col that means "injection"
    lang: str
    max_positive_rate: float | None = None  # guard against an inverted label mapping


PUBLIC_SOURCES = [
    Source("deepset/prompt-injections", "text", "label", 1, "en+de"),
    # Card: ~7k safe vs ~3k injection, so injections must be the minority class.
    Source("xTRam1/safe-guard-prompt-injection", "text", "label", 1, "en", max_positive_rate=0.5),
    Source("jackhhao/jailbreak-classification", "prompt", "type", "jailbreak", "en"),
]


def load_public_sources(sources: list[Source] = PUBLIC_SOURCES) -> pd.DataFrame:
    """Download the public datasets and map them to our schema (needs internet)."""
    from datasets import get_dataset_config_names, load_dataset  # lazy: not needed to serve

    frames = []
    for s in sources:
        # Some datasets publish several configurations; take "default" when there is a choice.
        configs = get_dataset_config_names(s.hf_id)
        config = "default" if "default" in configs or not configs else configs[0]
        for split_name, part in load_dataset(s.hf_id, config).items():
            frames.append(_to_schema(part.to_pandas(), s, split_name))
    return pd.concat(frames, ignore_index=True)


def _to_schema(df: pd.DataFrame, s: Source, split_name: str) -> pd.DataFrame:
    missing = {s.text_col, s.label_col} - set(df.columns)
    if missing:
        raise ValueError(f"{s.hf_id}: missing columns {missing}; found {list(df.columns)}")
    is_pos = df[s.label_col] == s.positive
    if not is_pos.any():
        values = df[s.label_col].unique()[:10]
        raise ValueError(f"{s.hf_id}: no row has {s.label_col}={s.positive!r}; values: {values}")
    rate = float(is_pos.mean())
    if s.max_positive_rate is not None and rate > s.max_positive_rate:
        raise ValueError(f"{s.hf_id}: {rate:.0%} positives, label mapping looks inverted")
    return pd.DataFrame({
        "text": df[s.text_col].astype(str),
        "label": is_pos.astype(int),
        "source": s.hf_id,
        "lang": s.lang,
        "family": "public",
        # Respect the providers' own test split; everything else is training material.
        "split": "test" if split_name == "test" else "train",
    })


_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Canonical form used only for duplicate detection (the model sees the raw text)."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip().lower()


def text_key(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()


def clean(
    df: pd.DataFrame, min_chars: int = 3, max_chars: int = 20_000
) -> tuple[pd.DataFrame, dict]:
    """Drop empty/huge texts, label conflicts and duplicates (including cross-split leaks)."""
    stats = {"input_rows": len(df)}
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len().between(min_chars, max_chars)]
    stats["after_length_filter"] = len(df)

    df["key"] = df["text"].map(text_key)
    # The same text labelled both ways is noise we cannot resolve: drop every copy.
    n_labels = df.groupby("key")["label"].transform("nunique")
    stats["label_conflicts_dropped"] = int((n_labels > 1).sum())
    df = df[n_labels == 1]

    # Exact duplicates: keep one copy, in the most protected split (see _SPLIT_PRIORITY).
    df = df.assign(_prio=df["split"].map(_SPLIT_PRIORITY))
    before = len(df)
    df = df.sort_values("_prio", kind="stable").drop_duplicates("key", keep="first")
    stats["duplicates_dropped"] = before - len(df)
    stats["output_rows"] = len(df)
    return df.drop(columns=["_prio"]).reset_index(drop=True), stats


def add_validation_split(df: pd.DataFrame, val_frac: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """Carve a validation set out of train, stratified by source and label."""
    df = df.copy()
    train_idx = df.index[df["split"] == "train"]
    strata = df.loc[train_idx, "source"] + "|" + df.loc[train_idx, "label"].astype(str)
    stratify = strata if strata.value_counts().min() >= 2 else df.loc[train_idx, "label"]
    _, val_idx = train_test_split(train_idx, test_size=val_frac, random_state=seed,
                                  stratify=stratify)
    df.loc[val_idx, "split"] = "val"
    return df


def build(
    public: pd.DataFrame, synthetic: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, dict]:
    df, stats = clean(pd.concat([public, synthetic], ignore_index=True))
    df = add_validation_split(df, seed=seed)
    df["label"] = df["label"].astype(int)
    return df[COLUMNS], stats


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Rows and injection rate per split and source: the first thing to look at."""
    return (df.groupby(["split", "source"])["label"]
              .agg(rows="size", injection_rate="mean")
              .round(3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="artifacts/data/dataset.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    public = load_public_sources()
    synthetic = synth.generate(synth.SynthConfig(seed=args.seed))
    df, stats = build(public, synthetic, seed=args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print("Cleaning:", stats)
    print(summary(df).to_string())
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
