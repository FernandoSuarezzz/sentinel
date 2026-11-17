"""Version 1 of Sentinel, kept as the BASELINE: word + character TF-IDF, then logistic regression.

Every serious ML project compares its model against a simple one. If the transformer cannot
clearly beat this, the extra complexity is not worth it. `sentinel.evaluate` trains this
baseline on exactly the same data as the transformer and reports both side by side.

It still works on its own, as in version 1 (deepset dataset only):
    python -m sentinel.baseline train
    python -m sentinel.baseline eval
    python -m sentinel.baseline check "Ignore all previous instructions"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

DEFAULT_THRESHOLD = 0.5
DATASET = "deepset/prompt-injections"
MODEL_PATH = os.path.join("models", "sentinel.joblib")
THRESHOLD_PATH = os.path.join("models", "threshold.json")


@dataclass
class Verdict:
    label: str  # "injection" or "benign"
    score: float  # estimated probability that the text is an injection


def build_pipeline() -> Pipeline:
    # Words catch phrases like "ignore previous instructions".
    # Character n-grams survive typos, spacing tricks and other languages.
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ])
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", classifier)])


def train(texts, labels) -> Pipeline:
    model = build_pipeline()
    model.fit(texts, labels)
    return model


def tune_threshold(texts, labels, cv=5, grid=None):
    """Picks the decision threshold using ONLY the training data (k-fold cross-validation),
    so the test set stays untouched. Returns (best_threshold, rows) where each row is
    (threshold, precision, recall, f1) measured on the held-out folds."""
    grid = grid or [round(0.1 + 0.05 * i, 2) for i in range(17)]  # 0.10 ... 0.90
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    scores = cross_val_predict(build_pipeline(), texts, labels, cv=folds,
                               method="predict_proba")[:, 1]
    rows = []
    for threshold in grid:
        predicted = [1 if s >= threshold else 0 for s in scores]
        p, r, f1, _ = precision_recall_fscore_support(
            labels, predicted, average="binary", pos_label=1, zero_division=0)
        rows.append((threshold, p, r, f1))
    # best F1; ties -> closest to 0.5
    best = max(rows, key=lambda row: (row[3], -abs(row[0] - DEFAULT_THRESHOLD)))
    return best[0], rows


def evaluate(model, texts, labels, threshold=DEFAULT_THRESHOLD) -> dict:
    """Precision / recall / F1 for the 'injection' class, plus the confusion matrix."""
    scores = model.predict_proba(texts)[:, 1]
    predicted = [1 if s >= threshold else 0 for s in scores]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {"precision": precision, "recall": recall, "f1": f1,
            "true_negatives": int(tn), "false_positives": int(fp),
            "false_negatives": int(fn), "true_positives": int(tp)}


def scores(model, texts) -> list[float]:
    return [float(s) for s in model.predict_proba(list(texts))[:, 1]]


def check(model, text, threshold=DEFAULT_THRESHOLD) -> Verdict:
    score = float(model.predict_proba([text])[0, 1])
    return Verdict("injection" if score >= threshold else "benign", score)


def save_model(model, path) -> None:
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


# ---- version 1 command line (deepset dataset only) -----------------------------------

def _load_deepset():
    from datasets import load_dataset  # imported here so the tests do not need the download

    ds = load_dataset(DATASET)
    train_split, test_split = ds["train"], ds["test"]
    return (list(train_split["text"]), list(train_split["label"]),
            list(test_split["text"]), list(test_split["label"]))


def _saved_threshold() -> float:
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH) as f:
            return json.load(f)["threshold"]
    return DEFAULT_THRESHOLD


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sentinel.baseline",
                                     description="Version 1: TF-IDF + logistic regression")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="train, choose the threshold by cross-validation, save both")
    ev = sub.add_parser("eval", help="evaluate the saved model on the held-out test split")
    ev.add_argument("--threshold", type=float, default=None, help="default: the saved threshold")
    ck = sub.add_parser("check", help="classify one text")
    ck.add_argument("text")
    ck.add_argument("--threshold", type=float, default=None, help="default: the saved threshold")
    args = parser.parse_args(argv)

    if args.command == "train":
        train_x, train_y, _, _ = _load_deepset()
        threshold, rows = tune_threshold(train_x, train_y)
        print("cross-validation on the training set:")
        print("  threshold  precision  recall  f1")
        for t, p, r, f1 in rows:
            marker = "  <- chosen" if t == threshold else ""
            print(f"  {t:9.2f}  {p:9.3f}  {r:6.3f}  {f1:.3f}{marker}")
        model = train(train_x, train_y)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        save_model(model, MODEL_PATH)
        with open(THRESHOLD_PATH, "w") as f:
            json.dump({"threshold": threshold,
                       "selected_by": "5-fold cross-validation on the training set, best F1"}, f)
        print(f"trained on {len(train_x)} examples, threshold {threshold}, saved to {MODEL_PATH}")
    elif args.command == "eval":
        _, _, test_x, test_y = _load_deepset()
        threshold = args.threshold if args.threshold is not None else _saved_threshold()
        print(f"threshold: {threshold}")
        for name, value in evaluate(load_model(MODEL_PATH), test_x, test_y, threshold).items():
            print(f"{name}: {value:.3f}" if isinstance(value, float) else f"{name}: {value}")
    else:
        threshold = args.threshold if args.threshold is not None else _saved_threshold()
        verdict = check(load_model(MODEL_PATH), args.text, threshold)
        print(f"{verdict.label} (score {verdict.score:.2f}, threshold {threshold})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
