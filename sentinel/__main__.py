"""Command line:  python -m sentinel train | eval | check "some text" """
import argparse
import json
import os
import sys

from .data import load_data
from .model import (DEFAULT_THRESHOLD, check, evaluate, load_model, save_model, train,
                    tune_threshold)

MODEL_PATH = os.path.join("models", "sentinel.joblib")
THRESHOLD_PATH = os.path.join("models", "threshold.json")


def saved_threshold():
    """The threshold chosen at training time (cross-validation), or the default 0.5."""
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH) as f:
            return json.load(f)["threshold"]
    return DEFAULT_THRESHOLD


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sentinel", description="Prompt-injection firewall")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="train, choose the threshold by cross-validation, save both")
    ev = sub.add_parser("eval", help="evaluate the saved model on the held-out test split")
    ev.add_argument("--threshold", type=float, default=None, help="default: the saved threshold")
    ck = sub.add_parser("check", help="classify one text")
    ck.add_argument("text")
    ck.add_argument("--threshold", type=float, default=None, help="default: the saved threshold")
    args = parser.parse_args(argv)

    if args.command == "train":
        train_x, train_y, _, _ = load_data()
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
        _, _, test_x, test_y = load_data()
        threshold = args.threshold if args.threshold is not None else saved_threshold()
        print(f"threshold: {threshold}")
        for name, value in evaluate(load_model(MODEL_PATH), test_x, test_y, threshold).items():
            print(f"{name}: {value:.3f}" if isinstance(value, float) else f"{name}: {value}")
    else:
        threshold = args.threshold if args.threshold is not None else saved_threshold()
        verdict = check(load_model(MODEL_PATH), args.text, threshold)
        print(f"{verdict.label} (score {verdict.score:.2f}, threshold {threshold})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
