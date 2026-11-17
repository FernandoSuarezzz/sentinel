"""Evaluate the model exactly as it ships (ONNX int8 + windowing) and benchmark a baseline.

Run:  python -m sentinel.evaluate --config configs/train.yaml

Steps
 1. Pick the decision threshold on the *validation* set: the lowest threshold whose false
    positive rate stays <= target_fpr (default 1%). The test sets are never used to tune.
 2. Measure on two held-out sets:
      test      - same attack styles as training, unseen texts and payload variants
      ood_test  - attack *families* never seen in training (generalisation)
 3. Compare with version 1 of Sentinel (TF-IDF + logistic regression, retrained on the same
    data, same threshold rule) and with an existing open-source detector.
 4. Measure CPU latency of PyTorch vs ONNX fp32 vs ONNX int8.
Writes artifacts/report/report.json and report.md (tables ready for the README).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from sentinel import baseline as baseline_v1
from sentinel.detector import Detector


def threshold_for_fpr(y: np.ndarray, p: np.ndarray, target_fpr: float) -> float:
    """Lowest threshold such that at most `target_fpr` of negatives score >= it."""
    negatives = np.sort(p[y == 0])[::-1]
    k = int(np.floor(target_fpr * len(negatives)))  # negatives we may let through
    if k >= len(negatives):
        return 0.5
    # Strictly above the (k+1)-th highest negative score -> at most k false positives.
    return float(np.clip(np.nextafter(negatives[k], np.inf), 1e-4, 0.9999))


def metrics_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")
    return {"n": len(y), "precision": precision, "recall": recall, "f1": f1,
            "fpr": fp / (fp + tn) if fp + tn else 0.0, "roc_auc": auc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def breakdown(part: pd.DataFrame, p: np.ndarray, threshold: float, by: str) -> pd.DataFrame:
    """Recall (attacks caught) and FPR (legit texts blocked) per group."""
    rows = []
    for key, idx in part.groupby(by).indices.items():
        y, s = part["label"].to_numpy()[idx], p[idx]
        pred = s >= threshold
        rows.append({by: key, "rows": len(idx),
                     "recall": pred[y == 1].mean() if (y == 1).any() else np.nan,
                     "fpr": pred[y == 0].mean() if (y == 0).any() else np.nan})
    return pd.DataFrame(rows).set_index(by).round(3)


def _batched(score: Callable[[list[str]], list[float]], texts: Sequence[str], bs: int = 32):
    out: list[float] = []
    for i in range(0, len(texts), bs):
        out.extend(score(list(texts[i:i + bs])))
    return np.asarray(out)


def _latency_ms(fn: Callable[[], object], runs: int = 100) -> dict:
    for _ in range(5):  # warm-up
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    q = statistics.quantiles(times, n=20)
    return {"p50": round(statistics.median(times), 1), "p95": round(q[18], 1)}


def _baseline_scorer(name: str):
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("text-classification", model=name, device=device, top_k=None,
                   truncation=True, max_length=512)
    positive = next(lbl for lbl in clf.model.config.id2label.values()
                    if "INJ" in lbl.upper() or "JAIL" in lbl.upper())

    def score(texts: list[str]) -> list[float]:
        return [next(d["score"] for d in out if d["label"] == positive) for out in clf(texts)]

    return score


def _md_table(rows: dict[str, dict], cols: list[str]) -> str:
    head = "| | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    body = "".join(
        f"| {name} | " + " | ".join(
            f"{m[c]:.3f}" if isinstance(m[c], float) else str(m[c]) for c in cols) + " |\n"
        for name, m in rows.items())
    return head + body


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Sentinel")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    df = pd.read_parquet(cfg["data_path"])
    onnx_dir, report_dir = Path(cfg["onnx_dir"]), Path(cfg["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    detector = Detector.from_dir(onnx_dir)

    def sentinel_score(texts: list[str]) -> list[float]:
        return [r.score for r in detector.scan_batch(texts)]

    # 1. Threshold from validation only.
    df["label"] = df["label"].astype(int)
    val = df[df["split"] == "val"]
    p_val = _batched(sentinel_score, val["text"].tolist())
    threshold = threshold_for_fpr(val["label"].to_numpy(), p_val, cfg["target_fpr"])
    detector.config.threshold = threshold
    detector.config.save(onnx_dir)  # the served model uses the calibrated threshold
    print(f"Threshold for FPR <= {cfg['target_fpr']:.0%} on validation: {threshold:.4f}")

    report: dict = {"threshold": threshold, "target_fpr": cfg["target_fpr"], "splits": {}}
    md = [f"Threshold {threshold:.3f} (chosen on validation for FPR <= "
          f"{cfg['target_fpr']:.0%}).\n"]

    # 3a. Our own baseline (version 1 of Sentinel): same training data, same threshold rule.
    #     This isolates the effect of the model itself: TF-IDF + logistic regression vs
    #     a fine-tuned transformer.
    train_part = df[df["split"] == "train"]
    v1 = baseline_v1.train(train_part["text"].tolist(), train_part["label"].tolist())
    v1_threshold = threshold_for_fpr(val["label"].to_numpy(),
                                     np.asarray(baseline_v1.scores(v1, val["text"])),
                                     cfg["target_fpr"])
    report["v1_threshold"] = v1_threshold
    md.append(f"Version 1 baseline threshold {v1_threshold:.3f} (same rule).\n")

    # 3b. An existing open-source detector, at its own default threshold.
    external = None if args.skip_baseline else _baseline_scorer(cfg["baseline_model"])
    for split in ("test", "ood_test"):
        part = df[df["split"] == split].reset_index(drop=True)
        y = part["label"].to_numpy()
        p = _batched(sentinel_score, part["text"].tolist())
        rows = {"Sentinel v2 (transformer, int8 ONNX)": metrics_at(y, p, threshold)}
        p_v1 = np.asarray(baseline_v1.scores(v1, part["text"]))
        rows["Sentinel v1 (TF-IDF + logistic regression)"] = metrics_at(y, p_v1, v1_threshold)
        if external is not None:
            pb = _batched(external, part["text"].tolist())
            rows["ProtectAI DeBERTa @0.5 (its default)"] = metrics_at(y, pb, 0.5)
        report["splits"][split] = rows
        md.append(f"\n### {split}\n\n" + _md_table(rows, ["precision", "recall", "f1", "fpr",
                                                          "roc_auc"]))
        group = "family" if split == "ood_test" else "source"
        for by in (group, "lang"):
            table = breakdown(part, p, threshold, by)
            table_v1 = breakdown(part, p_v1, v1_threshold, by)
            table = table.join(table_v1[["recall", "fpr"]], rsuffix="_v1")
            report["splits"][split][f"by_{by}"] = table.reset_index().to_dict(orient="records")
            md.append(f"\nBy {by} (v2, then v1 with _v1):\n\n{table.to_markdown()}\n")

    # 4. Latency on CPU, single request (what one API call costs).
    short = df.loc[df["split"] == "test", "text"].str.len().sort_values()
    short_text = df.loc[short.index[len(short) // 2], "text"]  # median-length text
    long_text = " ".join(df.loc[df["split"] == "test", "text"].head(60))[:6000]
    import onnxruntime as ort

    fp32 = Detector(ort.InferenceSession(str(onnx_dir / "model.onnx"),
                                         providers=["CPUExecutionProvider"]),
                    detector.tokenizer, detector.config)
    latency = {
        "ONNX int8, median text": _latency_ms(lambda: detector.scan(short_text)),
        "ONNX fp32, median text": _latency_ms(lambda: fp32.scan(short_text)),
        f"ONNX int8, long text ({detector.scan(long_text).chunks_scanned} windows)":
            _latency_ms(lambda: detector.scan(long_text), runs=30),
    }
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg["model_dir"])
        model = AutoModelForSequenceClassification.from_pretrained(cfg["model_dir"]).eval()
        enc = tok(short_text, truncation=True, max_length=cfg["max_length"], return_tensors="pt")

        def torch_once():
            with torch.no_grad():
                model(**enc)

        latency["PyTorch fp32, median text"] = _latency_ms(torch_once)
    except (ImportError, OSError) as exc:
        print(f"Skipping PyTorch latency: {exc}")
    report["latency_ms_cpu"] = latency
    md.append("\n### CPU latency per request (ms)\n\n| | p50 | p95 |\n|---|---|---|\n"
              + "".join(f"| {k} | {v['p50']} | {v['p95']} |\n" for k, v in latency.items()))
    export_report = onnx_dir / "export_report.json"
    if export_report.exists():
        report["export"] = json.loads(export_report.read_text())

    (report_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))
    (report_dir / "report.md").write_text("".join(md))
    print("".join(md))

    run_file = onnx_dir / "mlflow_run_id.txt"
    if run_file.exists():
        import mlflow

        with mlflow.start_run(run_id=run_file.read_text().strip()):
            mlflow.log_metric("threshold", threshold)
            for split, rows in report["splits"].items():
                for k, v in rows["Sentinel v2 (transformer, int8 ONNX)"].items():
                    if isinstance(v, float):
                        mlflow.log_metric(f"{split}_{k}", v)
            mlflow.log_artifacts(str(report_dir), "report")


if __name__ == "__main__":
    main()
