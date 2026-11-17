"""Fine-tune a pretrained multilingual encoder to classify text as SAFE or INJECTION.

Run:  python -m sentinel.train --config configs/train.yaml

Every run is tracked with MLflow (parameters, per-epoch metrics, dataset statistics), so
any number in the README can be traced back to the exact run that produced it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ID2LABEL = {0: "SAFE", 1: "INJECTION"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def compute_metrics(eval_pred) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(logits[:, 0] - logits[:, 1]))
    preds = (probs >= 0.5).astype(int)
    return {
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the Sentinel classifier")
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    import mlflow
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(cfg["seed"])
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "sentinel")
    df = pd.read_parquet(cfg["data_path"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    def as_dataset(split: str) -> Dataset:
        part = df.loc[df["split"] == split, ["text", "label"]].reset_index(drop=True)
        ds = Dataset.from_pandas(part)
        # Training truncates to one window; long texts are handled by windowing at serving.
        return ds.map(lambda b: tokenizer(b["text"], truncation=True,
                                          max_length=cfg["max_length"]),
                      batched=True, remove_columns=["text"])

    train_ds, val_ds = as_dataset("train"), as_dataset("val")
    print(f"train={len(train_ds)}  val={len(val_ds)}  device="
          f"{'cuda' if torch.cuda.is_available() else 'cpu'}")

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)

    training_args = TrainingArguments(
        output_dir=str(Path(cfg["model_dir"]) / "checkpoints"),
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,  # keep the epoch with the best validation F1
        metric_for_best_model="f1",
        save_total_limit=1,
        logging_steps=50,
        fp16=torch.cuda.is_available(),  # mixed precision: ~2x faster on a Colab T4
        report_to=["mlflow"],
        run_name="sentinel-train",
        seed=cfg["seed"],
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=compute_metrics)

    out = Path(cfg["model_dir"])
    # One MLflow run for the whole lifecycle: `sentinel.evaluate` later adds the test
    # metrics to this same run, using the id saved next to the model.
    with mlflow.start_run(run_name="sentinel") as run:
        # Prefixed so they never clash with the params Hugging Face logs on its own.
        mlflow.log_params({f"cfg.{k}": v for k, v in cfg.items()
                           if not k.endswith(("_dir", "_path"))})
        summary = df.groupby("split")["label"].agg(rows="size", injection_rate="mean")
        mlflow.log_dict(summary.reset_index().to_dict(orient="records"), "dataset_summary.json")
        mlflow.log_artifact(args.config)

        trainer.train()
        final = trainer.evaluate()
        print("Validation:", {k: round(v, 4) for k, v in final.items() if isinstance(v, float)})

        trainer.save_model(str(out))
        tokenizer.save_pretrained(str(out))
        (out / "mlflow_run_id.txt").write_text(run.info.run_id)
    print(f"Saved fine-tuned model to {out}")


if __name__ == "__main__":
    main()
