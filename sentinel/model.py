"""The classifier: word + character TF-IDF features, then logistic regression."""
from dataclasses import dataclass

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

DEFAULT_THRESHOLD = 0.5


@dataclass
class Verdict:
    label: str    # "injection" or "benign"
    score: float  # estimated probability that the text is an injection


def build_pipeline():
    # Words catch phrases like "ignore previous instructions".
    # Character n-grams survive typos, spacing tricks and other languages.
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ])
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", classifier)])


def train(texts, labels):
    model = build_pipeline()
    model.fit(texts, labels)
    return model


def tune_threshold(texts, labels, cv=5, grid=None):
    """Picks the decision threshold using ONLY the training data (k-fold cross-validation),
    so the test set stays untouched. Returns (best_threshold, rows) where each row is
    (threshold, precision, recall, f1) measured on the held-out folds."""
    grid = grid or [round(0.1 + 0.05 * i, 2) for i in range(17)]  # 0.10 ... 0.90
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    scores = cross_val_predict(build_pipeline(), texts, labels, cv=folds, method="predict_proba")[:, 1]
    rows = []
    for threshold in grid:
        predicted = [1 if s >= threshold else 0 for s in scores]
        p, r, f1, _ = precision_recall_fscore_support(
            labels, predicted, average="binary", pos_label=1, zero_division=0)
        rows.append((threshold, p, r, f1))
    best = max(rows, key=lambda row: (row[3], -abs(row[0] - DEFAULT_THRESHOLD)))  # best F1, ties -> closest to 0.5
    return best[0], rows


def evaluate(model, texts, labels, threshold=DEFAULT_THRESHOLD):
    """Precision / recall / F1 for the 'injection' class, plus the confusion matrix."""
    scores = model.predict_proba(texts)[:, 1]
    predicted = [1 if s >= threshold else 0 for s in scores]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {"precision": precision, "recall": recall, "f1": f1,
            "true_negatives": int(tn), "false_positives": int(fp),
            "false_negatives": int(fn), "true_positives": int(tp)}


def check(model, text, threshold=DEFAULT_THRESHOLD):
    score = float(model.predict_proba([text])[0, 1])
    return Verdict("injection" if score >= threshold else "benign", score)


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
