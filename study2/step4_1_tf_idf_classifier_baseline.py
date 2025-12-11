#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step_4_1_tf_idf_classifier_baseline.py

Baseline classifiers for Study 2:
For each topic and year (2015–2024), train a TF-IDF + LogisticRegression
classifier to distinguish CNN vs. Fox.

Data input:
    checkpoints/topics/topic_{topic}_{year}.csv
with columns: ts, year, network, sentence

Outputs:
    checkpoints/tfidf_baseline/{topic}/model_{topic}_{year}.joblib
    checkpoints/tfidf_baseline/{topic}/metrics_{topic}_{year}.json
"""

import os
import json
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from joblib import dump


# ----------------------------
# Config
# ----------------------------

DATA_DIR = "checkpoints/topics"
OUTPUT_ROOT = "checkpoints/tfidf_baseline"

TOPICS: List[str] = ["racism", "police", "immigration"]
YEARS: List[int] = list(range(2015, 2025))

RANDOM_STATE = 42
N_SPLITS = 3


# ----------------------------
# Utils
# ----------------------------

def load_topic_year_df(topic: str, year: int) -> pd.DataFrame:
    """Load topic-year CSV into DataFrame."""
    path = os.path.join(DATA_DIR, f"topic_{topic}_{year}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    # Basic cleaning
    df = df.dropna(subset=["sentence", "network"])
    df["sentence"] = df["sentence"].astype(str).str.strip()
    df = df[df["sentence"] != ""]
    return df


def encode_labels(network_series: pd.Series) -> np.ndarray:
    """
    Encode 'network' column to binary labels:
        CNN -> 0
        FOX -> 1  (treated as positive class)
    """
    net = network_series.str.lower().str.strip()
    mapping = {"cnn": 0, "fox": 1}
    if not set(net.unique()).issubset(mapping.keys()):
        raise ValueError(f"Unexpected network labels: {net.unique()}")
    return net.map(mapping).astype(int).values


def build_pipeline() -> Pipeline:
    """
    TF-IDF + LogisticRegression pipeline.

    We use:
      - Unigram + bigram
      - class_weight='balanced' to mitigate 9:1 imbalance
    """
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=3,
        strip_accents="unicode",
        lowercase=True,
    )

    clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    pipe = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf),
    ])
    return pipe


def cross_validate_topic_year(
    sentences: List[str],
    labels: np.ndarray,
    n_splits: int = N_SPLITS,
) -> Dict[str, float]:
    """
    3-fold stratified CV, return averaged metrics.
    Fox (label=1) treated as positive class.
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    accs, precs, recs, f1s = [], [], [], []

    for train_idx, val_idx in skf.split(sentences, labels):
        X_train = [sentences[i] for i in train_idx]
        X_val = [sentences[i] for i in val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        pipe = build_pipeline()
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val,
            y_pred,
            average="binary",
            pos_label=1,
            zero_division=0,
        )

        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    metrics = {
        "eval_accuracy": float(np.mean(accs)),
        "eval_precision": float(np.mean(precs)),
        "eval_recall": float(np.mean(recs)),
        "eval_f1": float(np.mean(f1s)),
        "n_samples": int(len(sentences)),
        "n_splits": int(n_splits),
    }
    return metrics


def train_full_model(
    sentences: List[str],
    labels: np.ndarray,
) -> Pipeline:
    """Train TF-IDF + LR on full data (used for saving baseline model)."""
    pipe = build_pipeline()
    pipe.fit(sentences, labels)
    return pipe


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ----------------------------
# Main
# ----------------------------

def main():
    ensure_dir(OUTPUT_ROOT)

    for topic in TOPICS:
        topic_out_dir = os.path.join(OUTPUT_ROOT, topic)
        ensure_dir(topic_out_dir)

        print(f"\n=== Topic: {topic} ===")

        for year in tqdm(YEARS, desc=f"Topic {topic}", ncols=80):
            try:
                df = load_topic_year_df(topic, year)
            except FileNotFoundError:
                print(f"[WARN] Missing file for topic={topic}, year={year}, skip.")
                continue

            if df.empty:
                print(f"[WARN] Empty data for topic={topic}, year={year}, skip.")
                continue

            sentences = df["sentence"].tolist()
            labels = encode_labels(df["network"])

            # 1) 3-fold CV metrics
            metrics = cross_validate_topic_year(sentences, labels, N_SPLITS)

            # 2) Train final model on full data
            model = train_full_model(sentences, labels)

            # 3) Save model
            model_path = os.path.join(
                topic_out_dir,
                f"tfidf_model_{topic}_{year}.joblib",
            )
            dump(model, model_path)

            # 4) Save metrics
            metrics_path = os.path.join(
                topic_out_dir,
                f"tfidf_metrics_{topic}_{year}.json",
            )
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "topic": topic,
                        "year": year,
                    } | metrics,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(
                f"[{topic} {year}] "
                f"acc={metrics['eval_accuracy']:.3f}, "
                f"prec={metrics['eval_precision']:.3f}, "
                f"rec={metrics['eval_recall']:.3f}, "
                f"f1={metrics['eval_f1']:.3f}"
            )


if __name__ == "__main__":
    main()
