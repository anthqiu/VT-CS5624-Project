import argparse
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


PROCESSED = Path("data/processed")
TABLES = Path("outputs/tables")
TABLES.mkdir(parents=True, exist_ok=True)

YEAR_MIN = 2015
YEAR_MAX = 2025


# ----------------------------
# Compatibility helper for old/new scikit-learn
# ----------------------------
def get_vocab_names(vectorizer: CountVectorizer):
    """
    Old sklearn: get_feature_names()
    New sklearn: get_feature_names_out()
    """
    if hasattr(vectorizer, "get_feature_names_out"):
        return vectorizer.get_feature_names_out()
    return vectorizer.get_feature_names()


# ----------------------------
# Text preprocessing (light but standard for LDA/lexical stats)
# ----------------------------
def basic_token_normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\\S+", " ", text)
    text = re.sub(r"[^a-z\\s']", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def load_corpus(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    required = {"network", "year", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Corpus missing required columns: {missing}")

    df = df.dropna(subset=["network", "year", "text"])
    df["network"] = df["network"].astype(str)
    df["text"] = df["text"].astype(str)
    df["year"] = df["year"].astype(int)

    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)]
    return df


# ----------------------------
# (A) Distinctive terms via log-odds
# ----------------------------
def log_odds_ratio_with_prior(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    alpha: float = 0.01,
) -> np.ndarray:
    """
    Compute log-odds ratio with a symmetric Dirichlet prior.
    Returns z-like scores (larger => more distinctive for A).
    """
    a = counts_a + alpha
    b = counts_b + alpha

    a0 = a.sum()
    b0 = b.sum()

    # Avoid division by zero edge cases
    a = np.clip(a, 1e-12, None)
    b = np.clip(b, 1e-12, None)
    a0 = max(a0, 1e-12)
    b0 = max(b0, 1e-12)

    log_odds = np.log(a / (a0 - a)) - np.log(b / (b0 - b))
    var = 1.0 / a + 1.0 / b
    z = log_odds / np.sqrt(var)
    return z


def compute_distinctive_terms_overall(
    df: pd.DataFrame,
    vectorizer: CountVectorizer,
    X,
    top_k: int = 50,
) -> pd.DataFrame:
    """
    Overall CNN vs Fox distinctive terms.
    """
    vocab = np.array(get_vocab_names(vectorizer))

    mask_cnn = df["network"].values == "CNN"
    mask_fox = df["network"].values == "Fox"

    X_cnn = X[mask_cnn].sum(axis=0)
    X_fox = X[mask_fox].sum(axis=0)

    counts_cnn = np.asarray(X_cnn).ravel()
    counts_fox = np.asarray(X_fox).ravel()

    z = log_odds_ratio_with_prior(counts_cnn, counts_fox, alpha=0.01)

    idx_cnn = np.argsort(z)[-top_k:][::-1]
    idx_fox = np.argsort(z)[:top_k]

    out = pd.DataFrame(
        {
            "top_cnn_term": vocab[idx_cnn],
            "cnn_z": z[idx_cnn],
            "top_fox_term": vocab[idx_fox],
            "fox_z": z[idx_fox],
        }
    )
    out.to_csv(TABLES / "study1_distinctive_terms_overall.csv", index=False)
    return out


def compute_distinctive_terms_by_year(
    df: pd.DataFrame,
    vectorizer: CountVectorizer,
    X,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Per-year CNN vs Fox distinctive terms.
    """
    vocab = np.array(get_vocab_names(vectorizer))
    rows = []

    for year in range(YEAR_MIN, YEAR_MAX + 1):
        sub_idx = df.index[df["year"] == year].to_numpy()
        if len(sub_idx) == 0:
            continue

        sub = df.loc[sub_idx]
        X_sub = X[sub_idx]

        mask_cnn = sub["network"].values == "CNN"
        mask_fox = sub["network"].values == "Fox"

        if mask_cnn.sum() == 0 or mask_fox.sum() == 0:
            continue

        counts_cnn = np.asarray(X_sub[mask_cnn].sum(axis=0)).ravel()
        counts_fox = np.asarray(X_sub[mask_fox].sum(axis=0)).ravel()

        z = log_odds_ratio_with_prior(counts_cnn, counts_fox, alpha=0.01)

        idx_cnn = np.argsort(z)[-top_k:][::-1]
        idx_fox = np.argsort(z)[:top_k]

        for t in idx_cnn:
            rows.append(
                {"year": year, "network_favored": "CNN", "term": vocab[t], "z": float(z[t])}
            )
        for t in idx_fox:
            rows.append(
                {"year": year, "network_favored": "Fox", "term": vocab[t], "z": float(z[t])}
            )

    out = pd.DataFrame(rows).sort_values(
        ["year", "network_favored", "z"],
        ascending=[True, True, False],
    )
    out.to_csv(TABLES / "study1_distinctive_terms_by_year.csv", index=False)
    return out


# ----------------------------
# (B) LDA topics
# ----------------------------
def fit_lda(
    X,
    n_topics: int = 20,
    random_state: int = 42,
) -> LatentDirichletAllocation:
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=20,
        learning_method="batch",
        random_state=random_state,
        evaluate_every=-1,
        n_jobs=1,  # stable on macOS
    )
    lda.fit(X)
    return lda


def top_words_per_topic(
    lda: LatentDirichletAllocation,
    feature_names: List[str],
    top_n: int = 15,
) -> pd.DataFrame:
    rows = []
    for k, topic_vec in enumerate(lda.components_):
        top_idx = np.argsort(topic_vec)[-top_n:][::-1]
        for rank, i in enumerate(top_idx, start=1):
            rows.append(
                {
                    "topic": k,
                    "rank": rank,
                    "term": feature_names[i],
                    "weight": float(topic_vec[i]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "study1_topics_topwords.csv", index=False)
    return out


def topic_prevalence_by_year_network(
    df: pd.DataFrame,
    doc_topic: np.ndarray,
) -> pd.DataFrame:
    """
    Mean topic proportions per (year, network).
    """
    topic_cols = [f"topic_{i}" for i in range(doc_topic.shape[1])]
    dt = pd.DataFrame(doc_topic, columns=topic_cols)
    meta = df[["year", "network"]].reset_index(drop=True)
    merged = pd.concat([meta, dt], axis=1)

    out = (
        merged.groupby(["year", "network"])[topic_cols]
        .mean()
        .reset_index()
        .sort_values(["year", "network"])
    )
    out.to_csv(TABLES / "study1_topic_prevalence_by_year_network.csv", index=False)
    return out


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Study 1 Step4 core results (lexical + LDA).")
    parser.add_argument(
        "--corpus",
        type=str,
        default=str(PROCESSED / "study1_corpus_2015_2025.parquet"),
    )
    parser.add_argument("--n-topics", type=int, default=20)
    parser.add_argument("--max-features", type=int, default=60000)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=0.6)
    parser.add_argument("--top-k-terms", type=int, default=50)
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus file: {corpus_path}")

    print(f"Loading corpus: {corpus_path}")
    df = load_corpus(corpus_path)

    df = df[df["network"].isin(["CNN", "Fox"])].copy()
    df["text_norm"] = df["text"].apply(basic_token_normalize)

    print("Building CountVectorizer...")
    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=(1, 1),
    )

    X = vectorizer.fit_transform(df["text_norm"])

    # ----- A) Lexical distinctive terms -----
    print("Computing overall distinctive terms (CNN vs Fox)...")
    compute_distinctive_terms_overall(
        df, vectorizer, X, top_k=args.top_k_terms
    )

    print("Computing per-year distinctive terms (CNN vs Fox)...")
    compute_distinctive_terms_by_year(
        df, vectorizer, X, top_k=20
    )

    # ----- B) LDA topics -----
    print(f"Fitting LDA with n_topics={args.n_topics}...")
    lda = fit_lda(X, n_topics=args.n_topics)

    feature_names = list(get_vocab_names(vectorizer))
    print("Saving top words per topic...")
    top_words_per_topic(lda, feature_names, top_n=15)

    print("Inferring document-topic distributions...")
    doc_topic = lda.transform(X)

    print("Saving topic prevalence by year/network...")
    topic_prevalence_by_year_network(df, doc_topic)

    print("\n=== Step4 core outputs generated ===")
    print("1) outputs/tables/study1_distinctive_terms_overall.csv")
    print("2) outputs/tables/study1_distinctive_terms_by_year.csv")
    print("3) outputs/tables/study1_topics_topwords.csv")
    print("4) outputs/tables/study1_topic_prevalence_by_year_network.csv")


if __name__ == "__main__":
    main()