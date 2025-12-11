import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# HuggingFace / PyTorch
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


PROCESSED = Path("data/processed")
TABLES = Path("outputs/tables")
MODELS = Path("outputs/models")

TABLES.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

YEAR_MIN = 2015
YEAR_MAX = 2025

# Proposal Study 1 keyword sets
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "racism": ["racism", "racist"],
    "police": ["police"],
    "immigration": ["immigration", "immigrant"],
}


# ----------------------------
# Utilities
# ----------------------------
def detect_device() -> str:
    # M-series Mac acceleration if available
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def simple_sentence_split(text: str) -> List[str]:
    """
    Lightweight sentence splitter without extra deps.
    Good enough for Study 1 poster replication.
    """
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []
    # Split on . ? ! ; plus ensure we keep reasonable chunks
    parts = re.split(r"(?<=[\.\?\!])\s+", text)
    # Filter very short fragments
    return [p.strip() for p in parts if len(p.strip()) >= 20]


def contains_any_keyword(sentence: str, keywords: List[str]) -> bool:
    s = sentence.lower()
    return any(k in s for k in keywords)


def map_sentiment_score(probs: np.ndarray) -> float:
    """
    Convert probabilities to a continuous sentiment score.
    Assumes label 1 = positive, label 0 = negative (SST-2 convention).
    Score in [-1, 1].
    """
    if probs.shape[-1] != 2:
        # fallback
        return 0.0
    p_neg, p_pos = float(probs[0]), float(probs[1])
    return p_pos - p_neg


# ----------------------------
# Load Study 1 corpus
# ----------------------------
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

    df = df[df["network"].isin(["CNN", "Fox"])].copy()
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)]
    return df


# ----------------------------
# Build keyword sentence dataset
# ----------------------------
def extract_keyword_sentences(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        network = r["network"]
        year = int(r["year"])
        text = r["text"]

        sentences = simple_sentence_split(text)
        if not sentences:
            continue

        for topic, kws in TOPIC_KEYWORDS.items():
            for sent in sentences:
                if contains_any_keyword(sent, kws):
                    rows.append(
                        {
                            "network": network,
                            "year": year,
                            "topic": topic,
                            "sentence": sent,
                        }
                    )

    out = pd.DataFrame(rows)
    return out


# ----------------------------
# RoBERTa fine-tuning (external sentiment dataset)
# ----------------------------
def compute_accuracy(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


def finetune_roberta_sentiment(
    model_name: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    max_train_samples: int,
    max_eval_samples: int,
):
    """
    Fine-tune RoBERTa-base on an external sentiment dataset (default SST-2),
    as required by proposal Study 1.
    """
    # This will download if not cached.
    ds = load_dataset("glue", "sst2")

    train_ds = ds["train"]
    eval_ds = ds["validation"]

    if max_train_samples and max_train_samples > 0:
        train_ds = train_ds.shuffle(seed=42).select(range(min(max_train_samples, len(train_ds))))
    if max_eval_samples and max_eval_samples > 0:
        eval_ds = eval_ds.select(range(min(max_eval_samples, len(eval_ds))))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tok(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=256)

    train_ds = train_ds.map(tok, batched=True)
    eval_ds = eval_ds.map(tok, batched=True)

    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in ("input_ids", "attention_mask", "label")])
    eval_ds = eval_ds.remove_columns([c for c in eval_ds.column_names if c not in ("input_ids", "attention_mask", "label")])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_accuracy,
    )

    trainer.train()
    metrics = trainer.evaluate()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save metrics
    pd.DataFrame([metrics]).to_csv(TABLES / "study1_sentiment_finetune_metrics.csv", index=False)

    return output_dir


# ----------------------------
# Inference on Study 1 keyword sentences
# ----------------------------
def infer_sentence_sentiment(
    sentences_df: pd.DataFrame,
    model_dir: Path,
    batch_size: int = 32,
) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    device = detect_device()
    model.to(device)
    model.eval()

    texts = sentences_df["sentence"].tolist()
    all_scores = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        batch_scores = [map_sentiment_score(p) for p in probs]
        all_scores.extend(batch_scores)

    out = sentences_df.copy()
    out["sentiment_score"] = all_scores
    return out


# ----------------------------
# SSP aggregation
# ----------------------------
def compute_yearly_means(sent_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        sent_df.groupby(["year", "topic", "network"])["sentiment_score"]
        .mean()
        .reset_index()
        .sort_values(["topic", "year", "network"])
    )
    out.to_csv(TABLES / "study1_sentiment_mean_by_year_topic_network.csv", index=False)
    return out


def compute_ssp(mean_df: pd.DataFrame) -> pd.DataFrame:
    pivot = mean_df.pivot_table(
        index=["year", "topic"],
        columns="network",
        values="sentiment_score",
        aggfunc="mean",
    ).reset_index()

    # Ensure both columns exist
    if "CNN" not in pivot.columns:
        pivot["CNN"] = np.nan
    if "Fox" not in pivot.columns:
        pivot["Fox"] = np.nan

    pivot["SSP"] = (pivot["CNN"] - pivot["Fox"]).abs()
    pivot = pivot.sort_values(["topic", "year"])

    pivot.to_csv(TABLES / "study1_ssp_timeseries.csv", index=False)
    return pivot


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Study 1 core method: RoBERTa sentiment + SSP time series.")
    parser.add_argument(
        "--corpus",
        type=str,
        default=str(PROCESSED / "study1_corpus_2015_2025.parquet"),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="roberta-base",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default=str(MODELS / "study1_roberta_sentiment"),
    )

    # Training controls (strict default = no sampling cap)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)

    # Pipeline toggles
    parser.add_argument("--skip-train", action="store_true", help="Use existing fine-tuned model if present.")
    parser.add_argument("--save-sentence-level", action="store_true", help="Save large sentence-level output CSV.")

    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus file: {corpus_path}")

    model_dir = Path(args.model_out)

    print(f"Loading corpus: {corpus_path}")
    df = load_corpus(corpus_path)

    print("Extracting keyword-containing sentences for Study 1 topics...")
    sent_df = extract_keyword_sentences(df)

    if sent_df.empty:
        raise RuntimeError("No keyword sentences found. Check corpus text or keyword rules.")

    print(f"Keyword sentence rows: {len(sent_df)}")

    if not args.skip_train or not model_dir.exists():
        print("Fine-tuning RoBERTa on external sentiment dataset (SST-2)...")
        finetune_roberta_sentiment(
            model_name=args.model_name,
            output_dir=model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
        )
    else:
        print(f"Skipping training. Using existing model at: {model_dir}")

    print("Running sentiment inference on Study 1 keyword sentences...")
    sent_scored = infer_sentence_sentiment(sent_df, model_dir=model_dir)

    if args.save_sentence_level:
        sent_scored.to_csv(TABLES / "study1_keyword_sentence_sentiment.csv", index=False)

    print("Aggregating sentiment means by year/topic/network...")
    mean_df = compute_yearly_means(sent_scored)

    print("Computing SSP time series...")
    ssp_df = compute_ssp(mean_df)

    print("\n=== Study 1 core outputs generated ===")
    print("1) outputs/tables/study1_sentiment_finetune_metrics.csv")
    print("2) outputs/tables/study1_sentiment_mean_by_year_topic_network.csv")
    print("3) outputs/tables/study1_ssp_timeseries.csv")
    if args.save_sentence_level:
        print("4) outputs/tables/study1_keyword_sentence_sentiment.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()