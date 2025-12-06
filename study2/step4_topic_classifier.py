import os
import json
import random
import numpy as np
import pandas as pd
import torch
from collections import Counter

from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

# BASE_MODEL_NAME = "bert-base-uncased"
BASE_MODEL_NAME = "roberta-base"

TOPIC_DIR = "checkpoints/topics"
MLM_ROOT = os.path.join("checkpoints/mlm", BASE_MODEL_NAME)
CLASSIFIER_ROOT = os.path.join("checkpoints/classifier", BASE_MODEL_NAME)

YEARS = list(range(2015, 2025))
from utils.config import TOPIC_KEYWORDS

RANDOM_SEED = 42
FOLDS = 3


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_tokenizer_and_model_for_topic(topic: str):
    mlm_dir = os.path.join(MLM_ROOT, topic)
    if not os.path.isdir(mlm_dir):
        raise FileNotFoundError(f"MLM checkpoint not found: {mlm_dir}")

    id2label = {0: "CNN", 1: "FOX"}
    label2id = {"CNN": 0, "FOX": 1}

    if BASE_MODEL_NAME.startswith("bert"):
        tokenizer = BertTokenizerFast.from_pretrained(mlm_dir)
        model = BertForSequenceClassification.from_pretrained(
            mlm_dir,
            num_labels=2,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained(mlm_dir)
        model = RobertaForSequenceClassification.from_pretrained(
            mlm_dir,
            num_labels=2,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )

    return tokenizer, model


def load_topic_year_dataframe(topic: str, year: int) -> pd.DataFrame:
    path = os.path.join(TOPIC_DIR, f"topic_{topic}_{year}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df = df.dropna(subset=["sentence", "network"])
    df["sentence"] = df["sentence"].astype(str).str.strip()

    def norm(x):
        x = str(x).upper()
        if "CNN" in x: return "CNN"
        if "FOX" in x: return "FOX"
        return None

    df["network_norm"] = df["network"].map(norm)
    df = df[df["network_norm"].isin(["CNN", "FOX"])]
    df["label"] = df["network_norm"].map({"CNN": 0, "FOX": 1})

    return df


def oversample_minority(df: pd.DataFrame):
    df_cnn = df[df.label == 0]
    df_fox = df[df.label == 1]

    if len(df_fox) == 0:
        return df  # fallback

    ratio = len(df_cnn) / len(df_fox)
    repeat_factor = max(1, int(ratio))

    df_fox_up = pd.concat([df_fox] * repeat_factor, ignore_index=True)
    balanced = pd.concat([df_cnn, df_fox_up], ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=42)

    print(f"Upsampling FOX: CNN={len(df_cnn)}, FOX={len(df_fox)} → FOX_up={len(df_fox_up)}")
    return balanced


def to_hf_dataset(df):
    return Dataset.from_pandas(df[["sentence", "label"]], preserve_index=False)


def tokenize_dataset(hf_ds, tokenizer, max_length=128):
    def tok(batch):
        enc = tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc["labels"] = batch["label"]
        return enc

    return hf_ds.map(tok, batched=True, remove_columns=["sentence", "label"])


def compute_class_weights(df: pd.DataFrame):
    counts = Counter(df["label"])
    total = counts[0] + counts[1]
    w_cnn = total / counts[0]
    w_fox = total / counts[1]
    return torch.tensor([w_cnn, w_fox], dtype=torch.float32)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs["logits"]

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = (preds == labels).mean()

    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }



def train_classifier_3fold(topic, year):
    print(f"\n===== {topic.upper()} YEAR {year} =====")

    df = load_topic_year_dataframe(topic, year)

    if len(df) < 300:
        print(f"[{topic}][{year}] Too few samples, skip.")
        return

    # Oversample FOX
    # df_balanced = oversample_minority(df)
    df_balanced = df
    ds = to_hf_dataset(df_balanced)

    N = len(ds)
    indices = np.arange(N)
    np.random.shuffle(indices)

    fold_size = N // FOLDS

    for fold in range(FOLDS):
        print(f"\n---- FOLD {fold} ----")

        start = fold * fold_size
        end = (fold + 1) * fold_size if fold < FOLDS - 1 else N

        test_idx = indices[start:end]
        remain_idx = np.concatenate([indices[:start], indices[end:]])

        val_size = len(remain_idx) // 8
        val_idx = remain_idx[:val_size]
        train_idx = remain_idx[val_size:]

        train_df = df_balanced.iloc[train_idx]
        val_df = df_balanced.iloc[val_idx]
        test_df = df_balanced.iloc[test_idx]

        train_ds = to_hf_dataset(train_df)
        val_ds = to_hf_dataset(val_df)
        test_ds = to_hf_dataset(test_df)

        tokenizer, model = get_tokenizer_and_model_for_topic(topic)

        train_tok = tokenize_dataset(train_ds, tokenizer)
        val_tok = tokenize_dataset(val_ds, tokenizer)
        test_tok = tokenize_dataset(test_ds, tokenizer)

        weights = compute_class_weights(train_df)

        data_collator = DataCollatorWithPadding(tokenizer)

        out_dir = os.path.join(CLASSIFIER_ROOT, topic, str(year), f"fold{fold}")
        os.makedirs(out_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=out_dir,
            overwrite_output_dir=True,
            num_train_epochs=6,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            warmup_steps=200,
            logging_steps=100,
            save_steps=200,
            eval_steps=200,
            save_total_limit=2,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            class_weights=weights,
        )

        trainer.train()

        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        test_tok.save_to_disk(os.path.join(out_dir, "test_dataset"))

        print(f"[Fold {fold}] Saved → {out_dir}")

        metrics = trainer.evaluate(test_tok)

        clean_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        clean_metrics["fold"] = fold
        clean_metrics["topic"] = topic
        clean_metrics["year"] = year

        metrics_path = os.path.join(out_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(clean_metrics, f, indent=4)

        print(f"[Fold {fold}] Final Test Metrics:", clean_metrics)
        print(f"[Fold {fold}] metrics.json written → {metrics_path}")


def main():
    set_seed(RANDOM_SEED)
    print("Topics:", TOPIC_KEYWORDS.keys())
    print("Years:", YEARS)

    for topic in TOPIC_KEYWORDS.keys():
        for year in YEARS:
            try:
                train_classifier_3fold(topic, year)
            except Exception as e:
                print(f"Error at [{topic}][{year}] → {e}")


if __name__ == "__main__":
    main()
