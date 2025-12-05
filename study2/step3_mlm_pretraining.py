import os
import random

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    RobertaTokenizerFast,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)


TOPIC_DIR = "checkpoints/topics"
OUTPUT_ROOT = "checkpoints/mlm"

BASE_MODEL_NAME = "bert-base-uncased"
# BASE_MODEL_NAME = "roberta-base"

RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_topics(topic_dir: str):
    topics = []
    if not os.path.isdir(topic_dir):
        raise FileNotFoundError(f"Topic directory not found: {topic_dir}")

    for fname in os.listdir(topic_dir):
        if not fname.startswith("topic_") or not fname.endswith(".csv"):
            continue

        basename = fname[:-4]
        if basename.count("_") == 1:
            topic = basename.split("_", 1)[1]
            topics.append(topic)

    topics = sorted(set(topics))
    return topics


def load_topic_corpus(topic: str) -> Dataset:
    path = os.path.join(TOPIC_DIR, f"topic_{topic}.csv")

    print(f"[{topic}] Loading corpus from {path}")
    df = pd.read_csv(path)

    df = df.dropna(subset=["sentence"])
    df["sentence"] = df["sentence"].astype(str).str.strip()
    df = df[df["sentence"] != ""]
    texts = df["sentence"].tolist()

    print(f"[{topic}] Total sentences: {len(texts)}")

    dataset = Dataset.from_dict({"text": texts})
    return dataset


def tokenize_function(examples, tokenizer, max_length: int = 128):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def train_mlm_for_topic(topic: str):
    print(f"\n========== Topic: {topic} ==========")
    dataset = load_topic_corpus(topic)

    if len(dataset) < 1000:
        print(f"[{topic}] Dataset small (<1000), using all for training, no eval split.")
        train_dataset = dataset
        eval_dataset = None
    else:
        split = dataset.train_test_split(test_size=0.05, seed=RANDOM_SEED)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"[{topic}] Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    print(f"[{topic}] Loading tokenizer & base model: {BASE_MODEL_NAME}")
    if BASE_MODEL_NAME.startswith("bert"):
        tokenizer = BertTokenizerFast.from_pretrained(BASE_MODEL_NAME)
        model = BertForMaskedLM.from_pretrained(BASE_MODEL_NAME)
    elif BASE_MODEL_NAME.startswith("roberta"):
        tokenizer = RobertaTokenizerFast.from_pretrained(BASE_MODEL_NAME)
        model = RobertaForMaskedLM.from_pretrained(BASE_MODEL_NAME)
    else:
        raise Exception(f"Unknown base model: {BASE_MODEL_NAME}")

    # Tokenization
    def _tok_fn(batch):
        return tokenize_function(batch, tokenizer, max_length=128)

    print(f"[{topic}] Tokenizing dataset...")
    train_dataset_tok = train_dataset.map(
        _tok_fn,
        batched=True,
        remove_columns=["text"],
        desc=f"Tokenizing train ({topic})",
    )
    if eval_dataset is not None:
        eval_dataset_tok = eval_dataset.map(
            _tok_fn,
            batched=True,
            remove_columns=["text"],
            desc=f"Tokenizing eval ({topic})",
        )
    else:
        eval_dataset_tok = None

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    output_dir = os.path.join(OUTPUT_ROOT, BASE_MODEL_NAME, topic)
    os.makedirs(output_dir, exist_ok=True)

    fp16 = torch.cuda.is_available()
    print(f"[{topic}] Using CUDA: {torch.cuda.is_available()}, fp16={fp16}")

    # 这里的超参数可以根据你的实际数据量调整
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        logging_steps=100,
        eval_strategy="epoch" if eval_dataset_tok is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=fp16,
        dataloader_num_workers=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tok,
        eval_dataset=eval_dataset_tok,
        data_collator=data_collator,
    )

    print(f"[{topic}] Starting MLM training...")
    trainer.train()
    print(f"[{topic}] Training done. Saving model to {output_dir}")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[{topic}] Saved model & tokenizer.\n")


def main():
    set_seed(RANDOM_SEED)

    print(f"Scanning topics in: {TOPIC_DIR}")
    topics = find_topics(TOPIC_DIR)

    print(f"Found topics: {topics}")

    for topic in topics:
        try:
            train_mlm_for_topic(topic)
        except Exception as e:
            print(f"[{topic}] ERROR during MLM training: {e}")

if __name__ == "__main__":
    main()