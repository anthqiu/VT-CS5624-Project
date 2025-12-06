import os
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
)

from utils.config import TOPIC_KEYWORDS  # 你的 topic 定义

# BASE_MODEL_NAME = "bert-base-uncased"
BASE_MODEL_NAME = "roberta-base"

CLASSIFIER_ROOT = os.path.join("checkpoints/classifier", BASE_MODEL_NAME)
IG_ROOT = os.path.join("checkpoints/ig", BASE_MODEL_NAME)

YEARS = list(range(2015, 2025))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IG_STEPS = 20
MAX_SAMPLES_PER_FOLD = 2000
MIN_TOKEN_COUNT = 8


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer_and_model(ckpt_dir: str):
    if BASE_MODEL_NAME.startswith("bert"):
        tokenizer = BertTokenizerFast.from_pretrained(ckpt_dir)
        model = BertForSequenceClassification.from_pretrained(ckpt_dir)
    elif BASE_MODEL_NAME.startswith("roberta"):
        tokenizer = RobertaTokenizerFast.from_pretrained(ckpt_dir)
        model = RobertaForSequenceClassification.from_pretrained(ckpt_dir)
    else:
        raise ValueError(f"Unknown BASE_MODEL_NAME: {BASE_MODEL_NAME}")

    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def integrated_gradients_single(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    steps: int = IG_STEPS,
):
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    emb_layer = model.get_input_embeddings()
    input_embeds = emb_layer(input_ids)          # [1, seq_len, hidden]
    baseline = torch.zeros_like(input_embeds)    # baseline = 全零 embedding

    delta = input_embeds - baseline
    total_grads = torch.zeros_like(input_embeds)

    for alpha in torch.linspace(0.0, 1.0, steps + 1)[1:]:
        scaled_embeds = baseline + alpha * delta
        scaled_embeds.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        outputs = model(
            inputs_embeds=scaled_embeds,
            attention_mask=attention_mask,
        )

        probs = torch.softmax(outputs.logits, dim=-1)
        prob_cnn = probs[:, 0].sum()

        grads = torch.autograd.grad(
            outputs=prob_cnn,
            inputs=scaled_embeds,
            retain_graph=False,
            create_graph=False,
        )[0]

        total_grads += grads

    avg_grads = total_grads / float(steps)
    ig = delta * avg_grads
    token_attr = ig.sum(dim=-1).squeeze(0)

    return token_attr.detach().cpu()


def merge_tokens_to_words_bert(tokens, attrs, special_ids, ids):
    merged = []
    current_pieces = []
    current_attr = 0.0

    for tid, tok, attr in zip(ids, tokens, attrs):
        if tid in special_ids:
            if current_pieces:
                word = "".join(current_pieces)
                merged.append((word, current_attr))
                current_pieces = []
                current_attr = 0.0
            continue

        if tok.startswith("##"):
            sub = tok[2:]
            current_pieces.append(sub)
            current_attr += attr
        else:
            if current_pieces:
                word = "".join(current_pieces)
                merged.append((word, current_attr))
            current_pieces = [tok]
            current_attr = attr

    if current_pieces:
        word = "".join(current_pieces)
        merged.append((word, current_attr))

    return merged


def merge_tokens_to_words_roberta(tokens, attrs, special_ids, ids):
    merged = []
    for tid, tok, attr in zip(ids, tokens, attrs):
        if tid in special_ids:
            continue
        word = tok.lstrip("Ġ").strip()
        if not word:
            continue
        merged.append((word, attr))
    return merged


def process_topic_year(topic: str, year: int):
    print(f"\n=== IG for topic={topic}, year={year} ===")

    fold_dirs = []
    for fold in range(3):
        ckpt_dir = os.path.join(CLASSIFIER_ROOT, topic, str(year), f"fold{fold}")
        test_dir = os.path.join(ckpt_dir, "test_dataset")
        if os.path.isdir(ckpt_dir) and os.path.isdir(test_dir):
            fold_dirs.append((fold, ckpt_dir, test_dir))
        else:
            print(f"[WARN] Missing fold {fold} for {topic}-{year}: {ckpt_dir} / {test_dir}")

    if not fold_dirs:
        print(f"[SKIP] No folds available for {topic}-{year}")
        return

    token_to_sum = defaultdict(float)
    token_to_count = defaultdict(int)

    for fold, ckpt_dir, test_dir in fold_dirs:
        print(f"  - Fold {fold}: {ckpt_dir}")

        tokenizer, model = load_tokenizer_and_model(ckpt_dir)
        special_ids = set(tokenizer.all_special_ids)

        ds = load_from_disk(test_dir)

        n_samples = len(ds)
        print(f"    Test samples: {n_samples}")

        if n_samples > MAX_SAMPLES_PER_FOLD:
            idx = np.random.choice(n_samples, size=MAX_SAMPLES_PER_FOLD, replace=False)
            idx = sorted(idx.tolist())
        else:
            idx = list(range(n_samples))

        for i in tqdm(idx, desc=f"IG fold {fold}", ncols=100):
            example = ds[i]

            input_ids = torch.tensor(example["input_ids"], dtype=torch.long).unsqueeze(0)
            attention_mask = torch.tensor(example["attention_mask"], dtype=torch.long).unsqueeze(0)

            # 计算 IG
            try:
                token_attr = integrated_gradients_single(
                    model,
                    input_ids,
                    attention_mask,
                    steps=IG_STEPS,
                )
            except Exception as e:
                print(f"    [WARN] IG failed on sample {i}: {e}")
                continue

            ids = example["input_ids"]
            tokens = tokenizer.convert_ids_to_tokens(ids)
            attrs = token_attr.tolist()

            # if BASE_MODEL_NAME.startswith("bert"):
            #     words = merge_tokens_to_words_bert(tokens, attrs, special_ids, ids)
            # elif BASE_MODEL_NAME.startswith("roberta"):
            #     words = merge_tokens_to_words_roberta(tokens, attrs, special_ids, ids)
            # else:
            # fallback:
            words = []
            for tid, tok, attr in zip(ids, tokens, attrs):
                if tid in special_ids:
                    continue
                w = tok.strip()
                if not w:
                    continue
                words.append((w, attr))

            for word, w_attr in words:
                w_norm = word.lower().strip()
                if not w_norm:
                    continue
                token_to_sum[w_norm] += float(w_attr)
                token_to_count[w_norm] += 1

        del model
        torch.cuda.empty_cache()

    if not token_to_sum:
        print(f"[WARN] No attributions collected for {topic}-{year}")
        return

    rows = []
    for tok, s in token_to_sum.items():
        c = token_to_count[tok]
        avg_attr = s / float(c)
        rows.append((tok, avg_attr, c))

    df = pd.DataFrame(rows, columns=["token", "avg_attr", "count"])
    df.sort_values("avg_attr", ascending=False, inplace=True)

    os.makedirs(IG_ROOT, exist_ok=True)
    out_path = os.path.join(IG_ROOT, f"{topic}_{year}_token_attributions.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")

    df = df[df["count"] >= MIN_TOKEN_COUNT]
    if df.empty:
        print(f"[WARN] After MIN_TOKEN_COUNT={MIN_TOKEN_COUNT} filter, no tokens left for {topic}-{year}")
        return
    print(f"[DONE] Saved IG token attribution → {out_path}")

    print("\n  Top 10 tokens (CNN-leaning, avg_attr > 0):")
    print(df.head(10))

    print("\n  Top 10 tokens (FOX-leaning, avg_attr < 0):")
    print(df.sort_values("avg_attr", ascending=True).head(10))


def main():
    set_seed(42)
    print("Using device:", DEVICE)
    print("Base model:", BASE_MODEL_NAME)
    print("Topics:", list(TOPIC_KEYWORDS.keys()))
    print("Years:", YEARS)

    for topic in TOPIC_KEYWORDS.keys():
        for year in YEARS:
            try:
                process_topic_year(topic, year)
            except Exception as e:
                print(f"[ERROR] topic={topic}, year={year}: {e}")


if __name__ == "__main__":
    main()
