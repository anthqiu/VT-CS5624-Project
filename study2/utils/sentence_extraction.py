from typing import List, Dict
import os
import re
import csv
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text))
    return text.strip()


def _process_chunk_to_file(args):
    """
    子进程处理一个 chunk：
    - 对每一行 transcript 切句
    - 写出到独立的 CSV 文件
    - 返回 (chunk_id, out_path, n_sentences)
    """
    chunk_id, chunk_df, output_dir = args

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"sentences_chunk_{chunk_id}.csv")

    n_sentences = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ts", "year", "network", "sentence"])

        for _, row in chunk_df.iterrows():
            text = normalize_text(row["text"])
            if not text:
                continue

            try:
                sentences = sent_tokenize(text)
            except Exception:
                # ⚠ 这里之前是 \\s+，会多一个反斜杠
                sentences = re.split(r"(?<=[.!?])\s+", text)

            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 5:
                    continue

                writer.writerow([row["ts"], row["year"], row["network"], sent])
                n_sentences += 1

    return chunk_id, out_path, n_sentences


def split_to_sentences(
    df: pd.DataFrame,
    output_dir: str,
    n_procs: int | None = None,
    min_chunk_size: int = 1000
):
    """
    多进程切句并写盘：
    - 不返回一个大 DataFrame，避免内存爆
    - 把句子按 chunk 写到多个 CSV 中
    - 返回 [(chunk_id, out_path, n_sentences), ...]
    """
    if n_procs is None:
        n_procs = max(cpu_count() - 1, 1)

    n_rows = len(df)
    if n_rows == 0:
        print("[split_to_sentences] empty DataFrame.")
        return []

    max_procs_by_data = max(1, n_rows // min_chunk_size)
    n_procs = min(n_procs, max_procs_by_data)

    print(f"[split_to_sentences] rows={n_rows}, n_procs={n_procs}")

    # 只保留必要列，减少进程间传输
    df = df[["ts", "year", "network", "text"]]

    chunks = np.array_split(df, n_procs)
    args_list = [(i, chunk, output_dir) for i, chunk in enumerate(chunks)]

    results = []
    with Pool(processes=n_procs) as pool:
        for res in tqdm(
            pool.imap_unordered(_process_chunk_to_file, args_list),
            total=len(args_list),
            desc="Splitting transcripts into sentences",
        ):
            results.append(res)

    # results: list[(chunk_id, out_path, n_sentences)]
    results.sort(key=lambda x: x[0])
    total_sent = sum(r[2] for r in results)
    print(f"[split_to_sentences] done. total sentences={total_sent}")

    return results


def filter_by_topic(df_sent: pd.DataFrame,
                    topic_keywords: Dict[str, List[str]],
                    year_filter: int = 2020):
    df_sent = df_sent[df_sent["year"] == year_filter]

    topic_dfs = {}
    for topic, kws in topic_keywords.items():
        pattern = r"|".join(re.escape(kw.lower()) for kw in kws)
        mask = df_sent["sentence"].str.lower().str.contains(pattern)
        topic_df = df_sent[mask].copy()
        topic_dfs[topic] = topic_df
        print(f"[{topic}] samples: {len(topic_df)}")
    return topic_dfs
