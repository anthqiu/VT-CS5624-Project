import pandas as pd
import os

from utils.data_prep import load_cnn_fox
from utils.sentence_extraction import split_to_sentences, filter_by_topic
from utils.config import TOPIC_KEYWORDS

if __name__ == "__main__":
    cnn_df, fox_df = load_cnn_fox()

    full_df = pd.concat([cnn_df, fox_df], ignore_index=True)
    del cnn_df
    del fox_df

    SENTENCES_OUTPUT = "checkpoints/sentences/"

    if not os.path.exists(SENTENCES_OUTPUT):
        os.makedirs(SENTENCES_OUTPUT, exist_ok=True)

    results = split_to_sentences(full_df, SENTENCES_OUTPUT)

    for cid, path, n_sent in results:
        print(f"chunk {cid}: {path}, sentences={n_sent}")
