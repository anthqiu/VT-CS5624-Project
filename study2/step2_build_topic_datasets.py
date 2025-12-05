import os
import pandas as pd
import gc

from utils.sentence_extraction import filter_by_topic
from utils.config import *

if __name__=="__main__":
    chunks = []
    for root, dirs, files in os.walk("./checkpoints/sentences/"):
        for file in files:
            print(f"file {os.path.join(root, file)}")
            chunks.append(pd.read_csv(os.path.join(root, file)))
    sentences = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    OUTPUT_DIR = "./checkpoints/topics/"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for year in range(2015,2025):
        print(f"year {year}")
        dataset=filter_by_topic(sentences, TOPIC_KEYWORDS, year)
        for key, df in dataset.items():
            df.to_csv(os.path.join(OUTPUT_DIR, f"topic_{key}_{year}.csv"), index=False)
        del dataset
        gc.collect()

    del sentences

    for topic in TOPIC_KEYWORDS:
        years = []
        for year in range(2015,2025):
            df = pd.read_csv(os.path.join(OUTPUT_DIR, f"topic_{topic}_{year}.csv"))
            years.append(df)
        topic_df: pd.DataFrame = pd.concat(years, ignore_index=True)
        topic_df.to_csv(os.path.join(OUTPUT_DIR, f"topic_{topic}.csv"), index=False)
