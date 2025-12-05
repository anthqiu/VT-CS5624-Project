from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

class TopicSentenceDataset(Dataset):
    def __init__(self, df, model_name="bert-base-uncased", max_length=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        # label: cnn -> 1, fox -> 0
        self.labels = (self.df["network"].str.lower() == "cnn").astype(int).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["sentence"]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
