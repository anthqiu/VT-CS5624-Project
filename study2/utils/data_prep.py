import pandas as pd

def load_cnn_fox(cnn_path = "../data/cnn.csv", fox_path = "../data/fox.csv"):
    def _load(path, network_name):
        df = pd.read_csv(path)
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df = df.dropna(subset=["ts", "text"])
        df["year"] = df["ts"].dt.year
        df["network"] = network_name
        return df

    cnn = _load(cnn_path, "cnn")
    fox = _load(fox_path, "fox")

    return cnn, fox