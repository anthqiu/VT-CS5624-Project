import argparse
from pathlib import Path

import pandas as pd


PROCESSED = Path("data/processed")
TABLES = Path("outputs/tables")
TABLES.mkdir(parents=True, exist_ok=True)

YEAR_MIN = 2015
YEAR_MAX = 2025


def load_corpus(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # Strict safety checks
    required = {"network", "publication_date", "year", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Corpus missing required columns: {missing}")

    df = df.dropna(subset=["network", "year", "text"])
    df["network"] = df["network"].astype(str)
    df["text"] = df["text"].astype(str)
    df["year"] = df["year"].astype(int)

    # Enforce Study 1 year bounds
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)]

    # If word_count already exists from Step2, trust it; otherwise compute
    if "word_count" not in df.columns:
        df["word_count"] = df["text"].str.split().apply(len)

    return df


def save_counts_by_year_network(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["year", "network"])
        .size()
        .reset_index(name="count")
        .sort_values(["year", "network"])
    )
    counts.to_csv(TABLES / "study1_counts_by_year_network.csv", index=False)
    return counts


def save_wordcount_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby(["year", "network"])["word_count"]
        .agg(["mean", "median", "std", "min", "max", "count"])
        .reset_index()
        .sort_values(["year", "network"])
    )
    stats.to_csv(TABLES / "study1_wordcount_stats_by_year_network.csv", index=False)
    return stats


def save_overlap_years(df: pd.DataFrame):
    years_by_net = (
        df.groupby("network")["year"]
        .apply(lambda s: sorted(set(s.tolist())))
        .to_dict()
    )

    cnn_years = set(years_by_net.get("CNN", []))
    fox_years = set(years_by_net.get("Fox", []))
    overlap = sorted(cnn_years.intersection(fox_years))

    out = pd.DataFrame(
        {
            "network": list(years_by_net.keys()) + ["OVERLAP_CNN_FOX"],
            "years": [",".join(map(str, v)) for v in years_by_net.values()] + [",".join(map(str, overlap))],
        }
    )
    out.to_csv(TABLES / "study1_year_coverage_and_overlap.csv", index=False)
    return out


def save_top_titles_preview(df: pd.DataFrame, n_per_year_net: int = 3):
    # Poster-friendly qualitative anchor, still strict (no model changes)
    rows = []
    for (year, net), sub in df.groupby(["year", "network"]):
        sample = sub.sample(n=min(n_per_year_net, len(sub)), random_state=42)
        for _, r in sample.iterrows():
            rows.append(
                {
                    "year": year,
                    "network": net,
                    "publication_date": str(r.get("publication_date", "")),
                    "title": str(r.get("title", "")),
                    "program_channel": str(r.get("program_channel", "")),
                    "word_count": int(r.get("word_count", 0)),
                }
            )

    out = pd.DataFrame(rows).sort_values(["year", "network"])
    out.to_csv(TABLES / "study1_sample_titles_by_year_network.csv", index=False)
    return out


def main():
    parser = argparse.ArgumentParser(description="Study 1 Step3 strict analysis outputs.")
    parser.add_argument(
        "--corpus",
        type=str,
        default=str(PROCESSED / "study1_corpus_2015_2025.parquet"),
        help="Path to Study 1 combined corpus parquet.",
    )
    parser.add_argument(
        "--n-sample-titles",
        type=int,
        default=3,
        help="How many sample titles per (year, network) for qualitative table.",
    )
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus file: {corpus_path}")

    print(f"Loading corpus: {corpus_path}")
    df = load_corpus(corpus_path)

    print("Saving counts by year/network...")
    save_counts_by_year_network(df)

    print("Saving wordcount stats by year/network...")
    save_wordcount_stats(df)

    print("Saving year coverage + overlap...")
    save_overlap_years(df)

    print("Saving qualitative sample title table...")
    save_top_titles_preview(df, n_per_year_net=args.n_sample_titles)

    print("\n=== Step3 outputs generated ===")
    print("outputs/tables/study1_counts_by_year_network.csv")
    print("outputs/tables/study1_wordcount_stats_by_year_network.csv")
    print("outputs/tables/study1_year_coverage_and_overlap.csv")
    print("outputs/tables/study1_sample_titles_by_year_network.csv")


if __name__ == "__main__":
    main()