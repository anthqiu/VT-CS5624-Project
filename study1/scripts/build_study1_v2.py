import argparse
import csv
import gzip
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# Increase CSV field limit to handle long transcript text.
csv.field_size_limit(sys.maxsize)

DATA_DIR = Path("data/raw_new/extracted")
OUTPUT_DIR = Path("data/processed")
TABLE_DIR = Path("outputs/tables")

# Study 1 bounds (inclusive).
YEAR_MIN = 2015
YEAR_MAX = 2025

# Basic timezone mapping to keep CNN datetime construction consistent.
TZ_MAP = {
    "ET": "America/New_York",
    "EST": "America/New_York",
    "EDT": "America/New_York",
    "CT": "America/Chicago",
    "CDT": "America/Chicago",
    "CST": "America/Chicago",
    "PT": "America/Los_Angeles",
    "PDT": "America/Los_Angeles",
    "PST": "America/Los_Angeles",
    "MT": "America/Denver",
    "MDT": "America/Denver",
    "MST": "America/Denver",
    "UTC": "UTC",
    "GMT": "UTC",
}


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    norm = unicodedata.normalize("NFKC", text)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def count_words(text: str) -> int:
    return len(text.split())


def build_datetime_from_parts(row: pd.Series) -> Optional[pd.Timestamp]:
    year = str(row.get("year") or "").strip()
    month = str(row.get("month") or "").strip()
    day = str(row.get("date") or "").strip()
    time_part = str(row.get("time") or "").strip()
    tz = str(row.get("timezone") or "").strip().upper()

    if not year or not month or not day:
        return None

    try:
        date_str = f"{year}-{int(month):02d}-{int(day):02d}"
    except Exception:
        return None

    if time_part:
        date_str = f"{date_str} {time_part}"

    try:
        ts = pd.to_datetime(date_str, errors="coerce")
    except Exception:
        return None

    if ts is pd.NaT:
        return None

    # Localize to timezone if known; otherwise leave naive.
    if tz in TZ_MAP:
        try:
            ts = ts.tz_localize(TZ_MAP[tz], nonexistent="shift_forward", ambiguous="NaT")
        except (TypeError, ValueError):
            pass

    return ts


def _best_existing_path(base: Path, stem: str) -> Optional[Path]:
    gz = base / f"{stem}.csv.gz"
    raw = base / f"{stem}.csv"
    if gz.exists():
        return gz
    if raw.exists():
        return raw
    return None


def load_cnn(cnn_max_rows: int) -> pd.DataFrame:
    cnn_dir = DATA_DIR / "cnn"
    paths = []
    for stem in ("cnn-7", "cnn-8"):
        p = _best_existing_path(cnn_dir, stem)
        if p:
            paths.append(p)

    usecols = {
        "url",
        "channel.name",
        "program.name",
        "year",
        "month",
        "date",
        "time",
        "timezone",
        "text",
        "subhead",
        "wordcount",
        "duration",
        "uid",
        "path",
    }

    dfs = []
    for p in paths:
        if not p.exists():
            print(f"[WARN] Missing CNN file: {p}", file=sys.stderr)
            continue

        header_cols = pd.read_csv(p, nrows=0, compression="infer").columns
        wanted = [c for c in header_cols if c in usecols]

        df = pd.read_csv(
            p,
            dtype=str,
            encoding="utf-8",
            keep_default_na=False,
            usecols=wanted,
            engine="c",
            compression="infer",
        )

        # Sprint sampling per file (0 disables).
        if cnn_max_rows and len(df) > cnn_max_rows:
            df = df.sample(n=cnn_max_rows, random_state=42)

        df["source_file"] = p.name
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)

    # Build publication_date.
    df_all["publication_date"] = df_all.apply(build_datetime_from_parts, axis=1)

    # Normalize text and compute word_count early for dedup logic.
    df_all["text"] = df_all.get("text", "").astype(str).apply(normalize_text)
    df_all["word_count_raw"] = df_all["text"].apply(count_words)

    return df_all


def dedup_cnn(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    def pick(group: pd.DataFrame) -> pd.Series:
        grp = group.copy()
        grp["has_text"] = grp["text"].str.len() > 0
        grp = grp.sort_values(
            by=["has_text", "word_count_raw"],
            ascending=[False, False],
        )
        return grp.iloc[0]

    with_url = df[df["url"].astype(str).str.strip() != ""]
    without_url = df[df["url"].astype(str).str.strip() == ""]

    deduped_with = (
        with_url.groupby("url", dropna=False, group_keys=False)
        .apply(pick)
        .reset_index(drop=True)
    )
    deduped = pd.concat([deduped_with, without_url], ignore_index=True)
    return deduped.reset_index(drop=True)


def read_text_file(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return normalize_text(f.read())
    return normalize_text(path.read_text(encoding="utf-8", errors="replace"))


def load_fox_metadata() -> pd.DataFrame:
    files = [
        (DATA_DIR / "fox" / "foxnews-transcript-urls-2025.csv", 0),  # preferred
        (DATA_DIR / "fox" / "foxnews-transcript.csv", 1),
    ]
    usecols = {
        "title",
        "description",
        "url",
        "publicationDate",
        "lastPublishedDate",
        "category",
        "html_file",
        "authors",
        "duration",
        "imageUrl",
        "isBreaking",
        "isLive",
    }

    records = []
    for path, priority in files:
        if not path.exists():
            print(f"[WARN] Missing Fox metadata file: {path}", file=sys.stderr)
            continue

        header_cols = pd.read_csv(path, nrows=0).columns
        wanted = [c for c in header_cols if c in usecols]

        df = pd.read_csv(
            path,
            dtype=str,
            encoding="utf-8",
            keep_default_na=False,
            usecols=wanted,
            engine="c",
        )
        df["priority"] = priority
        df["source_file"] = path.name
        records.append(df)

    if not records:
        return pd.DataFrame()

    df_all = pd.concat(records, ignore_index=True)
    return df_all


def pick_best_fox_row(group: pd.DataFrame) -> pd.Series:
    """
    Prefer lower priority (urls_2025), then newer date.
    Robust tz fix: vectorized UTC parse + naive conversion,
    avoid Series.where() on mixed timezone blocks.
    """
    group = group.copy()

    if "lastPublishedDate" not in group.columns:
        group["lastPublishedDate"] = ""
    if "publicationDate" not in group.columns:
        group["publicationDate"] = ""

    def to_utc_naive(series: pd.Series) -> pd.Series:
        dt = pd.to_datetime(series, errors="coerce", utc=True)
        return dt.dt.tz_convert("UTC").dt.tz_localize(None)

    parsed_last = to_utc_naive(group["lastPublishedDate"])
    parsed_pub = to_utc_naive(group["publicationDate"])

    group["best_date"] = parsed_last.fillna(parsed_pub)

    group = group.sort_values(
        by=["priority", "best_date"],
        ascending=[True, False],
    )
    return group.iloc[0].copy()


def merge_fox_metadata(df_meta: pd.DataFrame) -> pd.DataFrame:
    if df_meta.empty:
        return df_meta

    deduped = (
        df_meta.groupby("url", dropna=False, group_keys=False)
        .apply(pick_best_fox_row)
        .reset_index(drop=True)
    )

    def choose_pub(row: pd.Series) -> Optional[pd.Timestamp]:
        for field in ("lastPublishedDate", "publicationDate"):
            val = row.get(field, "")
            if val:
                ts = pd.to_datetime(val, errors="coerce", utc=True)
                if ts is not pd.NaT:
                    return ts.tz_convert("UTC").tz_localize(None)
        return None

    deduped["publication_date"] = deduped.apply(choose_pub, axis=1)
    return deduped


def resolve_fox_text(df_meta: pd.DataFrame) -> pd.DataFrame:
    if df_meta.empty:
        return df_meta

    text_dirs = [
        DATA_DIR / "fox" / "fnc_transcripts_text_2025",
        DATA_DIR / "fox" / "foxnews-transcript-text" / "text",
    ]

    texts = []
    word_counts = []
    is_full = []
    source_refs = []
    text_cache: Dict[str, Tuple[str, str]] = {}

    for _, row in df_meta.iterrows():
        html_file = str(row.get("html_file") or "").strip()
        stem = html_file

        if stem.endswith(".html") or stem.endswith(".htm"):
            stem = stem.rsplit(".", 1)[0]
        elif stem.endswith(".html.gz"):
            stem = stem[:-len(".html.gz")]

        if not stem:
            url = str(row.get("url") or "").strip().strip("/")
            if url:
                stem = Path(url).name

        content = ""
        source_ref = ""
        full = False

        if stem:
            if stem in text_cache:
                content, source_ref = text_cache[stem]
                full = bool(content)
            else:
                found_path: Optional[Path] = None
                for d in text_dirs:
                    for ext in (".txt", ".txt.gz"):
                        candidate = d / f"{stem}{ext}"
                        if candidate.exists():
                            found_path = candidate
                            break
                    if found_path:
                        break

                if found_path:
                    content = read_text_file(found_path)
                    source_ref = str(found_path)
                    full = True

                text_cache[stem] = (content, source_ref)

        texts.append(content)
        word_counts.append(count_words(content) if content else 0)
        is_full.append(full)
        source_refs.append(source_ref)

    df_meta = df_meta.copy()
    df_meta["text"] = texts
    df_meta["word_count_raw"] = word_counts
    df_meta["is_full_transcript_text"] = is_full
    df_meta["source_file_ref"] = source_refs
    return df_meta


def parse_category_name(val: str) -> str:
    if not isinstance(val, str):
        return ""
    m = re.search(r"name['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"]", val)
    if m:
        return m.group(1).strip()
    return val.strip() if val else ""


def filter_and_standardize(
    df_cnn: pd.DataFrame, df_fox: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def pick_title(row: pd.Series, alt_fields: Tuple[str, ...]) -> str:
        for f in ("title",) + alt_fields:
            if f in row and isinstance(row[f], str) and row[f].strip():
                return row[f].strip()
        return ""

    def pick_program(row: pd.Series, program_col: Optional[str], channel_col: Optional[str]) -> str:
        vals = []
        if program_col and program_col in row and isinstance(row[program_col], str) and row[program_col].strip():
            vals.append(row[program_col].strip())
        if channel_col and channel_col in row and isinstance(row[channel_col], str) and row[channel_col].strip():
            vals.append(row[channel_col].strip())
        return " | ".join(dict.fromkeys(vals))

    def finalize(
        df: pd.DataFrame,
        network: str,
        program_col: Optional[str] = None,
        channel_col: Optional[str] = None,
        alt_title_fields: Tuple[str, ...] = (),
    ) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        df["network"] = network

        # Normalize publication_date to UTC-naive to avoid tz-mismatch later.
        pub = pd.to_datetime(df["publication_date"], errors="coerce", utc=True)
        pub = pub.dt.tz_convert("UTC").dt.tz_localize(None)
        df["publication_date"] = pub

        df = df.dropna(subset=["publication_date"])
        df["year"] = df["publication_date"].dt.year
        df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)]

        df["text"] = df.get("text", "").astype(str).apply(normalize_text)
        df = df[df["text"].str.len() >= 200]

        df["word_count"] = df["text"].apply(count_words)
        df["title"] = df.apply(lambda r: pick_title(r, alt_title_fields), axis=1)
        df["program_channel"] = df.apply(lambda r: pick_program(r, program_col, channel_col), axis=1)

        if "source_file_ref" not in df.columns:
            df["source_file_ref"] = df.get("source_file", "")

        cols = [
            "network",
            "publication_date",
            "year",
            "title",
            "program_channel",
            "url",
            "text",
            "word_count",
            "source_file_ref",
        ]
        if "is_full_transcript_text" in df.columns:
            cols.append("is_full_transcript_text")

        return df[cols]

    # Enrich Fox program_channel from category if available.
    if not df_fox.empty and "category" in df_fox.columns:
        df_fox = df_fox.copy()
        df_fox["program_parsed"] = df_fox["category"].apply(parse_category_name)

    cnn_final = finalize(
        df_cnn,
        "CNN",
        program_col="program.name",
        channel_col="channel.name",
        alt_title_fields=("subhead",),
    )
    fox_final = finalize(
        df_fox,
        "Fox",
        program_col="program_parsed",
        channel_col=None,
        alt_title_fields=(),
    )
    return cnn_final, fox_final


def dedup_corpus(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.drop_duplicates(subset=["network", "publication_date", "title", "text"])


def save_outputs(cnn_df: pd.DataFrame, fox_df: pd.DataFrame, combined: pd.DataFrame):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cnn_df.to_parquet(OUTPUT_DIR / "cnn_clean.parquet", index=False)
    fox_df.to_parquet(OUTPUT_DIR / "fox_clean.parquet", index=False)
    combined.to_parquet(OUTPUT_DIR / "study1_corpus_2015_2025.parquet", index=False)

    summary = (
        combined.groupby(["year", "network"])
        .size()
        .reset_index(name="count")
        .sort_values(["year", "network"])
    )
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_DIR / "dataset_summary_by_year_network.csv", index=False)
    return summary


def qa_prints(cnn_df: pd.DataFrame, fox_df: pd.DataFrame, combined: pd.DataFrame):
    print("=== QA: Counts ===")
    print(f"CNN clean count: {len(cnn_df)}")
    print(f"Fox clean count: {len(fox_df)}")
    print(f"Combined corpus count: {len(combined)}")
    print()

    print("=== QA: Year coverage 2015-2025 ===")
    for year in range(YEAR_MIN, YEAR_MAX + 1):
        cnn_count = len(cnn_df[cnn_df["year"] == year]) if not cnn_df.empty else 0
        fox_count = len(fox_df[fox_df["year"] == year]) if not fox_df.empty else 0
        print(f"{year}: CNN={cnn_count}, Fox={fox_count}")
    print()

    overlap_years = sorted(set(cnn_df["year"]).intersection(set(fox_df["year"])))
    print("Years with both CNN and Fox:", overlap_years)
    print()

    def sample_preview(df: pd.DataFrame, network: str):
        print(f"--- Sample rows for {network} ---")
        if df.empty:
            print("No rows\n")
            return
        sample = df.sample(n=min(2, len(df)), random_state=42)
        for _, row in sample.iterrows():
            text_preview = row["text"][:200]
            print(
                f"{row['publication_date']}, title={row.get('title','')}, "
                f"program_channel={row.get('program_channel','')}, "
                f"word_count={row.get('word_count',0)}, text_preview={text_preview}"
            )
        print()

    sample_preview(cnn_df, "CNN")
    sample_preview(fox_df, "Fox")


def main():
    parser = argparse.ArgumentParser(description="Build Study 1 corpus (sprint-friendly, TXT-only Fox).")
    parser.add_argument(
        "--cnn-max-rows",
        type=int,
        default=200_000,
        help="Sprint cap per CNN file. Set to 0 to disable sampling.",
    )
    args = parser.parse_args()

    ensure_dirs()

    print("Loading CNN data...")
    cnn_raw = load_cnn(cnn_max_rows=args.cnn_max_rows)
    cnn_dedup = dedup_cnn(cnn_raw)

    print("Loading Fox metadata...")
    fox_meta = load_fox_metadata()
    fox_meta_dedup = merge_fox_metadata(fox_meta)

    print("Resolving Fox transcript text (TXT only)...")
    fox_with_text = resolve_fox_text(fox_meta_dedup)

    print("Standardizing and filtering...")
    cnn_clean, fox_clean = filter_and_standardize(cnn_dedup, fox_with_text)

    combined = pd.concat([cnn_clean, fox_clean], ignore_index=True, sort=False)
    combined = dedup_corpus(combined)

    print("Saving outputs...")
    save_outputs(cnn_clean, fox_clean, combined)

    print("Running QA checks...")
    qa_prints(cnn_clean, fox_clean, combined)


if __name__ == "__main__":
    main()