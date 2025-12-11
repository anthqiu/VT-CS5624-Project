from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


TABLES = Path("outputs/tables")
FIGS = Path("outputs/figs")
FIGS.mkdir(parents=True, exist_ok=True)


def _find_first(patterns):
    """Return first existing file that matches any glob pattern."""
    for pat in patterns:
        matches = sorted(TABLES.glob(pat))
        if matches:
            return matches[0]
    return None


def plot_counts_by_year_network():
    """
    Accepts either:
      - study1_counts_by_year_network.csv
      - dataset_summary_by_year_network.csv
    with columns: year, network, count
    """
    path = _find_first([
        "study1_counts_by_year_network*.csv",
        "dataset_summary_by_year_network*.csv",
    ])

    if not path:
        print("[WARN] counts file not found.")
        return

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "count" not in df.columns:
        # attempt to infer
        for c in df.columns:
            if c.lower() in ("n", "size", "counts", "num", "number"):
                df = df.rename(columns={c: "count"})
                break

    if not {"year", "network", "count"}.issubset(df.columns):
        # last resort: rebuild
        if {"year", "network"}.issubset(df.columns):
            df = df.groupby(["year", "network"]).size().reset_index(name="count")
        else:
            print(f"[WARN] Unexpected columns in {path.name}: {df.columns.tolist()}")
            return

    pivot = df.pivot_table(index="year", columns="network", values="count", aggfunc="sum").fillna(0).sort_index()

    plt.figure()
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], marker="o", linewidth=1.5, label=str(col))

    plt.title("Study 1 Corpus Size by Year and Network")
    plt.xlabel("Year")
    plt.ylabel("Document Count")
    plt.legend()
    plt.tight_layout()

    out = FIGS / "study1_fig1_counts_by_year_network.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[OK] Saved {out}")


def plot_distinctive_terms_overall(top_k=20):
    """
    Accepts:
      - study1_distinctive_terms_overall.csv
    columns: top_cnn_term, cnn_z, top_fox_term, fox_z

    If not found, we warn with a helpful hint.
    """
    path = _find_first(["study1_distinctive_terms_overall*.csv"])
    if not path:
        print("[WARN] distinctive terms overall file not found.")
        print("       If you haven't run Step4 lexical script, run:")
        print("       python -u scripts/study1_core.py  (or your Step4 core script)")
        return

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    needed = {"top_cnn_term", "cnn_z", "top_fox_term", "fox_z"}
    if not needed.issubset(df.columns):
        print(f"[WARN] Unexpected columns in {path.name}: {df.columns.tolist()}")
        return

    sub = df.head(top_k).copy()

    # CNN favored
    plt.figure()
    terms = list(sub["top_cnn_term"])[::-1]
    scores = list(sub["cnn_z"])[::-1]
    plt.barh(terms, scores)
    plt.title("Distinctive Terms (Overall) - CNN Favored")
    plt.xlabel("Log-odds Z (higher = more CNN-distinctive)")
    plt.tight_layout()

    out1 = FIGS / "study1_fig2a_distinctive_terms_cnn.png"
    plt.savefig(out1, dpi=300)
    plt.close()
    print(f"[OK] Saved {out1}")

    # Fox favored
    plt.figure()
    terms = list(sub["top_fox_term"])[::-1]
    scores = list(sub["fox_z"])[::-1]
    plt.barh(terms, scores)
    plt.title("Distinctive Terms (Overall) - Fox Favored")
    plt.xlabel("Log-odds Z (lower = more Fox-distinctive)")
    plt.tight_layout()

    out2 = FIGS / "study1_fig2b_distinctive_terms_fox.png"
    plt.savefig(out2, dpi=300)
    plt.close()
    print(f"[OK] Saved {out2}")


def plot_ssp_time_series():
    """
    Accepts:
      - study1_ssp_timeseries.csv

    We支持两种常见结构：

    1) 长表：
        year, network, ssp
        (可选 topic)
    2) 宽表：
        year, ... , cnn_ssp_like_col, fox_ssp_like_col
        例如:
          year, topic, ssp_cnn, ssp_fox
          year, topic, cnn, fox
          year, topic, CNN_mean, Fox_mean
    """
    path = _find_first(["study1_ssp_timeseries*.csv"])
    if not path:
        print("[WARN] SSP time series file not found.")
        return

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Ensure year
    if "year" not in df.columns:
        # try date-like column
        date_col = None
        for c in df.columns:
            if c.lower() in ("date", "publication_date", "datetime", "time"):
                date_col = c
                break
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            df["year"] = df[date_col].dt.year
        else:
            print(f"[WARN] No year/date column found in {path.name}.")
            return

    # ------- Case A: long format -------
    if "network" in df.columns:
        # find ssp col
        ssp_col = None
        for c in df.columns:
            if c.lower() in ("ssp", "ssp_score", "ssp_value"):
                ssp_col = c
                break
        if not ssp_col:
            # maybe named like mean_ssp
            for c in df.columns:
                if "ssp" in c.lower():
                    ssp_col = c
                    break

        if not ssp_col:
            print(f"[WARN] Could not find SSP column in long-format file {path.name}.")
            return

        # aggregate over topics if present
        agg = df.groupby(["year", "network"])[ssp_col].mean().reset_index(name="ssp")
        pivot = agg.pivot_table(index="year", columns="network", values="ssp", aggfunc="mean").sort_index()

    # ------- Case B: wide format -------
    else:
        # Try to detect CNN/Fox SSP-like columns
        cols_lower = {c.lower(): c for c in df.columns}

        # Candidate patterns
        cnn_candidates = []
        fox_candidates = []

        for c in df.columns:
            cl = c.lower()
            if "cnn" in cl:
                cnn_candidates.append(c)
            if "fox" in cl:
                fox_candidates.append(c)

        # If still empty, try exact "CNN"/"Fox"
        if not cnn_candidates and "cnn" in cols_lower:
            cnn_candidates = [cols_lower["cnn"]]
        if not fox_candidates and "fox" in cols_lower:
            fox_candidates = [cols_lower["fox"]]

        # Filter candidates to numeric-feasible cols
        def _pick_numeric_best(cands):
            out = []
            for c in cands:
                # ignore obvious non-metric fields
                if c.lower() in ("cnn-7", "cnn-8"):
                    continue
                out.append(c)
            return out

        cnn_candidates = _pick_numeric_best(cnn_candidates)
        fox_candidates = _pick_numeric_best(fox_candidates)

        if not cnn_candidates or not fox_candidates:
            print("[WARN] SSP wide-format detected but could not identify CNN/Fox SSP columns.")
            print(f"       Columns found: {df.columns.tolist()}")
            return

        # Choose first candidate by default
        cnn_col = cnn_candidates[0]
        fox_col = fox_candidates[0]

        # Coerce numeric
        df[cnn_col] = pd.to_numeric(df[cnn_col], errors="coerce")
        df[fox_col] = pd.to_numeric(df[fox_col], errors="coerce")

        # Aggregate over other fields (e.g., topic) by year
        agg = df.groupby("year")[[cnn_col, fox_col]].mean().reset_index()
        agg = agg.rename(columns={cnn_col: "CNN", fox_col: "Fox"})

        pivot = agg.set_index("year")[["CNN", "Fox"]].sort_index()

    # -------- Plot --------
    plt.figure()
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], marker="o", linewidth=1.5, label=str(col))

    plt.title("SSP Time Series by Year (Study 1)")
    plt.xlabel("Year")
    plt.ylabel("SSP (mean)")
    plt.legend()
    plt.tight_layout()

    out = FIGS / "study1_fig3_ssp_time_series.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[OK] Saved {out}")


def main():
    print("Plotting Study 1 poster figures...")
    plot_counts_by_year_network()
    plot_distinctive_terms_overall(top_k=20)
    plot_ssp_time_series()
    print("\nDone. Check outputs/figs/ for PNG files.")


if __name__ == "__main__":
    main()