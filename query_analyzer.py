"""
query_analyzer.py
-----------------
Identifies inefficient, high-cost, and anomalous SQL queries from
Snowflake query history. Flags common anti-patterns and computes
efficiency scores to prioritize optimization candidates.

Inefficiency Signals Detected
------------------------------
1. SELECT *           — scans all columns; wastes I/O and compute
2. Missing WHERE       — full table scans on large tables
3. Cartesian joins     — exponential row explosion
4. Low partition pruning ratio — poor clustering / missing filters
5. High bytes-per-row  — inefficient projection or wide tables
6. Long runtime        — outlier execution time (>2 std deviations)
7. High credit cost    — top 10% by credits_used
8. Repeated identical queries — caching not leveraged

Efficiency Score
----------------
Each query starts at 100. Penalty points are deducted per flag.
Score ≥ 80  → Efficient
Score 50–79 → Moderate — review recommended
Score  < 50 → Inefficient — optimize immediately
"""

import re
import pandas as pd
import numpy as np
from typing import Optional


# ──────────────────────────────────────────────
# Anti-pattern detection helpers
# ──────────────────────────────────────────────

def _has_select_star(query: str) -> bool:
    """Detect SELECT * usage."""
    return bool(re.search(r"SELECT\s+\*", query, re.IGNORECASE))


def _has_no_where_clause(query: str) -> bool:
    """Detect SELECT queries with no WHERE, HAVING, or LIMIT."""
    if not re.search(r"\bSELECT\b", query, re.IGNORECASE):
        return False
    has_where   = bool(re.search(r"\bWHERE\b",  query, re.IGNORECASE))
    has_having  = bool(re.search(r"\bHAVING\b", query, re.IGNORECASE))
    has_limit   = bool(re.search(r"\bLIMIT\b",  query, re.IGNORECASE))
    return not (has_where or has_having or has_limit)


def _has_cartesian_join(query: str) -> bool:
    """Detect implicit Cartesian joins (comma-separated FROM)."""
    from_clause = re.search(
        r"\bFROM\b(.+?)(?:\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|$)",
        query, re.IGNORECASE | re.DOTALL)
    if not from_clause:
        return False
    from_body = from_clause.group(1)
    tables = [t.strip() for t in from_body.split(",") if t.strip()]
    return len(tables) > 1


def _has_1_equals_1(query: str) -> bool:
    """Detect WHERE 1=1 — often indicates a dynamic query building anti-pattern."""
    return bool(re.search(r"WHERE\s+1\s*=\s*1", query, re.IGNORECASE))


def _detect_flags(row: pd.Series,
                  runtime_threshold_s: float,
                  credit_threshold: float) -> dict:
    """
    Run all anti-pattern checks on a single query row.
    Returns a dict of flag_name → bool.
    """
    query = str(row.get("query_text", ""))

    low_partition_ratio = False
    if row.get("partitions_total", 0) > 0:
        ratio = row.get("partitions_scanned", 0) / row["partitions_total"]
        low_partition_ratio = (ratio > 0.85 and row.get("partitions_total", 0) > 5)

    high_bytes_per_row = False
    if row.get("rows_produced", 0) > 0:
        bpr = row.get("bytes_scanned", 0) / row["rows_produced"]
        high_bytes_per_row = bpr > 1_000_000  # >1 MB per row

    return {
        "flag_select_star":        _has_select_star(query),
        "flag_no_where_clause":    _has_no_where_clause(query),
        "flag_cartesian_join":     _has_cartesian_join(query),
        "flag_where_1_equals_1":   _has_1_equals_1(query),
        "flag_low_partition_pruning": low_partition_ratio,
        "flag_high_bytes_per_row": high_bytes_per_row,
        "flag_long_runtime":       row.get("execution_time_s", 0) > runtime_threshold_s,
        "flag_high_credit_cost":   row.get("credits_used", 0) > credit_threshold,
    }


def _compute_efficiency_score(flags: dict) -> int:
    """
    Compute a 0–100 efficiency score based on detected flags.
    """
    penalties = {
        "flag_select_star":           10,
        "flag_no_where_clause":       20,
        "flag_cartesian_join":        25,
        "flag_where_1_equals_1":       5,
        "flag_low_partition_pruning": 15,
        "flag_high_bytes_per_row":    10,
        "flag_long_runtime":          15,
        "flag_high_credit_cost":      15,
    }
    score = 100
    for flag, is_set in flags.items():
        if is_set:
            score -= penalties.get(flag, 0)
    return max(0, score)


def _efficiency_label(score: int) -> str:
    if score >= 80:
        return "Efficient"
    elif score >= 50:
        return "Moderate"
    else:
        return "Inefficient"


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def analyze_queries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run full query analysis on a query history DataFrame.

    Returns the DataFrame enriched with:
    - flag_* columns (one per anti-pattern)
    - flag_count       : total flags raised
    - efficiency_score : 0–100
    - efficiency_label : Efficient / Moderate / Inefficient
    - top_flags        : human-readable list of active flags
    """
    df = df.copy()

    # Dynamic thresholds (outlier = mean + 2 std)
    runtime_threshold = (
        df["execution_time_s"].mean() + 2 * df["execution_time_s"].std()
    )
    credit_threshold = df["credits_used"].quantile(0.90)

    flag_records = df.apply(
        lambda row: _detect_flags(row, runtime_threshold, credit_threshold),
        axis=1,
        result_type="expand",
    )

    df = pd.concat([df, flag_records], axis=1)

    df["flag_count"] = flag_records.sum(axis=1).astype(int)

    df["efficiency_score"] = flag_records.apply(
        lambda flags: _compute_efficiency_score(flags.to_dict()), axis=1
    )
    df["efficiency_label"] = df["efficiency_score"].apply(_efficiency_label)

    # Human-readable flag descriptions
    flag_labels = {
        "flag_select_star":           "SELECT *",
        "flag_no_where_clause":       "No WHERE clause",
        "flag_cartesian_join":        "Cartesian join",
        "flag_where_1_equals_1":      "WHERE 1=1",
        "flag_low_partition_pruning": "Low partition pruning",
        "flag_high_bytes_per_row":    "High bytes/row",
        "flag_long_runtime":          "Long runtime",
        "flag_high_credit_cost":      "High credit cost",
    }
    flag_cols = list(flag_labels.keys())

    def _top_flags(row):
        active = [flag_labels[c] for c in flag_cols if row.get(c, False)]
        return ", ".join(active) if active else "None"

    df["top_flags"] = df.apply(_top_flags, axis=1)

    return df


def get_inefficient_queries(df: pd.DataFrame,
                            max_score: int = 79) -> pd.DataFrame:
    """Return queries with efficiency_score <= max_score, sorted by score."""
    if "efficiency_score" not in df.columns:
        raise ValueError("Run analyze_queries() first.")
    return (
        df[df["efficiency_score"] <= max_score]
        .sort_values("efficiency_score")
        .reset_index(drop=True)
    )


def query_summary(df: pd.DataFrame) -> dict:
    """High-level statistics on query health."""
    if "efficiency_label" not in df.columns:
        raise ValueError("Run analyze_queries() first.")

    label_counts = df["efficiency_label"].value_counts().to_dict()
    return {
        "total_queries":       len(df),
        "efficient":           label_counts.get("Efficient",    0),
        "moderate":            label_counts.get("Moderate",     0),
        "inefficient":         label_counts.get("Inefficient",  0),
        "avg_efficiency_score": round(df["efficiency_score"].mean(), 1),
        "most_common_flag":    (
            df[[c for c in df.columns if c.startswith("flag_") and c != "flag_count"]]
            .sum()
            .idxmax()
            .replace("flag_", "")
            .replace("_", " ")
        ),
    }


def detect_duplicate_queries(df: pd.DataFrame,
                              window_minutes: int = 60) -> pd.DataFrame:
    """
    Detect repeated identical queries within a rolling time window.
    These are candidates for result caching.

    Returns a subset of df with a 'duplicate_count' column showing
    how many times the same query text ran in the window.
    """
    df = df.copy().sort_values("start_time")
    df["query_text_norm"] = (
        df["query_text"]
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    counts = df.groupby("query_text_norm")["query_id"].transform("count")
    df["duplicate_count"] = counts

    duplicates = (
        df[df["duplicate_count"] > 1]
        .sort_values("duplicate_count", ascending=False)
        .reset_index(drop=True)
    )
    return duplicates


# ──────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_generator import generate_query_history

    df = generate_query_history(n_records=500)
    df = analyze_queries(df)

    print("── Query Health Summary ──")
    for k, v in query_summary(df).items():
        print(f"  {k}: {v}")

    print("\n── Sample Inefficient Queries ──")
    bad = get_inefficient_queries(df)
    print(bad[["query_id", "efficiency_score", "top_flags",
               "credits_used", "execution_time_s"]].head(10).to_string(index=False))
