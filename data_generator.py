"""
data_generator.py
-----------------
Generates synthetic Snowflake-style query history data for analysis and prototyping.
Simulates realistic distributions of query runtimes, compute credits, data scanned,
users, warehouses, and query types — closely mirroring real Snowflake QUERY_HISTORY views.
"""

import pandas as pd
import numpy as np
import random
import uuid
from datetime import datetime, timedelta


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

WAREHOUSES = [
    ("ANALYTICS_XS", "X-Small", 1),
    ("ANALYTICS_S",  "Small",   2),
    ("ANALYTICS_M",  "Medium",  4),
    ("ETL_L",        "Large",   8),
    ("ML_XL",        "X-Large", 16),
]

USERS = [
    "alice.chen", "bob.kumar", "carol.smith", "dan.patel",
    "eve.johnson", "frank.li", "grace.kim", "henry.müller",
    "irene.silva", "james.okafor"
]

TEAMS = {
    "alice.chen":    "Data Engineering",
    "bob.kumar":     "Analytics",
    "carol.smith":   "Data Science",
    "dan.patel":     "Analytics",
    "eve.johnson":   "Data Engineering",
    "frank.li":      "Data Science",
    "grace.kim":     "BI & Reporting",
    "henry.müller":  "BI & Reporting",
    "irene.silva":   "Analytics",
    "james.okafor":  "Data Engineering",
}

QUERY_TEMPLATES = [
    "SELECT * FROM orders WHERE created_at > '{date}' AND status = 'COMPLETE'",
    "SELECT user_id, COUNT(*) as cnt FROM events GROUP BY user_id ORDER BY cnt DESC",
    "SELECT a.*, b.revenue FROM customers a LEFT JOIN transactions b ON a.id = b.customer_id",
    "SELECT DATE_TRUNC('month', created_at) as month, SUM(amount) FROM payments GROUP BY 1",
    "SELECT * FROM raw_logs WHERE log_date = '{date}'",
    "SELECT product_id, SUM(quantity) FROM order_items GROUP BY product_id HAVING SUM(quantity) > 100",
    "WITH base AS (SELECT * FROM sessions) SELECT user_id, COUNT(*) FROM base GROUP BY 1",
    "SELECT DISTINCT customer_id FROM orders o JOIN returns r ON o.id = r.order_id",
    "UPDATE staging.temp_table SET processed = TRUE WHERE id IN (SELECT id FROM queue LIMIT 5000)",
    "INSERT INTO mart.daily_summary SELECT DATE(created_at), COUNT(*) FROM events GROUP BY 1",
    "SELECT * FROM large_fact_table",                          # intentionally inefficient
    "SELECT * FROM another_huge_table WHERE 1=1",             # intentionally inefficient
    "SELECT a.*, b.*, c.* FROM t1 a, t2 b, t3 c",            # cartesian-style
    "SELECT COUNT(*) FROM logs",
    "SELECT * FROM dim_product WHERE category IS NOT NULL",
]

QUERY_TYPES = ["SELECT", "SELECT", "SELECT", "SELECT", "SELECT",
               "INSERT", "UPDATE", "SELECT", "SELECT", "SELECT",
               "SELECT", "SELECT", "SELECT", "SELECT", "SELECT"]

STATUS_CHOICES = ["SUCCESS"] * 18 + ["FAILED"] * 1 + ["QUEUED"] * 1


def _credits_for_warehouse(warehouse_name: str, runtime_seconds: float) -> float:
    """Estimate Snowflake credits: 1 credit = 1 compute-hour for X-Small."""
    multiplier_map = {
        "ANALYTICS_XS": 1,
        "ANALYTICS_S":  2,
        "ANALYTICS_M":  4,
        "ETL_L":        8,
        "ML_XL":       16,
    }
    multiplier = multiplier_map.get(warehouse_name, 1)
    return round((runtime_seconds / 3600) * multiplier, 6)


def generate_query_history(n_records: int = 5000,
                           start_date: datetime = None,
                           end_date: datetime = None) -> pd.DataFrame:
    """
    Generate a synthetic Snowflake QUERY_HISTORY-style DataFrame.

    Parameters
    ----------
    n_records   : number of query records to generate
    start_date  : earliest query start time (default: 90 days ago)
    end_date    : latest query start time   (default: now)

    Returns
    -------
    pd.DataFrame with columns matching Snowflake QUERY_HISTORY schema
    """
    if start_date is None:
        start_date = datetime.utcnow() - timedelta(days=90)
    if end_date is None:
        end_date = datetime.utcnow()

    total_seconds = (end_date - start_date).total_seconds()

    records = []
    for _ in range(n_records):
        # ── timing ──────────────────────────────────────────────
        offset_s      = random.uniform(0, total_seconds)
        query_start   = start_date + timedelta(seconds=offset_s)

        # Bimodal runtime: most queries fast, a few are expensive
        if random.random() < 0.15:          # 15% are long-running / expensive
            runtime_s = np.random.lognormal(mean=4.5, sigma=1.2)   # ~90s–3000s
        else:
            runtime_s = np.random.lognormal(mean=1.5, sigma=1.0)   # ~1s–60s

        runtime_s   = max(0.1, round(runtime_s, 3))
        runtime_ms  = int(runtime_s * 1000)
        query_end   = query_start + timedelta(seconds=runtime_s)

        # ── warehouse & user ─────────────────────────────────────
        wh_name, wh_size, _ = random.choice(WAREHOUSES)
        user                 = random.choice(USERS)
        team                 = TEAMS[user]

        # ── bytes scanned ────────────────────────────────────────
        # Larger warehouses tend to scan more data
        wh_weight     = {"X-Small": 1, "Small": 2, "Medium": 4,
                         "Large": 8, "X-Large": 16}[wh_size]
        bytes_scanned = abs(np.random.lognormal(
            mean=20 + wh_weight * 0.3, sigma=1.5))
        bytes_scanned = int(bytes_scanned)

        # ── rows & partitions ─────────────────────────────────────
        rows_produced      = int(abs(np.random.lognormal(mean=8, sigma=2)))
        partitions_scanned = int(abs(np.random.lognormal(mean=3, sigma=1.5))) + 1
        partitions_total   = partitions_scanned + random.randint(0, 20)

        # ── credits ───────────────────────────────────────────────
        credits_used = _credits_for_warehouse(wh_name, runtime_s)

        # ── query text ────────────────────────────────────────────
        tmpl_idx  = random.randint(0, len(QUERY_TEMPLATES) - 1)
        query_txt = QUERY_TEMPLATES[tmpl_idx].format(
            date=query_start.strftime("%Y-%m-%d"))
        query_type = QUERY_TYPES[tmpl_idx]

        # ── status ────────────────────────────────────────────────
        status = random.choice(STATUS_CHOICES)
        if status == "FAILED":
            runtime_ms    = int(runtime_ms * 0.3)
            credits_used  = credits_used * 0.3
            bytes_scanned = 0

        records.append({
            "query_id":             str(uuid.uuid4()),
            "query_text":           query_txt,
            "query_type":           query_type,
            "user_name":            user,
            "team":                 team,
            "warehouse_name":       wh_name,
            "warehouse_size":       wh_size,
            "execution_time_ms":    runtime_ms,
            "execution_time_s":     round(runtime_ms / 1000, 3),
            "bytes_scanned":        bytes_scanned,
            "gb_scanned":           round(bytes_scanned / 1e9, 6),
            "rows_produced":        rows_produced,
            "partitions_scanned":   partitions_scanned,
            "partitions_total":     partitions_total,
            "credits_used":         round(credits_used, 6),
            "start_time":           query_start,
            "end_time":             query_end,
            "status":               status,
        })

    df = pd.DataFrame(records).sort_values("start_time").reset_index(drop=True)
    return df


def save_sample_data(path: str = "data/query_history.csv",
                     n_records: int = 5000) -> pd.DataFrame:
    """Generate and save sample data to CSV."""
    df = generate_query_history(n_records=n_records)
    df.to_csv(path, index=False)
    print(f"[data_generator] Saved {len(df)} records → {path}")
    return df


if __name__ == "__main__":
    save_sample_data()
