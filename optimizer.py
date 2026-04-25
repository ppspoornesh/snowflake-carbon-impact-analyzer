"""
optimizer.py
------------
Generates AI-driven optimization recommendations for inefficient
Snowflake SQL queries using rule-based heuristics and optional
LLM integration (OpenAI / any OpenAI-compatible endpoint).

Two modes
---------
1. Rule-based  (no API key needed)
   Uses detected flags from query_analyzer.py to produce structured,
   actionable recommendations. Always available.

2. LLM-assisted  (requires OPENAI_API_KEY environment variable)
   Sends the query + context to an LLM and returns a natural-language
   optimization suggestion with an estimated impact level.
   Falls back to rule-based if the API call fails.
"""

import os
import re
import json
import textwrap
from typing import Optional
import pandas as pd


# ──────────────────────────────────────────────
# Rule-based recommendation engine
# ──────────────────────────────────────────────

RULE_RECOMMENDATIONS: dict[str, dict] = {
    "flag_select_star": {
        "title":    "Replace SELECT * with explicit column list",
        "detail":   (
            "SELECT * reads every column from the table, increasing I/O and bytes scanned. "
            "Specify only the columns your downstream logic requires. "
            "Example: Replace `SELECT *` with `SELECT col_a, col_b, col_c`."
        ),
        "impact":   "High",
        "saving":   "Up to 40–70% reduction in bytes scanned and credit usage.",
    },
    "flag_no_where_clause": {
        "title":    "Add a WHERE clause to limit rows scanned",
        "detail":   (
            "Queries without a WHERE clause perform full table scans, consuming maximum compute. "
            "Add date range filters, partition keys, or status filters. "
            "Example: Add `WHERE created_at >= DATEADD(day, -30, CURRENT_DATE)`."
        ),
        "impact":   "High",
        "saving":   "Potentially 80–95% reduction in rows scanned for large tables.",
    },
    "flag_cartesian_join": {
        "title":    "Remove implicit Cartesian join (comma-separated FROM)",
        "detail":   (
            "Comma-separated table references in FROM produce a Cartesian product, "
            "multiplying rows exponentially. Use explicit JOIN ... ON syntax instead. "
            "Example: Replace `FROM t1, t2` with `FROM t1 JOIN t2 ON t1.id = t2.fk_id`."
        ),
        "impact":   "Critical",
        "saving":   "Eliminates exponential row explosion — often 1000x+ credit reduction.",
    },
    "flag_where_1_equals_1": {
        "title":    "Remove WHERE 1=1 placeholder",
        "detail":   (
            "WHERE 1=1 is a dynamic query building artefact. While it doesn't change "
            "results, it signals that the query may lack meaningful filters. "
            "Review the full query for missing predicates."
        ),
        "impact":   "Low",
        "saving":   "Minimal direct saving, but signals a query that likely needs additional filters.",
    },
    "flag_low_partition_pruning": {
        "title":    "Improve partition pruning (add cluster key filters)",
        "detail":   (
            "High partitions_scanned / partitions_total ratio suggests Snowflake cannot "
            "skip micro-partitions. Add filters on the table's cluster key columns "
            "(typically date or timestamp columns). "
            "Consider re-clustering the table if filters are already present."
        ),
        "impact":   "High",
        "saving":   "60–90% reduction in micro-partitions scanned.",
    },
    "flag_high_bytes_per_row": {
        "title":    "Reduce bytes scanned per row — check projections and wide tables",
        "detail":   (
            "Very high bytes-per-row indicates the query scans many wide columns "
            "for each output row. Select fewer columns, use column-store projection, "
            "or filter earlier in the pipeline using CTEs or subqueries."
        ),
        "impact":   "Medium",
        "saving":   "20–50% reduction in bytes scanned depending on column widths.",
    },
    "flag_long_runtime": {
        "title":    "Reduce execution time — consider query restructuring or larger warehouse",
        "detail":   (
            "This query's runtime is a statistical outlier (>2 std above mean). "
            "Investigate the query plan using EXPLAIN. "
            "Consider: (1) adding indexes or cluster keys, (2) breaking into smaller CTEs, "
            "(3) temporarily scaling up warehouse size, or (4) materializing intermediate results."
        ),
        "impact":   "Medium",
        "saving":   "Varies — profile with EXPLAIN before changing warehouse size.",
    },
    "flag_high_credit_cost": {
        "title":    "Reduce credit consumption — query is in top 10% by cost",
        "detail":   (
            "This query is among the most expensive by credit usage. "
            "Prioritize optimizing it first. Apply other recommendations above; "
            "also consider scheduling it during off-peak hours or using result caching."
        ),
        "impact":   "High",
        "saving":   "Typically 30–60% credit reduction after applying query-level optimizations.",
    },
}


def get_rule_based_recommendations(row: pd.Series) -> list[dict]:
    """
    Generate a list of recommendations for a single query row
    based on its flag columns.

    Parameters
    ----------
    row : pd.Series — a row from analyze_queries() output

    Returns
    -------
    List of recommendation dicts, sorted by impact (Critical > High > Medium > Low)
    """
    impact_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    recs = []

    for flag_col, rec in RULE_RECOMMENDATIONS.items():
        if row.get(flag_col, False):
            recs.append({
                "flag":    flag_col,
                "title":   rec["title"],
                "detail":  rec["detail"],
                "impact":  rec["impact"],
                "saving":  rec["saving"],
            })

    recs.sort(key=lambda r: impact_order.get(r["impact"], 99))
    return recs


# ──────────────────────────────────────────────
# LLM-assisted recommendations
# ──────────────────────────────────────────────

def _build_llm_prompt(row: pd.Series, flags: list[str]) -> str:
    """Build a focused prompt for the LLM optimizer."""
    flag_descriptions = [
        RULE_RECOMMENDATIONS[f]["title"]
        for f in flags
        if f in RULE_RECOMMENDATIONS
    ]

    prompt = textwrap.dedent(f"""
    You are a Snowflake SQL optimization expert with a focus on reducing compute costs
    and carbon emissions from cloud analytics workloads.

    A query has been flagged as inefficient. Here are the details:

    Query Text:
    {row.get('query_text', 'N/A')}

    Execution Stats:
    - Warehouse Size  : {row.get('warehouse_size', 'N/A')}
    - Runtime         : {row.get('execution_time_s', 0):.1f} seconds
    - Credits Used    : {row.get('credits_used', 0):.4f}
    - GB Scanned      : {row.get('gb_scanned', 0):.3f} GB
    - CO₂ Emitted     : {row.get('co2_g', 0):.2f} grams

    Detected Issues:
    {chr(10).join(f'- {d}' for d in flag_descriptions) if flag_descriptions else '- None detected by rules'}

    Please provide:
    1. A brief diagnosis of why this query is inefficient (2-3 sentences).
    2. A concrete, specific rewrite suggestion or optimization step.
    3. An estimated impact on credits and carbon reduction (Low / Medium / High).

    Be specific to the actual query text. Do not repeat generic advice already listed above.
    Format your response as JSON with keys: "diagnosis", "suggestion", "estimated_impact".
    """).strip()

    return prompt


def get_llm_recommendation(row: pd.Series,
                            flags: list[str],
                            api_key: Optional[str] = None,
                            model: str = "gpt-3.5-turbo") -> dict:
    """
    Get an LLM-generated optimization recommendation for a query.

    Requires: OPENAI_API_KEY environment variable or explicit api_key parameter.
    Falls back to rule-based recommendations if the API call fails.

    Parameters
    ----------
    row     : pd.Series — query row from analyze_queries()
    flags   : list of active flag column names
    api_key : OpenAI API key (defaults to OPENAI_API_KEY env var)
    model   : OpenAI model to use

    Returns
    -------
    dict with keys: diagnosis, suggestion, estimated_impact, source
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return {
            "diagnosis":        "LLM not configured — set OPENAI_API_KEY.",
            "suggestion":       "Using rule-based recommendations instead.",
            "estimated_impact": "N/A",
            "source":           "rule-based",
        }

    try:
        import openai
        client = openai.OpenAI(api_key=key)

        prompt   = _build_llm_prompt(row, flags)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400,
        )
        content = response.choices[0].message.content.strip()

        # Parse JSON response
        # Strip markdown code fences if present
        content = re.sub(r"```json\s*|\s*```", "", content).strip()
        result  = json.loads(content)
        result["source"] = "llm"
        return result

    except Exception as e:
        return {
            "diagnosis":        f"LLM call failed: {str(e)[:100]}",
            "suggestion":       "Falling back to rule-based recommendations.",
            "estimated_impact": "N/A",
            "source":           "fallback",
        }


# ──────────────────────────────────────────────
# Batch optimization report
# ──────────────────────────────────────────────

def generate_optimization_report(df: pd.DataFrame,
                                 top_n: int = 20,
                                 use_llm: bool = False) -> pd.DataFrame:
    """
    Generate an optimization report for the top-N most inefficient queries.

    Parameters
    ----------
    df     : DataFrame from analyze_queries() + enrich_dataframe()
    top_n  : number of queries to include in the report
    use_llm: whether to attempt LLM recommendations

    Returns
    -------
    DataFrame with one row per query and columns:
    query_id, efficiency_score, top_flags, recommendations_summary,
    credits_used, co2_g, estimated_saving
    """
    if "efficiency_score" not in df.columns:
        raise ValueError("Run analyze_queries() and enrich_dataframe() first.")

    candidates = (
        df[df["efficiency_score"] < 80]
        .nsmallest(top_n, "efficiency_score")
        .copy()
    )

    flag_cols = [c for c in df.columns if c.startswith("flag_") and c != "flag_count"]

    report_rows = []
    for _, row in candidates.iterrows():
        active_flags = [c for c in flag_cols if row.get(c, False)]
        recs         = get_rule_based_recommendations(row)
        llm_rec      = {}

        if use_llm and active_flags:
            llm_rec = get_llm_recommendation(row, active_flags)

        rec_summary = "; ".join(r["title"] for r in recs[:3]) if recs else "No issues"

        report_rows.append({
            "query_id":             row.get("query_id", ""),
            "user_name":            row.get("user_name", ""),
            "team":                 row.get("team", ""),
            "warehouse_name":       row.get("warehouse_name", ""),
            "efficiency_score":     row.get("efficiency_score", 0),
            "efficiency_label":     row.get("efficiency_label", ""),
            "top_flags":            row.get("top_flags", ""),
            "credits_used":         round(row.get("credits_used", 0), 5),
            "co2_g":                round(row.get("co2_g", 0), 3),
            "execution_time_s":     round(row.get("execution_time_s", 0), 1),
            "gb_scanned":           round(row.get("gb_scanned", 0), 4),
            "recommendations":      rec_summary,
            "llm_suggestion":       llm_rec.get("suggestion", "") if use_llm else "",
            "llm_impact":           llm_rec.get("estimated_impact", "") if use_llm else "",
        })

    return pd.DataFrame(report_rows)


# ──────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_generator  import generate_query_history
    from carbon_estimator import enrich_dataframe
    from query_analyzer   import analyze_queries

    df = generate_query_history(n_records=500)
    df = enrich_dataframe(df)
    df = analyze_queries(df)

    report = generate_optimization_report(df, top_n=10)
    print("── Optimization Report (Top 10 Inefficient Queries) ──")
    print(report[["query_id", "efficiency_score", "top_flags",
                  "co2_g", "recommendations"]].to_string(index=False))
