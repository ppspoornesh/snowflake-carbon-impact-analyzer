"""
carbon_estimator.py
-------------------
Estimates the carbon footprint of Snowflake analytics workloads
based on compute credit consumption.

Methodology
-----------
1. Credits Used  →  Compute-Hours
   Snowflake bills 1 credit per compute-hour for an X-Small warehouse.
   Larger warehouses consume proportionally more credits per hour.

2. Compute-Hours  →  Energy (kWh)
   A standard cloud server draws ~200–350W under load.
   With a typical data-center PUE (Power Usage Effectiveness) of ~1.5,
   we estimate 0.48 kWh per compute-hour.
   Reference: IEA Data Centres & Data Transmission Networks (2022)

3. Energy (kWh)  →  CO₂ (kg)
   Global average grid carbon intensity ≈ 0.475 kg CO₂e / kWh.
   For Azure (Pune region), a more conservative 0.708 kg CO₂e/kWh is
   available from the Microsoft Emissions Impact Dashboard.
   We expose both and default to the global average.

Assumptions & Limitations
--------------------------
- Credit-to-hour mapping assumes standard Snowflake on-demand pricing.
  Enterprise/business-critical multipliers are not modelled.
- Carbon intensity is a regional/temporal average; actual intensity
  varies with grid mix and time of day.
- Idle warehouse time (auto-suspend overhead) is not captured in
  QUERY_HISTORY and is excluded from this model.
- Network and storage I/O energy is not included.

Units
-----
- credits_used     : Snowflake compute credits (dimensionless)
- compute_hours    : hours of compute consumed
- energy_kwh       : kilowatt-hours of electricity
- co2_kg           : kilograms of CO₂ equivalent
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

KWH_PER_COMPUTE_HOUR: float = 0.48        # energy per credit-hour (PUE-adjusted)
CO2_KG_PER_KWH_GLOBAL: float = 0.475     # IPCC / IEA global average
CO2_KG_PER_KWH_AZURE_INDIA: float = 0.708  # Microsoft EID — India West region

# Warehouse size → credit multiplier (per Snowflake docs)
WAREHOUSE_CREDIT_MULTIPLIER: dict[str, float] = {
    "X-Small": 1,
    "Small":   2,
    "Medium":  4,
    "Large":   8,
    "X-Large": 16,
    "2X-Large": 32,
    "3X-Large": 64,
    "4X-Large": 128,
}


@dataclass
class CarbonConfig:
    """Configuration for the carbon estimation model."""
    kwh_per_compute_hour: float = KWH_PER_COMPUTE_HOUR
    co2_kg_per_kwh: float = CO2_KG_PER_KWH_GLOBAL
    label: str = "Global Average (IEA)"


PRESETS = {
    "global":      CarbonConfig(),
    "azure_india": CarbonConfig(co2_kg_per_kwh=CO2_KG_PER_KWH_AZURE_INDIA,
                                label="Azure India West (Microsoft EID)"),
}


# ──────────────────────────────────────────────
# Core estimation functions
# ──────────────────────────────────────────────

def credits_to_kwh(credits: float,
                   kwh_per_compute_hour: float = KWH_PER_COMPUTE_HOUR) -> float:
    """Convert Snowflake credits to kWh of energy consumed."""
    return credits * kwh_per_compute_hour


def kwh_to_co2(kwh: float,
               co2_per_kwh: float = CO2_KG_PER_KWH_GLOBAL) -> float:
    """Convert kWh to kg CO₂ equivalent."""
    return kwh * co2_per_kwh


def estimate_co2(credits: float,
                 config: CarbonConfig = None) -> dict:
    """
    Full pipeline: credits → compute-hours → kWh → CO₂.

    Returns a dict with intermediate values for transparency.
    """
    if config is None:
        config = CarbonConfig()

    compute_hours = credits  # 1 credit = 1 compute-hour (X-Small baseline)
    energy_kwh    = credits_to_kwh(credits, config.kwh_per_compute_hour)
    co2_kg        = kwh_to_co2(energy_kwh, config.co2_kg_per_kwh)

    return {
        "credits_used":    credits,
        "compute_hours":   compute_hours,
        "energy_kwh":      round(energy_kwh, 6),
        "co2_kg":          round(co2_kg, 6),
        "co2_g":           round(co2_kg * 1000, 3),
        "methodology":     config.label,
    }


# ──────────────────────────────────────────────
# DataFrame-level enrichment
# ──────────────────────────────────────────────

def enrich_dataframe(df: pd.DataFrame,
                     config: CarbonConfig = None) -> pd.DataFrame:
    """
    Add carbon impact columns to a query history DataFrame.

    Required columns: credits_used
    Optional (improves accuracy): warehouse_size

    Returns the original DataFrame with additional columns:
        compute_hours, energy_kwh, co2_kg, co2_g
    """
    if config is None:
        config = CarbonConfig()

    df = df.copy()

    df["compute_hours"] = df["credits_used"]
    df["energy_kwh"]    = df["compute_hours"] * config.kwh_per_compute_hour
    df["co2_kg"]        = df["energy_kwh"] * config.co2_kg_per_kwh
    df["co2_g"]         = df["co2_kg"] * 1000

    return df


def aggregate_carbon(df: pd.DataFrame,
                     group_by: str | list = None) -> pd.DataFrame:
    """
    Aggregate carbon metrics, optionally grouped by a column
    (e.g., 'user_name', 'team', 'warehouse_name', 'query_type').
    """
    required = ["credits_used", "energy_kwh", "co2_kg"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Run enrich_dataframe() first.")

    agg_cols = {
        "query_id":      "count",
        "credits_used":  "sum",
        "energy_kwh":    "sum",
        "co2_kg":        "sum",
        "execution_time_s": "sum",
        "gb_scanned":    "sum",
    }

    if group_by is None:
        # Total summary
        result = pd.DataFrame([{
            "total_queries":       len(df),
            "total_credits":       df["credits_used"].sum(),
            "total_energy_kwh":    df["energy_kwh"].sum(),
            "total_co2_kg":        df["co2_kg"].sum(),
            "total_co2_g":         df["co2_kg"].sum() * 1000,
            "avg_co2_g_per_query": df["co2_kg"].mean() * 1000,
        }])
    else:
        result = (
            df.groupby(group_by)
              .agg({k: v for k, v in agg_cols.items() if k in df.columns})
              .reset_index()
              .rename(columns={
                  "query_id":        "query_count",
                  "credits_used":    "total_credits",
                  "energy_kwh":      "total_energy_kwh",
                  "co2_kg":          "total_co2_kg",
                  "execution_time_s":"total_runtime_s",
                  "gb_scanned":      "total_gb_scanned",
              })
              .sort_values("total_co2_kg", ascending=False)
              .reset_index(drop=True)
        )
        result["total_co2_g"] = result["total_co2_kg"] * 1000
        result["pct_of_total"] = (
            result["total_co2_kg"] / result["total_co2_kg"].sum() * 100
        ).round(2)

    return result


def identify_high_impact_queries(df: pd.DataFrame,
                                 top_pct: float = 0.15) -> pd.DataFrame:
    """
    Identify the top `top_pct` of queries by CO₂ contribution.
    Default: top 15% — these typically account for 60–80% of total emissions.
    """
    if "co2_kg" not in df.columns:
        raise ValueError("Run enrich_dataframe() before calling this function.")

    n_top     = max(1, int(len(df) * top_pct))
    top_df    = df.nlargest(n_top, "co2_kg").copy()
    top_df["impact_rank"] = range(1, len(top_df) + 1)
    top_df["pct_of_total_co2"] = (
        top_df["co2_kg"] / df["co2_kg"].sum() * 100
    ).round(2)
    return top_df.reset_index(drop=True)


# ──────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_generator import generate_query_history

    df = generate_query_history(n_records=1000)
    df = enrich_dataframe(df)

    total = aggregate_carbon(df)
    print("\n── Total Carbon Footprint ──")
    print(total.T.to_string())

    by_team = aggregate_carbon(df, group_by="team")
    print("\n── Carbon by Team ──")
    print(by_team.to_string(index=False))

    top = identify_high_impact_queries(df, top_pct=0.15)
    print(f"\n── Top 15% High-Impact Queries ({len(top)} queries) ──")
    print(top[["query_id", "user_name", "credits_used",
               "co2_g", "pct_of_total_co2"]].head(10).to_string(index=False))
