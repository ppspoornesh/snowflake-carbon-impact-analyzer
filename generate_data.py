"""
generate_data.py
----------------
Convenience script to pre-generate sample data for the dashboard.
Run once before launching the dashboard if you want a static dataset.

    python generate_data.py
    python generate_data.py --records 10000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_generator import save_sample_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Snowflake query history.")
    parser.add_argument("--records", type=int, default=5000,
                        help="Number of query records to generate (default: 5000)")
    parser.add_argument("--output",  type=str, default="data/query_history.csv",
                        help="Output CSV path (default: data/query_history.csv)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = save_sample_data(path=args.output, n_records=args.records)

    print(f"\nSample statistics:")
    print(f"  Records     : {len(df):,}")
    print(f"  Date range  : {df['start_time'].min().date()} → {df['start_time'].max().date()}")
    print(f"  Total credits: {df['credits_used'].sum():.2f}")
    print(f"  Users       : {df['user_name'].nunique()}")
    print(f"  Warehouses  : {df['warehouse_name'].nunique()}")
