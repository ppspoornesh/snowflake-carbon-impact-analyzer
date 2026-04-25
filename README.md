# 🌱 Snowflake Carbon Impact Analyzer

**Estimate the carbon footprint of your Snowflake analytics workloads. Identify inefficient queries. Reduce compute costs and CO₂ emissions.**

---

## Overview

Large-scale cloud analytics workloads consume significant compute energy — and emit CO₂. This tool analyzes Snowflake query history to:

- **Estimate** the carbon footprint of each query, user, team, and warehouse
- **Identify** high-impact and inefficient queries using anti-pattern detection
- **Generate** AI-driven optimization recommendations
- **Visualize** compute and carbon metrics via an interactive Streamlit dashboard

This project was built to support sustainability goals in cloud-based data processing — making analytics teams more aware of the environmental cost of their queries and giving them actionable steps to reduce it.

---

## Features

| Feature | Description |
|---------|-------------|
| 🔋 Carbon Estimation | Credits → compute-hours → kWh → CO₂ (kg) |
| 🔍 Query Anti-Pattern Detection | SELECT *, missing WHERE, Cartesian joins, low partition pruning, and more |
| 📊 Interactive Dashboard | Streamlit app with filters, drill-down, and export |
| 💡 Optimization Recommendations | Rule-based + optional LLM-assisted suggestions |
| 📅 Timeline Analysis | Daily compute and carbon usage heatmaps |
| 📁 CSV Export | Download optimization reports for sharing |

---

## Project Structure

```
snowflake-carbon-impact-analyzer/
├── src/
│   ├── data_generator.py    # Generates synthetic Snowflake QUERY_HISTORY data
│   ├── carbon_estimator.py  # Credits → kWh → CO₂ estimation pipeline
│   ├── query_analyzer.py    # Anti-pattern detection + efficiency scoring
│   ├── optimizer.py         # Rule-based & LLM-assisted recommendations
│   └── dashboard.py         # Streamlit dashboard (main app)
├── data/
│   └── query_history.csv    # Generated after running generate_data.py
├── docs/
│   └── methodology.md       # Carbon estimation methodology + assumptions
├── generate_data.py         # CLI script to generate sample data
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/ppspoornesh/snowflake-carbon-impact-analyzer.git
cd snowflake-carbon-impact-analyzer
pip install -r requirements.txt
```

### 2. Generate sample data (optional)

```bash
python generate_data.py --records 5000
```

### 3. Launch the dashboard

```bash
streamlit run src/dashboard.py
```

Open `http://localhost:8501` in your browser.

---

## Using Real Snowflake Data

Export your query history from Snowflake:

```sql
SELECT
    QUERY_ID,
    QUERY_TEXT,
    QUERY_TYPE,
    USER_NAME,
    WAREHOUSE_NAME,
    WAREHOUSE_SIZE,
    EXECUTION_TIME / 1000        AS execution_time_s,
    EXECUTION_TIME               AS execution_time_ms,
    BYTES_SCANNED,
    BYTES_SCANNED / 1e9          AS gb_scanned,
    ROWS_PRODUCED,
    PARTITIONS_SCANNED,
    PARTITIONS_TOTAL,
    CREDITS_USED_CLOUD_SERVICES  AS credits_used,
    START_TIME,
    END_TIME,
    EXECUTION_STATUS             AS status
FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
WHERE START_TIME >= DATEADD(day, -90, CURRENT_TIMESTAMP)
ORDER BY START_TIME DESC;
```

Save as `data/query_history.csv` and upload via the dashboard sidebar.

> **Note:** Access to `SNOWFLAKE.ACCOUNT_USAGE` requires `ACCOUNTADMIN` or the `SNOWFLAKE` database privilege.

---

## Carbon Estimation Methodology

```
Credits Used
    ↓  (1 credit = 1 compute-hour, X-Small baseline)
Compute Hours
    ↓  (× 0.48 kWh/compute-hour, PUE-adjusted)
Energy (kWh)
    ↓  (× 0.475 kg CO₂/kWh, IEA global average)
CO₂ Emitted (kg)
```

See [docs/methodology.md](docs/methodology.md) for full derivation, assumptions, and limitations.

---

## LLM-Assisted Recommendations (Optional)

Set your OpenAI API key to enable AI-generated query rewrites:

```bash
export OPENAI_API_KEY=sk-...
```

The optimizer will send flagged queries to GPT with execution context and return specific rewrite suggestions. Falls back to rule-based recommendations if the API key is not set.

---

## Efficiency Scoring

Each query is scored 0–100. Penalties are applied for detected anti-patterns:

| Anti-Pattern | Penalty |
|---|---|
| SELECT * | −10 |
| No WHERE clause | −20 |
| Cartesian join | −25 |
| Low partition pruning | −15 |
| Long runtime (outlier) | −15 |
| High credit cost (top 10%) | −15 |

- **Score ≥ 80** → Efficient ✅
- **Score 50–79** → Moderate — review recommended ⚠️
- **Score < 50** → Inefficient — optimize immediately 🔴

---

## Tech Stack

- **Python** — core analysis and pipeline
- **Pandas / NumPy** — data processing
- **Streamlit** — dashboard UI
- **Plotly** — interactive charts
- **OpenAI** — optional LLM recommendations

---

## License

MIT License — free to use, modify, and distribute.

---

*Built to support sustainability in cloud analytics. Every optimized query is a step toward greener data.*
