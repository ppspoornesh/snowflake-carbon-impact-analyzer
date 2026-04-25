"""
dashboard.py
------------
Streamlit dashboard for the Snowflake Carbon Impact Analyzer.

Run with:
    streamlit run src/dashboard.py

Features
--------
- KPI cards: total credits, total CO₂, avg efficiency score, top inefficient queries
- Carbon impact by team, user, warehouse (bar charts)
- Efficiency distribution (histogram + pie)
- High-impact query table with drill-down
- Optimization recommendations per query
- Timeline of compute usage and emissions
- CSV export
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_generator   import generate_query_history
from carbon_estimator import enrich_dataframe, aggregate_carbon, identify_high_impact_queries
from query_analyzer   import analyze_queries, query_summary
from optimizer        import generate_optimization_report, get_rule_based_recommendations


# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Snowflake Carbon Impact Analyzer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colour palette
GREEN  = "#2ECC71"
AMBER  = "#F39C12"
RED    = "#E74C3C"
BLUE   = "#3498DB"
DARK   = "#2C3E50"
LIGHT  = "#ECF0F1"

st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .section-header {
        font-size: 1.1rem; font-weight: 600;
        border-left: 4px solid #2ECC71;
        padding-left: 10px; margin: 20px 0 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar — data source & filters
# ──────────────────────────────────────────────

st.sidebar.image("https://img.shields.io/badge/🌱-Carbon%20Analyzer-2ECC71?style=flat", use_container_width=True)
st.sidebar.title("⚙️ Configuration")

data_source = st.sidebar.radio(
    "Data Source",
    ["Generate Sample Data", "Upload CSV"],
    help="Use sample data for demo or upload real Snowflake QUERY_HISTORY export."
)

n_records = 3000
if data_source == "Generate Sample Data":
    n_records = st.sidebar.slider("Number of sample queries", 500, 10000, 3000, 500)


@st.cache_data(show_spinner="Loading and analysing query data...")
def load_and_process(source: str, n: int, uploaded=None) -> pd.DataFrame:
    if source == "Generate Sample Data":
        df = generate_query_history(n_records=n)
    else:
        if uploaded is None:
            return pd.DataFrame()
        df = pd.read_csv(uploaded, parse_dates=["start_time", "end_time"])

    df = enrich_dataframe(df)
    df = analyze_queries(df)
    return df


uploaded_file = None
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Snowflake QUERY_HISTORY CSV",
        type=["csv"],
        help="Export from: SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY"
    )
    if uploaded_file is None:
        st.info("📂 Upload a CSV file or switch to sample data to get started.")
        st.stop()

df = load_and_process(data_source, n_records, uploaded_file)

if df.empty:
    st.warning("No data available.")
    st.stop()


# ── Sidebar filters ──────────────────────────────

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filters")

all_teams = ["All"] + sorted(df["team"].unique().tolist())
sel_team  = st.sidebar.selectbox("Team", all_teams)

all_wh  = ["All"] + sorted(df["warehouse_name"].unique().tolist())
sel_wh  = st.sidebar.selectbox("Warehouse", all_wh)

all_users = ["All"] + sorted(df["user_name"].unique().tolist())
sel_user  = st.sidebar.selectbox("User", all_users)

status_options = sorted(df["status"].unique().tolist())
sel_status = st.sidebar.multiselect("Query Status", status_options, default=["SUCCESS"])

date_range = st.sidebar.date_input(
    "Date Range",
    value=(df["start_time"].min().date(), df["start_time"].max().date()),
)

# ── Apply filters ─────────────────────────────────
fdf = df.copy()
if sel_team  != "All": fdf = fdf[fdf["team"]           == sel_team]
if sel_wh    != "All": fdf = fdf[fdf["warehouse_name"] == sel_wh]
if sel_user  != "All": fdf = fdf[fdf["user_name"]      == sel_user]
if sel_status:         fdf = fdf[fdf["status"].isin(sel_status)]
if len(date_range) == 2:
    fdf = fdf[
        (fdf["start_time"].dt.date >= date_range[0]) &
        (fdf["start_time"].dt.date <= date_range[1])
    ]


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────

st.title("🌱 Snowflake Carbon Impact Analyzer")
st.markdown(
    "Visualize the **environmental footprint** of your analytics workloads. "
    "Identify inefficient queries and take action to reduce compute costs and CO₂ emissions."
)
st.markdown("---")


# ──────────────────────────────────────────────
# KPI Cards
# ──────────────────────────────────────────────

total_queries   = len(fdf)
total_credits   = fdf["credits_used"].sum()
total_co2_kg    = fdf["co2_kg"].sum()
total_energy    = fdf["energy_kwh"].sum()
avg_eff_score   = fdf["efficiency_score"].mean()
inefficient_pct = (fdf["efficiency_label"] == "Inefficient").mean() * 100

c1, c2, c3, c4, c5, c6 = st.columns(6)

def kpi(col, value, label, color=GREEN, suffix=""):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{color}">{value}{suffix}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

kpi(c1, f"{total_queries:,}",           "Total Queries")
kpi(c2, f"{total_credits:,.2f}",        "Credits Used",     BLUE)
kpi(c3, f"{total_energy:,.1f}",         "Energy (kWh)",     AMBER)
kpi(c4, f"{total_co2_kg:,.2f}",         "CO₂ Emitted (kg)", RED)
kpi(c5, f"{avg_eff_score:.0f}",         "Avg Efficiency",
    GREEN if avg_eff_score >= 75 else AMBER if avg_eff_score >= 50 else RED,
    "/100")
kpi(c6, f"{inefficient_pct:.1f}",       "Inefficient Queries", RED, "%")

st.markdown("<br>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Tab layout
# ──────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Carbon Overview",
    "⚠️ Inefficient Queries",
    "🔬 Query Analysis",
    "💡 Recommendations",
    "📅 Timeline",
])


# ── Tab 1: Carbon Overview ────────────────────

with tab1:
    st.markdown('<div class="section-header">Carbon Footprint by Team</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        by_team = aggregate_carbon(fdf, group_by="team")
        fig = px.bar(
            by_team, x="team", y="total_co2_kg",
            color="total_co2_kg",
            color_continuous_scale="RdYlGn_r",
            labels={"total_co2_kg": "CO₂ (kg)", "team": "Team"},
            title="CO₂ Emissions by Team"
        )
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        by_wh = aggregate_carbon(fdf, group_by="warehouse_name")
        fig2  = px.pie(
            by_wh, names="warehouse_name", values="total_co2_kg",
            title="CO₂ Distribution by Warehouse",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Compute vs. Carbon by User</div>', unsafe_allow_html=True)
    by_user = aggregate_carbon(fdf, group_by="user_name")
    fig3 = px.scatter(
        by_user, x="total_credits", y="total_co2_kg",
        size="query_count", color="total_co2_kg",
        hover_name="user_name", text="user_name",
        color_continuous_scale="RdYlGn_r",
        labels={"total_credits": "Credits Used", "total_co2_kg": "CO₂ (kg)"},
        title="User-level Compute vs. Carbon Bubble Chart"
    )
    fig3.update_traces(textposition="top center")
    fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)


# ── Tab 2: Inefficient Queries ───────────────

with tab2:
    st.markdown('<div class="section-header">High-Impact Inefficient Queries</div>', unsafe_allow_html=True)

    top15 = identify_high_impact_queries(fdf, top_pct=0.15)
    ineff = fdf[fdf["efficiency_label"] == "Inefficient"].copy()

    col_x, col_y = st.columns([2, 1])

    with col_x:
        show_cols = ["query_id", "user_name", "team", "warehouse_name",
                     "efficiency_score", "top_flags", "credits_used",
                     "co2_g", "execution_time_s", "gb_scanned"]
        available = [c for c in show_cols if c in ineff.columns]
        st.dataframe(
            ineff[available].sort_values("efficiency_score").head(50),
            use_container_width=True,
            height=380,
        )

    with col_y:
        label_counts = fdf["efficiency_label"].value_counts().reset_index()
        label_counts.columns = ["label", "count"]
        color_map = {"Efficient": GREEN, "Moderate": AMBER, "Inefficient": RED}
        fig_pie = px.pie(
            label_counts, names="label", values="count",
            color="label", color_discrete_map=color_map,
            title="Query Efficiency Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<div class="section-header">Top 15% High-Impact Queries (by CO₂)</div>', unsafe_allow_html=True)
    top_cols = ["impact_rank", "query_id", "user_name", "team",
                "credits_used", "co2_g", "pct_of_total_co2", "execution_time_s"]
    available_top = [c for c in top_cols if c in top15.columns]
    st.dataframe(top15[available_top].head(30), use_container_width=True)

    pct_of_total = top15["co2_g"].sum() / fdf["co2_g"].sum() * 100 if fdf["co2_g"].sum() > 0 else 0
    st.info(
        f"📌 The top 15% of queries ({len(top15)}) account for "
        f"**{pct_of_total:.1f}%** of total CO₂ emissions. "
        f"Optimizing these alone would deliver the largest environmental impact."
    )


# ── Tab 3: Query Analysis ─────────────────────

with tab3:
    st.markdown('<div class="section-header">Flag Frequency Analysis</div>', unsafe_allow_html=True)

    flag_cols = [c for c in fdf.columns if c.startswith("flag_") and c != "flag_count"]
    flag_sums = fdf[flag_cols].sum().reset_index()
    flag_sums.columns = ["Flag", "Count"]
    flag_sums["Flag"] = flag_sums["Flag"].str.replace("flag_", "").str.replace("_", " ").str.title()
    flag_sums = flag_sums[flag_sums["Count"] > 0].sort_values("Count", ascending=True)

    fig_flags = px.bar(
        flag_sums, x="Count", y="Flag", orientation="h",
        color="Count", color_continuous_scale="RdYlGn_r",
        title="Query Anti-Pattern Frequency",
        labels={"Count": "Number of Queries", "Flag": "Anti-Pattern"}
    )
    fig_flags.update_layout(plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    st.plotly_chart(fig_flags, use_container_width=True)

    st.markdown('<div class="section-header">Efficiency Score Distribution</div>', unsafe_allow_html=True)
    fig_hist = px.histogram(
        fdf, x="efficiency_score", nbins=20,
        color_discrete_sequence=[BLUE],
        title="Distribution of Query Efficiency Scores (0–100)",
        labels={"efficiency_score": "Efficiency Score", "count": "Number of Queries"}
    )
    fig_hist.add_vline(x=80, line_dash="dash", line_color=GREEN, annotation_text="Efficient threshold")
    fig_hist.add_vline(x=50, line_dash="dash", line_color=RED,   annotation_text="Inefficient threshold")
    fig_hist.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_hist, use_container_width=True)


# ── Tab 4: Recommendations ────────────────────

with tab4:
    st.markdown('<div class="section-header">AI-Driven Optimization Recommendations</div>', unsafe_allow_html=True)

    top_n_recs = st.slider("Number of queries to analyze", 5, 50, 15)
    report     = generate_optimization_report(fdf, top_n=top_n_recs, use_llm=False)

    if report.empty:
        st.success("✅ No inefficient queries found with the current filters!")
    else:
        st.dataframe(
            report[["query_id", "user_name", "team", "efficiency_score",
                    "top_flags", "credits_used", "co2_g", "recommendations"]],
            use_container_width=True,
            height=350,
        )

        # Drill-down: click a query to see detailed recommendations
        st.markdown('<div class="section-header">Query Drill-Down</div>', unsafe_allow_html=True)
        selected_qid = st.selectbox(
            "Select a query for detailed recommendations",
            options=report["query_id"].tolist()
        )

        if selected_qid:
            row = fdf[fdf["query_id"] == selected_qid].iloc[0]
            recs = get_rule_based_recommendations(row)

            st.code(row.get("query_text", "N/A"), language="sql")

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Efficiency Score", f"{row['efficiency_score']}/100")
            col_m2.metric("Credits Used",     f"{row['credits_used']:.5f}")
            col_m3.metric("CO₂ Emitted",      f"{row['co2_g']:.3f} g")

            if recs:
                for i, rec in enumerate(recs, 1):
                    color = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(rec["impact"], "⚪")
                    with st.expander(f"{color} [{rec['impact']}] {rec['title']}"):
                        st.write(rec["detail"])
                        st.success(f"💰 Estimated saving: {rec['saving']}")
            else:
                st.success("✅ No anti-patterns detected for this query.")

        # Export
        st.markdown("---")
        csv = report.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Optimization Report (CSV)",
            data=csv,
            file_name="optimization_report.csv",
            mime="text/csv",
        )


# ── Tab 5: Timeline ───────────────────────────

with tab5:
    st.markdown('<div class="section-header">Compute & Carbon Usage Over Time</div>', unsafe_allow_html=True)

    fdf["date"] = fdf["start_time"].dt.date

    daily = (
        fdf.groupby("date")
        .agg(
            queries      = ("query_id",      "count"),
            credits      = ("credits_used",  "sum"),
            energy_kwh   = ("energy_kwh",    "sum"),
            co2_kg       = ("co2_kg",        "sum"),
        )
        .reset_index()
    )

    fig_timeline = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Daily Credits Used", "Daily CO₂ Emitted (kg)"),
        vertical_spacing=0.12,
    )
    fig_timeline.add_trace(
        go.Bar(x=daily["date"], y=daily["credits"], name="Credits",
               marker_color=BLUE), row=1, col=1
    )
    fig_timeline.add_trace(
        go.Scatter(x=daily["date"], y=daily["co2_kg"], name="CO₂ (kg)",
                   mode="lines+markers", line=dict(color=RED, width=2)), row=2, col=1
    )
    fig_timeline.update_layout(
        height=500, plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True, hovermode="x unified"
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    st.markdown('<div class="section-header">Hourly Heatmap — Query Volume</div>', unsafe_allow_html=True)
    fdf["hour"] = fdf["start_time"].dt.hour
    fdf["dow"]  = fdf["start_time"].dt.day_name()

    heatmap_data = (
        fdf.groupby(["dow", "hour"])["query_id"]
        .count()
        .unstack(fill_value=0)
    )
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data = heatmap_data.reindex([d for d in day_order if d in heatmap_data.index])

    fig_hm = px.imshow(
        heatmap_data,
        color_continuous_scale="Blues",
        labels={"x": "Hour of Day", "y": "Day of Week", "color": "Query Count"},
        title="Query Volume Heatmap (Day × Hour)",
        aspect="auto",
    )
    st.plotly_chart(fig_hm, use_container_width=True)


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<center><small>🌱 Snowflake Carbon Impact Analyzer · "
    "Carbon methodology: IEA 2022 · "
    "Built with Streamlit, Plotly & Python</small></center>",
    unsafe_allow_html=True,
)
