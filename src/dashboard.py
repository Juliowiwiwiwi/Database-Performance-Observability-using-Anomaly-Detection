"""
Database Performance Observability - Anomaly Detection Dashboard
==================================================================

Interactive Streamlit dashboard over the outputs of v2model.py.

Folder layout expected (same project structure as v2model.py):
    <project_root>/DataSets/query_log.csv
    <project_root>/DataSets/system_metrics.csv
    <project_root>/src/v2model.py
    <project_root>/src/dashboard.py         <- this file
    <project_root>/src/Outputs/
        query_log_scored_v2.csv
        incidents_v2.csv

Run with:
    streamlit run dashboard.py

If Outputs/query_log_scored_v2.csv doesn't exist yet, run v2model.py first.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

 
# PATHS
 

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = SCRIPT_DIR / "Outputs"

SCORED_PATH = OUTPUTS_DIR / "query_log_scored_v2.csv"
INCIDENTS_PATH = OUTPUTS_DIR / "incidents_v2.csv"

st.set_page_config(
    page_title="DB Anomaly Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

 
# DATA LOADING (cached so filters don't re-read 800k+ rows every click)
 

NEEDED_COLS = [
    "query_id", "timestamp", "query_type", "table_name",
    "execution_time_ms", "rows_scanned", "rows_returned",
    "cpu_pct_at_time", "memory_pct_at_time", "lock_wait_count_at_time",
    "is_anomaly", "anomaly_type",
    "z_score", "ref_z_score", "bucket_z", "if_score", "risk_score",
    "predicted_anomaly", "root_cause_hint",
]


@st.cache_data(show_spinner="Loading scored query log...")
def load_scored():
    df = pd.read_csv(SCORED_PATH, usecols=lambda c: c in NEEDED_COLS,
                      parse_dates=["timestamp"])
    df["is_anomaly"] = df["is_anomaly"].astype(bool)
    df["predicted_anomaly"] = df["predicted_anomaly"].astype(bool)
    return df


@st.cache_data(show_spinner="Loading incidents...")
def load_incidents():
    df = pd.read_csv(INCIDENTS_PATH, parse_dates=["start", "end"])
    return df


if not SCORED_PATH.exists() or not INCIDENTS_PATH.exists():
    st.error(
        f"Couldn't find model outputs.\n\n"
        f"Expected:\n- `{SCORED_PATH}`\n- `{INCIDENTS_PATH}`\n\n"
        f"Run `python v2model.py` first to generate them."
    )
    st.stop()

df_all = load_scored()
incidents_all = load_incidents()

 
# SIDEBAR FILTERS
 

st.sidebar.title("Filters")

min_ts, max_ts = df_all["timestamp"].min(), df_all["timestamp"].max()
date_range = st.sidebar.slider(
    "Date range",
    min_value=min_ts.to_pydatetime(),
    max_value=max_ts.to_pydatetime(),
    value=(min_ts.to_pydatetime(), max_ts.to_pydatetime()),
    format="MM/DD HH:mm",
)

all_types = sorted(df_all["query_type"].unique())
selected_types = st.sidebar.multiselect("Query type", all_types, default=all_types)

all_tables = sorted(df_all["table_name"].unique())
selected_tables = st.sidebar.multiselect("Table", all_tables, default=all_tables)

anomaly_filter = st.sidebar.radio(
    "Show", ["All queries", "Flagged anomalies only", "Ground-truth anomalies only"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Ground truth (`is_anomaly`) comes from the synthetic data generator's "
    "injected incidents. `predicted_anomaly` is the model's output. In a real "
    "deployment you would only have the model's predictions."
)

# apply filters
mask = (
    (df_all["timestamp"] >= pd.Timestamp(date_range[0]))
    & (df_all["timestamp"] <= pd.Timestamp(date_range[1]))
    & (df_all["query_type"].isin(selected_types))
    & (df_all["table_name"].isin(selected_tables))
)
if anomaly_filter == "Flagged anomalies only":
    mask &= df_all["predicted_anomaly"]
elif anomaly_filter == "Ground-truth anomalies only":
    mask &= df_all["is_anomaly"]

df = df_all.loc[mask]

incidents = incidents_all[
    (incidents_all["start"] <= pd.Timestamp(date_range[1]))
    & (incidents_all["end"] >= pd.Timestamp(date_range[0]))
    & (incidents_all["query_type"].isin(selected_types))
    & (incidents_all["table_name"].isin(selected_tables))
]

 
# HEADER + HEADLINE METRICS
 

st.title("Database Performance Observability — Anomaly Detection")
st.caption(
    "Two-layer detector (adaptive + reference z-score baselines, system-wide "
    "aggregate layer, Isolation Forest) with tuned ensemble threshold, "
    "evaluated against injected ground-truth incidents."
)

tp = ((df["predicted_anomaly"]) & (df["is_anomaly"])).sum()
fp = ((df["predicted_anomaly"]) & (~df["is_anomaly"])).sum()
fn = ((~df["predicted_anomaly"]) & (df["is_anomaly"])).sum()
precision = tp / (tp + fp) if (tp + fp) else np.nan
recall = tp / (tp + fn) if (tp + fn) else np.nan
f1 = 2 * precision * recall / (precision + recall) if (precision and recall) else np.nan

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Queries (filtered)", f"{len(df):,}")
c2.metric("Flagged anomalies", f"{int(df['predicted_anomaly'].sum()):,}")
c3.metric("Grouped incidents", f"{len(incidents):,}")
c4.metric("Precision", f"{precision:.1%}" if not np.isnan(precision) else "—")
c5.metric("Recall", f"{recall:.1%}" if not np.isnan(recall) else "—")
c6.metric("F1", f"{f1:.2f}" if not np.isnan(f1) else "—")

st.markdown("---")

 
# TABS
 

tab_scatter, tab_box, tab_timeline, tab_incidents, tab_explorer = st.tabs(
    ["Latency Scatter", "Box Plots by Type", "Timeline", "Incidents", "Query Explorer"]
)

# --- TAB 1: SCATTER ---
with tab_scatter:
    st.subheader("Query latency over time")
    st.caption(
        "Each point is a query. Red = flagged anomaly. Normal points are "
        "downsampled for rendering speed; anomalies are always shown in full."
    )

    normal_pts = df[~df["predicted_anomaly"]]
    if len(normal_pts) > 15000:
        normal_pts = normal_pts.sample(15000, random_state=1)
    anomaly_pts = df[df["predicted_anomaly"]]

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=normal_pts["timestamp"], y=normal_pts["execution_time_ms"],
        mode="markers", name="Normal",
        marker=dict(size=4, color="#4C72B0", opacity=0.25),
    ))
    fig.add_trace(go.Scattergl(
        x=anomaly_pts["timestamp"], y=anomaly_pts["execution_time_ms"],
        mode="markers", name="Flagged anomaly",
        marker=dict(size=6, color="#C44E52", opacity=0.8),
        customdata=anomaly_pts[["query_type", "table_name", "root_cause_hint"]],
        hovertemplate=(
            "<b>%{customdata[0]}</b> on %{customdata[1]}<br>"
            "%{y:.1f} ms<br>%{customdata[2]}<extra></extra>"
        ),
    ))
    fig.update_yaxes(type="log", title="Execution time (ms, log scale)")
    fig.update_xaxes(title="Time")
    fig.update_layout(height=550, legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, width='stretch')

# --- TAB 2: BOX PLOTS ---
with tab_box:
    st.subheader("Latency distribution by query type")
    st.caption(
        "Shows whether a query type is *naturally* slow (e.g. large report "
        "aggregations) versus individual outliers — the key distinction "
        "between 'normal but slow' and 'anomalous'."
    )
    order = (
        df.groupby("query_type")["execution_time_ms"].median()
        .sort_values(ascending=False).index.tolist()
    )
    fig = px.box(
        df, x="query_type", y="execution_time_ms", category_orders={"query_type": order},
        points=False, color="query_type",
    )
    fig.update_yaxes(type="log", title="Execution time (ms, log scale)")
    fig.update_xaxes(title="Query type")
    fig.update_layout(height=550, showlegend=False)
    st.plotly_chart(fig, width='stretch')

# --- TAB 3: TIMELINE ---
with tab_timeline:
    st.subheader("Anomalies over time vs. system load")
    bucket = st.select_slider("Bucket size", options=["15min", "30min", "1h", "3h"], value="1h")
    hourly = df.set_index("timestamp").resample(bucket).agg(
        anomaly_count=("predicted_anomaly", "sum"),
        avg_cpu=("cpu_pct_at_time", "mean"),
        avg_lock=("lock_wait_count_at_time", "mean"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly["timestamp"], y=hourly["anomaly_count"],
        name="Anomalies flagged", marker_color="#C44E52", opacity=0.7,
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=hourly["timestamp"], y=hourly["avg_cpu"],
        name="Avg CPU %", line=dict(color="#4C72B0"), yaxis="y2",
    ))
    fig.update_layout(
        height=550,
        yaxis=dict(title="Anomalies flagged"),
        yaxis2=dict(title="Avg CPU %", overlaying="y", side="right", range=[0, 100]),
        legend=dict(orientation="h", y=1.05),
        xaxis=dict(title="Time"),
    )
    st.plotly_chart(fig, width='stretch')

# --- TAB 4: INCIDENTS ---
with tab_incidents:
    st.subheader(f"Grouped incidents ({len(incidents):,} in current filter)")
    st.caption(
        "Consecutive flagged queries on the same table/query type are merged "
        "into one incident, so one persistent root cause shows as one row "
        "instead of thousands of individual alerts."
    )

    if incidents.empty:
        st.info("No incidents match the current filters.")
    else:
        top_n = st.slider("Show top N by severity", 5, min(100, len(incidents)), 20)
        top = incidents.sort_values("peak_severity", ascending=False).head(top_n).copy()
        top["label"] = top["query_type"] + " / " + top["table_name"]
        top = top.sort_values("start")
        # px.timeline needs a nonzero duration to render a visible bar
        top["end_display"] = top.apply(
            lambda r: r["end"] if r["end"] > r["start"] else r["start"] + pd.Timedelta(minutes=5),
            axis=1,
        )

        fig = px.timeline(
            top, x_start="start", x_end="end_display", y="label",
            color="peak_severity", color_continuous_scale="Reds",
            hover_data=["query_count", "duration_minutes", "true_anomaly_rate", "top_root_cause"],
        )
        fig.update_yaxes(autorange="reversed", title="")
        fig.update_layout(height=max(400, 28 * len(top)))
        st.plotly_chart(fig, width='stretch')

        st.markdown("**Incident details**")
        display_cols = [
            "query_type", "table_name", "start", "end", "duration_minutes",
            "query_count", "peak_severity", "true_anomaly_rate", "top_root_cause",
        ]
        st.dataframe(
            top[display_cols].sort_values("peak_severity", ascending=False),
            width='stretch', hide_index=True,
        )

# --- TAB 5: QUERY EXPLORER ---
with tab_explorer:
    st.subheader("Inspect individual flagged queries")
    st.caption("Filtered by the sidebar. Sorted by risk score, most severe first.")

    flagged = df[df["predicted_anomaly"]].sort_values("risk_score", ascending=False)
    display_cols = [
        "timestamp", "query_type", "table_name", "execution_time_ms",
        "rows_scanned", "rows_returned", "cpu_pct_at_time",
        "memory_pct_at_time", "lock_wait_count_at_time",
        "risk_score", "root_cause_hint", "is_anomaly", "anomaly_type",
    ]
    st.dataframe(
        flagged[display_cols].head(500),
        width='stretch', hide_index=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk score", min_value=0, max_value=float(flagged["risk_score"].max() or 1),
            ),
        },
    )
    st.caption(f"Showing top 500 of {len(flagged):,} flagged queries matching filters.")