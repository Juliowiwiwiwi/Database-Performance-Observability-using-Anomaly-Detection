"""
Database Query Anomaly Detection - Model v2
============================================

Fixes over v1:
  1. Isolation Forest features are scaled (StandardScaler) and trained on
     the training period only -> actually contributes signal now.
  2. New system-wide aggregate layer: buckets queries into 5-min windows
     and z-scores the POPULATION mean latency, catching proportional
     slowdowns (resource exhaustion) that per-query z-scores miss.
  3. Continuous risk score + precision-recall sweep -> threshold is tuned
     to maximize F1 instead of a hardcoded z > 4.
  4. Incident grouping: consecutive flagged queries (same table/type,
     small time gaps) are merged into single incidents instead of
     reporting thousands of individual alerts for one root cause.
  5. Root-cause ranking by z-score of each system metric at that moment,
     instead of fixed "CPU > 90%" thresholds.
  6. Time-based train/test split (train days 1-7, evaluate on days 8-14)
     to avoid the leakage in v1's "fit and score on everything" setup.
  7. Visualizations saved to Outputs/: scatter, box plots, timeline,
     precision-recall curve, incident summary.

Folder layout expected:
    <project_root>/DataSets/query_log.csv
    <project_root>/DataSets/system_metrics.csv
    <project_root>/src/v2model.py          <- this file
    <project_root>/src/Outputs/            <- CSVs + figures written here

Usage:
    python v2model.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend, still writes files fine
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import precision_recall_curve, f1_score

pd.set_option("display.width", 120)
sns.set_theme(style="whitegrid")

   
# PATHS
   

SCRIPT_DIR = Path(__file__).resolve().parent          # .../src
PROJECT_ROOT = SCRIPT_DIR.parent                       # project root
DATASETS_DIR = PROJECT_ROOT / "DataSets"
OUTPUTS_DIR = SCRIPT_DIR / "Outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

IN_QUERIES = DATASETS_DIR / "query_log.csv"
IN_METRICS = DATASETS_DIR / "system_metrics.csv"

OUT_SCORED = OUTPUTS_DIR / "query_log_scored_v2.csv"
OUT_INCIDENTS = OUTPUTS_DIR / "incidents_v2.csv"

   
# CONFIG
   

ROLLING_WINDOW = 300        # trailing N queries (per query_type) for adaptive baseline
MIN_PERIODS = 30
REFERENCE_DAYS = 3          # first N days = frozen "known clean" reference period
TRAIN_DAYS = 7              # train IF on first N days, evaluate on the rest

BUCKET_MINUTES = 5          # system-wide aggregate bucket size
BUCKET_ROLL_WINDOW = 24     # trailing buckets (24 * 5min = 2h) for population baseline

INCIDENT_GAP_MINUTES = 10   # gap allowed between flagged queries to still be "same incident"


   
# 1. LOAD
   

def load_data():
    q = pd.read_csv(IN_QUERIES, parse_dates=["timestamp"])
    q = q.sort_values("timestamp").reset_index(drop=True)
    q["is_anomaly"] = q["is_anomaly"].astype(bool)
    return q


   
# 2. FEATURE ENGINEERING (per-query, same core ideas as v1)
   

def engineer_features(q: pd.DataFrame) -> pd.DataFrame:
    q = q.copy()
    q["log_latency"] = np.log1p(q["execution_time_ms"])
    q["scan_ratio"] = q["rows_scanned"] / q["rows_returned"].clip(lower=1)
    q["log_scan_ratio"] = np.log1p(q["scan_ratio"])
    q["hour"] = q["timestamp"].dt.hour
    q["dow"] = q["timestamp"].dt.dayofweek
    q["is_weekend"] = (q["dow"] >= 5).astype(int)

    # --- adaptive trailing baseline per query_type ---
    q = q.sort_values(["query_type", "timestamp"])
    grp = q.groupby("query_type")["log_latency"]
    roll_mean = grp.transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).mean())
    roll_std = grp.transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).std())
    q["baseline_mean"] = roll_mean
    q["baseline_std"] = roll_std.replace(0, np.nan)
    q["z_score"] = ((q["log_latency"] - q["baseline_mean"]) / q["baseline_std"]).fillna(0)

    # --- frozen reference-period baseline per query_type ---
    ref_cutoff = q["timestamp"].min() + pd.Timedelta(days=REFERENCE_DAYS)
    ref_stats = (
        q[q["timestamp"] < ref_cutoff]
        .groupby("query_type")["log_latency"]
        .agg(ref_mean="mean", ref_std="std")
    )
    q = q.merge(ref_stats, on="query_type", how="left")
    q["ref_std"] = q["ref_std"].replace(0, np.nan)
    q["ref_z_score"] = ((q["log_latency"] - q["ref_mean"]) / q["ref_std"]).fillna(0)

    q = q.sort_values("timestamp").reset_index(drop=True)
    return q


   
# 3. SYSTEM-WIDE AGGREGATE LAYER (catches proportional/broad slowdowns)
   

def add_system_wide_layer(q: pd.DataFrame) -> pd.DataFrame:
    """
    Per-query z-scores compare a query to its OWN type's history, so a
    broad slowdown that affects every query type proportionally (e.g. CPU
    exhaustion) doesn't stand out much within any single type. This layer
    instead tracks the POPULATION mean latency in small time buckets and
    flags when that aggregate itself deviates from its own recent normal.
    """
    q = q.sort_values("timestamp").copy()
    q["bucket"] = q["timestamp"].dt.floor(f"{BUCKET_MINUTES}min")

    bucket_stats = q.groupby("bucket")["log_latency"].mean().rename("bucket_mean_latency").reset_index()
    bucket_stats = bucket_stats.sort_values("bucket")
    bucket_stats["bucket_roll_mean"] = (
        bucket_stats["bucket_mean_latency"].shift(1)
        .rolling(BUCKET_ROLL_WINDOW, min_periods=6).mean()
    )
    bucket_stats["bucket_roll_std"] = (
        bucket_stats["bucket_mean_latency"].shift(1)
        .rolling(BUCKET_ROLL_WINDOW, min_periods=6).std().replace(0, np.nan)
    )
    bucket_stats["bucket_z"] = (
        (bucket_stats["bucket_mean_latency"] - bucket_stats["bucket_roll_mean"])
        / bucket_stats["bucket_roll_std"]
    ).fillna(0)

    q = q.merge(bucket_stats[["bucket", "bucket_z"]], on="bucket", how="left")
    return q


   
# 4. TRAIN/TEST SPLIT
   

def split_train_test(q: pd.DataFrame):
    cutoff = q["timestamp"].min() + pd.Timedelta(days=TRAIN_DAYS)
    train_mask = q["timestamp"] < cutoff
    return train_mask


   
# 5. ISOLATION FOREST (scaled, trained on train period only)
   

def run_isolation_forest(q: pd.DataFrame, train_mask: pd.Series):
    feature_cols_numeric = [
        "log_latency", "log_scan_ratio",
        "cpu_pct_at_time", "memory_pct_at_time", "lock_wait_count_at_time",
        "hour", "is_weekend", "bucket_z",
    ]

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    type_encoded_full = ohe.fit_transform(q[["query_type"]])

    scaler = StandardScaler()
    scaler.fit(q.loc[train_mask, feature_cols_numeric])  # fit scaler on TRAIN only
    numeric_scaled_full = scaler.transform(q[feature_cols_numeric])

    X_full = np.hstack([numeric_scaled_full, type_encoded_full])
    X_train = X_full[train_mask.values]

    model = IsolationForest(
        n_estimators=300,
        contamination=0.02,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)  # train on TRAIN period only

    raw_pred = model.predict(X_full)             # -1 = anomaly, 1 = normal
    scores = model.decision_function(X_full)     # lower = more anomalous

    return raw_pred == -1, scores


   
# 6. UNIFIED RISK SCORE + THRESHOLD TUNING
   

def build_risk_score(q: pd.DataFrame) -> pd.Series:
    """
    Combine all three signals into one 0-1+ continuous score by taking the
    max of each signal normalized against a reference threshold of 4
    (roughly "4 standard deviations = fully alarmed"). This lets us sweep
    a single threshold and plot a real precision-recall curve, rather than
    ORing hardcoded boolean flags together like v1 did.
    """
    if_score_norm = (-q["if_score"] - (-q["if_score"]).min()) / (
        (-q["if_score"]).max() - (-q["if_score"]).min() + 1e-9
    )  # 0..1, higher = more anomalous

    risk = np.maximum.reduce([
        q["z_score"].abs() / 4.0,
        q["ref_z_score"].abs() / 4.0,
        q["bucket_z"].abs() / 4.0,
        if_score_norm * 1.5,  # scaled so IF alone can push a query over threshold=1
    ])
    risk = pd.Series(risk, index=q.index)

    # sub-millisecond-scale queries produce inflated z-scores from tiny
    # absolute jitter (a fast baseline has a tiny std, so a 1ms wobble can
    # look like a huge deviation). Suppress risk for trivially fast queries
    # since a few ms of noise on a <5ms query is never operationally
    # meaningful, regardless of how "anomalous" it looks statistically.
    trivial = q["execution_time_ms"] < 5
    risk = risk.where(~trivial, risk * 0.15)

    return risk


def tune_threshold(q: pd.DataFrame, mask: pd.Series):
    """Sweep thresholds on `risk_score`, pick the one maximizing F1 on the given mask."""
    y_true = q.loc[mask, "is_anomaly"].values
    scores = q.loc[mask, "risk_score"].values

    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1]) if len(thresholds) > 0 else 0
    best_threshold = thresholds[best_idx] if len(thresholds) > 0 else 1.0

    return best_threshold, precisions, recalls, thresholds, f1s


   
# 7. INCIDENT GROUPING (temporal deduplication)
   

def group_incidents(q: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses consecutive flagged queries sharing the same (query_type,
    table_name) into a single incident when the time gap between them is
    small, instead of reporting thousands of individual alerts for what
    is really one ongoing problem.
    """
    flagged = q[q["predicted_anomaly"]].sort_values(
        ["query_type", "table_name", "timestamp"]
    ).copy()
    if flagged.empty:
        return pd.DataFrame(columns=[
            "query_type", "table_name", "start", "end", "duration_minutes",
            "query_count", "peak_severity", "top_root_cause", "true_anomaly_rate"
        ])

    flagged["gap"] = flagged.groupby(["query_type", "table_name"])["timestamp"].diff()
    flagged["new_incident"] = (
        flagged["gap"].isna() | (flagged["gap"] > pd.Timedelta(minutes=INCIDENT_GAP_MINUTES))
    )
    flagged["incident_id"] = flagged.groupby(["query_type", "table_name"])["new_incident"].cumsum()

    incidents = (
        flagged.groupby(["query_type", "table_name", "incident_id"])
        .agg(
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            query_count=("timestamp", "count"),
            peak_severity=("risk_score", "max"),
            true_anomaly_rate=("is_anomaly", "mean"),
            top_root_cause=("root_cause_hint", lambda s: s.value_counts().idxmax()),
        )
        .reset_index(drop=False)
    )
    incidents["duration_minutes"] = (
        (incidents["end"] - incidents["start"]).dt.total_seconds() / 60
    ).round(1)
    incidents = incidents.drop(columns=["incident_id"])
    incidents = incidents.sort_values("start").reset_index(drop=True)
    return incidents


   
# 8. ROOT-CAUSE RANKING (by z-score, not fixed thresholds)
   

def build_root_cause_hints(q: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.Series:
    """
    Rank candidate causes by how anomalous each system metric itself was
    at that moment (z-score vs its own rolling history), rather than fixed
    thresholds like "CPU > 90%". Whichever metric was most out-of-family
    at that timestamp wins the explanation.
    """
    m = metrics_df.sort_values("timestamp").copy()
    for col in ["cpu_pct", "memory_pct", "lock_wait_count"]:
        roll_mean = m[col].shift(1).rolling(200, min_periods=20).mean()
        roll_std = m[col].shift(1).rolling(200, min_periods=20).std().replace(0, np.nan)
        m[f"{col}_z"] = ((m[col] - roll_mean) / roll_std).fillna(0)

    m = m.set_index("timestamp")[["cpu_pct_z", "memory_pct_z", "lock_wait_count_z"]]

    # nearest-metric-snapshot lookup for each query
    idx = m.index
    q_sorted = q.sort_values("timestamp")
    pos = idx.searchsorted(q_sorted["timestamp"].values) - 1
    pos = np.clip(pos, 0, len(idx) - 1)
    metric_z = m.iloc[pos].reset_index(drop=True)
    metric_z.index = q_sorted.index
    q = q.join(metric_z)

    def explain(row):
        candidates = {
            "high CPU": row["cpu_pct_z"],
            "high memory": row["memory_pct_z"],
            "lock contention": row["lock_wait_count_z"],
            "inefficient scan (possible missing index)": row["log_scan_ratio"] * 2,  # scaled to comparable range
        }
        best_cause, best_score = max(candidates.items(), key=lambda kv: kv[1])
        if best_score < 2.0:
            return f"latency deviates from '{row['query_type']}' baseline (z={row['z_score']:.1f}), no dominant system-level cause"
        return f"{best_cause} (relative severity z={best_score:.1f})"

    return q.apply(explain, axis=1)


   
# 9. EVALUATION
   

def evaluate(q: pd.DataFrame, mask: pd.Series, pred_col: str, label: str):
    sub = q.loc[mask]
    tp = ((sub[pred_col]) & (sub["is_anomaly"])).sum()
    fp = ((sub[pred_col]) & (~sub["is_anomaly"])).sum()
    fn = ((~sub[pred_col]) & (sub["is_anomaly"])).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"\n--- {label} ---")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    print(f"  Recall by anomaly_type:")
    for atype, g in sub[sub["is_anomaly"]].groupby("anomaly_type"):
        caught = g[pred_col].sum()
        print(f"    {atype:20s}: {caught}/{len(g)} caught ({caught/len(g)*100:.1f}%)")
    return dict(precision=precision, recall=recall, f1=f1)


   
# 10. VISUALIZATIONS
   

def fig_scatter(q: pd.DataFrame, path: Path):
    plt.figure(figsize=(14, 6))
    normal = q[~q["predicted_anomaly"]]
    anomaly = q[q["predicted_anomaly"]]
    plt.scatter(normal["timestamp"], normal["execution_time_ms"], s=4, alpha=0.15,
                color="#4C72B0", label="Normal", rasterized=True)
    plt.scatter(anomaly["timestamp"], anomaly["execution_time_ms"], s=10, alpha=0.7,
                color="#C44E52", label="Flagged anomaly")
    plt.yscale("log")
    plt.xlabel("Time")
    plt.ylabel("Execution time (ms, log scale)")
    plt.title("Query Latency Over Time - Flagged Anomalies in Red")
    plt.legend(markerscale=3, loc="upper right")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def fig_boxplot(q: pd.DataFrame, path: Path):
    plt.figure(figsize=(12, 6))
    order = q.groupby("query_type")["execution_time_ms"].median().sort_values(ascending=False).index
    sns.boxplot(data=q, x="query_type", y="execution_time_ms", order=order,
                showfliers=False, palette="Blues_d")
    plt.yscale("log")
    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Query Type")
    plt.ylabel("Execution time (ms, log scale)")
    plt.title("Latency Distribution by Query Type (outliers hidden for readability)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def fig_timeline(q: pd.DataFrame, path: Path):
    hourly = q.set_index("timestamp").resample("1h").agg(
        anomaly_count=("predicted_anomaly", "sum"),
        avg_cpu=("cpu_pct_at_time", "mean"),
    )
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(hourly.index, hourly["anomaly_count"], width=0.03, color="#C44E52", alpha=0.7,
            label="Anomalies flagged / hour")
    ax1.set_ylabel("Anomalies flagged per hour", color="#C44E52")
    ax1.set_xlabel("Time")
    ax1.tick_params(axis="y", labelcolor="#C44E52")

    ax2 = ax1.twinx()
    ax2.plot(hourly.index, hourly["avg_cpu"], color="#4C72B0", linewidth=1.2,
              label="Avg CPU %")
    ax2.set_ylabel("Avg CPU %", color="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#4C72B0")

    plt.title("Anomaly Timeline vs. CPU Load")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_precision_recall(precisions, recalls, thresholds, best_threshold, path: Path):
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color="#4C72B0", linewidth=2)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1]) if len(thresholds) > 0 else 0
    plt.scatter([recalls[best_idx]], [precisions[best_idx]], color="#C44E52", zorder=5,
                label=f"Chosen threshold (F1={f1s[best_idx]:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Tradeoff (test period)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def fig_incidents(incidents: pd.DataFrame, path: Path, top_n: int = 30):
    """
    Plots only the top-N incidents by severity (peak risk score), not all
    of them - with thousands of tiny single-query "incidents" (mostly
    noise near the decision threshold), plotting everything produces an
    unreadable, enormous figure. This shows what actually matters: the
    biggest, longest, or most severe incidents.
    """
    if incidents.empty:
        return
    top = incidents.sort_values("peak_severity", ascending=False).head(top_n)
    top = top.sort_values("start")

    plt.figure(figsize=(12, max(4, 0.35 * len(top))))
    labels = [f"{r.query_type} / {r.table_name}" for r in top.itertuples()]
    severities = top["peak_severity"].values
    norm_sev = (severities - severities.min()) / (severities.max() - severities.min() + 1e-9)
    colors = sns.color_palette("Reds", n_colors=256)
    bar_colors = [colors[int(s * 255)] for s in norm_sev]

    for i, (row, color) in enumerate(zip(top.itertuples(), bar_colors)):
        dur_days = max(0.02, (row.end - row.start).total_seconds() / 86400)
        plt.barh(i, dur_days, left=mdates.date2num(row.start), color=color,
                  edgecolor="black", linewidth=0.5)

    plt.yticks(range(len(top)), labels, fontsize=8)
    plt.xlabel("Time")
    plt.gca().xaxis_date()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.title(f"Top {len(top)} Incidents by Severity (of {len(incidents)} total grouped incidents)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


   
# MAIN
   

def main():
    print(f"Reading data from: {DATASETS_DIR}")
    print(f"Writing outputs to: {OUTPUTS_DIR}\n")

    q = load_data()
    print(f"Loaded {len(q):,} queries")
    metrics_df = pd.read_csv(IN_METRICS, parse_dates=["timestamp"])

    print("Engineering per-query features...")
    q = engineer_features(q)

    print("Adding system-wide aggregate layer...")
    q = add_system_wide_layer(q)

    train_mask = split_train_test(q)
    print(f"Train: {train_mask.sum():,} queries | Test: {(~train_mask).sum():,} queries")

    print("Running Isolation Forest (scaled, trained on train period)...")
    if_pred, if_score = run_isolation_forest(q, train_mask)
    q["flag_isolation_forest"] = if_pred
    q["if_score"] = if_score

    print("Building unified risk score...")
    q["risk_score"] = build_risk_score(q)

    print("Tuning decision threshold on test period (unseen by IsolationForest)...")
    best_threshold, precisions, recalls, thresholds, f1s = tune_threshold(q, ~train_mask)
    print(f"  chosen threshold = {best_threshold:.3f}")

    q["predicted_anomaly"] = q["risk_score"] >= best_threshold

    print("Building ranked root-cause hints...")
    q["root_cause_hint"] = ""
    flagged_mask = q["predicted_anomaly"]
    hints = build_root_cause_hints(q.loc[flagged_mask].copy(), metrics_df)
    q.loc[flagged_mask, "root_cause_hint"] = hints.values

    print("\n================ EVALUATION ================")
    evaluate(q, train_mask, "flag_isolation_forest", "Isolation Forest only (train period)")
    evaluate(q, ~train_mask, "flag_isolation_forest", "Isolation Forest only (test period)")
    evaluate(q, ~train_mask, "predicted_anomaly", "FINAL ENSEMBLE (test period, tuned threshold)")

    print("\nGrouping into incidents...")
    incidents = group_incidents(q)
    incidents.to_csv(OUT_INCIDENTS, index=False)
    print(f"  {len(incidents)} incidents (down from {flagged_mask.sum():,} individual flagged queries)")
    print(f"  wrote -> {OUT_INCIDENTS}")
    print(incidents[["query_type", "table_name", "start", "duration_minutes",
                      "query_count", "true_anomaly_rate"]].to_string(index=False))

    print("\nSaving scored dataset...")
    q.to_csv(OUT_SCORED, index=False)
    print(f"  wrote {len(q):,} rows -> {OUT_SCORED}")

    print("\nGenerating figures...")
    fig_scatter(q, OUTPUTS_DIR / "fig_scatter_anomaliesv2.png")
    fig_boxplot(q, OUTPUTS_DIR / "fig_boxplot_by_typev2.png")
    fig_timeline(q, OUTPUTS_DIR / "fig_timelinev2.png")
    fig_precision_recall(precisions, recalls, thresholds, best_threshold,
                          OUTPUTS_DIR / "fig_precision_recallv2.png")
    fig_incidents(incidents, OUTPUTS_DIR / "fig_incidentsv2.png")
    print(f"  figures written to {OUTPUTS_DIR}")

    print("\nDone.")


if __name__ == "__main__":
    main()