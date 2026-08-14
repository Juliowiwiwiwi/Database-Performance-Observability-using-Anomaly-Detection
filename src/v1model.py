"""
Database Query Anomaly Detection - Model v1
============================================

Two-layer detector:
  1. Rolling per-query-type z-score baseline (explainable, adapts over time)
  2. Isolation Forest on multivariate features (catches anomalies a single
     z-score would miss, e.g. normal latency but abnormal CPU correlation)

A query is flagged FINAL ANOMALY if either layer fires. Evaluated against
the injected ground truth (is_anomaly / anomaly_type) from the generator.

Usage:
    python detect_anomalies.py
Inputs:
    query_log.csv, system_metrics.csv (from generate_dataset.py)
Outputs:
    query_log_scored.csv  - original data + scores + predictions
    prints evaluation metrics (precision/recall/F1, per-incident recall)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder

pd.set_option("display.width", 120)

IN_QUERIES = "DataSets/query_log.csv"
IN_METRICS = "DataSets/system_metrics.csv"
OUT_SCORED = "src/Outputs/query_log_scored.csv"

ROLLING_WINDOW = 300     # trailing N queries (per query_type) for adaptive baseline
Z_THRESHOLD = 4.0        # z-score above this -> flagged by baseline layer
IF_CONTAMINATION = 0.02  # expected anomaly rate, matches generator's ~2%
MIN_PERIODS = 30         # need at least this many prior queries to trust the baseline


  
# 1. LOAD
  

def load_data():
    q = pd.read_csv(IN_QUERIES, parse_dates=["timestamp"])
    q = q.sort_values("timestamp").reset_index(drop=True)
    return q


  
# 2. FEATURE ENGINEERING
  

def engineer_features(q: pd.DataFrame) -> pd.DataFrame:
    q = q.copy()

    # log-transform latency: raw latency is heavily right-skewed, which
    # distorts both z-scores and Isolation Forest's distance geometry
    q["log_latency"] = np.log1p(q["execution_time_ms"])

    # scan efficiency: how many rows scanned per row actually returned.
    # a sudden jump here is the classic "missing index" fingerprint
    q["scan_ratio"] = q["rows_scanned"] / q["rows_returned"].clip(lower=1)
    q["log_scan_ratio"] = np.log1p(q["scan_ratio"])

    # time-of-day / day-of-week, so the model doesn't confuse "expected
    # peak-hour slowness" with an anomaly
    q["hour"] = q["timestamp"].dt.hour
    q["dow"] = q["timestamp"].dt.dayofweek
    q["is_weekend"] = (q["dow"] >= 5).astype(int)

    # --- rolling per-query-type baseline (adaptive, explainable layer) ---
    # shift(1) so a query is never compared against itself / future queries
    # (no leakage) - this also naturally "adapts as workload changes" since
    # it's a trailing window, not a fixed historical average
    q = q.sort_values(["query_type", "timestamp"])
    grp = q.groupby("query_type")["log_latency"]

    roll_mean = grp.transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).mean()
    )
    roll_std = grp.transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).std()
    )

    q["baseline_mean"] = roll_mean
    q["baseline_std"] = roll_std.replace(0, np.nan)
    q["z_score"] = (q["log_latency"] - q["baseline_mean"]) / q["baseline_std"]

    # cold-start rows (not enough history yet) get z_score = 0 (can't judge them)
    q["z_score"] = q["z_score"].fillna(0)

    # --- frozen reference-period baseline ---
    # the trailing rolling window is great for adapting to gradual workload
    # shifts, but it has a blind spot: if a regression is PERSISTENT (e.g. a
    # missing index that never gets fixed), the rolling window eventually
    # "learns" the slow behavior as the new normal and stops flagging it.
    # to catch sustained regime shifts, also compare against a baseline
    # frozen from an early, presumed-clean reference period.
    ref_cutoff = q["timestamp"].min() + pd.Timedelta(days=3)
    ref_stats = (
        q[q["timestamp"] < ref_cutoff]
        .groupby("query_type")["log_latency"]
        .agg(ref_mean="mean", ref_std="std")
    )
    q = q.merge(ref_stats, on="query_type", how="left")
    q["ref_std"] = q["ref_std"].replace(0, np.nan)
    q["ref_z_score"] = (q["log_latency"] - q["ref_mean"]) / q["ref_std"]
    q["ref_z_score"] = q["ref_z_score"].fillna(0)

    q = q.sort_values("timestamp").reset_index(drop=True)
    return q


  
# 3. LAYER 1: ROLLING Z-SCORE BASELINE
  

def flag_zscore_layer(q: pd.DataFrame) -> pd.Series:
    # flagged if EITHER the adaptive trailing baseline OR the frozen
    # reference baseline says this query is unusual
    return (q["z_score"].abs() > Z_THRESHOLD) | (q["ref_z_score"].abs() > Z_THRESHOLD)


  
# 4. LAYER 2: ISOLATION FOREST (multivariate)
  

def flag_isolation_forest(q: pd.DataFrame):
    feature_cols_numeric = [
        "log_latency", "log_scan_ratio",
        "cpu_pct_at_time", "memory_pct_at_time", "lock_wait_count_at_time",
        "hour", "is_weekend",
    ]

    # one-hot encode query_type so the model can learn "normal" separately
    # per query shape (a JOIN and a SELECT are naturally different scales)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    type_encoded = ohe.fit_transform(q[["query_type"]])
    type_cols = [f"type_{c}" for c in ohe.categories_[0]]

    X = np.hstack([q[feature_cols_numeric].values, type_encoded])

    model = IsolationForest(
        n_estimators=200,
        contamination=IF_CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    raw_pred = model.predict(X)          # -1 = anomaly, 1 = normal
    scores = model.decision_function(X)  # lower = more anomalous

    is_anomaly = raw_pred == -1
    return is_anomaly, scores, model


  
# 5. ROOT-CAUSE HINT (simple rule-based explanation per flagged row)
  

def explain_row(row) -> str:
    reasons = []
    if row["cpu_pct_at_time"] > 90:
        reasons.append(f"high CPU ({row['cpu_pct_at_time']:.0f}%)")
    if row["memory_pct_at_time"] > 90:
        reasons.append(f"high memory ({row['memory_pct_at_time']:.0f}%)")
    if row["lock_wait_count_at_time"] > 10:
        reasons.append(f"lock contention (waits={row['lock_wait_count_at_time']:.1f})")
    if row["scan_ratio"] > 50:
        reasons.append(f"inefficient scan (scanned {row['scan_ratio']:.0f}x rows returned -> possible missing index)")
    if not reasons:
        reasons.append(f"latency deviates from '{row['query_type']}' baseline (z={row['z_score']:.1f}) with no clear system-level cause")
    return "; ".join(reasons)


  
# 6. EVALUATION AGAINST GROUND TRUTH
  

def evaluate(q: pd.DataFrame, pred_col: str, label_col: str = "is_anomaly"):
    tp = ((q[pred_col]) & (q[label_col])).sum()
    fp = ((q[pred_col]) & (~q[label_col])).sum()
    fn = ((~q[pred_col]) & (q[label_col])).sum()
    tn = ((~q[pred_col]) & (~q[label_col])).sum()

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"\n--- {pred_col} vs ground truth ---")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")

    # recall broken down by injected incident type - the most useful
    # slide for a manager ("we catch 95% of missing-index regressions")
    print(f"\n  Recall by anomaly_type:")
    for atype, sub in q[q[label_col]].groupby("anomaly_type"):
        caught = sub[pred_col].sum()
        print(f"    {atype:20s}: {caught}/{len(sub)} caught ({caught/len(sub)*100:.1f}%)")

    return dict(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


  
# MAIN
  

def main():
    print("Loading data...")
    q = load_data()
    print(f"  {len(q):,} queries loaded")

    print("Engineering features...")
    q = engineer_features(q)

    print("Running Layer 1: rolling z-score baseline...")
    q["flag_zscore"] = flag_zscore_layer(q)

    print("Running Layer 2: Isolation Forest...")
    is_anom_if, if_scores, model = flag_isolation_forest(q)
    q["flag_isolation_forest"] = is_anom_if
    q["if_anomaly_score"] = if_scores

    # final ensemble: flagged by either layer
    q["predicted_anomaly"] = q["flag_zscore"] | q["flag_isolation_forest"]

    # ground truth column comes in as string "True"/"False" from CSV -> ensure bool
    q["is_anomaly"] = q["is_anomaly"].astype(bool)

    # root-cause explanation for anything flagged
    q["root_cause_hint"] = ""
    flagged_mask = q["predicted_anomaly"]
    q.loc[flagged_mask, "root_cause_hint"] = q.loc[flagged_mask].apply(explain_row, axis=1)

    # --- evaluation ---
    evaluate(q, "flag_zscore")
    evaluate(q, "flag_isolation_forest")
    results = evaluate(q, "predicted_anomaly")

    print("\nSaving scored dataset...")
    q.to_csv(OUT_SCORED, index=False)
    print(f"  wrote {len(q):,} rows -> {OUT_SCORED}")

    print("\nSample flagged anomalies with root-cause hints:")
    sample = q[q["predicted_anomaly"]].sample(min(5, flagged_mask.sum()), random_state=1)
    for _, row in sample.iterrows():
        print(f"  [{row['timestamp']}] {row['query_type']} on {row['table_name']}: "
              f"{row['execution_time_ms']:.0f}ms -> {row['root_cause_hint']}")

    return q, results


if __name__ == "__main__":
    main()