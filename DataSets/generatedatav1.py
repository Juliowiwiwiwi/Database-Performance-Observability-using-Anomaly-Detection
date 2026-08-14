"""
Synthetic Database Performance Dataset Generator
=================================================

Generates two correlated tables for a database observability / anomaly
detection project:

1. system_metrics.csv - time series of CPU, memory, locks, connections, etc.
2. query_log.csv       - individual query executions, with latency that is a
                          function of query complexity + concurrent system
                          load + injected anomalies.

Ground truth columns (is_anomaly, anomaly_type) are included so you can
evaluate precision/recall of your detection model later. Just remember not
to feed those two columns into your model as features - they're for
evaluation only.

Usage:
    python generate_dataset.py
Outputs:
    system_metrics.csv
    query_log.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

 
# CONFIG
 

SEED = 42
rng = np.random.default_rng(SEED)

START = datetime(2026, 1, 5, 0, 0, 0)   # a Monday
DAYS = 14
METRIC_INTERVAL_SEC = 30                # system metrics sampled every 30s
AVG_QUERIES_PER_MIN_BASE = 40           # avg query rate at low-traffic time

OUT_METRICS = "system_metrics.csv"
OUT_QUERIES = "query_log.csv"

# Query templates: name -> (base_latency_ms_mean, base_latency_ms_std,
#                            base_rows_scanned_mean, selectivity)
# selectivity = rows_returned / rows_scanned, roughly
QUERY_TEMPLATES = {
    "SELECT_simple":      dict(base_ms=8,   std_ms=3,   rows_scanned=200,   selectivity=0.6),
    "SELECT_filtered":    dict(base_ms=15,  std_ms=6,   rows_scanned=1500,  selectivity=0.05),
    "JOIN_2table":        dict(base_ms=45,  std_ms=15,  rows_scanned=5000,  selectivity=0.02),
    "JOIN_multi":         dict(base_ms=120, std_ms=40,  rows_scanned=20000, selectivity=0.01),
    "AGGREGATE_groupby":  dict(base_ms=200, std_ms=70,  rows_scanned=50000, selectivity=0.005),
    "AGGREGATE_report":   dict(base_ms=800, std_ms=250, rows_scanned=200000, selectivity=0.001),
    "UPDATE_row":         dict(base_ms=10,  std_ms=4,   rows_scanned=1,     selectivity=1.0),
    "DELETE_batch":       dict(base_ms=60,  std_ms=20,  rows_scanned=3000,  selectivity=1.0),
}
TEMPLATE_NAMES = list(QUERY_TEMPLATES.keys())
# relative frequency each template is called (SELECTs much more common)
TEMPLATE_WEIGHTS = np.array([30, 25, 15, 8, 10, 3, 6, 3], dtype=float)
TEMPLATE_WEIGHTS /= TEMPLATE_WEIGHTS.sum()

TABLES = ["users", "orders", "order_items", "payments", "products",
          "sessions", "inventory", "shipments"]

 
# 1. SYSTEM METRICS TIME SERIES
 

def generate_system_metrics():
    n_points = int(DAYS * 24 * 3600 / METRIC_INTERVAL_SEC)
    timestamps = [START + timedelta(seconds=i * METRIC_INTERVAL_SEC) for i in range(n_points)]

    cpu = np.zeros(n_points)
    mem = np.zeros(n_points)
    lock_waits = np.zeros(n_points)
    connections = np.zeros(n_points)
    disk_io = np.zeros(n_points)
    cache_hit = np.zeros(n_points)

    # anomaly injection windows (defined once, reused by query generator too)
    anomaly_windows = build_anomaly_windows()

    for i, ts in enumerate(timestamps):
        hour = ts.hour + ts.minute / 60
        dow = ts.weekday()  # 0=Mon .. 6=Sun

        # --- daily/weekly seasonality (business hours load) ---
        # peak around 10am-6pm on weekdays, quiet at night/weekends
        daily_curve = np.exp(-((hour - 14) ** 2) / (2 * 5 ** 2))  # gaussian bump centered 2pm
        weekend_factor = 0.35 if dow >= 5 else 1.0
        load_factor = daily_curve * weekend_factor  # 0..1

        base_cpu = 15 + 55 * load_factor
        base_mem = 40 + 30 * load_factor
        base_conn = 10 + 90 * load_factor
        base_lock = 0.5 + 3 * load_factor
        base_io = 50 + 200 * load_factor
        base_cache = 0.97 - 0.05 * load_factor

        # --- noise ---
        cpu[i] = base_cpu + rng.normal(0, 3)
        mem[i] = base_mem + rng.normal(0, 2)
        connections[i] = max(0, base_conn + rng.normal(0, 5))
        lock_waits[i] = max(0, base_lock + rng.normal(0, 0.5))
        disk_io[i] = max(0, base_io + rng.normal(0, 15))
        cache_hit[i] = np.clip(base_cache + rng.normal(0, 0.01), 0.7, 0.999)

        # --- apply anomaly windows that affect system metrics ---
        for win in anomaly_windows:
            if win["start"] <= ts <= win["end"]:
                if win["type"] == "resource_exhaustion":
                    cpu[i] = min(99, cpu[i] + win["intensity"] * 40)
                    mem[i] = min(98, mem[i] + win["intensity"] * 25)
                elif win["type"] == "lock_contention":
                    lock_waits[i] += win["intensity"] * 25
                    connections[i] += win["intensity"] * 20

        cpu[i] = np.clip(cpu[i], 1, 99)
        mem[i] = np.clip(mem[i], 5, 98)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "cpu_pct": cpu.round(2),
        "memory_pct": mem.round(2),
        "active_connections": connections.round(0).astype(int),
        "lock_wait_count": lock_waits.round(2),
        "disk_io_mbps": disk_io.round(2),
        "cache_hit_ratio": cache_hit.round(4),
    })
    return df, anomaly_windows


 
# 2. ANOMALY WINDOW DEFINITIONS (ground truth events)
 

def build_anomaly_windows():
    """
    Defines specific injected incidents across the 14-day period.
    Each window has a type, a time range, an intensity (0-1), and for
    query-level anomalies, which template(s)/table(s) are affected.
    """
    windows = []

    # 1) Missing index incident: JOIN_2table on 'orders' suddenly gets much
    #    slower starting day 4, and stays slow (regression, not transient)
    windows.append(dict(
        type="missing_index",
        start=START + timedelta(days=4, hours=9),
        end=START + timedelta(days=14),  # persists till end of dataset
        intensity=1.0,
        template="JOIN_2table",
        table="orders",
    ))

    # 2) Lock contention incident: batch job holds locks, blocking UPDATE_row
    #    and SELECT_filtered queries for ~45 min on day 6
    windows.append(dict(
        type="lock_contention",
        start=START + timedelta(days=6, hours=2, minutes=0),
        end=START + timedelta(days=6, hours=2, minutes=45),
        intensity=1.0,
        template=None,  # affects any query hitting the locked table
        table="payments",
    ))

    # 3) Resource exhaustion: CPU/memory spike (e.g. bad deploy, runaway
    #    process) on day 9 afternoon, degrades ALL query types broadly
    windows.append(dict(
        type="resource_exhaustion",
        start=START + timedelta(days=9, hours=13, minutes=0),
        end=START + timedelta(days=9, hours=14, minutes=30),
        intensity=1.0,
        template=None,
        table=None,
    ))

    # 4) Second, smaller lock contention blip on day 11 (shorter, tests
    #    detection sensitivity)
    windows.append(dict(
        type="lock_contention",
        start=START + timedelta(days=11, hours=16, minutes=10),
        end=START + timedelta(days=11, hours=16, minutes=25),
        intensity=0.7,
        template=None,
        table="inventory",
    ))

    return windows


def in_window(ts, windows, wtype=None):
    for w in windows:
        if w["start"] <= ts <= w["end"]:
            if wtype is None or w["type"] == wtype:
                return w
    return None


 
# 3. QUERY LOG GENERATION
 

def generate_query_log(metrics_df, anomaly_windows):
    metrics_df = metrics_df.set_index("timestamp")
    records = []
    query_id_counter = 1

    ts = START
    end = START + timedelta(days=DAYS)

    while ts < end:
        hour = ts.hour + ts.minute / 60
        dow = ts.weekday()
        daily_curve = np.exp(-((hour - 14) ** 2) / (2 * 5 ** 2))
        weekend_factor = 0.35 if dow >= 5 else 1.0
        load_factor = daily_curve * weekend_factor

        queries_this_minute = rng.poisson(AVG_QUERIES_PER_MIN_BASE * (0.3 + 1.7 * load_factor))

        # look up nearest system metric snapshot
        nearest_metric_ts = metrics_df.index.asof(ts)
        if pd.isna(nearest_metric_ts):
            nearest_metric_ts = metrics_df.index[0]
        cpu = metrics_df.loc[nearest_metric_ts, "cpu_pct"]
        mem = metrics_df.loc[nearest_metric_ts, "memory_pct"]
        lock_waits = metrics_df.loc[nearest_metric_ts, "lock_wait_count"]

        for _ in range(queries_this_minute):
            q_ts = ts + timedelta(seconds=int(rng.integers(0, 60)))
            template_name = rng.choice(TEMPLATE_NAMES, p=TEMPLATE_WEIGHTS)
            tpl = QUERY_TEMPLATES[template_name]
            table = rng.choice(TABLES)

            # base latency: lognormal-ish so there's a natural long tail
            # (this is what makes "normal but slow" queries realistic)
            base_latency = max(1, rng.normal(tpl["base_ms"], tpl["std_ms"]))
            rows_scanned = max(1, int(rng.normal(tpl["rows_scanned"], tpl["rows_scanned"] * 0.3)))
            rows_returned = max(1, int(rows_scanned * tpl["selectivity"] * rng.uniform(0.7, 1.3)))

            # --- system-load correlation (legitimate slowdown, not "anomaly" per se) ---
            cpu_penalty = 1 + max(0, (cpu - 60) / 100) * 1.5   # latency grows once CPU > 60%
            latency = base_latency * cpu_penalty

            is_anomaly = False
            anomaly_type = "none"

            # --- injected anomaly: missing index ---
            win = in_window(q_ts, anomaly_windows, "missing_index")
            if win and template_name == win["template"] and table == win["table"]:
                latency *= rng.uniform(6, 10)          # 6-10x slower
                rows_scanned = int(rows_scanned * rng.uniform(8, 15))  # full-scan-like behavior
                is_anomaly = True
                anomaly_type = "missing_index"

            # --- injected anomaly: lock contention ---
            win = in_window(q_ts, anomaly_windows, "lock_contention")
            if win and (win["table"] is None or table == win["table"]):
                if rng.random() < 0.8:  # most (not all) queries on that table are blocked
                    latency += rng.uniform(500, 3000)  # blocked waiting on lock
                    is_anomaly = True
                    anomaly_type = "lock_contention"

            # --- injected anomaly: resource exhaustion (broad effect) ---
            win = in_window(q_ts, anomaly_windows, "resource_exhaustion")
            if win:
                latency *= rng.uniform(3, 6)
                is_anomaly = True
                anomaly_type = "resource_exhaustion"

            # --- natural outliers that are NOT anomalies: e.g. a big report
            # query is just legitimately slow because it's a big report ---
            if template_name == "AGGREGATE_report" and rng.random() < 0.15:
                latency *= rng.uniform(1.5, 2.5)  # occasionally a bigger report run
                # NOT marked anomalous - this is the "normal but slow" case

            latency = max(0.5, latency)

            records.append(dict(
                query_id=query_id_counter,
                timestamp=q_ts,
                query_type=template_name,
                table_name=table,
                execution_time_ms=round(latency, 2),
                rows_scanned=rows_scanned,
                rows_returned=rows_returned,
                cpu_pct_at_time=round(cpu, 2),
                memory_pct_at_time=round(mem, 2),
                lock_wait_count_at_time=round(lock_waits, 2),
                is_anomaly=is_anomaly,
                anomaly_type=anomaly_type,
            ))
            query_id_counter += 1

        ts += timedelta(minutes=1)

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    return df


 
# MAIN
 

def main():
    print("Generating system metrics...")
    metrics_df, anomaly_windows = generate_system_metrics()
    metrics_df.to_csv(OUT_METRICS, index=False)
    print(f"  wrote {len(metrics_df):,} rows -> {OUT_METRICS}")

    print("Generating query log...")
    query_df = generate_query_log(metrics_df, anomaly_windows)
    query_df.to_csv(OUT_QUERIES, index=False)
    print(f"  wrote {len(query_df):,} rows -> {OUT_QUERIES}")

    n_anom = query_df["is_anomaly"].sum()
    print(f"\nTotal queries: {len(query_df):,}")
    print(f"Injected anomalies: {n_anom:,} ({n_anom/len(query_df)*100:.2f}%)")
    print(query_df["anomaly_type"].value_counts())

    print("\nAnomaly windows (ground truth incidents):")
    for w in anomaly_windows:
        print(f"  [{w['type']}] {w['start']} -> {w['end']} "
              f"(table={w.get('table')}, template={w.get('template')})")


if __name__ == "__main__":
    main()