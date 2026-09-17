# Database Performance Observability using Anomaly Detection

A hybrid database performance monitoring and anomaly detection system that identifies abnormal query behavior, groups related performance issues into incidents, and provides possible root-cause signals.

The system uses **statistical baselines, Isolation Forest, system-level metrics, and rule-based root-cause analysis** to distinguish naturally slow queries from queries that are behaving abnormally for their workload.

> **Status:** Functional prototype / research project
> **Data:** Synthetic database workload
> **Detection:** Unsupervised anomaly detection + statistical baselines
> **Dashboard:** Streamlit + Plotly

---

## Overview

Database queries can be slow for many different reasons. A query taking several seconds is not necessarily a problem if it is a complex reporting query that normally takes several seconds.

The challenge is identifying when a query is **abnormally slow relative to its own historical behavior and the current system conditions**.

This project addresses that problem by combining multiple detection signals:

1. **Adaptive rolling z-score** — detects sudden changes relative to recent behavior of the same query type.
2. **Frozen reference baseline** — detects persistent regressions that could eventually be absorbed by a rolling baseline.
3. **System-wide bucket z-score** — detects periods where database performance deteriorates across multiple query types.
4. **Isolation Forest** — detects unusual combinations of multiple features.
5. **Root-cause analysis** — ranks system metrics that were most anomalous when a query was flagged.
6. **Incident grouping** — combines individual anomaly alerts into larger performance incidents.

The resulting risk signals are combined into a unified **risk score**, which is thresholded to determine whether a query should be flagged.

---

## Architecture

```text
                 ┌──────────────────────┐
                 │   Synthetic Data     │
                 │                      │
                 │ Query Logs + Metrics │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Feature Engineering  │
                 │                      │
                 │ • Log latency        │
                 │ • Scan ratio         │
                 │ • Time features      │
                 │ • System metrics     │
                 │ • Rolling baseline   │
                 │ • Reference baseline │
                 │ • Bucket statistics  │
                 └──────────┬───────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    ┌──────────────────┐       ┌──────────────────┐
    │ Statistical      │       │ Isolation Forest  │
    │ Detection        │       │                  │
    │                  │       │ Multivariate     │
    │ Rolling Z-score  │       │ anomaly detection│
    │ Reference Z-score│       └────────┬─────────┘
    │ Bucket Z-score   │                │
    └─────────┬────────┘                │
              └─────────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │    Unified Risk      │
                 │       Score          │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Threshold Selection  │
                 │      using F1        │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Anomaly Detection    │
                 └──────────┬───────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    ┌──────────────────┐       ┌──────────────────┐
    │ Incident         │       │ Root Cause       │
    │ Grouping         │       │ Ranking          │
    └─────────┬────────┘       └────────┬─────────┘
              └─────────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │ Streamlit Dashboard  │
                 │       + Plotly        │
                 └──────────────────────┘
```

---

# 1. Synthetic Data Generation

The project uses synthetic database workload data to simulate realistic query behavior and performance incidents.

Two types of data are generated:

### Query logs

Each query execution contains information such as:

* Timestamp
* Query type
* Table/query context
* Execution time
* Rows scanned
* Rows returned
* Anomaly information for evaluation

### System metrics

System-level measurements are generated at regular intervals, including:

* CPU utilization
* Memory utilization
* Active connections
* Lock wait count
* Disk I/O
* Cache hit ratio

The synthetic workload contains both normal behavior and injected performance problems.

### Simulated scenarios

Examples include:

* **Missing index** — increased scan ratio and query latency
* **Lock contention** — increased lock waits and query latency
* **Resource exhaustion** — high CPU/system-wide degradation
* **Naturally complex queries** — queries that are slow but represent expected behavior

Ground-truth anomaly labels are available in the synthetic dataset, but they are **not used to train the anomaly detector**. They are used to evaluate the system after detection.

---

# 2. Feature Engineering

Raw database telemetry is transformed into features that provide more useful statistical and behavioral signals.

## Log-transformed latency

Database latency is typically right-skewed, with many fast queries and a smaller number of very slow queries.

The system uses:

```python
log_latency = np.log1p(execution_time_ms)
```

This reduces the influence of extreme values and provides a more suitable distribution for statistical analysis.

---

## Scan ratio

The system calculates:

```text
scan_ratio = rows_scanned / rows_returned
```

A high scan ratio can indicate inefficient query execution.

For example:

```text
50,000 rows scanned
100 rows returned

scan_ratio = 500
```

The log-transformed version is also used:

```python
log_scan_ratio = np.log1p(scan_ratio)
```

This helps handle the potentially large range of scan ratios.

---

## Time-based features

Temporal features include:

* Hour of day
* Day of week
* Weekend indicator

These provide workload context because database behavior can naturally vary depending on the time and day.

---

## System metrics at query time

The nearest system-metric observation is associated with each query.

This provides context such as:

```text
Query latency:       800 ms
CPU:                  92%
Memory:               87%
Lock waits:           18
```

This helps determine whether an abnormal query occurred during broader system pressure.

---

# 3. Adaptive Rolling Baseline

One of the main components is a **per-query-type rolling baseline**.

Instead of comparing every query against a single global latency distribution, queries are compared against other queries of the same type.

For each query type, the system maintains a trailing window of approximately **300 queries**.

The z-score is calculated as:

```text
z = (current_value - rolling_mean) / rolling_std
```

using log-transformed latency.

### Why query-type-specific baselines?

Consider:

```text
SELECT_simple       → normally 10–30 ms
JOIN_2table         → normally 50–100 ms
AGGREGATE_report    → normally 1,000–3,000 ms
```

A global latency threshold would incorrectly treat the reporting query as abnormal.

A per-query-type baseline instead asks:

> "Is this query unusual compared with other queries of this type?"

---

## Why a rolling baseline?

Database workloads change over time.

A fixed historical baseline could generate large numbers of alerts after legitimate workload changes.

A rolling baseline adapts to recent behavior.

However, this introduces another problem:

> A persistent performance regression can eventually become part of the rolling baseline.

For example, if a query becomes permanently slower, the rolling average may gradually shift toward the new slower behavior.

To address this, the system uses a second baseline.

---

# 4. Frozen Reference Baseline

A **reference baseline** is calculated from the initial period of the dataset and then kept fixed.

The reference baseline provides a long-term comparison point for each query type.

This creates two different perspectives:

```text
Rolling baseline
        ↓
"What has this query been doing recently?"

Reference baseline
        ↓
"What did healthy behavior originally look like?"
```

This allows the system to detect persistent regressions that an adaptive rolling baseline might eventually absorb.

### Trade-off

The reference baseline assumes that its initial period represents relatively healthy behavior.

In a production implementation, this baseline would ideally be validated or approved before being frozen.

---

# 5. System-Wide Bucket Detection

Per-query-type detection is complemented by a system-wide signal.

Queries are grouped into **5-minute buckets**, regardless of query type.

The system calculates the average log-latency for each bucket and monitors its deviation from recent bucket behavior.

This helps detect situations where:

> Multiple query types become slower at approximately the same time.

For example:

```text
14:00–14:05 → normal
14:05–14:10 → CPU spike + widespread latency increase
14:10–14:15 → elevated latency
```

A query might not look extremely unusual compared with other queries of its own type, but a system-wide performance degradation can still be detected through the bucket-level signal.

---

# 6. Isolation Forest

The statistical signals are complemented by an **Isolation Forest** anomaly detector.

Isolation Forest is an unsupervised machine-learning algorithm designed to identify observations that are unusual within a dataset.

### Core idea

Anomalous observations tend to be easier to isolate using random splits.

A simplified process is:

```text
Random feature
      ↓
Random split
      ↓
Split dataset
      ↓
Repeat recursively
      ↓
Measure how quickly a point is isolated
```

An observation that can be isolated using relatively few splits is considered more unusual.

The project uses:

```python
IsolationForest(
    n_estimators=300,
    contamination=0.02,
    max_samples="auto",
    random_state=42
)
```

### Parameters

**300 trees**

More trees generally provide more stable estimates at the cost of additional computation.

**2% contamination**

The synthetic dataset contains an anomaly rate close to 2%, so this value is used as an estimate of expected contamination.

**`max_samples="auto"`**

Scikit-learn uses up to 256 samples per tree when the dataset is sufficiently large.

**`random_state=42`**

Provides reproducible results across runs.

---

# 7. Feature Scaling

The Isolation Forest model is trained using standardized features.

```python
scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted **only on the training data**.

The same transformation is then applied to the test data.

This avoids using information from the evaluation period when calculating the transformation parameters.

---

# 8. Training and Evaluation

The dataset is divided chronologically.

```text
Days 1–7
   ↓
Training

Days 8–14
   ↓
Evaluation
```

The Isolation Forest is fitted using the training period.

The evaluation period remains unseen during model training.

The synthetic anomaly labels are used **only after generating predictions** to calculate evaluation metrics.

Therefore, the anomaly detection model itself remains unsupervised.

---

# 9. Unified Risk Score

The system combines several anomaly signals into one risk score.

The normalized signals include:

```text
Adaptive z-score
Reference z-score
System-wide bucket z-score
Isolation Forest score
```

The z-score components are normalized using:

```python
abs(z_score) / 4.0
```

Therefore:

```text
z = 4  → normalized value = 1
z = 8  → normalized value = 2
```

The Isolation Forest score is inverted and rescaled because more negative Isolation Forest decision scores represent more anomalous observations.

The final risk score is based on the strongest signal:

```python
risk_score = max(
    z_score_norm,
    ref_z_score_norm,
    bucket_z_norm,
    if_score_norm
)
```

### Why `max()`?

The detection layers are designed to catch different types of problems.

For example:

```text
Rolling z-score
→ sudden query-specific regression

Reference baseline
→ persistent regression

Bucket z-score
→ system-wide degradation

Isolation Forest
→ unusual multivariate combination
```

Using the maximum allows a strong signal from any layer to contribute directly to the final risk score.

---

# 10. Fast-Query Noise Suppression

Very fast queries can have extremely small variance.

For example:

```text
2 ms
3 ms
2 ms
```

A small absolute change can produce a large statistical z-score even though the operational impact is negligible.

The system therefore reduces the risk score for queries below a small latency threshold:

```python
if execution_time_ms < 5:
    risk_score *= 0.15
```

This is intended to reduce noise from statistically unusual but operationally insignificant fast queries.

---

# 11. Threshold Optimization

The risk score is continuous, so a threshold is required to convert it into:

```text
Normal
     vs.
Anomaly
```

Rather than selecting an arbitrary threshold, the system evaluates different thresholds using the synthetic ground-truth labels.

Precision and recall are calculated across possible thresholds.

The F1 score is then used to select a threshold balancing the two:

```text
F1 = 2 × precision × recall
     ─────────────────────────
       precision + recall
```

The resulting tuned threshold is approximately:

```text
0.7
```

on the current synthetic dataset.

### Evaluation result

The current evaluation reports approximately:

```text
Precision: 0.80
Recall:    0.74
```

This means that, on the current synthetic evaluation set:

* Approximately 80% of flagged queries correspond to labeled anomalies.
* Approximately 74% of labeled anomalies are detected.

These metrics are specific to the current synthetic dataset and should not be interpreted as production performance.

---

# 12. Incident Grouping

A database incident can generate thousands of anomalous queries.

Treating every query as an independent alert would create excessive noise.

The system therefore groups related anomaly events into incidents.

Queries are grouped using:

```text
Query type
Table/context
Timestamp
```

A gap of more than approximately **10 minutes** between related anomalous queries starts a new incident.

For each incident, the system calculates information such as:

* Start time
* End time
* Duration
* Number of anomalous queries
* Peak risk score
* Anomaly rate
* Root-cause hint

This converts a large collection of individual alerts into a smaller set of higher-level incidents.

---

# 13. Root Cause Analysis

After detecting an anomaly, the system attempts to identify which metric was most abnormal around the event.

Metrics considered include:

* CPU utilization
* Memory utilization
* Lock waits
* Scan efficiency

These are compared against their respective statistical baselines.

The system then ranks the available signals based on their deviation.

For example:

```text
CPU z-score       = 4.2
Memory z-score    = 1.8
Lock wait z-score = 0.7
Scan ratio score  = 2.1

Top signal → CPU
```

### Important limitation

This is **root-cause indication, not causal proof**.

A high CPU score does not prove that CPU caused the query slowdown. It only indicates that CPU was unusually high when the anomaly occurred.

A production system would require additional telemetry and causal analysis to establish the actual cause.

---

# 14. Dashboard

The project includes a Streamlit dashboard with interactive Plotly visualizations.

The dashboard provides views including:

### Query latency

Shows query performance over time and highlights detected anomalies.

### Anomaly scatter plot

Visualizes normal and anomalous observations to make unusual behavior easier to inspect.

### Query-type distributions

Box plots show how latency varies across different query types.

This helps distinguish:

```text
Naturally slow query type
        vs.
Abnormally slow query
```

### Timeline

The dashboard provides an incident/anomaly timeline and system metric overlays to identify periods of broader performance degradation.

### Root-cause information

Detected incidents can be examined alongside system-level signals to understand which metrics were most abnormal.

---

# Project Structure

```text
DatabasePerformanceObservabilityusingAnomalyDetection/
│
├── data/
│   └── generated datasets
│
├── models/
│   └── trained model artifacts
│
├── notebooks/
│   └── EDA and experiments
│
├── src/
│   ├── data generation
│   ├── feature engineering
│   ├── anomaly detection
│   ├── incident grouping
│   ├── root-cause analysis
│   └── dashboard
│
├── requirements.txt
└── README.md
```

> The exact files may vary as the project continues to evolve.

---

# Tech Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Isolation Forest

### Statistics

* SciPy
* Statistical z-score baselines

### Visualization

* Plotly
* Matplotlib
* Seaborn

### Dashboard

* Streamlit

---

# Key Results

On the current synthetic evaluation dataset:

| Metric                  |           Result |
| ----------------------- | ---------------: |
| Precision               |            ~0.80 |
| Recall                  |            ~0.74 |
| Detection approach      |           Hybrid |
| ML algorithm            | Isolation Forest |
| Training                |     Unsupervised |
| Evaluation period       |        Days 8–14 |
| Isolation Forest trees  |              300 |
| Estimated contamination |               2% |

The system is designed to detect multiple classes of database performance problems rather than relying on a single latency threshold.

---

# What Makes the Approach Different?

A simple monitoring system might use:

```text
if latency > 500 ms:
    alert()
```

This approach ignores context.

This project instead asks:

```text
Is this query unusually slow
for this type of query?

        +

Is it also unusual compared
with the original healthy baseline?

        +

Did the entire system become slower?

        +

Does its combination of features
look anomalous?

        ↓

Unified Risk Score
        ↓
Anomaly / Normal
        ↓
Incident + Root-Cause Signals
```

This makes the system more suitable for distinguishing **expected slow workloads from abnormal performance regressions**.

---

# Limitations

This is currently a prototype built using synthetic data.

Important limitations include:

* Synthetic workloads may not represent all production database behaviors.
* The frozen reference baseline assumes its initial period represents healthy behavior.
* Root-cause analysis indicates correlation rather than causation.
* Threshold tuning depends on synthetic ground-truth labels.
* Real production systems generally lack complete anomaly labels, making recall difficult to measure.
* The current implementation is batch-oriented rather than a true streaming system.
* The current system focuses on anomaly detection rather than automatically fixing database problems.

---

# Future Work

Potential improvements include:

* Real database telemetry integration
* Streaming / near-real-time detection
* PostgreSQL/MySQL query log integration
* Human-approved baseline initialization
* More advanced root-cause analysis
* Query-plan analysis
* Database-specific features such as index usage and execution plans
* Alerting integrations
* Incident persistence and tracking
* Online model/baseline updates
* Evaluation using real production incidents

---

# Running the Dashboard

Create and activate a virtual environment:

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
streamlit run .\src\dashboard.py
```

The dashboard should then be available at:

```text
http://localhost:8501
```

> **Important:** `dashboard.py` should be launched with `streamlit run`, not `python dashboard.py`, because Streamlit requires its own runtime for sessions, widgets, and dashboard state.

---

# Project Status

🚧 **Functional Prototype**

Current implementation includes:

* [x] Synthetic workload generation
* [x] Query/system metric correlation
* [x] Feature engineering
* [x] Rolling query-type baselines
* [x] Frozen reference baselines
* [x] System-wide bucket detection
* [x] Isolation Forest anomaly detection
* [x] Feature scaling
* [x] Unified risk scoring
* [x] Threshold optimization
* [x] Incident grouping
* [x] Root-cause signal ranking
* [x] Interactive Streamlit dashboard
* [x] Plotly visualizations

### Next major milestone

**Real-time / streaming database performance observability.**
