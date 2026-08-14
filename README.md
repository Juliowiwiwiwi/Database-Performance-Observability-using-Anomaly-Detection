# Database Performance Observability using Anomaly Detection (WORK IN PROGRESS - CURRENTLY BUILDING)

## Overview

A machine learning-based system for monitoring database performance, detecting abnormal query behavior, and identifying possible root causes.

The project uses synthetic database query logs and system metrics to simulate realistic database workloads and performance issues.

## Pipeline

Synthetic Data
→ Feature Engineering
→ Anomaly Detection
→ Root Cause Analysis
→ Visualization / Dashboard

## Main Components

### 1. Synthetic Data Generation
Generate correlated query logs and system metrics with realistic workload patterns.

Simulated anomalies include:
- Missing index
- Lock contention
- High CPU / resource exhaustion
- Naturally complex queries that are slow but not anomalous

### 2. Feature Engineering
Create features such as:
- Query execution time
- Rows scanned / rows returned
- Query-template statistics
- Rolling latency statistics
- CPU and memory usage
- Lock waits
- Active connections
- Time-based features

### 3. Anomaly Detection
Use:
- Statistical baseline / rolling z-score
- Isolation Forest

The goal is to detect queries that behave abnormally compared to their historical behavior.

### 4. Root Cause Analysis
Use rule-based correlation to identify likely causes such as:
- High CPU
- Memory pressure
- Lock contention
- Possible missing index

### 5. Visualization
Visualize:
- Query latency over time
- Detected anomalies
- System metrics
- Query-type performance
- Root-cause correlations

A Streamlit dashboard may be added later.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Matplotlib / Seaborn
- Plotly
- Streamlit

## Project Structure

data/        - Generated datasets
src/         - Main project code
notebooks/   - EDA and experiments
models/      - Trained models
dashboard/   - Streamlit dashboard

## Current Status

🚧 Project setup / data generation

Next:
1. Generate synthetic data
2. Validate the generated data
3. Perform EDA
4. Build feature engineering pipeline
5. Implement anomaly detection
6. Add root-cause analysis
7. Build visualization/dashboard