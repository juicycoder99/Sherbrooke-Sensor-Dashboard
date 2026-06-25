# Sherbrooke Sensor Dashboard

An interactive Streamlit dashboard for monitoring, visualising, and analysing environmental
sensor data (temperature, humidity, moisture, and gas levels), with built-in anomaly detection.

The full application is in [`Dashboard.py`](Dashboard.py).

## Overview

The dashboard ingests a time-series of environmental sensor readings and turns it into an
explorable monitoring tool. It loads two datasets (a clean baseline and an anomaly-injected
variant), cleans and indexes them by timestamp, and exposes a range of interactive views for
trend analysis, correlation, and anomaly detection. The data is a synthetic Sherbrooke sensor
dataset published on Kaggle, designed to mimic realistic environmental behaviour.

## Features

- **Live overview** — current-time sidebar, dataset switching, and an at-a-glance statistical summary.
- **Gas-level trends** — multiple views of gas readings over selectable timeframes (daily, monthly, yearly).
- **Environmental analysis** — temperature, humidity, and moisture trends, plus seasonal breakdowns.
- **Correlation analysis** — relationships between gas levels and the other environmental variables.
- **Anomaly detection** — three selectable methods (IQR, Z-Score, and a trained **XGBoost** model with
  a tuned decision threshold), applied over the full series or a custom date range.
- **Reporting** — downloadable summaries and report exports from the sidebar.

## Tech stack

Python · Streamlit · pandas · NumPy · matplotlib · seaborn · Altair · scikit-learn · XGBoost · Kaggle API

## Running it

```bash
pip install -r requirements.txt
streamlit run Dashboard.py
```

The app pulls its dataset from Kaggle, so set your Kaggle credentials (`KAGGLE_USERNAME` and
`KAGGLE_KEY`) in Streamlit secrets before running.

## Files

| File | Description |
|------|-------------|
| `Dashboard.py` | The full Streamlit dashboard application |
| `PROJECT_BRIEF.pdf` | Project brief (goals, objectives, outcomes) |
| `requirements.txt` | Python dependencies |
| `xgb_anomaly_model.joblib` | Trained XGBoost anomaly-detection model |
| `xgb_best_threshold.joblib` | Tuned decision threshold for the anomaly model |
