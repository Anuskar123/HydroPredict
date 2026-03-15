# ⚡ HydroPredict AI — Climate-Aware Renewable Grid Optimizer

**AI Hackathon 2026 | Embark College, Pulchowk**

An AI-powered forecasting dashboard that predicts short-term hydropower generation capacity by analyzing weather and river level data across Nepal. Designed to help grid operators anticipate drops in energy production and activate backup measures before blackouts occur.

## The Problem

Nepal's electricity grid depends on hydropower for ~90% of its supply. Climate variability — droughts, erratic monsoons, temperature shifts — directly threatens grid stability. Grid operators today react *after* shortages occur, leading to load shedding and economic losses.

## The Solution

HydroPredict AI predicts energy *supply* 72 hours in advance by modeling the chain:

```
Weather → River Flow → Turbine Output → Grid Alert
```

Operators see a live dashboard with color-coded alerts and can run "what-if" scenarios (e.g., "what if rainfall drops 50%?") to plan ahead.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run everything (data generation → model training → dashboard)
python run.py
```

The dashboard opens at **http://localhost:8501**.

## Manual Setup

```bash
# Generate data
python -m src.data_generator

# Train model
python -m src.model

# Launch dashboard
streamlit run app.py
```

## Project Structure

```
├── app.py                  # Streamlit dashboard (main UI)
├── run.py                  # One-command setup script
├── requirements.txt        # Python dependencies
├── src/
│   ├── data_generator.py   # Realistic Nepal weather + hydro data
│   ├── data_processing.py  # Feature engineering pipeline
│   └── model.py            # XGBoost training & evaluation
├── data/                   # Generated datasets (CSV)
├── models/                 # Trained model artifacts
└── assets/                 # Static assets
```

## Datasets Used

| Dataset | Source | Role |
|---------|--------|------|
| **Nepal Multi-District Weather** | Local `data/nepali_multi_district.csv` | **REAL** historical weather (2020-2025) covering Temp, Precip, Humidity. |
| River Flow & Generation | Physics-based Simulation | Modeled target variables derived from real weather inputs. |

> **Data Strategy:** The model trains on **real-world weather observations** from 2020 to 2025. Since granular hourly generation data is proprietary, we generate the target variable (`generation_mw`) using a standard hydrological physics model ($Q = f(Precip, Temp, Elevation)$) to create a ground-truth dataset for the AI to learn.

## Features

- **72-Hour Forecast** — Hourly generation predictions with color-coded bars
- **Grid Alert System** — Green/Yellow/Red status with actionable recommendations
- **What-If Scenarios** — Sliders to simulate drought, flooding, or temperature shifts
- **Multi-District View** — All 5 river systems at a glance with utilization metrics
- **Model Transparency** — R², MAE, feature importances displayed in dashboard

## Tech Stack

- **Model:** XGBoost (time-series regression with lag/rolling features)
- **Dashboard:** Streamlit + Plotly
- **Data:** pandas + NumPy

## Team

AI Hackathon 2026, Embark College, Pulchowk
# HydroPredict
