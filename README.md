
# 🏥 Respiratory ED Demand Forecasting Dashboard  
### Advanced Analytics Project – FH Südwestfalen  

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![Statsmodels](https://img.shields.io/badge/Statsmodels-SARIMAX-green.svg)
![scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)

---

# 📌 Project Overview

This project develops an interactive forecasting dashboard to analyze and predict respiratory emergency department (ED) demand using:

- 📊 Vaping sales data (local dataset)
- 🌍 Delphi FluView ILINet data (via API)
- 📈 Statistical and machine learning forecasting models
- 🔍 Classification-based surge detection
- 🎛 Scenario-based sensitivity analysis

---

# 🔁 Application Data Flow

## STEP 1 — Load Local Vaping Dataset

Function: `load_vaping_csv()`

- Reads `vaping_data.csv`
- Converts Date column using `dayfirst=True`
- Forces weekly Sunday alignment (`W-SUN`)

---

## STEP 2 — Sidebar Controls

Users define:
- Region
- Start Date
- End Date
- Forecast Horizon (weeks)
- What-if Demand (%)

---

## STEP 3 — Fetch Respiratory Proxy (Delphi API)

Function: `fetch_fluview(start_epiweek, end_epiweek, region)`

- Calls Delphi API (`source=fluview`)
- Retrieves `weighted_ili` or `ili`
- Converts epiweek → Sunday date

---

## STEP 4 — Build Master Dataset

```
master = ensure_weekly(vape.merge(ed, on="Date", how="left"))
```

---

# 📊 TAB 1 — Exploratory Data Analysis

Displays:
- Summary statistics
- Trend plots
- Histograms
- Correlation heatmap
- Scatter relationships

---

# 📈 TAB 2 — Modeling & Forecasting

Models:
- Seasonal Naïve
- SARIMAX
- Prophet (optional)
- Gradient Boosting

Regression Metrics:
- MAE
- MSE
- RMSE
- RMSLE
- R²

Classification Metrics:
- Accuracy
- Precision
- Recall
- AUC

High-demand threshold default = 75th percentile of training data.

---

# 📉 TAB 3 — Trend & What-if Analysis

Uses seasonal decomposition (period=52).

Scenario simulation:

X_scenario = X × (1 + pct/100)

Displays baseline vs scenario forecast comparison.

---

# 🛠️ How to Run Locally

1. Clone repository  
`git clone https://github.com/yourusername/yourrepo.git`

2. Navigate into project  
`cd yourrepo`

3. Create virtual environment  
`python -m venv .venv`

4. Activate environment  

Windows:
`.venv\Scripts\activate`

Mac/Linux:
`source .venv/bin/activate`

5. Install dependencies  
`pip install -r requirements.txt`

6. Run application  
`streamlit run app.py`

Open in browser:  
http://localhost:8501

---

# 📦 Project Structure

```
AA_PROJECT_FINAL/
├── app.py
├── vaping_data.csv
├── requirements.txt
├── README.md
└── .venv/ (ignored)
```

---

# ⚠ Limitations

- Observational data (not causal inference)
- Short horizons may limit classification reliability
- SARIMAX assumes linear relationships

---

# 👩‍🎓 Academic Context

Advanced Analytics  
FH Südwestfalen University  
Master’s Program  

---

# 📄 License

Academic project — educational use only.
