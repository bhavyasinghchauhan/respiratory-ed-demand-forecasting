import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")


# Optional dependency: Prophet
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False


# -------------------------------------------------------------------
# App config / constants
# -------------------------------------------------------------------
st.set_page_config(page_title="Respiratory ED Forecasting Dashboard", layout="wide")
st.title("Respiratory ED Forecasting Dashboard")

# Prefer secrets/env, but keep fallback to avoid breaking current runs.
DEFAULT_DELPHI_KEY = "551821194a0d5"
DELPHI_EPIDATA_KEY = (
    st.secrets.get("DELPHI_EPIDATA_KEY")
    if hasattr(st, "secrets")
    else None
) or os.getenv("DELPHI_EPIDATA_KEY") or DEFAULT_DELPHI_KEY


# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------
def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def epiweek_from_date(d: pd.Timestamp) -> int:
    return int(pd.Timestamp(d).strftime("%G%V"))


def epiweek_to_date(epiweek: int) -> pd.Timestamp:
    s = str(int(epiweek))
    y = int(s[:4])
    w = int(s[4:])
    return pd.to_datetime(f"{y}-W{w:02d}-7", format="%G-W%V-%u")


def ensure_weekly(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df = df.set_index(date_col).asfreq("W-SUN")
    return df.reset_index()


def clamp_date(d: pd.Timestamp, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.Timestamp:
    return min(max(d, lo), hi)


def rmsle_safe(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def mape_safe(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-9)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def mase_safe(y_true, y_pred, y_train, seasonality: int = 52) -> float:
    """
    MASE = MAE(model) / MAE(naive)

    Naive baseline:
    - seasonal naive with period=52 when possible
    - otherwise a lag-1 naive baseline
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    model_mae = float(mean_absolute_error(y_true, y_pred))

    if len(y_train) > seasonality:
        naive_in = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    elif len(y_train) > 1:
        naive_in = np.abs(y_train[1:] - y_train[:-1])
    else:
        return np.nan

    naive_denom = float(np.mean(naive_in)) if len(naive_in) else np.nan
    if np.isnan(naive_denom) or naive_denom < 1e-12:
        return np.nan

    return float(model_mae / naive_denom)


def regression_metrics(y_true, y_pred, y_train_for_mase=None) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    out = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "RMSLE": rmsle_safe(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": mape_safe(y_true, y_pred),
    }
    out["MASE"] = (
        mase_safe(y_true, y_pred, y_train_for_mase, seasonality=52)
        if y_train_for_mase is not None
        else np.nan
    )
    return out


def classification_metrics_from_regression(y_true, y_pred, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)

    auc = np.nan
    fpr = tpr = None
    try:
        if len(np.unique(y_true_bin)) == 2:
            auc = float(roc_auc_score(y_true_bin, y_pred))
            fpr, tpr, _ = roc_curve(y_true_bin, y_pred)
    except Exception:
        pass

    return {
        "threshold": float(threshold),
        "confusion_matrix": cm,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "auc": auc,
        "roc_fpr": fpr,
        "roc_tpr": tpr,
        "y_true_bin": y_true_bin,
        "y_pred_bin": y_pred_bin,
    }


def add_lagged_exog(
    df: pd.DataFrame,
    exog_cols,
    lags=(1, 2, 3, 4),
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create lag features for each exogenous column.

    Returns:
      - updated df
      - list of generated lag column names
    """
    df = df.copy()
    lag_cols: list[str] = []

    for col in exog_cols:
        for lag in lags:
            name = f"{col}_lag{lag}"
            df[name] = df[col].shift(lag)
            lag_cols.append(name)

    return df, lag_cols


# -------------------------------------------------------------------
# Delphi (FluView / ILINet)
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_fluview(
    start_epiweek: int,
    end_epiweek: int,
    region: str = "nat",
) -> pd.DataFrame | None:
    url = "https://api.delphi.cmu.edu/epidata/api.php"
    params = {
        "source": "fluview",
        "regions": region,
        "epiweeks": f"{start_epiweek}-{end_epiweek}",
        "auth": DELPHI_EPIDATA_KEY,
    }

    r = requests.get(url, params=params, timeout=60)
    data = r.json()

    if data.get("result") != 1 or "epidata" not in data:
        return None

    df = pd.DataFrame(data["epidata"])
    if df.empty:
        return None

    df["Date"] = df["epiweek"].apply(epiweek_to_date)

    if "weighted_ili" in df.columns:
        df = df[["Date", "weighted_ili"]].rename(columns={"weighted_ili": "Resp_ED_Proxy"})
    elif "ili" in df.columns:
        df = df[["Date", "ili"]].rename(columns={"ili": "Resp_ED_Proxy"})
    else:
        return None

    return ensure_weekly(df, "Date")


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_vaping_csv() -> pd.DataFrame:
    app_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(app_dir, "vaping_data.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing vaping CSV at: {path}\n"
            "Place 'vaping_data.csv' in the same folder as app.py."
        )

    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError("Vaping CSV must contain a 'Date' column.")

    df["Date"] = safe_to_datetime(df["Date"])
    df = df.dropna(subset=["Date"])
    return ensure_weekly(df, "Date")


try:
    vape_all = load_vaping_csv()
except Exception as e:
    st.error(str(e))
    st.stop()

data_min = pd.to_datetime(vape_all["Date"].min())
data_max = pd.to_datetime(vape_all["Date"].max())

default_start = clamp_date(pd.to_datetime("2019-01-01"), data_min, data_max)
default_end = clamp_date(pd.to_datetime("2023-12-31"), data_min, data_max)


# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------
st.sidebar.header("Controls")

region = st.sidebar.text_input(
    "Delphi FluView region (e.g., nat, hhs1, hhs2...)",
    value="nat",
)

start_date = st.sidebar.date_input(
    "Start Date",
    value=default_start.date(),
    min_value=data_min.date(),
    max_value=data_max.date(),
)
end_date = st.sidebar.date_input(
    "End Date",
    value=default_end.date(),
    min_value=data_min.date(),
    max_value=data_max.date(),
)

horizon = st.sidebar.number_input(
    "Forecast horizon (weeks)",
    min_value=4,
    max_value=26,
    value=8,
    step=1,
)

what_if_pct = st.sidebar.slider(
    "What-if: change vaping demand (%)",
    min_value=-50,
    max_value=100,
    value=0,
    step=5,
)

start_ts = pd.to_datetime(start_date)
end_ts = pd.to_datetime(end_date)

if start_ts > end_ts:
    st.error("Start date must be before end date.")
    st.stop()


# Filter vaping to date range
vape = vape_all[(vape_all["Date"] >= start_ts) & (vape_all["Date"] <= end_ts)].copy()

# Fetch ED/ILI proxy
start_epi = epiweek_from_date(start_ts)
end_epi = epiweek_from_date(end_ts)
ed = fetch_fluview(start_epi, end_epi, region=region)

if ed is None:
    st.error("Failed to fetch ED/ILI proxy from Delphi FluView. Check region code or API key.")
    st.stop()

ed = ed[(ed["Date"] >= start_ts) & (ed["Date"] <= end_ts)].copy()

master = ensure_weekly(vape.merge(ed, on="Date", how="left"), "Date")

# Identify numeric vaping predictors
vape_exog_cols = [
    c for c in vape.columns
    if c != "Date" and pd.api.types.is_numeric_dtype(vape[c])
]
if not vape_exog_cols:
    st.error("No numeric vaping columns found to use as predictors.")
    st.stop()

# Lag features for predictors
master_lagged, vape_lag_cols = add_lagged_exog(master, vape_exog_cols, lags=(1, 2, 3, 4))


# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1) EDA Summary",
        "2) Models & Forecasting",
        "3) Comparison, Seasonality & What-if",
        "4) Project Guide",
    ]
)

# =========================
# TAB 1 — EDA
# =========================
with tab1:
    st.subheader("EDA: Vaping Dataset, ED/ILI Dataset, and Master Table")

    c1, c2, c3 = st.columns(3)
    c1.metric("Vaping rows", len(vape))
    c2.metric("ED/ILI rows", len(ed))
    c3.metric("Master rows", len(master))

    st.divider()

    colL, colR = st.columns(2, gap="large")
    with colL:
        st.markdown("### Vaping summary")
        st.dataframe(vape[vape_exog_cols].describe().T, use_container_width=True, height=320)
    with colR:
        st.markdown("### ED/ILI summary")
        st.dataframe(ed[["Resp_ED_Proxy"]].describe().T, use_container_width=True, height=320)

    st.divider()

    colL, colR = st.columns(2, gap="large")

    vape_y = "Total_K" if "Total_K" in vape_exog_cols else vape_exog_cols[0]

    fig_vape_trend = px.line(vape, x="Date", y=vape_y, title=f"Vaping trend: {vape_y}")
    fig_vape_trend.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))

    fig_ed_trend = px.line(ed, x="Date", y="Resp_ED_Proxy", title="ED/ILI proxy trend")
    fig_ed_trend.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))

    colL.plotly_chart(fig_vape_trend, use_container_width=True)
    colR.plotly_chart(fig_ed_trend, use_container_width=True)

    st.divider()

    colL, colR = st.columns(2, gap="large")

    fig_vape_hist = px.histogram(vape, x=vape_y, nbins=40, title=f"Distribution: {vape_y}")
    fig_vape_hist.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))

    fig_ed_hist = px.histogram(ed, x="Resp_ED_Proxy", nbins=40, title="Distribution: ED/ILI proxy")
    fig_ed_hist.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))

    colL.plotly_chart(fig_vape_hist, use_container_width=True)
    colR.plotly_chart(fig_ed_hist, use_container_width=True)

    st.divider()

    st.write("### Missing values (master)")
    miss = master_lagged.isna().sum().sort_values(ascending=False)
    st.dataframe(miss[miss > 0].to_frame("missing_count"), use_container_width=True)

    st.write("### Correlation heatmap (numeric columns incl. lags)")
    corr = master_lagged.drop(columns=["Date"]).corr(numeric_only=True)
    fig_corr = px.imshow(corr, text_auto=False, aspect="auto", title="Correlation heatmap")
    fig_corr.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_corr, use_container_width=True)

    st.write("### Relationship: vaping vs ED/ILI proxy")
    xcol = "Total_K" if "Total_K" in vape_exog_cols else vape_exog_cols[0]
    fig_rel = px.scatter(master, x=xcol, y="Resp_ED_Proxy", trendline="ols", title=f"{xcol} vs Resp_ED_Proxy")
    fig_rel.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_rel, use_container_width=True)

# =========================
# TAB 2 — Models & Forecasting
# =========================
with tab2:
    st.subheader("Models, error scores, and evaluation")

    dfm_all = master_lagged.dropna(subset=["Resp_ED_Proxy"]).copy()
    dmin = pd.to_datetime(dfm_all["Date"].min())
    dmax = pd.to_datetime(dfm_all["Date"].max())
    max_train = dmax - pd.Timedelta(weeks=int(horizon))

    if max_train <= dmin:
        st.warning("Not enough data for train/test with the selected horizon. Extend the date range.")
        st.stop()

    default_model_start = clamp_date(start_ts, dmin, dmax)

    model_start = st.date_input(
        "Model Start Date (training begins)",
        value=default_model_start.date(),
        min_value=dmin.date(),
        max_value=dmax.date(),
    )

    train_end = st.date_input(
        "Training End Date (test uses next horizon weeks)",
        value=max_train.date(),
        min_value=dmin.date(),
        max_value=max_train.date(),
    )

    dfm = dfm_all[dfm_all["Date"] >= pd.to_datetime(model_start)].copy()
    train = dfm[dfm["Date"] <= pd.to_datetime(train_end)].copy()
    test = dfm[dfm["Date"] > pd.to_datetime(train_end)].head(int(horizon)).copy()

    if len(test) == 0 or len(train) < 60:
        st.warning("Not enough train/test rows. Choose an earlier training end or widen the date range.")
        st.stop()

    y_train = train["Resp_ED_Proxy"].astype(float)
    y_test = test["Resp_ED_Proxy"].astype(float)

    exog_cols = vape_lag_cols
    X_train = train[exog_cols].astype(float)
    X_test = test[exog_cols].astype(float)

    default_thr = float(np.nanpercentile(y_train.values, 75))
    thr = st.number_input(
        "High-demand threshold (for confusion matrix / AUC)",
        value=default_thr,
        help=(
            "Default is the 75th percentile of training data. "
            "This classification view is a secondary operational add-on; the main task is regression forecasting."
        ),
    )

    # Classification evaluation window: try to ensure both classes appear
    avail_df = dfm[dfm["Date"] > pd.to_datetime(train_end)].copy()
    available_after = int(avail_df.shape[0])

    if available_after <= 0:
        st.warning("No rows available after Training End Date for classification evaluation.")
        st.stop()

    min_clf = int(horizon)
    max_clf = max(min_clf, min(104, available_after))

    if max_clf == min_clf:
        clf_window = min_clf
        st.info(
            f"Only {available_after} week(s) available after Training End Date, "
            f"so the classification window is fixed at {clf_window}."
        )
    else:
        clf_window = st.slider(
            "Classification eval window (weeks)",
            min_value=min_clf,
            max_value=max_clf,
            value=min(52, max_clf),
            step=1,
        )

    chosen = int(clf_window)
    while True:
        tmp = avail_df.head(chosen)
        y_tmp = tmp["Resp_ED_Proxy"].astype(float).values
        y_bin = (y_tmp >= float(thr)).astype(int)
        if len(np.unique(y_bin)) == 2 or chosen >= max_clf:
            break
        chosen = min(max_clf, chosen + 4)

    test_clf = avail_df.head(chosen).copy()
    y_test_clf = test_clf["Resp_ED_Proxy"].astype(float)
    X_test_clf = test_clf[exog_cols].astype(float)

    y_true_bin_clf = (y_test_clf.values >= float(thr)).astype(int)
    high_eval = int(np.sum(y_true_bin_clf == 1))
    low_eval = int(np.sum(y_true_bin_clf == 0))
    tot_eval = max(1, high_eval + low_eval)

    st.caption(
        f"Classification window: {len(test_clf)} weeks | "
        f"High={high_eval} ({high_eval/tot_eval:.0%}), Low={low_eval} ({low_eval/tot_eval:.0%}). "
        "ROC/AUC requires both classes."
    )

    preds_reg: dict[str, np.ndarray | None] = {}
    preds_clf: dict[str, np.ndarray | None] = {}

    with st.spinner("Running models..."):
        # 0) ED-only baseline
        try:
            sarima = sm.tsa.SARIMAX(
                y_train,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 52),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            preds_reg["SARIMA (ED-only)"] = np.asarray(sarima.forecast(steps=len(y_test)))
            preds_clf["SARIMA (ED-only)"] = np.asarray(sarima.forecast(steps=len(y_test_clf)))
        except Exception:
            preds_reg["SARIMA (ED-only)"] = None
            preds_clf["SARIMA (ED-only)"] = None

        # 1) Seasonal naive
        season = 52

        def seasonal_naive(y_series: pd.Series, n_steps: int) -> np.ndarray:
            if len(y_series) >= season:
                base = y_series.iloc[-season:].values
                reps = int(np.ceil(n_steps / len(base)))
                return np.tile(base, reps)[:n_steps]
            return np.repeat(float(y_series.iloc[-1]), n_steps)

        preds_reg["Seasonal Naive"] = seasonal_naive(y_train, len(y_test))
        preds_clf["Seasonal Naive"] = seasonal_naive(y_train, len(y_test_clf))

        # 2) SARIMAX with exogenous lag features
        try:
            sarimax = sm.tsa.SARIMAX(
                y_train,
                exog=X_train,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 52),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            preds_reg["SARIMAX (ED + vaping lags)"] = np.asarray(
                sarimax.forecast(steps=len(y_test), exog=X_test)
            )
            preds_clf["SARIMAX (ED + vaping lags)"] = np.asarray(
                sarimax.forecast(steps=len(y_test_clf), exog=X_test_clf)
            )
        except Exception:
            preds_reg["SARIMAX (ED + vaping lags)"] = None
            preds_clf["SARIMAX (ED + vaping lags)"] = None

        # 3) Prophet (optional)
        if PROPHET_AVAILABLE:
            try:
                dfp = (
                    train[["Date", "Resp_ED_Proxy"] + exog_cols]
                    .copy()
                    .rename(columns={"Date": "ds", "Resp_ED_Proxy": "y"})
                )
                m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
                for c in exog_cols:
                    m.add_regressor(c)
                m.fit(dfp)

                future_reg = test[["Date"] + exog_cols].copy().rename(columns={"Date": "ds"})
                future_clf = test_clf[["Date"] + exog_cols].copy().rename(columns={"Date": "ds"})

                preds_reg["Prophet (ED + vaping lags)"] = m.predict(future_reg)["yhat"].values
                preds_clf["Prophet (ED + vaping lags)"] = m.predict(future_clf)["yhat"].values
            except Exception:
                preds_reg["Prophet (ED + vaping lags)"] = None
                preds_clf["Prophet (ED + vaping lags)"] = None
        else:
            preds_reg["Prophet (not installed)"] = None
            preds_clf["Prophet (not installed)"] = None

        # 4) Gradient Boosting (target lags + exog lags)
        try:
            tmp = dfm[["Date", "Resp_ED_Proxy"] + exog_cols].copy().sort_values("Date")
            tmp["y_lag1"] = tmp["Resp_ED_Proxy"].shift(1)
            tmp["y_lag2"] = tmp["Resp_ED_Proxy"].shift(2)
            tmp["y_lag4"] = tmp["Resp_ED_Proxy"].shift(4)
            tmp = tmp.dropna()

            tmp_train = tmp[tmp["Date"] <= pd.to_datetime(train_end)]
            tmp_test_reg = tmp[tmp["Date"] > pd.to_datetime(train_end)].head(int(horizon))
            tmp_test_clf = tmp[tmp["Date"] > pd.to_datetime(train_end)].head(int(len(test_clf)))

            feat_cols = ["y_lag1", "y_lag2", "y_lag4"] + exog_cols
            gbr = GradientBoostingRegressor(random_state=42)
            gbr.fit(tmp_train[feat_cols].values, tmp_train["Resp_ED_Proxy"].values)

            preds_reg["Gradient Boosting (lags + exog lags)"] = gbr.predict(tmp_test_reg[feat_cols].values)
            preds_clf["Gradient Boosting (lags + exog lags)"] = gbr.predict(tmp_test_clf[feat_cols].values)
        except Exception:
            preds_reg["Gradient Boosting (lags + exog lags)"] = None
            preds_clf["Gradient Boosting (lags + exog lags)"] = None

    # Metrics summary
    results = []
    for name, yhat_reg in preds_reg.items():
        if yhat_reg is None:
            continue
        yhat_clf = preds_clf.get(name)
        if yhat_clf is None:
            continue

        n_reg = min(len(y_test), len(yhat_reg))
        reg = regression_metrics(
            y_test.values[:n_reg],
            np.asarray(yhat_reg)[:n_reg],
            y_train_for_mase=y_train.values,
        )

        n_clf = min(len(y_test_clf), len(yhat_clf))
        clf = classification_metrics_from_regression(
            y_test_clf.values[:n_clf],
            np.asarray(yhat_clf)[:n_clf],
            threshold=float(thr),
        )

        results.append(
            {
                "Model": name,
                **reg,
                "Accuracy": clf["accuracy"],
                "Precision": clf["precision"],
                "Recall": clf["recall"],
                "AUC": clf["auc"],
            }
        )

    if not results:
        st.error("No models produced predictions. Check date range, missing values, and optional installs.")
        st.stop()

    results_df = pd.DataFrame(results).sort_values("RMSE", ascending=True).reset_index(drop=True)
    best_model = results_df.iloc[0]["Model"]
    st.success(f"Best model (lowest RMSE): **{best_model}**")

    st.write("### Regression metrics")
    st.dataframe(
        results_df[["Model", "MAE", "RMSE", "MAPE", "MASE", "MSE", "RMSLE", "R2"]],
        use_container_width=True,
    )

    st.write("### RMSE by model")
    st.plotly_chart(px.bar(results_df, x="Model", y="RMSE", title="RMSE by Model (best first)"), use_container_width=True)

    st.write("### ED-only vs ED+vaping lags")
    impact_cols = ["Model", "RMSE", "MAE", "MAPE", "MASE"]
    impact_df = results_df[
        results_df["Model"].isin(["SARIMA (ED-only)", "SARIMAX (ED + vaping lags)"])
    ][impact_cols].copy()

    if impact_df.shape[0] == 2:
        st.dataframe(impact_df, use_container_width=True)
        st.plotly_chart(px.bar(impact_df, x="Model", y="RMSE", title="ED-only vs ED+vaping lags (RMSE)"), use_container_width=True)
    else:
        st.info("Impact comparison not available (SARIMA or SARIMAX failed for the current selection).")

    st.divider()

    st.write("### Forecast vs Actual (test horizon)")
    best_pred_reg = preds_reg[best_model]
    n_reg = min(len(y_test), len(best_pred_reg))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test["Date"].iloc[:n_reg], y=y_test.iloc[:n_reg], mode="lines+markers", name="Actual"))
    fig.add_trace(
        go.Scatter(
            x=test["Date"].iloc[:n_reg],
            y=np.asarray(best_pred_reg)[:n_reg],
            mode="lines+markers",
            name=f"Forecast ({best_model})",
        )
    )
    fig.update_layout(title=f"Forecast vs Actual — {best_model}", xaxis_title="Date", yaxis_title="Resp_ED_Proxy")
    st.plotly_chart(fig, use_container_width=True)

    st.write("### Confusion matrix + ROC/AUC (best model)")
    best_pred_clf = preds_clf[best_model]
    n_clf = min(len(y_test_clf), len(best_pred_clf))
    clf_best = classification_metrics_from_regression(
        y_test_clf.values[:n_clf],
        np.asarray(best_pred_clf)[:n_clf],
        threshold=float(thr),
    )

    cm = clf_best["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=["Actual Low (0)", "Actual High (1)"], columns=["Pred Low (0)", "Pred High (1)"])
    st.dataframe(cm_df, use_container_width=True)

    fig_cm = px.imshow(cm, text_auto=True, aspect="auto", title="Confusion Matrix (Best model)")
    st.plotly_chart(fig_cm, use_container_width=True)

    if clf_best["roc_fpr"] is not None and not np.isnan(clf_best["auc"]):
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=clf_best["roc_fpr"],
                y=clf_best["roc_tpr"],
                mode="lines",
                name=f"ROC (AUC={clf_best['auc']:.3f})",
            )
        )
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
        fig_roc.update_layout(title="ROC Curve (Best model)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)
    else:
        st.info(
            "ROC/AUC needs both classes present in the evaluation window. "
            "Try expanding the overall date range or adjusting the threshold."
        )

    st.divider()

    st.write(f"### Next {int(horizon)}-week forecast (post training end)")
    future_dates = pd.date_range(pd.to_datetime(train_end) + pd.Timedelta(days=7), periods=int(horizon), freq="W-SUN")

    # Simple operational assumption: hold exog lags constant at last training row
    last_exog = train.iloc[-1][exog_cols].astype(float).values
    X_future = pd.DataFrame([last_exog] * int(horizon), columns=exog_cols, index=future_dates)

    future_pred = None

    if preds_reg.get("SARIMAX (ED + vaping lags)") is not None:
        try:
            sarimax_full = sm.tsa.SARIMAX(
                y_train,
                exog=X_train,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 52),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            future_pred = sarimax_full.forecast(steps=int(horizon), exog=X_future)
        except Exception:
            future_pred = None

    if future_pred is None:
        future_pred = np.repeat(float(y_train.iloc[-1]), int(horizon))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=train["Date"].tail(52), y=y_train.tail(52), mode="lines", name="History (last 52w)"))
    fig2.add_trace(go.Scatter(x=future_dates, y=future_pred, mode="lines+markers", name="Forecast"))
    fig2.update_layout(title=f"Future Forecast (next {int(horizon)} weeks)", xaxis_title="Date", yaxis_title="Resp_ED_Proxy")
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# TAB 3 — Rolling CV + Seasonality + What-if
# =========================
@st.cache_data(show_spinner=False)
def rolling_origin_cv(
    df: pd.DataFrame,
    exog_cols: list[str],
    horizon: int,
    initial_train_weeks: int,
    step_weeks: int,
    max_folds: int,
    include_prophet: bool,
) -> pd.DataFrame:
    """
    Rolling-origin cross-validation.

    Returns a fold-level metrics table for each model.
    """
    df = df.dropna(subset=["Resp_ED_Proxy"]).copy().sort_values("Date")
    df = df.dropna(subset=exog_cols).copy()

    if df.shape[0] < initial_train_weeks + horizon + 5:
        return pd.DataFrame()

    models = [
        "Seasonal Naive",
        "SARIMA (ED-only)",
        "SARIMAX (ED + vaping lags)",
        "Gradient Boosting (lags + exog lags)",
    ]
    if include_prophet and PROPHET_AVAILABLE:
        models.append("Prophet (ED + vaping lags)")

    rows = []
    season = 52

    def seasonal_naive(arr: np.ndarray, n_steps: int) -> np.ndarray:
        if len(arr) >= season:
            base = arr[-season:]
            reps = int(np.ceil(n_steps / len(base)))
            return np.tile(base, reps)[:n_steps]
        return np.repeat(float(arr[-1]), n_steps)

    fold = 0
    start_train_end = initial_train_weeks - 1

    while True:
        train_end_idx = start_train_end + fold * step_weeks
        test_start = train_end_idx + 1
        test_end = test_start + horizon

        if test_end > df.shape[0]:
            break

        fold += 1
        if fold > max_folds:
            break

        train = df.iloc[:test_start].copy()
        test = df.iloc[test_start:test_end].copy()

        y_train = train["Resp_ED_Proxy"].astype(float).values
        y_test = test["Resp_ED_Proxy"].astype(float).values
        X_train = train[exog_cols].astype(float).values
        X_test = test[exog_cols].astype(float).values

        # Seasonal naive
        yhat = seasonal_naive(y_train, horizon)
        rows.append({"Fold": fold, "Model": "Seasonal Naive", **regression_metrics(y_test, yhat, y_train)})

        # SARIMA (ED-only)
        try:
            sarima = sm.tsa.SARIMAX(
                y_train,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 52),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            yhat = np.asarray(sarima.forecast(steps=horizon))
            rows.append({"Fold": fold, "Model": "SARIMA (ED-only)", **regression_metrics(y_test, yhat, y_train)})
        except Exception:
            pass

        # SARIMAX
        try:
            sarimax = sm.tsa.SARIMAX(
                y_train,
                exog=X_train,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 52),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            yhat = np.asarray(sarimax.forecast(steps=horizon, exog=X_test))
            rows.append({"Fold": fold, "Model": "SARIMAX (ED + vaping lags)", **regression_metrics(y_test, yhat, y_train)})
        except Exception:
            pass

        # Gradient Boosting
        try:
            tmp = train[["Date", "Resp_ED_Proxy"] + exog_cols].copy().sort_values("Date")
            tmp["y_lag1"] = tmp["Resp_ED_Proxy"].shift(1)
            tmp["y_lag2"] = tmp["Resp_ED_Proxy"].shift(2)
            tmp["y_lag4"] = tmp["Resp_ED_Proxy"].shift(4)
            tmp = tmp.dropna()

            feat_cols = ["y_lag1", "y_lag2", "y_lag4"] + exog_cols
            gbr = GradientBoostingRegressor(random_state=42)
            gbr.fit(tmp[feat_cols].values, tmp["Resp_ED_Proxy"].values)

            tmp2 = df.iloc[:test_end].copy()
            tmp2["y_lag1"] = tmp2["Resp_ED_Proxy"].shift(1)
            tmp2["y_lag2"] = tmp2["Resp_ED_Proxy"].shift(2)
            tmp2["y_lag4"] = tmp2["Resp_ED_Proxy"].shift(4)
            tmp2 = tmp2.dropna(subset=feat_cols + ["Resp_ED_Proxy"])
            ttest = tmp2[tmp2["Date"].isin(test["Date"])].copy()

            if ttest.shape[0] == horizon:
                yhat = gbr.predict(ttest[feat_cols].values)
                rows.append({"Fold": fold, "Model": "Gradient Boosting (lags + exog lags)", **regression_metrics(y_test, yhat, y_train)})
        except Exception:
            pass

        # Prophet (optional)
        if include_prophet and PROPHET_AVAILABLE:
            try:
                dfp = (
                    train[["Date", "Resp_ED_Proxy"] + exog_cols]
                    .copy()
                    .rename(columns={"Date": "ds", "Resp_ED_Proxy": "y"})
                )
                mprop = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
                for c in exog_cols:
                    mprop.add_regressor(c)
                mprop.fit(dfp)

                future = test[["Date"] + exog_cols].copy().rename(columns={"Date": "ds"})
                yhat = mprop.predict(future)["yhat"].values
                rows.append({"Fold": fold, "Model": "Prophet (ED + vaping lags)", **regression_metrics(y_test, yhat, y_train)})
            except Exception:
                pass

    return pd.DataFrame(rows)


with tab3:
    st.subheader("Rolling CV + trends/seasonality + what-if")

    st.markdown("### Rolling-origin cross-validation")
    st.caption("Backtests models across multiple train/test splits to estimate out-of-sample performance.")

    cv_col1, cv_col2, cv_col3, cv_col4 = st.columns(4)
    with cv_col1:
        initial_train_weeks = st.number_input("Initial train size (weeks)", min_value=104, max_value=260, value=156, step=4)
    with cv_col2:
        step_weeks = st.number_input("Step size (weeks)", min_value=1, max_value=26, value=4, step=1)
    with cv_col3:
        max_folds = st.number_input("Max folds", min_value=3, max_value=20, value=8, step=1)
    with cv_col4:
        include_prophet = st.checkbox("Include Prophet in CV (slower)", value=False)

    with st.spinner("Running rolling CV..."):
        cv_df = rolling_origin_cv(
            master_lagged.copy(),
            exog_cols=vape_lag_cols,
            horizon=int(horizon),
            initial_train_weeks=int(initial_train_weeks),
            step_weeks=int(step_weeks),
            max_folds=int(max_folds),
            include_prophet=bool(include_prophet),
        )

    if cv_df.empty:
        st.warning("Not enough data for rolling-origin CV with current settings. Try smaller initial train or a wider date range.")
    else:
        st.write("#### Fold-level results")
        st.dataframe(cv_df[["Fold", "Model", "RMSE", "MAE", "MAPE", "MASE", "R2"]].sort_values(["Model", "Fold"]), use_container_width=True)

        st.write("#### CV summary (mean across folds)")
        cv_summary = (
            cv_df.groupby("Model", as_index=False)[["RMSE", "MAE", "MAPE", "MASE", "R2"]]
            .mean()
            .sort_values("RMSE", ascending=True)
        )
        st.dataframe(cv_summary, use_container_width=True)

        st.plotly_chart(px.box(cv_df, x="Model", y="RMSE", title="Rolling CV RMSE distribution by model"), use_container_width=True)
        st.plotly_chart(px.box(cv_df, x="Model", y="MAPE", title="Rolling CV MAPE distribution by model"), use_container_width=True)

        best_cv = cv_summary.iloc[0]["Model"]
        st.success(f"Best model by rolling-CV RMSE: **{best_cv}**")

    st.divider()

    st.subheader("Trends and seasonality")
    series = master.dropna(subset=["Resp_ED_Proxy"]).set_index("Date")["Resp_ED_Proxy"].asfreq("W-SUN")

    if series.dropna().shape[0] < 120:
        st.info("Not enough data points for reliable seasonal decomposition (need ~2+ years weekly).")
    else:
        decomp = seasonal_decompose(series, model="additive", period=52)

        trend = decomp.trend.dropna()
        st.plotly_chart(px.line(x=trend.index, y=trend.values, title="Trend component"), use_container_width=True)

        if len(trend) >= 12:
            recent = trend.iloc[-12:]
            slope = float((recent.iloc[-1] - recent.iloc[0]) / max(1, len(recent) - 1))
            direction = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "flat")
            st.markdown(
                f"""
**Trend summary**
- Recent direction (~12 weeks): **{direction}**
- Avg. change/week (trend component): **{slope:.4f}**
- Latest trend level: **{float(recent.iloc[-1]):.4f}**
                """.strip()
            )

        seasonal = decomp.seasonal.dropna()
        st.plotly_chart(px.line(x=seasonal.index, y=seasonal.values, title="Seasonality component (period=52)"), use_container_width=True)

        amp = float(seasonal.max() - seasonal.min())
        peak_date = seasonal.idxmax()
        trough_date = seasonal.idxmin()
        st.markdown(
            f"""
**Seasonality summary**
- Peak-to-trough swing: **{amp:.4f}**
- Peak around: **{peak_date.date()}**
- Low around: **{trough_date.date()}**
            """.strip()
        )

    st.divider()

    st.subheader("What-if: changing vaping demand signals")

    dfw = master_lagged.dropna(subset=["Resp_ED_Proxy"] + vape_exog_cols).copy()
    y = dfw["Resp_ED_Proxy"].astype(float)
    X_base_raw = dfw[vape_exog_cols].astype(float)

    h = int(horizon)
    last_date = dfw["Date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=7), periods=h, freq="W-SUN")

    scale = 1.0 + (what_if_pct / 100.0)
    last_raw = X_base_raw.iloc[-1] * scale
    _ = pd.DataFrame([last_raw.values] * h, columns=vape_exog_cols, index=future_dates)  # kept for clarity

    last_lag_row = dfw.iloc[-1][vape_lag_cols].astype(float).values
    X_future_lags = pd.DataFrame([last_lag_row] * h, columns=vape_lag_cols, index=future_dates)

    baseline_fc = None
    scenario_fc = None
    with st.spinner("Building what-if forecast..."):
        try:
            X_full = dfw[vape_lag_cols].astype(float)
            sar = sm.tsa.SARIMAX(
                y,
                exog=X_full,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 52),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)

            baseline_fc = sar.forecast(
                steps=h,
                exog=pd.DataFrame([dfw.iloc[-1][vape_lag_cols].astype(float).values] * h, columns=vape_lag_cols),
            )
            scenario_fc = sar.forecast(steps=h, exog=X_future_lags.values)
        except Exception:
            baseline_fc = None
            scenario_fc = None

    if baseline_fc is None or scenario_fc is None:
        st.info("What-if forecast unavailable (SARIMAX fit failed for current range).")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dfw["Date"].tail(52), y=y.tail(52), mode="lines", name="History (last 52w)"))
        fig.add_trace(go.Scatter(x=future_dates, y=baseline_fc, mode="lines+markers", name="Baseline forecast"))
        fig.add_trace(go.Scatter(x=future_dates, y=scenario_fc, mode="lines+markers", name=f"What-if forecast ({what_if_pct:+d}%)"))
        fig.update_layout(title="What-if forecast (updates as slider moves)", xaxis_title="Date", yaxis_title="Resp_ED_Proxy")
        st.plotly_chart(fig, use_container_width=True)

        delta = np.asarray(scenario_fc) - np.asarray(baseline_fc)
        avg_delta = float(np.mean(delta))
        pct_change = float(np.mean(delta) / (np.mean(baseline_fc) + 1e-9) * 100.0)

        st.markdown(
            f"""
**What-if summary (Demand% = {what_if_pct:+d}%):**
- Vaping demand signals scaled by **{scale:.2f}×**
- Avg. forecast change over next {h} weeks: **{avg_delta:+.4f}**
- Avg. % change vs baseline: **{pct_change:+.2f}%**
            """.strip()
        )

# =========================
# TAB 4 — Guide
# =========================
with tab4:
    st.subheader("Project Description & How to Use the Dashboard")
    st.markdown(
        """
### Goal
Forecast respiratory ED demand proxy (ILINet weighted ILI / ILI via Delphi FluView) using vaping sales as potential leading indicators.

### Modeling
- Lagged vaping regressors (1–4 weeks) to capture lead effects.
- ED-only baseline (SARIMA) vs ED+vaping (SARIMAX) to show incremental value.
- Rolling-origin cross-validation (Tab 3) for more stable out-of-sample comparisons.

### Metrics
Primary:
- RMSE, MAE, MAPE, MASE

Additional:
- MSE, RMSLE, R²

### Classification add-on
- Converts continuous demand into High vs Low using a threshold (default: 75th percentile of training).
- Confusion matrix + accuracy/precision/recall + ROC-AUC (requires both classes).

### Tabs
1) EDA — trends, distributions, correlations  
2) Models — forecasts + errors + ED-only vs ED+vaping impact  
3) Rolling CV + trend/seasonality + what-if simulations  
4) Guide — overview and definitions
"""
    )
