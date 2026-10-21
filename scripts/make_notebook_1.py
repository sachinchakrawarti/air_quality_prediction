"""
make_notebook_1.py
------------------
Builds notebook/notebook.ipynb — 20 code cells, end-to-end AQI prediction.

Sections:
    01_imports
    02_data_processing
    03_data_visualization_eda
    04_feature_engineering
    05_train_test_split
    06_model_training
    07_model_testing
    08_comparison_with_base_paper
    09_save_models_and_summary

Usage:
    python scripts/make_notebook_1.py
"""

import os
import nbformat as nbf

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, "notebook")
OUT_PATH     = os.path.join(NOTEBOOK_DIR, "notebook.ipynb")
os.makedirs(NOTEBOOK_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Notebook skeleton
# ----------------------------------------------------------------------
nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

cells = []
def md(t):   cells.append(nbf.v4.new_markdown_cell(t.strip()))
def code(t): cells.append(nbf.v4.new_code_cell(t.strip()))

# ======================================================================
# TITLE
# ======================================================================
md("""
# Seasonal and Pollutant-Driven AQI Prediction Across Indian Cities Using Random Forest Regression

**Reference paper:** Gupta et al. (2023), *Journal of Environmental and Public Health*

**20 code cells** — one end-to-end workflow:

| # | Section |
|---|---|
| 01 | `01_imports` |
| 02–05 | `02_data_processing` |
| 06–10 | `03_data_visualization_eda` |
| 11–12 | `04_feature_engineering` |
| 13 | `05_train_test_split` |
| 14–17 | `06_model_training` |
| 18–19 | `07_model_testing` |
| 20 | `08_comparison_with_base_paper` + `09_save_models_and_summary` |

> **Run from inside `notebook/`.** `DATA_DIR = "../dataset"` resolves relative to the working directory.
""")

# ======================================================================
# CODE CELL 01 — imports
# ======================================================================
md("## 01 — `01_imports`")
code("""
import os, warnings, joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import KNNImputer

import shap

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

DATA_DIR  = "../dataset"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Imports OK")
print("DATA_DIR :", os.path.abspath(DATA_DIR))
print("MODEL_DIR:", os.path.abspath(MODEL_DIR))
""")

# ======================================================================
# CODE CELL 02 — load
# ======================================================================
md("## 02 — `02_data_processing` (load)")
code("""
cd = pd.read_csv(os.path.join(DATA_DIR, "city_day.csv"))
print("Raw shape:", cd.shape)
print("Columns:", list(cd.columns))
display(cd.head())
""")

# ======================================================================
# CODE CELL 03 — parse date
# ======================================================================
md("## 03 — `02_data_processing` (parse date → month, season)")
code("""
cd["Date"]  = pd.to_datetime(cd["Date"])
cd["month"] = cd["Date"].dt.month
cd["season"] = cd["month"].apply(
    lambda m: "winter"        if m in [12, 1, 2]   else
              "summer"        if m in [3, 4, 5]    else
              "monsoon"       if m in [6, 7, 8, 9] else
              "post-monsoon")

if "Xylene" in cd.columns:
    cd = cd.drop(columns=["Xylene"])

print("After date parsing:", cd.shape)
""")

# ======================================================================
# CODE CELL 04 — city filter
# ======================================================================
md("## 04 — `02_data_processing` (city filter — ≥ 500 valid AQI rows)")
code("""
valid  = cd.dropna(subset=["AQI"])["City"].value_counts()
CITIES = valid[valid >= 500].index.tolist()
print(f"{len(CITIES)} cities selected: {CITIES}")
cd = cd[cd["City"].isin(CITIES)].copy()
print("Rows after city filter:", len(cd))
""")

# ======================================================================
# CODE CELL 05 — impute + drop NaN AQI
# ======================================================================
md("## 05 — `02_data_processing` (drop NaN AQI + impute pollutants)")
code("""
POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
              "CO", "SO2", "O3", "Benzene", "Toluene"]

# ---- Drop rows where the TARGET (AQI) is NaN ----
before = len(cd)
cd = cd.dropna(subset=["AQI"]).reset_index(drop=True)
print(f"Dropped {before - len(cd)} rows with NaN AQI. Remaining: {len(cd)}")

# ---- Re-filter cities (some may fall below 500 after AQI drop) ----
counts = cd["City"].value_counts()
CITIES = counts[counts >= 500].index.tolist()
cd = cd[cd["City"].isin(CITIES)].copy()
print(f"{len(CITIES)} cities remaining: {CITIES}")
print("Rows after re-filter:", len(cd))

# ---- Impute pollutants per city ----
print("\\nMissing % BEFORE imputation:")
print((cd[POLLUTANTS].isnull().sum() / len(cd) * 100).round(2).to_string())

frames = []
for city, g in cd.groupby("City"):
    g = g.sort_values("Date").copy()

    # forward-fill then back-fill within city
    g[POLLUTANTS] = g[POLLUTANTS].ffill().bfill()

    # columns KNN can handle vs. columns still fully NaN
    still_nan_cols = [c for c in POLLUTANTS if g[c].isnull().all()]
    knn_cols       = [c for c in POLLUTANTS if c not in still_nan_cols and g[c].isnull().any()]

    if knn_cols:
        imputed = KNNImputer(n_neighbors=5).fit_transform(g[knn_cols])
        g[knn_cols] = pd.DataFrame(imputed, columns=knn_cols, index=g.index)

    # fully-NaN columns → global median fallback
    for c in still_nan_cols:
        g[c] = cd[c].median()

    frames.append(g)

cd = pd.concat(frames).reset_index(drop=True)

print("\\nMissing % AFTER imputation:")
print((cd[POLLUTANTS].isnull().sum() / len(cd) * 100).round(2).to_string())
print("Total missing remaining:", int(cd[POLLUTANTS].isnull().sum().sum()))
print("Final shape:", cd.shape)
""")

# ======================================================================
# CODE CELL 06 — Fig 1
# ======================================================================
md("## 06 — `03_data_visualization_eda` (Fig 1 — AQI distribution)")
code("""
fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=cd, x="City", y="AQI", ax=ax)
ax.set_title("Fig 1: AQI Distribution per City")
plt.xticks(rotation=45)
plt.tight_layout(); plt.show()
""")

# ======================================================================
# CODE CELL 07 — Fig 2
# ======================================================================
md("## 07 — `03_data_visualization_eda` (Fig 2 — monthly trend)")
code("""
monthly = cd.groupby(["City", "month"])["AQI"].mean().reset_index()
fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(data=monthly, x="month", y="AQI", hue="City", marker="o", ax=ax)
ax.set_title("Fig 2: Monthly AQI Trend per City")
ax.set_xticks(range(1, 13))
plt.tight_layout(); plt.show()
""")

# ======================================================================
# CODE CELL 08 — Fig 3
# ======================================================================
md("## 08 — `03_data_visualization_eda` (Fig 3 — seasonal bar)")
code("""
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=cd, x="season", y="AQI", estimator=np.mean,
            order=["winter", "summer", "monsoon", "post-monsoon"], ax=ax)
ax.set_title("Fig 3: Mean AQI by Season")
plt.tight_layout(); plt.show()
""")

# ======================================================================
# CODE CELL 09 — Fig 4
# ======================================================================
md("## 09 — `03_data_visualization_eda` (Fig 4 — correlation heatmap)")
code("""
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(cd[POLLUTANTS + ["AQI"]].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax)
ax.set_title("Fig 4: Pollutant Correlation Heatmap")
plt.tight_layout(); plt.show()
""")

# ======================================================================
# CODE CELL 10 — Fig 5 + sample rows
# ======================================================================
md("## 10 — `03_data_visualization_eda` (Fig 5 + sample rows)")
code("""
fig, ax = plt.subplots(figsize=(8, 4))
cd["AQI_Bucket"].value_counts().plot.bar(ax=ax, color="steelblue")
ax.set_title("Fig 5: AQI_Bucket Distribution")
plt.tight_layout(); plt.show()

print("Sample rows per city:")
display(cd.groupby("City").head(2)[["City", "Date", "PM2.5", "PM10", "NO2", "AQI"]])
""")

# ======================================================================
# CODE CELL 11 — one-hot season
# ======================================================================
md("## 11 — `04_feature_engineering` (one-hot season)")
code("""
cd = pd.get_dummies(cd, columns=["season"], prefix="season")
SEASON_COLS = [c for c in cd.columns if c.startswith("season_")]

FEATURES_POLL_ONLY = POLLUTANTS
FEATURES_FULL      = POLLUTANTS + ["month"] + SEASON_COLS

print("Poll-only :", FEATURES_POLL_ONLY)
print("Full      :", FEATURES_FULL)
""")

# ======================================================================
# CODE CELL 12 — describe
# ======================================================================
md("## 12 — `04_feature_engineering` (feature summary)")
code("""
display(cd[FEATURES_FULL + ["AQI"]].describe().T.round(2))
""")

# ======================================================================
# CODE CELL 13 — split
# ======================================================================
md("## 13 — `05_train_test_split` (chronological 80/20 per city)")
code("""
splits = {}
for city in CITIES:
    g   = cd[cd["City"] == city].sort_values("Date").reset_index(drop=True)
    cut = int(len(g) * 0.8)
    tr, te = g.iloc[:cut], g.iloc[cut:]

    assert tr["AQI"].notna().all(), f"{city}: NaN in train AQI"
    assert te["AQI"].notna().all(), f"{city}: NaN in test AQI"
    assert len(tr) > 0 and len(te) > 0, f"{city}: empty split"

    splits[city] = {"train": tr, "test": te}
    print(f"{city:15s} train={len(tr):4d}  test={len(te):4d}")

print(f"\\n{len(splits)} cities split successfully")
""")

# ======================================================================
# CODE CELL 14 — Linear Regression
# ======================================================================
md("## 14 — `06_model_training` (Linear Regression baseline)")
code("""
PARAM_GRID = {
    "n_estimators":      [200, 300],
    "max_depth":         [None, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
    "max_features":      ["sqrt", None],
}

results, best = [], {}

for city in CITIES:
    tr, te = splits[city]["train"], splits[city]["test"]

    y_tr = tr["AQI"].to_numpy(dtype=float)
    y_te = te["AQI"].to_numpy(dtype=float)

    feats    = FEATURES_POLL_ONLY
    X_tr_raw = tr[feats].to_numpy(dtype=float)
    X_te_raw = te[feats].to_numpy(dtype=float)

    if np.isnan(X_tr_raw).any() or np.isnan(X_te_raw).any():
        raise ValueError(f"{city}: NaN in features — re-check cell 05")

    scaler = StandardScaler().fit(X_tr_raw)
    m      = LinearRegression().fit(scaler.transform(X_tr_raw), y_tr)
    pred   = m.predict(scaler.transform(X_te_raw))
    mae    = mean_absolute_error(y_te, pred)

    results.append({"City": city, "Model": "LR",
                    "Accuracy": (1 - mae) * 100,
                    "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
                    "MAE": mae, "R2": r2_score(y_te, pred)})

    if city not in best:
        best[city] = {}
    best[city]["LR"] = {"model": m, "scaler": scaler, "features": feats}

print(f"LR done for {len(CITIES)} cities")
""")

# ======================================================================
# CODE CELL 15 — Untuned RFR
# ======================================================================
md("## 15 — `06_model_training` (Untuned RFR)")
code("""
for city in CITIES:
    tr, te = splits[city]["train"], splits[city]["test"]

    y_tr = tr["AQI"].to_numpy(dtype=float)
    y_te = te["AQI"].to_numpy(dtype=float)

    feats    = FEATURES_POLL_ONLY
    X_tr_raw = tr[feats].to_numpy(dtype=float)
    X_te_raw = te[feats].to_numpy(dtype=float)

    scaler = StandardScaler().fit(X_tr_raw)
    m      = RandomForestRegressor(n_estimators=100, random_state=42,
                                   n_jobs=-1).fit(scaler.transform(X_tr_raw), y_tr)
    pred   = m.predict(scaler.transform(X_te_raw))
    mae    = mean_absolute_error(y_te, pred)

    results.append({"City": city, "Model": "RFR-untuned",
                    "Accuracy": (1 - mae) * 100,
                    "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
                    "MAE": mae, "R2": r2_score(y_te, pred)})

    best[city]["RFR-untuned"] = {"model": m, "scaler": scaler, "features": feats}

print(f"Untuned RFR done for {len(CITIES)} cities")
""")

# ======================================================================
# CODE CELL 16 — Tuned RFR (pollutants only)
# ======================================================================
md("## 16 — `06_model_training` (Tuned RFR — pollutants only)")
code("""
for city in CITIES:
    tr, te = splits[city]["train"], splits[city]["test"]

    y_tr = tr["AQI"].to_numpy(dtype=float)
    y_te = te["AQI"].to_numpy(dtype=float)

    feats    = FEATURES_POLL_ONLY
    X_tr_raw = tr[feats].to_numpy(dtype=float)
    X_te_raw = te[feats].to_numpy(dtype=float)

    scaler = StandardScaler().fit(X_tr_raw)
    gs = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        PARAM_GRID,
        cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    gs.fit(scaler.transform(X_tr_raw), y_tr)

    m    = gs.best_estimator_
    pred = m.predict(scaler.transform(X_te_raw))
    mae  = mean_absolute_error(y_te, pred)

    results.append({"City": city, "Model": "RFR-tuned-poll",
                    "Accuracy": (1 - mae) * 100,
                    "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
                    "MAE": mae, "R2": r2_score(y_te, pred)})

    best[city]["RFR-tuned-poll"] = {"model": m, "scaler": scaler, "features": feats}
    print(f"  {city:15s} tuned-poll done")

print(f"Tuned RFR (pollutants only) done for {len(CITIES)} cities")
""")

# ======================================================================
# CODE CELL 17 — Tuned RFR (pollutants + seasonal)
# ======================================================================
md("## 17 — `06_model_training` (Tuned RFR — pollutants + seasonal) ← main model")
code("""
for city in CITIES:
    tr, te = splits[city]["train"], splits[city]["test"]

    y_tr = tr["AQI"].to_numpy(dtype=float)
    y_te = te["AQI"].to_numpy(dtype=float)

    feats    = FEATURES_FULL
    X_tr_raw = tr[feats].to_numpy(dtype=float)
    X_te_raw = te[feats].to_numpy(dtype=float)

    scaler = StandardScaler().fit(X_tr_raw)
    gs = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        PARAM_GRID,
        cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    gs.fit(scaler.transform(X_tr_raw), y_tr)

    m    = gs.best_estimator_
    pred = m.predict(scaler.transform(X_te_raw))
    mae  = mean_absolute_error(y_te, pred)

    results.append({"City": city, "Model": "RFR-tuned-seasonal",
                    "Accuracy": (1 - mae) * 100,
                    "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
                    "MAE": mae, "R2": r2_score(y_te, pred)})

    best[city]["RFR-tuned-seasonal"] = {"model": m, "scaler": scaler, "features": feats}
    print(f"  {city:15s} tuned-seasonal done")

print(f"Tuned RFR (pollutants + seasonal) done for {len(CITIES)} cities")
""")

# ======================================================================
# CODE CELL 18 — metrics pivot
# ======================================================================
md("## 18 — `07_model_testing` (metrics tables)")
code("""
res_df = pd.DataFrame(results)

print("=" * 60)
print("ACCURACY (higher = better)")
print("=" * 60)
display(res_df.pivot(index="City", columns="Model", values="Accuracy").round(2))

print("\\n" + "=" * 60)
print("RMSE (lower = better)")
print("=" * 60)
display(res_df.pivot(index="City", columns="Model", values="RMSE").round(3))

print("\\n" + "=" * 60)
print("R² (higher = better)")
print("=" * 60)
display(res_df.pivot(index="City", columns="Model", values="R2").round(3))

print(f"\\nRow count: {len(res_df)}  (expected {4 * len(CITIES)})")
print("Models:", sorted(res_df["Model"].unique()))
""")

# ======================================================================
# CODE CELL 19 — figures + SHAP
# ======================================================================
md("## 19 — `07_model_testing` (Figs 6–9)")
code("""
# Fig 6 — Seasonal gain
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(CITIES)); w = 0.35
a = [res_df[(res_df.City == c) & (res_df.Model == "RFR-tuned-poll")]["Accuracy"].values[0]     for c in CITIES]
b = [res_df[(res_df.City == c) & (res_df.Model == "RFR-tuned-seasonal")]["Accuracy"].values[0] for c in CITIES]
ax.bar(x - w/2, a, w, label="Pollutants only",       color="#4C72B0")
ax.bar(x + w/2, b, w, label="Pollutants + Seasonal", color="#DD8452")
ax.set_xticks(x); ax.set_xticklabels(CITIES, rotation=45)
ax.set_ylabel("Accuracy (%) = (1 − MAE) × 100")
ax.set_title("Fig 6: Seasonal Features Gain")
ax.legend(); plt.tight_layout(); plt.show()

# Fig 7 — Actual vs Predicted
n = len(CITIES); cols = 2; rows = (n + 1) // 2
fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows)); axes = axes.flatten()
for i, city in enumerate(CITIES):
    te  = splits[city]["test"]; art = best[city]["RFR-tuned-seasonal"]
    pr  = art["model"].predict(art["scaler"].transform(te[art["features"]].to_numpy(dtype=float)))
    axes[i].scatter(te["AQI"], pr, alpha=0.5, s=15)
    lim = [0, max(te["AQI"].max(), pr.max())]
    axes[i].plot(lim, lim, "r--")
    axes[i].set_title(city); axes[i].set_xlabel("Actual"); axes[i].set_ylabel("Predicted")
for j in range(n, len(axes)): axes[j].axis("off")
plt.suptitle("Fig 7: Actual vs Predicted AQI")
plt.tight_layout(); plt.show()

# Fig 8 — SHAP heatmap
shap_rows = []
for city in CITIES:
    art  = best[city]["RFR-tuned-seasonal"]
    X_te = art["scaler"].transform(splits[city]["test"][art["features"]].to_numpy(dtype=float))
    sv   = shap.TreeExplainer(art["model"]).shap_values(X_te)
    for feat, val in zip(art["features"], np.abs(sv).mean(axis=0)):
        shap_rows.append({"City": city, "Feature": feat, "SHAP": float(val)})

heat = pd.DataFrame(shap_rows).pivot(index="City", columns="Feature", values="SHAP").fillna(0)
plt.figure(figsize=(12, 5))
sns.heatmap(heat, cmap="YlOrRd", annot=True, fmt=".2f")
plt.title("Fig 8: SHAP Importance — Cities × Features")
plt.tight_layout(); plt.show()

# Fig 9 — Beeswarm for best city
top_city = (res_df[res_df.Model == "RFR-tuned-seasonal"]
            .sort_values("Accuracy", ascending=False).iloc[0]["City"])
art  = best[top_city]["RFR-tuned-seasonal"]
X_te = art["scaler"].transform(splits[top_city]["test"][art["features"]].to_numpy(dtype=float))
sv   = shap.TreeExplainer(art["model"]).shap_values(X_te)
print("Best city:", top_city)
shap.summary_plot(sv, X_te, feature_names=art["features"])
""")

# ======================================================================
# CODE CELL 20 — compare + save
# ======================================================================
md("## 20 — `08_comparison_with_base_paper` + `09_save_models_and_summary`")
code("""
# ---- Comparison with Gupta et al. ----
base_paper = pd.DataFrame({
    "City": ["New Delhi", "Bangalore", "Kolkata", "Hyderabad"],
    "Base_RFR_imbalanced": [79.4764, 67.7038, 90.9700, 78.3672],
    "Base_RFR_balanced":   [84.7284, 90.3071, 93.7438, 97.6080],
})

# Normalize names in case your dataset uses "Delhi" not "New Delhi"
name_map = {"Delhi": "New Delhi"}
your = (res_df[res_df.Model == "RFR-tuned-seasonal"][["City", "Accuracy"]]
        .rename(columns={"Accuracy": "Your_RFR_seasonal"}))
your["City"] = your["City"].replace(name_map)

comp = base_paper.merge(your, on="City", how="left")
comp["Δ_vs_imbalanced"] = comp["Your_RFR_seasonal"] - comp["Base_RFR_imbalanced"]
comp["Δ_vs_balanced"]   = comp["Your_RFR_seasonal"] - comp["Base_RFR_balanced"]

print("Comparison with Gupta et al. (2023):")
display(comp.round(2))

# ---- Save artifacts for Streamlit UI ----
for city in CITIES:
    art = best[city]["RFR-tuned-seasonal"]
    joblib.dump(art["model"],    f"{MODEL_DIR}/rfr_{city}.pkl")
    joblib.dump(art["scaler"],   f"{MODEL_DIR}/scaler_{city}.pkl")
    joblib.dump(art["features"], f"{MODEL_DIR}/features_{city}.pkl")

print("\\nModels saved to:", os.path.abspath(MODEL_DIR))

# ---- Summary ----
mean_gain = (res_df[res_df.Model == "RFR-tuned-seasonal"]["Accuracy"].mean()
             - res_df[res_df.Model == "RFR-tuned-poll"]["Accuracy"].mean())

print("\\n=== FINAL SUMMARY ===")
print(f"Cities analysed   : {len(CITIES)}")
print(f"Best accuracy     : {res_df['Accuracy'].max():.2f}%  "
      f"({res_df.loc[res_df['Accuracy'].idxmax(), 'City']})")
print(f"Mean seasonal gain: {mean_gain:.2f} pp")
""")

# ======================================================================
# Save
# ======================================================================
nb["cells"] = cells
nbf.write(nb, OUT_PATH)

n_code = sum(1 for c in cells if c.cell_type == "code")
n_md   = sum(1 for c in cells if c.cell_type == "markdown")
print(f"✅ Wrote {OUT_PATH}")
print(f"   {len(cells)} cells total  ({n_code} code, {n_md} markdown)")
assert n_code == 20, f"Expected 20 code cells, got {n_code}"
print("   ✅ exactly 20 code cells confirmed")