"""
train_aruco_height_model.py

Requirements:
  pip install numpy pandas scikit-learn matplotlib joblib

Input CSV columns required:
  u_center, v_center, Z_measured, Z_true

Outputs:
  - saved models: polynomial_model.joblib and rf_model.joblib
  - some diagnostic plots saved to disk
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

# ---------- User settings ----------
CSV_PATH = "dataset.csv"   # your logged csv
OUTPUT_DIR = "height_model_out"
TEST_SIZE = 0.2
RANDOM_STATE = 42
# Use Z_measured (depth/fusion) as additional input? set True to include it
USE_Z_MEASURED = True
# -----------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) Load data
df = pd.read_csv(CSV_PATH)

# rename to expected columns if necessary - adjust if your columns differ
# expected: u_center, v_center, Z_measured, Z_true
# If you recorded base_Z_fused as "base_Z_fused", change below:
# df = df.rename(columns={'base_Z_fused':'Z_measured', 'base_Z_depth':'Z_measured', ...})

required_cols = ["u_center", "v_center", "Z_measured", "Z_true"]
for c in required_cols:
    if c not in df.columns:
        raise RuntimeError(f"CSV missing required column: {c}")

# drop rows with NaNs in important fields
df = df.dropna(subset=required_cols).reset_index(drop=True)

# Features: either [u,v] (only), or [u,v,Z_measured]
if USE_Z_MEASURED:
    X = df[["u_center", "v_center", "Z_measured"]].values
else:
    X = df[["u_center", "v_center"]].values
y = df["Z_true"].values   # target is ground truth height

# normalize pixel coords optionally (improves numeric conditioning)
# scale u,v to [-1,1] based on image size (if you know width/height), otherwise use mean/std:
u_mean, u_std = X[:,0].mean(), X[:,0].std() if X[:,0].std()>0 else 1.0
v_mean, v_std = X[:,1].mean(), X[:,1].std() if X[:,1].std()>0 else 1.0
X_norm = X.copy()
X_norm[:,0] = (X[:,0] - u_mean) / u_std
X_norm[:,1] = (X[:,1] - v_mean) / v_std
if USE_Z_MEASURED and X.shape[1] > 2:
    z_mean, z_std = X[:,2].mean(), X[:,2].std() if X[:,2].std()>0 else 1.0
    X_norm[:,2] = (X[:,2] - z_mean) / z_std

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ------------------ Option A: 2D Polynomial surface (Ridge) ------------------
# We fit a polynomial of degree 2 or 3 (quadratic usually enough).
poly_degree = 3
poly_model = make_pipeline(PolynomialFeatures(degree=poly_degree, include_bias=False),
                           Ridge(alpha=1e-3))

poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)
print(f"[Polynomial deg={poly_degree}] Test MSE: {mse_poly:.6f}  R2: {r2_poly:.4f}")

joblib.dump({
    "model": poly_model,
    "u_mean": u_mean, "u_std": u_std,
    "v_mean": v_mean, "v_std": v_std,
    "z_mean": (z_mean if USE_Z_MEASURED else None),
    "z_std": (z_std if USE_Z_MEASURED else None),
    "use_z_measured": USE_Z_MEASURED
}, os.path.join(OUTPUT_DIR, "polynomial_model.joblib"))


# ------------------ Option B: RandomForestRegressor (flexible) ------------------
# Quick hyperparameter search
param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [8, 15, None],
    "min_samples_leaf": [1, 3]
}
# We'll use a simple RandomForest and GridSearchCV (small grid)
rf = RandomForestRegressor(random_state=RANDOM_STATE)
# Use a simple grid search on small data; if large data, you can skip GridSearch
grid = GridSearchCV(rf, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"[RandomForest best] params={grid.best_params_} Test MSE: {mse_rf:.6f}  R2: {r2_rf:.4f}")

joblib.dump({
    "model": best_rf,
    "u_mean": u_mean, "u_std": u_std,
    "v_mean": v_mean, "v_std": v_std,
    "z_mean": (z_mean if USE_Z_MEASURED else None),
    "z_std": (z_std if USE_Z_MEASURED else None),
    "use_z_measured": USE_Z_MEASURED
}, os.path.join(OUTPUT_DIR, "rf_model.joblib"))


# ------------------ Diagnostics plots ------------------
def save_scatter(y_true, y_pred, title, filename):
    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    mx = max(y_true.max(), y_pred.max())
    mn = min(y_true.min(), y_pred.min())
    plt.plot([mn, mx], [mn, mx], 'k--')
    plt.xlabel("Z_true")
    plt.ylabel("Z_pred")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

save_scatter(y_test, y_pred_poly, "Poly predicted vs true", "poly_scatter.png")
save_scatter(y_test, y_pred_rf, "RF predicted vs true", "rf_scatter.png")

# Also make uv heatmap of residuals for the better model (pick RF if lower mse else poly)
better_model = "rf" if mse_rf < mse_poly else "poly"
if better_model == "rf":
    y_pred_all = best_rf.predict(X_norm)
else:
    y_pred_all = poly_model.predict(X_norm)

residuals = y - y_pred_all
# plot residuals as scatter in pixel space (de-normalize first)
u_pixels = X[:,0]  # original pixel u (not normalized)
v_pixels = X[:,1]
plt.figure(figsize=(8,6))
sc = plt.scatter(u_pixels, v_pixels, c=residuals, s=20, cmap='RdBu_r')
plt.colorbar(sc, label='residual (Z_true - Z_pred) [m]')
plt.gca().invert_yaxis()
plt.xlabel("u (pixels)")
plt.ylabel("v (pixels)")
plt.title(f"Residuals heat (better_model={better_model})")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residuals_uv.png"))
plt.close()

print(f"Saved models & diagnostic plots to {OUTPUT_DIR}")
