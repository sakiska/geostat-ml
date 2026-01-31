# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 15:18:36 2025

@author: sakis
"""
import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("Suppl_material.csv")
xcol = "Easting" if "Easting" in df.columns else "X"
ycol = "Northing" if "Northing" in df.columns else "Y"

x = df[xcol].values
y = df[ycol].values

elements = ["Cu", "Pb", "Zn"]
models = ["spherical", "exponential", "gaussian"]

def loo_ok_rmse(vals, variogram_model):
    preds = np.zeros_like(vals, dtype=float)

    for i in range(len(vals)):
        mask = np.ones(len(vals), dtype=bool)
        mask[i] = False

        ok = OrdinaryKriging(
            x[mask], y[mask], vals[mask],
            variogram_model=variogram_model,
            verbose=False, enable_plotting=False
        )

        z_pred, z_var = ok.execute("points", np.array([x[i]]), np.array([y[i]]))
        preds[i] = float(z_pred)

    rmse = np.sqrt(mean_squared_error(vals, preds))
    mae  = mean_absolute_error(vals, preds)
    r2   = r2_score(vals, preds)
    me   = np.mean(preds - vals)   # bias

    return rmse, mae, r2, me

all_results = []

for elem in elements:
    vals = df[elem].values.astype(float)

    for m in models:
        rmse, mae, r2, me = loo_ok_rmse(vals, m)
        all_results.append([elem, m, rmse, mae, r2, me])

res_df = pd.DataFrame(
    all_results,
    columns=["Element", "VariogramModel", "RMSE", "MAE", "R2", "ME(bias)"]
)

print(res_df)

best = res_df.loc[res_df.groupby("Element")["RMSE"].idxmin()]
print("\nBEST MODELS (min RMSE):")
print(best)
