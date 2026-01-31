# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 22:08:33 2025

@author: XxX
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("Suppl_material.csv")
X = df[["X", "Y", "Z", "dist_fault",
        "Lith_Schist", "Lith_AndesiteTuff",
        "slope", "aspect"]]

y = df["Pb"]   # ➜ Change for each element

loo = LeaveOneOut()
preds = np.zeros(len(df))

for train_idx, test_idx in loo.split(X):
    
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_test = X.iloc[test_idx]

    model = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    preds[test_idx] = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y, preds))
mae  = mean_absolute_error(y, preds)
r2   = r2_score(y, preds)
me   = np.mean(preds - y)

print("RANDOM FOREST - Pb LOOCV")
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 :", r2)
print("ME :", me)