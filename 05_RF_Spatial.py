# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 09:03:32 2025

@author: sakis
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("Suppl_material.csv")
feature_cols = [
    "dist_to_Zn_hotspot", 
    "Zn_knn_mean",
    "Zn_knn_std",  # Change these features based on Table 2
]

X = df[feature_cols]
y = df["Zn"] # ➜ Change for each element

rf = RandomForestRegressor(
    n_estimators=500,
    max_features="sqrt",
    random_state=42
)

loo = LeaveOneOut()

preds = []
actual = []

for train_idx, test_idx in loo.split(X):
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    preds.append(y_pred[0])
    actual.append(y_test.values[0])

preds = np.array(preds)
actual = np.array(actual)

rmse = np.sqrt(mean_squared_error(actual, preds))
mae = mean_absolute_error(actual, preds)
r2 = r2_score(actual, preds)
me = np.mean(preds - actual)

print("RF (minimal spatial features) - Zn LOOCV")
print("RMSE:", rmse)
print("MAE :", mae)
print("R2  :", r2)
print("ME  :", me)