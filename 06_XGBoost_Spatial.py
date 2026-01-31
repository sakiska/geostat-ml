# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 14:28:46 2025

@author: sakis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

df = pd.read_csv("Suppl_material.csv")
feature_cols = [
    "dist_to_Zn_hotspot",
    "Zn_knn_mean",
    "Zn_knn_std" ,   # Change these features based on Table 2
]

X = df[feature_cols]
y = df["Zn"] # ➜ Change for each element

loo = LeaveOneOut()

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    reg_lambda=1.0,
    reg_alpha=0.0,
    random_state=42
)

preds = []
actual = []

for train_idx, test_idx in loo.split(X):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test,  y_test  = X.iloc[test_idx],  y.iloc[test_idx]

    xgb.fit(X_train, y_train)
    pred = xgb.predict(X_test)

    preds.append(pred[0])
    actual.append(y_test)

preds = np.array(preds)
actual = np.array(actual)

rmse = np.sqrt(mean_squared_error(actual, preds))
mae = mean_absolute_error(actual, preds)
r2  = r2_score(actual, preds)
me  = np.mean(preds - actual)

print("XGBoost (minimal spatial features) - Zn LOOCV")
print("RMSE:", rmse)
print("MAE :", mae)
print("R2  :", r2)
print("ME  :", me)