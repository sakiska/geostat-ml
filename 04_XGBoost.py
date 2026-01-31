# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 10:11:05 2025

@author: sakis
"""

from xgboost import XGBRegressor

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

df = pd.read_csv("Suppl_material.csv")

X = df[["X", "Y", "Z", "dist_fault", "Lith_Schist", "Lith_AndesiteTuff", "slope", "aspect"]]
y = df["Zn"] # ➜ Change for each element

loo = LeaveOneOut()

preds = []
actual = []

for train_idx, test_idx in loo.split(X):
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    
    preds.append(y_pred[0])
    actual.append(y_test.values[0])

preds = np.array(preds)
actual = np.array(actual)

rmse = np.sqrt(mean_squared_error(actual, preds))
mae = mean_absolute_error(actual, preds)
r2 = r2_score(actual, preds)
me = np.mean(preds - actual)

print("XGBoost - Zn LOOCV")
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 :", r2)
print("ME :", me)
