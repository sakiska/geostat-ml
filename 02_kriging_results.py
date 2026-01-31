# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 15:15:54 2025

@author: sakis
"""

import pandas as pd
from pykrige.ok import OrdinaryKriging

df = pd.read_csv("Suppl_material.csv")
xcol = "Easting" if "Easting" in df.columns else "X"
ycol = "Northing" if "Northing" in df.columns else "Y"

x = df[xcol].values
y = df[ycol].values

best_models = {"Cu":"exponential","Pb":"spherical","Zn":"spherical"}

for elem, vmodel in best_models.items():
    vals = df[elem].values.astype(float)

    ok = OrdinaryKriging(
        x, y, vals,
        variogram_model=vmodel,
        verbose=False, enable_plotting=False
    )

    params = ok.variogram_model_parameters 

    sill   = params[0]
    vrange = params[1]
    nugget = params[2]

    print(f"\n{elem}  ({vmodel})")
    print("  sill   =", sill)
    print("  range  =", vrange)
    print("  nugget =", nugget)