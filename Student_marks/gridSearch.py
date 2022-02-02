#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 20:45:38 2022

@author: erik
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge,Lasso, LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.metrics import mean_squared_error

datafile = "Student_Marks.csv"
df = pd.read_csv(datafile)

kf = KFold()

ridgeAlpha = np.geomspace(0.1, 3, 10)
lassoAlpha =np.geomspace(1e-9, 1e0, num=10)

regressors = [ [Ridge, ridgeAlpha], [Lasso, lassoAlpha],[LinearRegression, None]  ]

results= []
for  regressor, alpha in regressors:

    estimator = Pipeline([("scaler", StandardScaler()),
            ("polynomial_features", PolynomialFeatures()),
            ("regression", regressor())])

    params = {
        'polynomial_features__degree': [1, 2, 3],
        'regression__alpha': alpha
    }
    
    filteredParams =  {k: v for k, v in  params.items() if v is not None}
    
    grid = GridSearchCV(estimator, filteredParams, cv=kf)
    X = df.drop("Marks",axis=1)
    y = df.Marks
    grid.fit(X, y)
    y_pred = grid.predict(X)
    msq_error = mean_squared_error(y, y_pred)
    results.append([regressor.__name__, msq_error])
