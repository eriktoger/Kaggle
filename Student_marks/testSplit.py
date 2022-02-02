#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 20:49:40 2022

@author: erik
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
linear = LinearRegression()

sc_X = StandardScaler()
datafile = "Student_Marks.csv"
df = pd.read_csv(datafile)

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2,include_bias=False)


features = ["number_courses","time_study"]
pf.fit(df[ features ])


feat_array = pf.transform(df[features])
newDf =pd.DataFrame(feat_array,columns=pf.get_feature_names_out())
newDf["Marks"] = df["Marks"]


trials = 10
total_msq = 0;
while trials >0:
    train, test = train_test_split(df, test_size=0.2)
    
    x_train = train.iloc[:,:-1]
    y_train = train.iloc[:,-1:]
    
    x_test = test.iloc[:,:-1]
    y_test = test.iloc[:,-1:]
    
    
    transformed_train = sc_X.fit_transform(x_train)
    transformed_test = sc_X.transform(x_test)
    
    regressor = LinearRegression()
    
    regressor.fit(x_train,y_train.values.ravel())
    y_pred = regressor.predict(x_test)
    msq_error = mean_squared_error(y_test, y_pred)
    total_msq = total_msq + msq_error
    trials = trials - 1

testSplit_msq_error = total_msq /10