#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

datafile = "Student_Marks.csv"
df = pd.read_csv(datafile)

# I want to compare SVR vs LinearRegression
# I want to x2 and sqrt one of the variables and see

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
svr =SVR(kernel = 'rbf', gamma ='scale')
linear = LinearRegression()
regressors = [svr,linear]
polynomials = [1,0.5,2]


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()


lowest_msq = 9999999
bestSettings = []

for regressor in regressors:
    for p1 in polynomials:
        for p2 in polynomials:
            trials = 10
            current_msq = 0
            while trials > 0:
                
                train, test = train_test_split(df, test_size=0.2)

                x_train = train.iloc[:,:-1]
                y_train = train.iloc[:,-1:]

                x_test = test.iloc[:,:-1]
                y_test = test.iloc[:,-1:]
                
                poly_train = x_train.copy()
                poly_test = x_test.copy()
                
                poly_train["number_courses"] = x_train["number_courses"]**p1
                poly_test["number_courses"] = x_test["number_courses"]**p1
                
                poly_train["time_study"] = x_train["time_study"]**p2
                poly_test["time_study"] = x_test["time_study"]**p2
                
                transformed_train = sc_X.fit_transform(poly_train)
                transformed_test = sc_X.transform(poly_test)
                
                regressor.fit(transformed_train,y_train.values.ravel())
                y_pred = regressor.predict(transformed_test)
                msq_error = mean_squared_error(y_test, y_pred)
                
                current_msq = current_msq + msq_error
                trials = trials - 1
            
        
            if current_msq < lowest_msq:
                lowest_msq = current_msq
                bestSettings = [regressor,p1,p2]


results = [lowest_msq / 10, bestSettings]
