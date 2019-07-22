#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 09:55:14 2019

@author: erik
"""


# I really should go back and do this one again.
# compare it with some other kernels!
#the same with other.


import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

features = list(train)

#remove id and pricing
featuresX = features[1:-1] 
X_train = train[featuresX]
y_train = train[['SalePrice']]
X_train['MSSubClass'] = X_train['MSSubClass'].apply(str) # seem to work but gives warning
test['MSSubClass'] = test['MSSubClass'].apply(str)
id_nr = test['Id']
test= test[featuresX] # drop id

X_train['train'] = 1
test['train'] = 0
combined = pd.concat([X_train,test])

#what to do if onehotencoder encodes different?
# https://medium.com/@vaibhavshukla182/how-to-solve-mismatch-in-train-and-test-set-after-categorical-encoding-8320ed03552f
ohe = pd.get_dummies(combined)
X_train = ohe[ohe['train'] == 1 ]
test = ohe[ohe['train'] == 0 ]
X_train.drop(['train'],axis = 1, inplace = True)
test.drop(['train'],axis = 1, inplace = True)


#replace nan
for column in X_train:
    if X_train[column].isnull().any():
       X_train[column].fillna((X_train[column].mean()), inplace=True) 
       #print('{0} has {1} null values'.format(column, X_train[column].isnull().sum()))

for column in test:
    if test[column].isnull().any():
       test[column].fillna((test[column].mean()), inplace=True) 
       #print('{0} has {1} null values'.format(column, test[column].isnull().sum()))


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# fit calculats the parametes, mean ans standard deviation in s'this case
#transform applies them and thus transforms
X_train = sc_X.fit_transform(X_train)
test = sc_X.transform(test)

'''
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, gamma ='scale')
classifier.fit(X_train, y_train)
'''

'''
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
'''

'''
from sklearn.linear_model import Lasso, LassoCV
classifier = Lasso(alpha =0.001, random_state=1)
classifier.fit(X_train,y_train)
'''

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(test)

answer = pd.concat( [id_nr,pd.Series(data = y_pred, name = 'SalePrice')], axis = 1)

df = pd.DataFrame (data = answer ,columns = ['Id','SalePrice'])
df.to_csv(path_or_buf = "~/Downloads/Machine learning/Kaggle/House Prices/hpGNB.csv" , index = False)
#

#drop id
# labelencode mssubclass
#labelencode mszoning
# lotFrontaage replace nan with mean
# label encodestreet just 2 so change pave to 0 and grvl to 1
# drop ally? or set nan to 0 and rest to 1? or nan to string NA and labelencode
#lotshape labelencode?
# landcountor labelencode?
#utilities labelencode
#lotconfig labelencode
#landslope labelencode
#neigherborhood labelencode
#condition1 onehot encode
# condition2 onehot encode
# bldgType onehotencode
#housestyle onehotencode (same as mssubclass?)
#roofstyle onehotencode
# Roofmatl onehoteencode (need this and roofstye?)
#exterior1 onehotencode
#exterior2 onehotencode (often same as 1)
#masvnrArea onehoteencode



# create a minimalistsk model of the 10 features i think is the most important?
