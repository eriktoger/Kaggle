# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:36:26 2019

@author: jv5191
"""
# i should redo this one
# age should be classes. 0-5 6-12 12-15 16-24 or something
# create IsAlone from parch/familysize/kids 
# use cabin/title

## https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
#ensembling.py has some code in it from this notebook


import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')



# only use the relevent features
trainSlim = train[['Pclass','Sex','Age','SibSp','Parch','Fare']] # fare?
y_train = train[ ['Survived'] ]
testSlim = test[['Pclass', 'Sex','Age','SibSp','Parch','Fare']] # fare?




from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

#onehotencoder = OneHotEncoder(categories = [0])
#trainSlim = onehotencoder.fit_transform(trainSlim).toarray()

ct = ColumnTransformer(
    [('oh_enc', OneHotEncoder(sparse=False), [0,1]),],  # the column numbers I want to apply this to
    remainder='passthrough'  # This leaves the rest of my columns in place
)
trainSlim = ct.fit_transform(trainSlim) # Notice the output is a string
testSlim = ct.fit_transform(testSlim)


# remove 1 from pClass 0 and 1  from sex 3
import numpy as np
trainSlim = trainSlim[:, [1,2,4,5,6,7,8]]
testSlim = testSlim[:, [1,2,4,5,6,7,8]]

# nan with mean
import numpy.ma as ma
trainSlim = np.where(np.isnan(trainSlim), ma.array(trainSlim, mask=np.isnan(trainSlim)).mean(axis=0), trainSlim)
testSlim = np.where(np.isnan(testSlim), ma.array(testSlim, mask=np.isnan(testSlim)).mean(axis=0), testSlim)


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# fit calculats the parametes, mean ans standard deviation in s'this case
#transform applies them and thus transforms
trainSlim = sc_X.fit_transform(trainSlim)
testSlim = sc_X.transform(testSlim)

# Fitting Logistic Regression to the Training set

'''
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(trainSlim, y_train)
'''

'''
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p =2)
classifier.fit(trainSlim, y_train)
'''


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, gamma ='scale')
classifier.fit(trainSlim, y_train)

'''
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(trainSlim, y_train)
'''

'''
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(trainSlim, y_train)
'''

'''
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state =0 )
classifier.fit(trainSlim, y_train)
'''

# Predicting the Test set results
y_pred = classifier.predict(testSlim)
y_pred.resize( (418,1) )
testPI = test.iloc[:,[0]].values
testPI.resize( (418,1) )
y_pred = np.array(y_pred)
testPI = np.array(testPI)

answer = np.concatenate( (testPI,y_pred) , axis = 1)


#y_pred.resize( (418,1) )
#testPI.resize( (418,1))
df = pd.DataFrame (data = answer ,columns = ['PassengerId','Survived'])
df.to_csv(path_or_buf = "~/Downloads/Machine learning/Kaggle/Titanic/titanicSVM4.csv" , index = False)
#df = pd.DataFrame({'name': ['Raphael', 'Donatello'],
#                   'mask': ['red', 'purple'],
#                    'weapon': ['sai', 'bo staff']})
#df.to_csv(index=False)