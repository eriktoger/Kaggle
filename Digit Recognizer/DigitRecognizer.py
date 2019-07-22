#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:13:52 2019

@author: erik
"""
#https://www.kaggle.com/c/digit-recognizer/discussion/61480
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv('train.csv')
X_train = train.iloc[:, 1:].values
y_train = train.iloc[:,0].values
test = pd.read_csv('test.csv')

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, gamma ='scale')
classifier.fit(X_train, y_train)

test = test.iloc[:].values
y_pred = classifier.predict(test)

idx = np.arange(1, len(y_pred)+1)
idx.resize(28000,1)
y_pred.resize(28000,1)

answer = np.concatenate( (idx,y_pred) , axis = 1)


#y_pred.resize( (418,1) )
#testPI.resize( (418,1))
df = pd.DataFrame (data = answer ,columns = ['ImageId','Label'])
df.to_csv(path_or_buf = "~/Downloads/Machine learning/Kaggle/Digit Recognizer/answer.csv" , index = False)