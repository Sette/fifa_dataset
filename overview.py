#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:57:10 2019

@author: bruno
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Path of the file to read
fifa_filepath = "data.csv"
# Read the file into a variable iris_data
data = pd.read_csv(fifa_filepath)
# Print the first 5 rows of the data
data.head()


y = data.Position

data = data.iloc[: ,28:-2]

data = pd.concat([data,y],axis=1)

# making new data frame with dropped NA values 
data = data.dropna(axis = 0, how ='any') 

y = data.Position

data.drop(columns=["Position"],inplace=True)
  
def extract_value_from(Value):
    try:
        out = Value.split('+')[0]
        return float(out)
    except:
        return(Value)


Columns = data.iloc[:,:27].columns.to_list()
[data.update(data[Column_Name].apply(lambda x: extract_value_from(x)))for Column_Name in Columns]

data = data.apply(pd.to_numeric) 

print(data.info())

le = preprocessing.LabelEncoder()
print("Number of Class: ",y.nunique())
y = le.fit_transform(y)
list(le.classes_)

from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
lasso = linear_model.Lasso()
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(data, y, 
                                                    test_size=0.20, 
                                                    random_state=42 )
print(X_train.shape)
print(X_test.shape)

print(X_train.info())

dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test,label=y_test)

param = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 27}  # the number of classes that exist in this datset
num_round = 200  # the number of training iterations

bst = xgb.train(param, dtrain, num_round)
bst.dump_model('dump.raw.txt')
preds = bst.predict(dtest)

best_preds = np.asarray([np.argmax(line) for line in preds])

cf = confusion_matrix(y_test, best_preds)

accuracy_score(y_test, best_preds)