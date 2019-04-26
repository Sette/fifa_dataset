#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:05:07 2019

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


df2 = data.loc[:, 'Crossing':'Release Clause']
df1 = data[['Age', 'Overall', 'Value', 'Wage', 'Preferred Foot', 'Skill Moves', 'Position', 'Height', 'Weight']]
df = pd.concat([df1, df2], axis=1)

df = df.dropna()

def value_to_int(df_value):
    try:
        value = float(df_value[1:-1])
        suffix = df_value[-1:]

        if suffix == 'M':
            value = value * 1000000
        elif suffix == 'K':
            value = value * 1000
    except ValueError:
        value = 0
    return value
  
df['Value_float'] = df['Value'].apply(value_to_int)
df['Wage_float'] = df['Wage'].apply(value_to_int)
df['Release_Clause_float'] = df['Release Clause'].apply(lambda m: value_to_int(m))

def weight_to_int(df_weight):
    value = df_weight[:-3]
    return value
  
df['Weight_int'] = df['Weight'].apply(weight_to_int)
df['Weight_int'] = df['Weight_int'].apply(lambda x: int(x))

def height_to_int(df_height):
    try:
        feet = int(df_height[0])
        dlm = df_height[-2]

        if dlm == "'":
            height = round((feet * 12 + int(df_height[-1])) * 2.54, 0)
        elif dlm != "'":
            height = round((feet * 12 + int(df_height[-2:])) * 2.54, 0)
    except ValueError:
        height = 0
    return height

df['Height_int'] = df['Height'].apply(height_to_int)


df = df.drop(['Value', 'Wage', 'Release Clause', 'Weight', 'Height'], axis=1)

le_foot = preprocessing.LabelEncoder()
df["Preferred Foot"] = le_foot.fit_transform(df["Preferred Foot"].values)


for i in ['ST', 'CF', 'LF', 'LS', 'LW', 'RF', 'RS', 'RW']:
  df.loc[df.Position == i , 'Pos'] = 'Strikers' 

for i in ['CAM', 'CDM', 'LCM', 'CM', 'LAM', 'LDM', 'LM', 'RAM', 'RCM', 'RDM', 'RM']:
  df.loc[df.Position == i , 'Pos'] = 'Midfielder' 

for i in ['CB', 'LB', 'LCB', 'LWB', 'RB', 'RCB', 'RWB','GK']:
  df.loc[df.Position == i , 'Pos'] = 'Defender' 

df.drop(columns=["Position","Pos"],inplace=True)

plt.figure(figsize=(12, 8))
fig = sns.countplot(x = 'Pos', data =df)

plt.figure(figsize=(12, 8))

# Set up the matplotlib figure
f, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=False)
sns.despine(left=True)

sns.boxplot('Pos', 'Overall', data = df, ax=axes[0, 0])
sns.boxplot('Pos', 'Age', data = df, ax=axes[0, 1])

sns.boxplot('Pos', 'Height_int', data = df, ax=axes[1, 1])
sns.boxplot('Pos', 'Weight_int', data = df, ax=axes[1, 0])


from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
lasso = linear_model.Lasso()
from sklearn.model_selection import train_test_split

le_class = preprocessing.LabelEncoder()

df['Pos'] = le_class.fit_transform(df['Pos'])

y = df["Pos"]

X_train, X_test, y_train, y_test = train_test_split(df, y, 
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
    'num_class': 3}  # the number of classes that exist in this datset
num_round = 50  # the number of training iterations

bst = xgb.train(param, dtrain, num_round)
bst.dump_model('dump.raw.txt')
preds = bst.predict(dtest)
preds_train = bst.predict(dtrain)

best_preds = np.asarray([np.argmax(line) for line in preds])

best_preds_train = np.asarray([np.argmax(line) for line in preds_train])

cf_train = confusion_matrix(y_train, best_preds_train)

accuracy_score(y_train, best_preds_train)

cf = confusion_matrix(y_test, best_preds)

accuracy_score(y_test, best_preds)


