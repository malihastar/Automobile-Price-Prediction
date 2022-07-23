# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 23:26:09 2021

@author: lenovo
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor

df=pd.read_csv("Data.csv")

df.drop('Unnamed: 0', inplace=True, axis=1)
x=df.drop(labels=['Fiyat'],  axis=1)


y=df['Fiyat'].values
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.3,random_state=20)

bag_model = BaggingRegressor(bootstrap_features = True)

bag_params = {"n_estimators": range(2,30)}
bag_cv_model = GridSearchCV(bag_model, bag_params, cv = 10)


bag_cv_model.fit(x_train, y_train)

predication_test=bag_cv_model.predict(x_test)


print("Accuracy=",bag_cv_model.score(x_test, y_test))

# Evaluating the Algorithm
bag_cv_model.best_params_
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,predication_test))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predication_test))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predication_test)))

# ccuracy= 0.9195281853726677
# Mean Absolute Error: 8.948096881310788
# Mean Squared Error: 197.62776393420683
# Root Mean Squared Error: 14.058014224427533

# bag_cv_model.best_params_
# Out[12]: {'n_estimators': 24}