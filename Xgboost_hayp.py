# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 20:22:24 2021

@author: lenovo
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import metrics

df=pd.read_csv("Data.csv")

df.drop('Unnamed: 0', inplace=True, axis=1)
x=df.drop(labels=['Fiyat'],  axis=1)


y=df['Fiyat'].values
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.3,random_state=20)


params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],
 'n_estimators'     : [100, 200, 500, 1000],
    
}

xgb_model=XGBRegressor()

random_search=RandomizedSearchCV(xgb_model,param_distributions=params,n_iter=5,n_jobs=-1,cv=5,verbose=3)
random_search.fit(x_train, y_train)

random_search.best_estimator_
random_search.best_params_

model=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.7, enable_categorical=False,
              gamma=0.0, gpu_id=-1, importance_type=None,
              interaction_constraints='', learning_rate=0.1, max_delta_step=0,
              max_depth=8, min_child_weight=3,
              monotone_constraints='()', n_estimators=200, n_jobs=4,
              num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
              validate_parameters=1, verbosity=None)
model = model.fit(x_train,y_train)
from sklearn.model_selection import cross_val_score
score=cross_val_score(model,x,y,cv=10)

predication_test = model.predict(x_test)
print("Accuracy=",model.score(x_test, y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,predication_test))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predication_test))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predication_test)))

# Accuracy= 0.9337412993567273
# Mean Absolute Error: 7.74601967504125
# Mean Squared Error: 162.72230109333717
# Root Mean Squared Error: 12.756265170234474

# random_search.best_estimator_
# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=0.4, gamma=0.4,
#              importance_type='gain', learning_rate=0.05, max_delta_step=0,
#              max_depth=6, min_child_weight=1, missing=None, n_estimators=500,
#              n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#              silent=None, subsample=1, verbosity=1)

# random_search.best_params_
# {'colsample_bytree': 0.4,
#  'gamma': 0.4,
#  'learning_rate': 0.05,
#  'max_depth': 6,
#  'min_child_weight': 1,
#  'n_estimators': 500}