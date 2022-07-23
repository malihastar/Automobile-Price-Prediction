
"""
Created on Wed Oct 27 19:29:19 2021

@author: lenovo
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
df=pd.read_csv("Data.csv")
df.drop('Unnamed: 0', inplace=True, axis=1)

y=df['Fiyat'].values

df.head()

x=df.drop(labels=['Fiyat'],  axis=1)
x_tarin, x_test, y_train, y_test=train_test_split(x,y, test_size=0.3,random_state=20)

params = {'max_depth': list(range(1,20)),
            'max_features': [3,5,10,15,20,30],
            'n_estimators' : [100, 200,300, 400,500,1000]}

model=RandomForestRegressor(random_state = 42)

randomF_cv_model = GridSearchCV(model, 
                           params, 
                           cv = 10, 
                            n_jobs = -1)
randomF_cv_model.fit(x_tarin, y_train)
predication_test=randomF_cv_model.predict(x_test)

from sklearn import metrics

print("Accuracy=",randomF_cv_model.score(x_test, y_test))

# Evaluating the Algorithm

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,predication_test))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predication_test))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predication_test)))


# randomF_cv_model.best_estimator_
# randomF_cv_model.best_params_
# Accuracy= 0.9263511184497717
# Mean Absolute Error: 8.216794025520688
# Mean Squared Error: 180.87157403411211
# Root Mean Squared Error: 13.448850286701541


# randomF_cv_model.best_params_
# Out[213]: {'max_depth': 9, 'max_features': 15, 'n_estimators': 300}