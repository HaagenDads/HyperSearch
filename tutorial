## HyperSearch Tutorial (v1.1.3a)
## In this guide, I'll help you through the inhouse library 'HyperSearch', developped in june 2018. This library contains functions
## used to effectively optimize hyperparameters. It was first conceived for XGBoost, but any machine learning problem, from regression 
## to classification, could make use of these functions. It is built to be as flexible as possible, and yet be able to run with very 
## little user input. Precision is still proportional to the runtime, but a user can adjust it according to their needs.

## XGBoost Regression example
## First step involves importing your packages and your datasets. For this example, we'll use the public Boston Housing dataset.


import numpy as np
import pandas as pd
import xgboost as xgb

import HyperSearch.core as hyp

from sklearn import datasets
boston = datasets.load_boston()

X = pd.DataFrame(boston.data) 
y = pd.DataFrame(boston.target)

X.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRARIO','B','LSTAT']
y.columns = ['PRICE']

## Data preparation has already been done in this dataset, but XGBoost requires One Hot Encoding for categorical variables, 
## though it doesn't care about normalization.

## In order to evaluate true model performance, one ought to split the dataset in training vs validation sets. This step should not
## be overlooked, and the split has to be carefully done.


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.4, random_state=200)

## A - Naive XGBoost regression
## This block performs a naive XGBoost regression on the price. In addition to giving poor performances, its overfitting yields
## an RMSE twice as large on validation set.

xgboost_reg = xgb.XGBRegressor(objective='reg:linear', random_state=413, n_jobs=10)
xgboost_reg.fit(Xt, yt)

yt_pred = xgboost_reg.predict(Xt)
yv_pred = xgboost_reg.predict(Xv)

print('RMSE-training: ' + str(np.sqrt(mean_squared_error(yt, yt_pred))))
print('RMSE-validation: ' + str(np.sqrt(mean_squared_error(yv, yv_pred))))

#  >> RMSE-training: 1.2816630767101036
#  >> RMSE-validation: 3.3019777310411556

## B - Selective optimisation
## Let's add some regularization and try to find the optimal min_child_weight and max_depth.

params = {}

s = {'estimator':xgb.XGBRegressor(objective='reg:linear', random_state=413, n_jobs=10),
     'scoring':mean_squared_error}

variable_tuner = hyp.Variable_tuning(Xt, yt, s)
params = variable_tuner.train(var=['min_child_weight', 'max_depth'], initial=[1, 3], 
                              params=params, n_steps=2, broad=2, cv=3)
                              
#  >> Step 1: 
#  >>   <cross-validating for 8 dispositions>
#  >>   error: [15.828309], ntrees: 100
#  >>   min_child_weight: 1, max_depth: 3
#  >> Step 2: 
#  >>  <cross-validating for 3 dispositions>
#  >>   error: [15.828309], ntrees: 100
#  >>   min_child_weight: 1, max_depth: 3

params["min_child_weight"] = 2
params["max_depth"] = 6
xgboost_reg = xgb.XGBRegressor(objective='reg:linear', **params)
xgboost_reg.fit(Xt, yt)

yt_pred = xgboost_reg.predict(Xt)
yv_pred = xgboost_reg.predict(Xv)

print('RMSE-training: ' + str(np.sqrt(mean_squared_error(yt, yt_pred))))
print('RMSE-validation: ' + str(np.sqrt(mean_squared_error(yv, yv_pred))))

#  >> RMSE-training: 0.39283209044243983
#  >> RMSE-validation: 3.0614298694548543

## Though the RMSE on validation set decreased, it's still evident that some hard overfitting is at play here

## C - Complete optimization

params = {'colsample_bytree': 1,
          'learning_rate': 0.1,
          'max_depth': 6,
          'min_child_weight': 2,
          'n_estimators': 100,
          'reg_alpha': 0,
          'reg_lambda': 1,
          'subsample': 1}

s = {'estimator':xgb.XGBRegressor(objective='reg:linear', random_state=413, n_jobs=10),
     'scoring':mean_squared_error,
     'early_stop':50}

p = hyp.Protocol()
p.n_estimator.skip = True
p.gamma.skip = True

params = hyp.xgboost_smart_search(Xt, yt, params, structure=s, protocol=p)

#  >> _____________________________________
#  >> max_depth x min_child_weight
#  >> Broad: 0 - Precision steps: 2
#  >> Step 1: 
#  >>  <cross-validating for 8 dispositions>
#  >>  14.59953075, ntrees: 100
#  >>   max_depth: 6, min_child_weight: 2
#  >> Step 2: 
#  >>  <cross-validating for 3 dispositions>
#  >>  14.59953075, ntrees: 100
#  >>   max_depth: 6, min_child_weight: 2
#  >> _____________________________________
#  >> subsample x colsample_bytree
#  >> Broad: 0 - Precision steps: 2
#  >> Step 1: 
#  >>  <cross-validating for 9 dispositions>
#  >>  14.003031750000002, ntrees: 100
#  >>   subsample: 1.0, colsample_bytree: 0.9090909090909091
#  >> Step 2: 
#  >>  <cross-validating for 15 dispositions>
#  >>  13.900939750000001, ntrees: 100
#  >>   subsample: 0.9090909090909091, colsample_bytree: 0.9393939393939394
#  >>  <cross-validating for 3 dispositions>
#  >>  13.900939750000001, ntrees: 100
#  >>   subsample: 0.9090909090909091
#  >> _____________________________________
#  >> reg_alpha x reg_lambda
#  >> Broad: 0 - Precision steps: 2
#  >> Step 1: 
#  >>  <cross-validating for 5 dispositions>
#  >>  13.97565925, ntrees: 100
#  >>   reg_alpha: 0.001, reg_lambda: 1.0
#  >> Step 2: 
#  >>  <cross-validating for 5 dispositions>
#  >>  13.97565925, ntrees: 100
#  >>   reg_alpha: 0.001, reg_lambda: 1.0
#  >> _____________________________________
#  >> learning_rate
#  >> Broad: 0 - Precision steps: 2
#  >> Step 1: 
#  >>  <cross-validating for 5 dispositions>
#  >>  13.900939750000001, ntrees: 100
#  >>   learning_rate: 0.1
#  >> Step 2: 
#  >>  <cross-validating for 5 dispositions>
#  >>  13.900939750000001, ntrees: 100
#  >>   learning_rate: 0.1

params
{'colsample_bytree': 0.9393939393939394,
 'learning_rate': 0.1,
 'max_depth': 6,
 'min_child_weight': 2,
 'n_estimators': 100,
 'reg_alpha': 0,
 'reg_lambda': 1.0,
 'subsample': 0.9090909090909091}

xgboost_reg = xgb.XGBRegressor(objective='reg:linear', **params)
xgboost_reg.fit(Xt, yt)

yt_pred = xgboost_reg.predict(Xt)
yv_pred = xgboost_reg.predict(Xv)

print('RMSE-training: ' + str(np.sqrt(mean_squared_error(yt, yt_pred))))
print('RMSE-validation: ' + str(np.sqrt(mean_squared_error(yv, yv_pred))))

#  >> RMSE-training: 0.32315447531043684
#  >> RMSE-validation: 2.9335956144246693

## Note that very small sample size will cause very high variance in validation RMSE

## Certain pairs might not be of interest for a certain task. One can modify the default protocol to deactivate certain parameter
## search, increase cross validation, or look for less precise hyperparameters.

## In this last panel, the we'd like to run the protocol a few times to converge on a set of hyperparameters. 'overnight_tuning' 
## gives us this opportunity
## +++ Uses early stopping (only add one parameter to the structure)

## Creating the protocol with default values
p = hyp.Protocol()

## Changing a parameter in a step
p.n_estimator.skip = True
p.gamma.skip = True

## Requesting a grid search for a certain step
p.subsample_coltree.manual_range = {'subsample': [0.8, 0.9, 1.0], 'colsample_bytree': np.arange(0.6, 1.1, 0.1)}

## Requesting a random search for a certain step
p.subsample_coltree.manual_range = {'subsample': [0.8, 0.9, 1.0], 'colsample_bytree': np.arange(0.6, 1.1, 0.001)}
p.subsample_coltree.n_iter = 30

## Requesting a smart search for a certain step (by default)
p.subsample_coltree.lookup_grid = [9, 2.0]
p.subsample_coltree.psteps = 1

## Change the number of folds for one step
p.subsample_coltree.cv = 4

## Change the number of folds for all steps
p.set_cv(5)

initial_params = {'colsample_bytree': 0.94,
                 'learning_rate': 0.1,
                 'max_depth': 6,
                 'min_child_weight': 2,
                 'n_estimators': 500,
                 'reg_alpha': 0,
                 'reg_lambda': 1.0,
                 'subsample': 0.91
}

s = {'estimator': xgb.XGBRegressor(objective='reg:linear', random_state=413, n_jobs=10),
     'scoring': mean_squared_error,
     'early_stop': 15}

params = hyp.overnight_tuning(Xt, yt, initial_params, protocol=p, structure=s, num_rounds=3)

#  >> _____________________________________
#  >> max_depth x min_child_weight
#  >> Broad: 0 - Precision steps: 2
#  >> Step 1: 
#  >>  <cross-validating for 8 dispositions>
#  >>  14.4265948, ntrees: 80
#  >>   max_depth: 5, min_child_weight: 2
#  >>  <cross-validating for 2 dispositions>
#  >>  
#  >>   [ ... ]
#  >>  
#  >>   learning_rate: 0.1
#  >> Final step: Checking for optimal final number of trees.
#  >>  13.2628254, ntrees: 122
#  >> Best trees: 122



## GAM Regression example
## For every estimator, the structure remains the same, albeit simpler. For GAMs, the model's hyperparameters can be determined
## completly through 'tune_variable'.

import pygam as pyg

initialParams = {
     'n_splines': 8,
     'lam': 150
}
​
s = {'estimator':pyg.LinearGAM(),
     'scoring': mean_squared_error}

var_tuner = hyp.Variable_tuning(Xt, yt, s)
params = var_tuner.train(var=['n_splines','lam'], initial=[8,40], 
                         params=initialParams, n_steps=3, broad=True, cv=6)
                         
#  >> Step 1: 
#  >>  <cross-validating for 81 dispositions>
#  >>   error: [17.695642]
#  >>   n_splines: 16, lam: 64.0
#  >>  <cross-validating for 5 dispositions>
#  >>   error: [17.525548]
#  >>   n_splines: 32
#  >>  <cross-validating for 5 dispositions>
#  >>   error: [17.373667]
#  >>   n_splines: 51
#  >> Step 2: 
#  >>  <cross-validating for 81 dispositions>
#  >>   error: [17.328910]
#  >>   n_splines: 50, lam: 76.8
#  >>  <cross-validating for 5 dispositions>
#  >>   error: [17.308551]
#  >>   lam: 92.16
#  >>  <cross-validating for 5 dispositions>
#  >>   error: [17.300287]
#  >>   lam: 110.592
#  >>  <cross-validating for 5 dispositions>
#  >>   error: [17.299557]
#  >>   lam: 123.86304


## Frequent errors
## The most commun mistakes could come from bad input. Make sure that the format is exact for every input, since most are 
## wildly different from other inputs.

## An error would occur if an unrecognized hyperparameter is given, though does not cause any problem if an hyperparameter 
## is missing. As long as the format is respected, only surplus hyperparameters are problematic

## An error would occur if data isn't proprely preprocessed. One Hot Encoding is instrumental for most estimators, as all fitting 
## is done on numerical types.

## If a variable is categorial, make sure that it isn't treated as an ordinal variable






## END
