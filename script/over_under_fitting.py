"""
# The aim  of this is to use two regularisation methods,Lasso and Ridge regression to solve the problem of under or 
# over fitting. RMSE and MEA are used to evaluate model performance on test anmd training set
"""

# importing the libaries needed and modules needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pandas import read_csv, DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# importing analysis set data prepared earlier
data = pd.read_csv("../Data/fibro_analysis_set.csv", header=0)

# dropping all missing observations to obtain a complete data set
fibro_data = data.dropna()

# specifying the IV and DV
pdx = fibro_data[["age", "fibro_6m", "delta_fvc", "base_fvcpp"]]
pdy = fibro_data["base_fibro"]

# split the data into train set(80%) and test set(20%)
X_train, X_test, y_train, y_test = train_test_split(
    pdx, pdy, test_size=0.2, random_state=0
)

# RIDGE REGRESSION
# specifying a regularisation strength, aplha = 0.001
lr = linear_model.Ridge(alpha=0.001)

# train the model
lr.fit(X_train, y_train)

# make predictions on training set
y_train_pred = lr.predict(X_train)

# make predictions on the test set
y_test_pred = lr.predict(X_test)

# Evaluating models on train and test set
print("------ Ridge Regression (MAE and RMSE)------")
print("Train MAE: ", metrics.mean_absolute_error(y_train, y_train_pred))
print("Train RMSE: ", np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
print("Test MAE: ", metrics.mean_absolute_error(y_test, y_test_pred))
print("Test RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
print("Ridge Coef: ", lr.coef_)

# LASSO REGRESSION
lr = linear_model.Lasso(alpha=0.001)
lr.fit(X_train, y_train)

# predicting on trainingh set
y_train_pred = lr.predict(X_train)

# predicting on test set
y_test_pred = lr.predict(X_test)
print("----- LASSO Regression(MAE and RMSE) -----")
print("Train MAE: ", metrics.mean_absolute_error(y_train, y_train_pred))
print("Train RMSE: ", np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
print("Test MAE: ", metrics.mean_absolute_error(y_test, y_test_pred))
print("Test RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
print("LASSO Coef: ", lr.coef_)





