"""
 The aim of this script is to applly simple Linear regression using scikitlearn. 
 This investigate how age affects baseline fibrocyte
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
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# importing analaysis set data prepared earlier
fibro_data = pd.read_csv("../Data/fibro_analysis_set.csv", header=0)

# dropping all missing values and creating a new data frame called subset1
subset1 = fibro_data.dropna()

# data preparation
y = subset1["base_fibro"].values

# calculating the number of observations
observation = len(subset1)

# reshaping the IV into 2 dimensions
x = subset1["age"].values.reshape(observation, 1)

# split the data into train set(80%) and test set(20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# setting  up the regression model to train the algorithm
regressor = LinearRegression()
regressor.fit(x_train, y_train)
print("Intercept:", regressor.intercept_)
print("Intercept:", regressor.coef_)

# Making predictions
y_pred = regressor.predict(x_test)

# building a data frame for actual and predicted values
df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(df.head(3))

# Evaluating the the regression algorithm
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Square Error:", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Square Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
