"""
# The aim  of this is to investigate assumptions behind linear model
"""

# importing packages
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
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import plot_leverage_resid2

# importing analysis set data prepared earlier
data = pd.read_csv("../Data/fibro_analysis_set.csv", header=0)

# dropping all missing observations to obtain a complete data set
fibro_data = data.dropna()

# specifying the IV and DV
pdx = fibro_data[["age", "fibro_6m", "delta_fvc", "base_fvcpp"]]
pdy = fibro_data["base_fibro"]

# split the data into train set(80%) and test set(20%)
x_train, x_test, y_train, y_test = train_test_split(
    pdx, pdy, test_size=0.2, random_state=0
)

# creating a fitted model
lm = sm.OLS(y_train, x_train).fit()
print(lm.summary())

# Outlier detection,plotting normalised residuals vs leverage
fig, ax = plt.subplots()
fig = plot_leverage_resid2(lm, ax=ax)
plt.show(fig)

#  Find outliers
test = lm.outlier_test()
# print(test)
print("Bad data points(bonf(p) <0.05)")
print(test[test["bonf(p)"] < 0.05])

# variance check
plt.plot(lm.resid, "o")
plt.title("residual plot")
plt.ylabel("residual")
plt.xlabel("observatoion number")
plt.show()

# normnality of residuals
plt.hist(lm.resid)
plt.show()
