"""
The aim is to transform the IV and investigate if that captures patterns  better
# in the data set and improves the fit of the model
"""
# importing libraries
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

# importing analysis set data prepared earlier
data = pd.read_csv("../Data/fibro_analysis_set.csv", header=0)

# dropping all missing observations to obtain a complete data set
fibro_data = data.dropna()

# providing data for the regressor as np arrays,the arrays is required to be in two dimension for the x input
observation = len(fibro_data)
x = fibro_data["age"].values.reshape(observation, 1)

y = fibro_data["base_fibro"].values

# split the data into train set(80%) and test set(20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# fitting the model for degree one
regressor_degree_one = LinearRegression()
regressor_degree_one.fit(x_train, y_train)

# defining transformation for quadratic and cubic IV
quad_featurizer = PolynomialFeatures(degree=2)
cubic_featurizer = PolynomialFeatures(degree=3)

# transforming x_train to quadratic
x_train_quad = quad_featurizer.fit_transform(x_train)

# transforming x_train to cubic
x_train_cubic = cubic_featurizer.fit_transform(x_train)

# fitting a quadratic term in the model for IV
regressor_quad = LinearRegression()
regressor_quad.fit(x_train_quad, y_train)

# fitting a cubic term in the model for IV
regressor_cubic = LinearRegression()
regressor_cubic.fit(x_train_cubic, y_train)

# Visualising the fit
plt.scatter(x_train, y_train, label="degree one")  # this plots the x and y scatter
plt.plot(
    x_train, regressor_degree_one.predict(x_train), color="blue", zorder=1
)  # this adds the linear fitted line
plt.plot(
    x_train, regressor_quad.predict(x_train_quad), c="r", marker="d"
)  # this now adds the quadratic line
plt.plot(
    x_train, regressor_cubic.predict(x_train_cubic), c="y", marker="o"
)  # this now adds a cubic line
plt.legend(["degree1(fitted)", "quadratic", "cubic"])
plt.xlabel("Age")
plt.ylabel("baseline fibro")
plt.show()
plt.savefig("../Results/polynomnial_investigation.png")

# Estimating the different Coefficient of determination(R squared) for the first,quadratic and cubic fit
R_squared_one = regressor_degree_one.score(x_train, y_train)
print("Coefficient of determination(degree one):", R_squared_one)
R_squared_quad = regressor_quad.score(x_train_quad, y_train)
print("Coefficient of determination(quadrtatic):", R_squared_quad)
R_squared_cubic = regressor_cubic.score(x_train_cubic, y_train)
print("Coefficient of determination(cubic):", R_squared_cubic)
