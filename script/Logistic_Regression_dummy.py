"""
The aim is to fit a logistic regression model(univariate and multivariate) aand practise one dummy coding
"""
# importing the libaries needed and modules needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pandas import read_csv, DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, classification_report

# importing  the data set
df = pd.read_csv("../Data/fibro_dataset.csv", header=0)
subset = df[
    [
        "diseaase_prog_new",
        "d_smoking",
        "age",
        "d_male",
        "base_fibro",
        "bmi",
        "base_dlcopp",
        "base_fvcpp",
    ]
]
subset = subset.dropna()

# creating dummy variables for gender and smoking status using pd.get_dummy() function
dummy_smoker = pd.get_dummies(subset["d_smoking"], prefix="smoker")
dummy_gender = pd.get_dummies(subset["d_male"], prefix="gender")

# creating a list of cointinuous variables
continuous_columns = ["age", "base_fibro", "bmi", "base_dlcopp", "base_fvcpp"]
subset_continuous = subset[continuous_columns]

# concatenating the dummy variables with the continuous variables
subset_data_new = pd.concat(
    [dummy_gender, dummy_smoker, subset_continuous, subset["diseaase_prog_new"]], axis=1
)

# split data between train and test set. 70-30 ration and use random_state = 42 for reproducibility
x_train, x_test, y_train, y_test = train_test_split(
    subset_data_new.drop(["diseaase_prog_new"], axis=1),
    subset_data_new["diseaase_prog_new"],
    train_size=0.7,
    random_state=42,
)
# Fitting the model on the training data
logistic_model = sm.Logit(y_train, x_train).fit()
print("Regression for moldel 1")
print(logistic_model.summary())

# Remove one extra columns for all the categorical variables for which dummies have been created else this will produce NANs
remove_extra_dummy_columns = ["gender_0.0", "smoker_0.0"]
remove_cols_insg = []
remove_cols = list(set(remove_extra_dummy_columns))

# logistic_model1 = sm.Logit(y_train,sm.add_constant(x_train.drop(remove_cols,axis=1))).fit()
logistic_model1 = sm.Logit(y_train, x_train.drop(remove_cols, axis=1)).fit()
print("Regression for moldel 2")
print(logistic_model1.summary())
