"""
The aim is to fit a logistic regression model(univariate and multivariate) and estimate the coefficient of each parameter
"""
# importing the libaries needed and modules needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import sys
import os
from pandas import read_csv, DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing

# importing  the data set
df = pd.read_csv("../Data/fibro_dataset.csv", header=0)

# creating a subset of variables
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

# dropping all missing observations to obtain a complete data set
df1 = subset.dropna()

# Data exploration. countinng the number of progresses and non progressed cases
progression = df1["diseaase_prog_new"].value_counts()
print(progression)
sns.countplot(x="diseaase_prog_new", data=df1, palette="hls")
plt.show()

# analying those progressed and those not progressed
count_prog_yes = len(df1[df1["diseaase_prog_new"] == 1])
count_prog_no = len(df1[df1["diseaase_prog_new"] == 0])
pct_prog_yes = count_prog_yes / (count_prog_yes + count_prog_no) * 100
print(pct_prog_yes)
pct_prog_no = count_prog_no / (count_prog_yes + count_prog_no) * 100
print(pct_prog_no)

# calculating the mean of each numerical variables grouped by progression status
dframe = df1.groupby("diseaase_prog_new").mean()
print(dframe)

# calculating categorical means of other categorical variables
smoking_summary = df1.groupby("d_smoking").mean()
print(smoking_summary)
gender_summary = df1.groupby("d_male").mean()
print(gender_summary)

# Visualisating if gender/ smoking status is a good predictor of disease progression
pd.crosstab(df1.d_male, df1.diseaase_prog_new).plot(kind="bar")
plt.xlabel("Gender")
plt.ylabel("progression")
plt.show()
# plt.savefig(prog_freq)
pd.crosstab(df1.d_smoking, df1.diseaase_prog_new).plot(kind="bar")
plt.xlabel("smoker")
plt.ylabel("progression")
plt.show()

# fitting the model(univariate)
observation = len(df1)
y = df1["diseaase_prog_new"].values  # outome
x = df1["base_fibro"].values.reshape(observation, 1)  # IV
logit_model = sm.Logit(y, x)
result = logit_model.fit()
print(result.summary())

# fitting a miultivariate model
y = df1["diseaase_prog_new"].values  # outome
x = df1[["base_fibro", "bmi", "d_male", "base_fvcpp", "base_dlcopp"]].values
logistic_model1 = sm.Logit(y, x).fit()
print(logistic_model1.summary())
