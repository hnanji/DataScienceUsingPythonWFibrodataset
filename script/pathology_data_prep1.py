"""
Pathology Node project:
Aim: To import the data, clean, relabel with meaningfull names, create and analysis set to be used and prepare for discriptive statistics
"""
# importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pandas import read_csv, DataFrame

# defining functions
# defining a function that labels disease stutus as 1 = ipf and 0 =nsip, those with missing disease status forms the control group
def disease_label(x):
    if x == 1:
        return "ipf"
    elif x == 0:
        return "nsip"
    else:
        return "Control"


# defining a function that labels gender as 1=male, 0= female
def gender_label(x):
    if x == 1:
        return "Male"
    elif x == 0:
        return "Female"
    else:
        return "Missing"


# defining a function that label smoking status as 1=smoker and 0=never, any  NaNs are labelled missing
def smoking_label(x):
    if x == 1:
        return "Smoker"
    elif x == 0:
        return " Never"
    else:
        return "Missing"


# defining a function  that counts the total number of participants
def total_participants(x):
    return len(x)


# importing data set needed
fibro_data = pd.read_csv("../Data/fibro_dataset.csv", header=0)


# exploring the data set
print(fibro_data.head(3))
print(fibro_data.info())

# display the number of observations and number of variables
print(fibro_data.shape)

# check the nature of the data set
print(type(fibro_data))

# check index
print(fibro_data.index)

# check variable formats
print(fibro_data.dtypes)

# show the variable names
print(fibro_data.columns)

# Returns a data frame with variables and the number of missing values
missing_status_report = pd.DataFrame(fibro_data.isnull().sum())
print(missing_status_report)

# renaming some variables with more meanigful names
fibro_data.rename(columns={"d_male": "gender"}, inplace=True)
fibro_data.rename(columns={"d_smoking": "smokingstatus"}, inplace=True)
fibro_data.rename(columns={"d_type": "type"}, inplace=True)
fibro_data.rename(columns={"age": "age"}, inplace=True)
fibro_data.rename(
    columns={"diseaase_prog_new": "disprog", "disease_mortality_new": "dismortality"},
    inplace=True,
)
fibro_data.rename(
    columns={
        "disease_PROG": "progression",
        "disease_MORTcens": "dismortens",
        "DOD": "dateofdeath",
    },
    inplace=True,
)
# fibro_data.rename(columns={'disease_PROG' :'progression','disease_MORTcens':'dis_mortcens', 'DOD':'dateofdeath'}, inplace=True)
fibro_data.rename(columns={"d_ipf": "status"}, inplace=True)

# this line of code creates a new variable status
fibro_data["status"] = fibro_data["status"].apply(disease_label)

# this creates a new variable newgender
fibro_data["newgender"] = fibro_data["gender"].apply(gender_label)

# this line of code will create a new variable newsmoker
fibro_data["newsmoker"] = fibro_data["smokingstatus"].apply(smoking_label)

# creating a subset of variables to summarise
fibro_subset = fibro_data[
    [
        "age",
        "status",
        "newgender",
        "base_fibro",
        "fibro_6m",
        "delta_fvc",
        "base_dlcopp",
        "base_fvcpp",
        "base_fev1_fvc",
        "newsmoker",
    ]
]

# saving data set to re reused later
fibro_subset.to_csv("../Data/fibro_analysis_set.csv", index=True, header=True)

# checking the counts of males and females
print(fibro_subset["newgender"].value_counts())

# exploring participants with disease status
print(fibro_subset["status"].value_counts())

# counting participants with missiing disease status
print(fibro_subset["status"].isnull().sum())

# exploring smoking status
print(fibro_subset["newsmoker"].value_counts())

# checking missing smoking status
print(fibro_subset["newsmoker"].isnull().sum())

# counting the total particpants in study. this uses the function defined above
total_pop = total_participants(fibro_subset)

print(total_pop)

# summarising age of particicipants split by disease status. this returns just the count,. mean, std, min and max .results outputted into a csv file
age = np.round(fibro_subset["age"].groupby([fibro_subset["status"]]).describe(), 2)[
    ["count", "mean", "std", "min", "max"]
]
age.to_csv("../Results/summary_of_age.csv", index=True, header=True)
print(age)

# summarising baseline fibrocyte  of particicipants split by disease status
base_fibro = np.round(
    fibro_subset["base_fibro"].groupby([fibro_subset["status"]]).describe(), 2
)[["count", "mean", "std", "min", "max"]]
base_fibro.to_csv("../Results/baseline_fibrocyte.csv", index=True, header=True)
print(base_fibro)

# summarising 6 month fibrocyte  of particicipants split by disease status
six_month_fibrocyte = np.round(
    fibro_subset["fibro_6m"].groupby([fibro_subset["status"]]).describe(), 2
)[["count", "mean", "std", "min", "max"]]
six_month_fibrocyte.to_csv("../Results/Six_month_frbocyte.csv", index=True, header=True)
print(six_month_fibrocyte)

# summarising absolute change in fibrocyte  of particicipants split by disease status.use the absolute value of delta_fvc
absolute_change = np.round(
    fibro_subset["delta_fvc"].abs().groupby([fibro_subset["status"]]).describe(), 2
)[["count", "mean", "std", "min", "max"]]
absolute_change.to_csv("../Results/Absolute_change.csv", index=True, header=True)
print(absolute_change)

# summarising forced expiratory value of particicipants split by disease status
forced_expiratory = np.round(
    fibro_subset["base_fev1_fvc"].groupby([fibro_subset["status"]]).describe(), 2
)[["count", "mean", "std", "min", "max"]]
forced_expiratory.to_csv("../Results/forced_expiratory.csv", index=True, header=True)
print(forced_expiratory)

# summarising forced expiratory value of particicipants split by disease status
fvcpp = np.round(
    fibro_subset["base_fvcpp"].groupby([fibro_subset["status"]]).describe(), 2
)[["count", "mean", "std", "min", "max"]]
fvcpp.to_csv("../Results/forced_vital_capacity.csv", index=True, header=True)
print(fvcpp)

# summarising forced expiratory value of particicipants split by disease status
dlcopp = np.round(
    fibro_subset["base_dlcopp"].groupby([fibro_subset["status"]]).describe(), 2
)[["count", "mean", "std", "min", "max"]]
dlcopp.to_csv("../Results/dlcopp.csv", index=True, header=True)
print(dlcopp)

# tabulating smoking status by disease status
smoking_count = fibro_subset["newsmoker"].value_counts()
smoking_count.to_csv("../Results/count_of_smokers.csv", index=True, header=True)
print(smoking_count)
smoking_by_status = pd.crosstab(
    fibro_subset.newsmoker, fibro_subset.status, margins=True
)
smoking_by_status.to_csv(
    "../Results/smoking_by_disease_status.csv", index=True, header=True
)
print(smoking_by_status)
