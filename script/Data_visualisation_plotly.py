"""
 The aim of this script is to visualise the data and save the results in an output foldrer titled Result
"""

# Importing packages 
import pandas as pd
import numpy as np
import plotly.graph_objs as go # for graphing
import cufflinks as cf
import matplotlib.pyplot as plt
import sys
import os
from pandas import read_csv, DataFrame
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
import plotly.express  as px

cf.go_offline()
init_notebook_mode(connected = True)

# importing the data set needed
fibro_subset = pd.read_csv('../Data/fibro_analysis_set.csv', header = 0)

# box plot of baseline/6month fibrocyte according to ipf status
fig =px.box(fibro_subset,x='status',y='base_fibro')
fig.write_image('../Results/fig0.jpeg')
fig =px.box(fibro_subset,x='status',y='fibro_6m')
fig.write_image('../Results/fig1.jpeg')

# histogram if basesline and 6 months fibrocyte
fig =px.histogram(fibro_subset,x='base_fibro', color = 'status')
fig.write_image('../Results/fig2.jpeg')
fig =px.histogram(fibro_subset,x='fibro_6m', color = 'status')
fig.write_image('../Results/fig3.jpeg')

# scatter plot showing variation of and fibrocyte
fig =px.scatter(fibro_subset,x='age',y='base_fibro', color = 'status')
fig.write_image('../Results/fig4.jpeg')

# box plot showing distribution of smokining status according to disease status
fig =px.bar(fibro_subset,x='newsmoker', color='status')
fig.write_image('../Results/fig5.jpeg')

# box plot showing distribution of smokining status
fig =px.bar(fibro_subset,x='newgender', color='status')
fig.write_image('../Results/fig6.jpeg')

# summary table
df1 = (
    fibro_subset.dropna()
    .groupby("status")
    .agg(
        {
            "age": ["mean", "median", "min", "max", "std"],
            "base_fibro": ["mean", "median", "min", "max", "std"],
            "fibro_6m": ["mean", "median", "min", "max", "std"],
        }
    )
    .astype(int)
)

summary_data = df1.T
summary_data
m = summary_data.reset_index()
col_rename = {
    "level_0": "variable",
    "level_1": "Statistics",
    "ipf": "ipf",
    "nsip": "nsip",
}
new_data = m.rename(columns=col_rename)


fig = go.Figure(
    data=[
        go.Table(
            header=dict(
                values=list(new_data.columns), fill_color="paleturquoise", align="left"
            ),
            cells=dict(
                values=[
                    new_data.variable,
                    new_data.Statistics,
                    new_data.ipf,
                    new_data.nsip,
                ],
                fill_color="lavender",
                align="left",
            ),
        )
    ]
)

fig.write_image("../Results/fig7.jpeg")

