"""
 The aim of this script is to visualise data using matplib, superimposing plots
"""

# Importing packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# importing the data set needed
df = pd.read_csv('../Data/fibro_dataset.csv', header = 0)

# Histogram of age by disease status
"""subseting observations into two data frames for easy plotting
subsetting a only those with idpf ==1
ubsetting a only those with idpf ==0
"""
df_idpf1 = df[df['d_ipf']==1] 
df_idpf0 = df[df['d_ipf']==0] 
plt.hist(df_idpf1['age'],20, label = 'ipf') 
plt.hist(df_idpf0['age'],20, label = 'nsip') 
plt.xlabel('Age')
plt.ylabel('count')
plt.title('Distribution of Age by disease status')
plt.legend()
plt.show()

# Histogram of baseline fibrcyte by disease status
plt.hist(df_idpf1['base_fibro'],15, label = 'ipf') 
plt.hist(df_idpf0['base_fibro'],15, label = 'nsip') 
plt.xlabel('baseline fibrocyte')
plt.ylabel('count')
plt.title('Distribution of baseline fibriocyte by disease status')
plt.legend()
plt.show()

