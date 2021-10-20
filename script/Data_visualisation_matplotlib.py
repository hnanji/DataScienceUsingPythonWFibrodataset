"""
 The aim of this script is to visualise data using matplib
"""

# Importing packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# importing the data set needed
df = pd.read_csv('../Data/fibro_dataset.csv', header = 0)

# plotting scatter plots
plt.scatter(df['age'], df['base_fibro'])
plt.xlabel('Age')
plt.ylabel('baseline fobrocytes')
plt.title('variation of age and fibricyte')
plt.show()

plt.scatter(df['age'], df['base_fibro'])
plt.xlabel('Age')
plt.ylabel('baseline fobrocytes')
plt.title('variation of age and fibricyte')
plt.show()

plt.scatter(df['base_dlcopp'], df['base_fibro'])
plt.xlabel('base_dlcopp')
plt.ylabel('baseline fobrocytes')
plt.title('variation of base_dlcopp and fibricyte')
plt.show()

plt.scatter(df['base_fvcpp'], df['base_fibro'])
plt.xlabel('base_fvcpp')
plt.ylabel('baseline fobrocytes')
plt.title('variation of base_fvcpp and fibricyte')
plt.show()

# plotting histograms
plt.hist(df['age'], 25) # changing the bib size
plt.show()

plt.hist(df['base_fibro'], 25) # changing the bib size
plt.show()

plt.hist(df['base_dlcopp'], 25) # changing the bib size
plt.show()

plt.hist(df['base_fvcpp'], 25) # changing the bib size
plt.show()

# box plots
subset = df[['age', 'bmi','base_dlcopp','base_fibro']]
subset.boxplot()
plt.show()

#bar plots
plt.bar(df['d_ipf'],df['base_fvcpp'], color = 'green')
plt.show()
