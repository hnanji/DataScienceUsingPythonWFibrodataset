"""
 The aim of this script is to implement classificaton in scikit learn and statmodel
"""

# Importing packages
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import roc_curve, auc

# importing the data set needed
df = pd.read_csv("../Data/fibro_dataset.csv")

# creating a subset of the data
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
df1 = pd.DataFrame(subset.dropna())  # dropping obsertvations with missing values

# specifying the I|V and DV
x = df1[
    ["d_smoking", "age", "d_male", "base_fibro", "bmi", "base_dlcopp", "base_fvcpp"]
]
y = df1["diseaase_prog_new"]

# fitting a Logistic regression classifier
classifier1 = LogisticRegression()
classifier1.fit(x, y)

# data split
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Making Predictions of the outcome on the test data
y_predicted = classifier1.predict(x_test)

# use the score function to get the mean predicted score
mean_predicted_value_train = classifier1.score(x_train, y_train)
print(mean_predicted_value_train)

# Binary Classification performance. Contigency table comparing the predidted outcome and the outcome on the test data set
print(len(y_test))
print(len(y_predicted))
confusion_matrix = confusion_matrix(y_test, y_predicted)
print(confusion_matrix)

# visualising the confusion matrix
plt.matshow(confusion_matrix)
plt.title("confusion matrix")
plt.colorbar()
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()

# propoprtion of classifier that are correct
Accuracy = accuracy_score(y_test, y_predicted)
print("Accuracy:", Accuracy)

# classification report
report = metrics.classification_report(y_test, y_predicted)
print(report)

# calulate the predicticted probability
pred_probabilities = classifier1.predict_proba(x_test)

# estimating ROC statististics and plotting ROC curve
false_positive_rate, recall, thresholds = roc_curve(y_test, pred_probabilities[:, 1])
roc_auc = auc(false_positive_rate, recall)
plt.title("Receiver Operating Characteristic")
plt.plot(false_positive_rate, recall, "b", label="AUC = %0.2f" % roc_auc)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel("Recall")
plt.xlabel("Fall-out")
plt.show()

# fitting a Logistic regression using model
Xc= sm.add_constant(x_train)
logistic_regression=sm.Logit(y_train,Xc)

# fitted model
fitted_model = logistic_regression.fit()
# examining model cofficients
print(fitted_model.summary())


