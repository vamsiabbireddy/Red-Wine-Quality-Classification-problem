#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:12:26 2020

@author: egreddy
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

a = pd.read_csv("/Users/egreddy/Downloads/taskFile1580882753721")

X = a.copy()

len(X.columns)
y = X.iloc[:,11:]
del X["quality (Target)"]
#Checking for the null values
a.isna().sum()

#Checking for the outliers

plt.boxplot(X["fixed acidity"])
plt.boxplot(X["volatile acidity"])
plt.boxplot(X["citric acid"])
plt.boxplot(X["residual sugar"])#jghjghj
plt.boxplot(X["chlorides"])#kj.jkk

plt.boxplot(X["free sulfur dioxide"])

plt.boxplot(X["total sulfur dioxide"])
plt.boxplot(X["density"])#ujkuf
plt.boxplot(X["pH"])#vbbsd
plt.boxplot(X["sulphates"])
plt.boxplot(X["alcohol"])



per = [0,1,2,5,6,9,10]
for i in per :
	percentiles = X.iloc[:,i].quantile([0.005,0.957]).values
	X.iloc[:,i] =  X.iloc[:,i].clip(percentiles[0], percentiles[1])
    
    
percentiles = X.iloc[:,3].quantile([0.0,0.9015]).values
X.iloc[:,3] =  X.iloc[:,3].clip(percentiles[0], percentiles[1])  
    
percentiles = X.iloc[:,4].quantile([0.01,0.93]).values
X.iloc[:,4] =  X.iloc[:,4].clip(percentiles[0], percentiles[1]) 

percentiles = X.iloc[:,7].quantile([0.025,0.95]).values
X.iloc[:,7] =  X.iloc[:,7].clip(percentiles[0], percentiles[1]) 

percentiles = X.iloc[:,8].quantile([0.015,0.95]).values
X.iloc[:,8] =  X.iloc[:,8].clip(percentiles[0], percentiles[1]) 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)


y_pred_test = classifier.predict(X_test)

y_pred_train = classifier.predict(X_train)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

cm_test = confusion_matrix(y_test, y_pred_test)
cm_train = confusion_matrix(y_train, y_pred_train)

accuracy=accuracy_score(y_test, y_pred_test) 

accuracy_1=accuracy_score(y_train, y_pred_train) 

class_report=classification_report(y_test, y_pred_test)

"""Here we got accuracies near for both trainn and test but the accuracies are very low"""
"""
So here we came to know that by using tghe logistic 
regression we didn't got good accuracy.So, we have to try for other methods
"""

#DECISION TREE

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_test = classifier.predict(X_test)

y_pred_train = classifier.predict(X_train)

DecisionTreeClassifier()

cm_test = confusion_matrix(y_test, y_pred_test)
cm_train = confusion_matrix(y_train, y_pred_train)

accuracy=accuracy_score(y_test, y_pred_test) 

accuracy_1=accuracy_score(y_train, y_pred_train) 

class_report=classification_report(y_test, y_pred_test)


"""Here we got train accuracy 1 but the test accuracy is very low
"""

#SVM
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

classifier.coef_
classifier.intercept_

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred_train = classifier.predict(X_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, y_pred)

accuracy=accuracy_score(y_test, y_pred) 
accuracy1=accuracy_score(y_train, y_pred_train) 


#Naive Bayes

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_test = classifier.predict(X_test)
y_pred_train = classifier.predict(X_train)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

cm_test = confusion_matrix(y_test, y_pred_test)
cm_train = confusion_matrix(y_train, y_pred_train)

accuracy=accuracy_score(y_test, y_pred_test) 

accuracy_1=accuracy_score(y_train, y_pred_train) 

class_report=classification_report(y_test, y_pred_test)

#Random Forest
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train)


# Use the forest's predict method on the test data
predictions = np.array([list(rf.predict(X_train))])#, index = y_train.index)[0]
# Calculate the absolute errors
errors = abs(predictions.T - y_train.values)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors /np.array([list( y_train.values)]))
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

predictions = np.array([list(rf.predict(X_test))])#, index = y_train.index)[0]
# Calculate the absolute errors
errors = abs(predictions.T - y_test.values)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors /np.array([list( y_test.values)]))
# Calculate and display accuracy
accuracy1 = 100 - np.mean(mape)
print('Accuracy test:', round(accuracy1, 2), '%.')




















