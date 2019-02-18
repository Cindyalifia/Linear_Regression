# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 13:04:03 2019

@author: PERSONALISE NOTEBOOK
"""

#Simple Linear Regression 

#importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[: , 1].values

#Splitting the dataset into the Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Feature Scalling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() 
X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.transform(X_test)"""

#Fitting simple linear regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) # Train set

#Predicting the test result 
y_pred = regressor.predict(X_test)

#Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red') # Coordinate X and y train
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #predict of X_train which is y_train
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red') #Cordinate X and y test
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #predict of X_train which is y_train
plt.title('Salary vs Experience (Test Set)') # The blue line is the same line with X and y train
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()








