# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:38:14 2023

@author: elena

esempio regressione su salary data SENZA MACHINE LEARNING
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data from Kaggle CSV file
#df = pd.read_csv('https://www.kaggle.com/vihansp/salary-data-simple-linear-regression')
df = pd.read_csv("Salary_Data.csv")
df.head()

# Extract input and output variables
x = df['YearsExperience'].values.reshape(-1, 1)
y = df['Salary'].values.reshape(-1, 1)

# Create linear regression object and fit the model
reg = LinearRegression().fit(x, y)

# Predict the y-values using the trained model
y_pred = reg.predict(x)

# Plot the data points and the linear regression line
plt.scatter(x, y, color='blue')
plt.plot(x, y_pred, color='red')

# Add labels and a title to the plot
plt.xlabel('Input Variable')
plt.ylabel('Output Variable')
plt.title('Simple Linear Regression')

# Display the plot
plt.show()

print("Coefficient of determination: %.2f" % r2_score(y, y_pred))


