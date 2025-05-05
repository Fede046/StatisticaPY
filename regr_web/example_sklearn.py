# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:33:49 2023

@author: elena

esempio di regressione lineare usando sklearn

con vettore di pochi dati
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])

# Reshape data
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# Create linear regression object and fit the model
reg = LinearRegression().fit(x, y)

# Predict the y-values using the trained model
y_pred = reg.predict(x)

print(reg.intercept_,reg.coef_)
# Plot the data points and the linear regression line
plt.scatter(x, y, color='blue')
plt.plot(x, y_pred, color='red')

# Add labels and a title to the plot
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple Linear Regression')

# Display the plot
plt.show()

