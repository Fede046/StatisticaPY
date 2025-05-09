#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#In alternativa guardare simple_linear_regression_for_salary_data
#Le 4 cose che bisogna fare
#Calcolare corfficiente retta-> sm.ols
#Disegnare retta
#Calcolare R^2->Summary dell'output di sm.ols
#Normalità residui->Summary dell'output di sm.ols

"""
Created on Sun Apr  2 10:29:54 2023

@author: elenalolipiccolomini

https://www.statsmodels.org/dev/examples/notebooks/generated/predict.html
"""
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm

plt.rc("figure", figsize=(16, 8))
plt.rc("font", size=14)

nsample = 50
sig = 0.25
x1 = np.linspace(0, 20, nsample)
X = np.column_stack((x1, np.sin(x1), (x1 - 5) ** 2))
X = sm.add_constant(X)
beta = [5.0, 0.5, 0.5, -0.02]
y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)

olsmod = sm.OLS(y, X)
olsres = olsmod.fit()
print(olsres.summary())

#R-square mi dice quanto è valido,
#vedo la bontà del modello
#prediction in sample

xn=np.linspace(x1[0],x1[49],100)

ypred = olsres.params[0]+olsres.params[1]*xn

#guardo i coefficient

#per vedere se i residui sono normali guardo JB
#JB se è sopra 0.5 i valori sono normali
#graphical visualization

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(x1, y, "o", label="Data")
ax.plot(x1, y_true, "b-", label="True")
ax.plot(xn,ypred, "r", label="OLS prediction")
ax.legend(loc="best")
plt.show()

#prediction with formulas

from statsmodels.formula.api import ols

data = {"x1": x1, "y": y}

res = ols("y ~ x1 + np.sin(x1) + I((x1-5)**2)", data=data).fit()


res.params

