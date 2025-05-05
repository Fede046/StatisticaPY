# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:38:45 2023

@author: elena
"""

##################################################################################################
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

fig = plt.figure(figsize=(8, 6))
crime_data = sm.datasets.statecrime.load_pandas()
results = smf.ols('murder ~ hs_grad + urban + poverty + single',  data=crime_data.data).fit()
sm.graphics.plot_regress_exog(results, 'poverty', fig=fig)
plt.show()

fig = plt.figure(figsize=(8, 6))
fig = sm.qqplot(results.resid, line='45')
plt.show()

from scipy.stats import shapiro
 
print(shapiro(results.resid))