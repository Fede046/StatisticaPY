#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 09:48:43 2025

@author: elenalolipiccolomini
"""

# importing required libraries 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

"""# Loading Dataset """

# importing iris flower data from sklearn 
iris_df = sns.load_dataset("iris")

# converting data into a pandas compatible dataFrame 
#iris_df = pd.DataFrame(data=np.c_[iris_data['data'], iris_data['target']], columns=iris_data['feature_names'] + ['target'])

#iris_df  = pd.read_csv("iris.csv")
"""# Getting familiar with the Data

After loading the data, the very first thing we can do is to look at the data. Exploring some insights of the data by looking 
at rows and columns(features). Most importantly, we need a cleaned data to further apply Expolatory data analysis and machine learning algorithms.
"""

# printing first few lines of the iris flower data 
iris_df.head()

# printing the concise summary of the dataframe such as index dtype, columns and non-null values
iris_df.info()

# alternative: usign '.isnull()' function checking if we have any null values in our dataset. 
iris_df.isnull().sum()

# We are having 3 different iris flower species and they are encoded with class labels. We can change 
# this to categorical labels. This will help us in Expolatory data analysis 
#iris_df['target'] = iris_df['target'].replace({0.0:'setosa', 1.0:'versicolor', 2.0:'virginica'})

# looking at some descriptive statistics
iris_df.describe()