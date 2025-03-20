#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 11:04:53 2025

@author: elenalolipiccolomini
"""

import pandas as pd
import numpy as np

series_np = pd.Series(np.array([10,20,30,40,50,60]))
series_np 

series_np.values

series_np[2]


df = pd.DataFrame([["Fred",80],["Jill",90]],columns=["student", "grade"])

print(df.head())
print(df.tail())
print(df.describe)

#Normally Pandas dataframe operations create a new dataframe. 
#But we can use inplace=True in some operations to update the existing dataframe without having to make a new one.


df.set_index("student",inplace=True)

#Add a column to the dataframe

df['birthdate']=['1970-01-12', '1972-05-12']
df.columns

#select a column from the dataframe

grade=df['grade']
grade

#Add rows to the data frame by creating a new one (df2) and the appending it to create df3
df2 = pd.DataFrame([[70,'1980-11-12'],[97, '1984-11-01']],index=["Costas", "Ilya"], columns=["grade", "birthdate"])
df2

df3=pd.concat([df,df2])
df3

#select rows from the dataframe

df3.iloc[0:2]

#####################################################################

#Create anew data frame

mcu_data = {'Title': ['Ant-Man and the Wasp', 'Avengers: Infinity War', 'Black Panther', 'Thor: Ragnarok', 
              'Spider-Man: Homecoming', 'Guardians of the Galaxy Vol. 2'],
            'Year':[2018, 2018, 2018, 2017, 2017, 2017],
            'Studio':['Beuna Vista', 'Beuna Vista', 'Beuna Vista', 'Beuna Vista', 'Sony', 'Beuna Vista'],
            'Rating': [np.nan, np.nan, 0.96, 0.92, 0.92, 0.83]}

df_mcu = pd.DataFrame(mcu_data)
df_mcu.describe

df_mcu.shape
df_mcu.columns 
df_mcu.index
df_mcu.values 

df_mcu['Title']
df_mcu[['Title', 'Rating']]


df_mcu['Title'][:2]
df_mcu.iloc[0,1]
df_mcu.iloc[[2,4,5]]

df_mcu

