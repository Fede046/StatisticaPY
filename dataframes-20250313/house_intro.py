#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 09:50:15 2025

@author: elenalolipiccolomini
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

housing=pd.read_csv("housing.csv")

housing.info()
housing.head()

#EDA - Exploratory data analysis

# statistical exploration of data
housing.describe()
