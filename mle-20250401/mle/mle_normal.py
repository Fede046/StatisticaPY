# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:29:45 2023

stima parametri media e varianza di una distribuzione normale.
"""

import numpy as np
from scipy.stats import norm

# Define the true parameter values
mu_true = 2.0
sigma_true = 1.5

# Generate some random data from the normal distribution


data = norm.rvs(loc=mu_true, scale=sigma_true, size=1000,random_state=100)

# Define the log-likelihood function for the normal distribution
def log_likelihood(theta, data):
    mu, sigma = theta
    log_lik = np.sum(norm.logpdf(data, loc=mu, scale=sigma))
    return log_lik

# Define the function to maximize the log-likelihood
def neg_log_likelihood(theta, data):
    return -log_likelihood(theta, data)

# Use scipy's minimize function to find the maximum likelihood estimate
from scipy.optimize import minimize

# Set the initial guess for the parameter values
theta_0 = [1.0, 1.0]

# Find the maximum likelihood estimate
result = minimize(neg_log_likelihood, theta_0, args=(data,), method='Nelder-Mead')

# Print the results
print("True parameter values: mu={}, sigma={}".format(mu_true, sigma_true))
print("MLE parameter values: mu={}, sigma={}".format(result.x[0], result.x[1]))
