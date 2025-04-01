# -*- coding: utf-8 -*-
"""

Questa parte è in esame 100


stima parametri media e varianza di una distribuzione normale.
"""

import numpy as np
from scipy.stats import norm

# Define the true parameter values
mu_true = 2.0
sigma_true = 1.5

# Generate some random data from the normal distribution
#estrae un simple random sample della distribuzione

data = norm.rvs(loc=mu_true, scale=sigma_true, size=1000,random_state=100)
#random state serve per fissare il seme generatore nella generazione di numeri casulai,
#non importante il numero, ma ci permette di generare a sessa sequenza di generazione
#di numeri casuale

# Define the log-likelihood function for the normal distribution
#mi calcolo il logarimo della funzione di lighliwood, funzione che dipende da due 
#paramteri theta 1 e theta 2,
#Questa funzione ha bisogna di avere i dati e il tipo di distribuzione a cui i dati 
#appartengono
def log_likelihood(theta, data):
    mu, sigma = theta
    #pdf della distibuzione normale, mentre la media e la deviazione 
    #standard vemgono lasciate ingognite.
    #Questi valori del logratimo vengono sommate
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
