# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:17:43 2023

Stima del parametro p di una distribuzione binomiale
"""


#################################################################################################

# VERSION 2 
# stimo solo il parametro theta della distribuzione binomiale
#La funzione binomiale ha due paramtri, è una funzione discreta di probabilità, che rappresenta n processi di bernulli
#dove ciascun processo ha una probabilità di successo. 


import numpy as np
from scipy.stats import binom

# Define the true parameter values
#Suppongo di conoscere uno dei due process (n) e stimare
n = 100
p_true = 0.3 #deve essere <3 pk probabilità

# Generate some random data from the binomial distribution
#Mi genera un campione 
#n a sx dell'uguale è il nnome del parametro, quello che in ogni funzione binom 
#una sorta di etichetta, una stringa costante,
#il secondo n è il nome della vriabile asseganta in precedenza
#quelli che prima si chaimavano loc e scale
data = binom.rvs(n=n, p=p_true, size=1000, random_state=100)

# Define the log-likelihood function for the binomial distribution
#devo identificare theta che ha solo un valore
def log_likelihood(theta, data):
    n = 100
    #adesso il vettore dei parametri ha soltanto p
    P = theta
    #perchè una funzione discreta utilizzo logpmf
    log_lik = np.sum(binom.logpmf(data, n=n, p=P))
    return log_lik

# Define the function to maximize the log-likelihood
def neg_log_likelihood(theta, data):
    return -log_likelihood(theta, data)

# Use scipy's minimize function to find the maximum likelihood estimate
from scipy.optimize import minimize

# Set the initial guess for the parameter value
theta_0 = 0.5

# Find the maximum likelihood estimate
result = minimize(neg_log_likelihood, theta_0, args=(data,), method='Nelder-Mead')

# Print the results
#n non è stato stimato l'ho scritto, di solito è meglio stimare un parametro 
#rispetto che a due
print("True parameter values: n={}, p={}".format(n, p_true))
print("MLE parameter values: n={}, p={}".format(n, result.x[0]))

"""
n this program, we first define the true parameter values for the binomial distribution (n and p_true). We then generate some random data from the binomial distribution using these parameter values.

Next, we define the log-likelihood function for the binomial distribution, which takes a parameter value (theta) and the data as input, and returns the log-likelihood of the data given the parameter value. We also define the neg_log_likelihood function, which simply negates the log-likelihood function so that we can use it with scipy's minimize function.

We then use scipy's minimize function to find the maximum likelihood estimate of the parameter value. We set the initial guess for the parameter value to 0.5, and use the Nelder-Mead method for optimization.

Finally, we print out the true parameter values and the maximum likelihood estimate.
"""
