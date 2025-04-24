import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize

#%%
#Esercizio 1
import numpy as np
# Definisco i valori reali dei parametri
n = 20
p_true = 0.55

# Genero alcuni dati casuali dalla distribuzione binomiale
data = stats.binom.rvs(n=n, p=p_true, size=1000, random_state=100)
#Con questo abbiamo generato il nostro vettore di variabili aleatorie
# Definisco la funzione di log-verosimiglianza per la distribuzione binomiale
def log_likelihood(theta, data):
    n = 20
    p = theta
    log_lik = np.sum(stats.binom.logpmf(data, n=n, p=p))
    return log_lik

# Definisco la funzione che voglio minimizzare (la log-verosimiglianza negativa)
# Siccome n è noto e dobbiamo stimare solo p, p lo chiamiamo theta (parametro da stimare)
def neg_log_likelihood(theta, data):
    return -log_likelihood(theta, data)

# Importo la funzione minimize di scipy per trovare la MLE
from scipy.optimize import minimize

# Imposto un valore iniziale per il parametro da stimare
theta_0 = 0.5

# Trovo la stima di massima verosimiglianza
#neg_log_likelihood -> funzione da minimizzare
result = minimize(neg_log_likelihood, theta_0, args=(data,), method='Nelder-Mead')

# Stampo i risultati
print(f"Valori reali dei parametri: n={n}, p={p_true}")
print(f"Valore stimato tramite MLE: n={n}, p={result.x[0]}")


#%%
#Esercizio 3
#In questo caso stimiamo la media mu e la deviazioen standard


# Definiamo i valori veri dei parametri
mu_true = 1
sigma_true = 1

# Generiamo i nostri valori
#scale -> deviazione standard
vettore_aleatorio = stats.norm.rvs(loc=mu_true, scale=sigma_true, size=30, random_state=10)

# Definiamo la funzione log_likelihood per la normale 
def log_likelihood(theta, data):
    mu, sigma = theta
    log_lik = np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma))
    return log_lik

# Definiamo la log-likelihood negativa
def neg_log_likelihood(theta, data):
    return -log_likelihood(theta, data)

# Troviamo il massimo valore dei parametri, che non massimizzino la funzione
theta_0 = [3.0, 2.0]   #guess_inziale 
result = minimize(neg_log_likelihood, theta_0, args=(vettore_aleatorio,), method='Nelder-Mead')


# Stampiamo i risultati
print(f"Valori reali dei parametri: mu={mu_true}, sigma={sigma_true}")
print(f"Valore stimato tramite MLE: mu={result.x[0]}, sigma={result.x[1]}")

#%%
#Esericizio 3 bis
#Metodo tramite il calcolo a mano esplicito degli stimatori
 
mu_hat = np.mean(vettore_aleatorio)
sigma2_hat = np.mean ((vettore_aleatorio - mu_hat)**2)

print(f"Stima della media della normale: {mu_hat} ")
print(f"Stima della varianza della normale: {np.sqrt(sigma2_hat)} ")
