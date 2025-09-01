import scipy.stats as stats
import numpy as np
from scipy.optimize import minimize


#Generare un campione casuale di 20 elementi da una distribuzione bino
#miale con p=0.55, n=100, utilizzando come seme generatore 10. Stimare
# con il metodo MLE il valore di p supponendo noto n.

np.random.seed(10)

campione = np.random.binomial(n=100, p=0.55,size=20)
print(campione)

#trovo il log della L(tetha)
def log_likeH(theta,campione):
    n = 100
    p = theta
    log_like = np.sum(stats.binom.logpmf(campione,n= n, p=p))
    return log_like

#Faccio il -log(L(Thetha)) -> Mimimizzo 
#Minimizzazione della fuznione di verosmiglianza

def neg_log_like(theta,campione):
    return -log_likeH(theta, campione)

#non dato da esercizio (inventato)
#Ipotesi iniziale
theta0 = 0.6

res = minimize(neg_log_like,theta0,args=(campione,),method='Nelder-Mead')


print(res.x[0])

#%%
import scipy.stats as stats
import numpy as np
from scipy.optimize import minimize


#Generare un campione casuale di 30 elementi da una distribuzione normale
# con media mu= 1 e deviazione standard std=1, utilizzando come seme
# generatore 10. Stimare con il metodo MLE il valore di mu e std.

np.random.seed(10)

campione = np.random.normal(size=30,loc = 1,scale=1)

def log_likeH(theta,campione):
    mu,ds = theta
    logL = np.sum(stats.norm.logpdf(campione,loc=mu,scale=ds))
    return logL

def neg_logL(theta,campione):
    return -log_likeH(theta, campione)

#Stima 0 (ipotizzo)
theta0 = [1,1]

res = minimize(neg_logL,theta0,args=(campione,),method='Nelder-Mead')

print(res.x[0],res.x[1])


#%%
import scipy.stats as stats
import numpy as np
from scipy.optimize import minimize

#Generare un campione casuale di 100 elementi da una distribuzione normale con parametri μ=1.5
# e σ=1.5
#, utilizzando come seme generatore 100. Stimare con il metodo MLE i valori di μ
# e σ
#.
#{ N.B. Utilizzare come valori iniziale μ0=1
#, σ0=1
# e come metodo per l’ottimizzazione method="Nelder-Mead"}.

np.random.seed(100)

campione = np.random.normal(loc=1.5,scale=1.5,size=100)

def log_likH(theta,campione):
    mu,ds = theta
    log = np.sum(stats.norm.logpdf(campione,loc=mu, scale=ds))
    return log

def neg_logLike1(theta,campione):
    return -log_likH(theta, campione)


#Valori iniz
theta0 = [1,1]

#res
res = minimize(neg_logLike1,theta0,args=(campione,),method="Nelder-Mead")
print(res.x[0])
print(res.x[1])










































