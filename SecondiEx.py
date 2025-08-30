
#%%

import scipy.stats as stats
#La concentrazione di zuccheri in una bevanda non puo superare il 10%
 #con una tolleranza del 5%. Il test di ipotesi su un campione di 70 bevande
 #restutuisce un p-value p = 0.01. Posso certificare l’azienda


#P value accettata H0 se maggiore di 0.05 altrimenti rifiuto -> rifiuto

#%%

import scipy.stats as stats

#La concentrazione di zuccheri in una bevanda non puo superare il 10%.
#Quale istruzione e’ corretta per fare un test di ipotesi su 70 bevande per
#la certificazione?

rvs = stats.uniform.rvs(size=70,random_state=10)
tt = stats.ttest_1samp(rvs, popmean=0.1,alternative='greater')
print(tt[1])





