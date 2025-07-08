#Test di Ipotesi

#La concentrazione di zuccheri in una bevanda non può superare il 10%.
#Quale istruzione è corretta per fare un test di ipotesi su 70 bevande per la cerificazione
#H0 <=0.1
#Ha >0.1
import scipy.stats as stats
import numpy as np

confidenza = 0.95
ds = 1.5
x_bar = 0.95
n = 90
quantile = stats.norm.ppf(1-(1-confidenza)/2)

#mergine d'errore
margine = quantile*(ds/np.sqrt(n))

#Intervallo di conf
Inter = (x_bar-margine,x_bar+margine)
print(Inter)

