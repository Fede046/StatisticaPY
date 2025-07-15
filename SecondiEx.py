


import scipy.stats as stats
import numpy as np
zalfamezzi = stats.norm.ppf(1-(1-0.95)/2)
margine_di_errore = 1.5/np.sqrt(90)
x_bar = 0.95
interva = (x_bar-zalfamezzi*errore_marginale,x_bar+zalfamezzi*errore_marginale) 

