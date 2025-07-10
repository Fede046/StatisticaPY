import numpy as np
import scipy.stats as stats
np.random.seed(42)  # Per riproducibilità

#H0 la bevanda non può superare il 10%
#Fatta su 70 bevande
#La concentrazione di zuccheri in una bevanda non puo superare il 10%.
# Quale istruzione e’ corretta per fare un test di ipotesi su 70 bevande per
 #la certificazione?

rvs = stats.uniform.rvs(size = 70,random_state = 3)
rvs = np.random.normal(0.1, 0.02, 70)

tt = stats.ttest_1samp(rvs, popmean=0.1,alternative='greater')
print(tt.pvalue)

