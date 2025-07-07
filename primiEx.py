
import scipy.stats as stats
import numpy as np

#Un particolare numero telefonico `e usato per ricevere sia fax che
#chiamate vocali. Se il 10% delle chiamate `e costituito da fax, e considerando 100 chiamate, qual `e la probabilit`a che:
#– 15 chiamate siano fax


#calcolo della pmf
#da utilizzare nel caso binomiale 
#  stats.binom.pmf(k,n(args),p(kwds))
#La PMF indica una funzione che associa ad ogni possibile valore di una variabile casuale discreta la
#probabilità che tale variabile associa quel valore specifico.
#In poche parole in base all probabilità che capiti un evento,
# mi dice la probabilità del che quella successione di eventi è capitata

#PMF (Probability Mass Function): Restituisce la probabilità di un singolo valore in una distribuzione discreta.

numeroPMF = stats.binom.pmf(15,100,0.1)

#calcolo dell media PMF (mu)
mediaPMF   = stats.binom.mean(100,0.1)

#calcolo varianza PMF (alfa quadro)
varianzaPMF = stats.binom.var(100,0.1)



print(numeroPMF)
print(mediaPMF)
print(varianzaPMF)


#%%
import scipy.stats as stats
import numpy as np
#PDF 

#Ti dice la probabilità di <=x, quindi sommi tutte le probabilità fino a quel punto 
#Per una distribuzione continua, la cdf è l'area sotto la pdf fino a quel punto 
#Probabilità che ci siano meno di 100 nascite di femmine su 500 nascite
#n=500
#p(nascita femmine) = 49.5 %
#k=100 (considera come minore di k)

numeroCDF = stats.binom.cdf(100,500,0.4905)

print(numeroCDF)


#%%


import scipy.stats as stats
import numpy as np
#Distribuzione di Poisson
#questa non è una distribuzione binomiale!!!!
#Questa distribuzione è legata e eventi rari quali l'arrivo di un clienet ein banca.
#lambda = è la media dell'evento nell'intervallo di tempo [0,1]
#X = numero di eventi che accadono nell'unità di tempo

#Calcolo la pmf di x


# Probabilità che arrivino 23 clienti tra le 12 e le 13
lambda1 = 20  # 20 clienti/ora
prob_23 = stats.poisson.pmf(23, lambda1)

# Probabilità che arrivino 10 clienti tra le 12 e le 12:30
lambda2 = 20 * 0.5  # 20 clienti/ora * 0.5 ore = 10
prob_10 = stats.poisson.pmf(10, lambda2)

print(f"Probabilità che arrivino 23 clienti tra le 12 e le 13: {prob_23:.4f}")
print(f"Probabilità che arrivino 10 clienti tra le 12 e le 12:30: {prob_10:.4f}")


#%%

#Distribuzione normale
#La media dei voti degli studenti di Informatica hanno una distribuzione normale con
#media 24.5 e deviazione standard 2.

#a) Qual `e la probabilit`a che che la media dei voti nello scorso anno
#accademico sia minore di 25?

#Calcolo la cdf nel caso della distribuzione normale 
#k=25
#media(mu) = 24.5
#deviazioni standard (scarto quadratico) (alfa) = 2
numNormCDF = stats.norm.cdf(25,24.5,2)
print(numNormCDF)


#b) Qual `e la probabilit`a che che la media dei voti nello scorso anno
#accademico sia compresa fra 25 e 26?


#P(25<x<26) -> P(x<=26) - P(x<=25)

probXmin26 = stats.norm.cdf(26,24.5,2) 


probXmin25 = stats.norm.cdf(25,24.5,2)

probXtra25e26 =probXmin26 - probXmin25

print(probXtra25e26)

#%%


#Un modello di auto h`a una velocit`a massima con distribuzione normale
#con media 180 km/h e deviazione standard 2 km/h. Eseguendo un test su
#una macchina,:
#Qual `e la probabilit`a che abbia una velocit`a massima minore di 181.5
#km/h?
prob181 = stats.norm.cdf(181.5,180,2)
print(prob181)
#Qual `e la probabilit`a che la velocit`a massima sia compresa fra 177 e
#182 km/h?
print(stats.norm.cdf(182,180,2)-stats.norm.cdf(177,180,2))


#qual `e la probabilit`a che la velocit`a massima sia compresa fra 180 e
#182 km/h?
print(stats.norm.cdf(182,180,2)-stats.norm.cdf(180,180,2))



#%%

#Distribuzione Esponenziale

#In una banca arrivano in media 12 clienti all'ora. Qual è la 
#probabilità che arrivi il primo clienete in meno 30 min?
#In generale l'evento è un arrivo, quindi si misura il tempo prima dell'arrivo dell'evento
#lambda = indica la media degli eventi nell'unità di tempo 
#lambda = 12 clienti all'ora
#lambdaSGN = 12/2 clienti in 30 min
# X = tempo prima che arrivi il primo cliente (exp(lambdaSGN))

#Devo trovare P(X<=30)
primoMeno30min = stats.expon.cdf(30,12/2)
print(primoMeno30min)


#%%
#Test di Ipotesi
#p-value -> deve essere essere >=0.5 ALTRIMENTO Ho vieni rigettata
 
#La concentrazione di zuccheri in una bevanda non puo superare il 10%
#on una tolleranza del 5%. Il test di ipotesi su un campione di 70 bevande
#restutuisce un p-value p = 0.01. Posso certificare l’azienda?
#Ho è rigettata perchè p-value è minore di 0.5


#La concentrazione di zuccheri in una bevanda non puo superare il 10%.
#Quale istruzione e’ corretta per fare un test di ipotesi su 70 bevande per
#la certificazione?

#1)Formulo un Hp nulla H0
#H0: media= 0.1 
#2) Formulo una Hp alternativa Ha (va contro l'Hp nulla)
#Ha: media>0.1

#Questa formula 
#4) Calcola la media campionaria 
#5) Confronta il valore della statistica calcolata 
#con quello della H0 e calcola il p-value

np.random.seed(42)  # Per riproducibilità
media = 0.1
dev_std = 0.02 #ipotizzata
campione = np.random.normal(media, dev_std, 70)
#per la formula ho bisogno di un array di campioni creo un campione test

risultato = stats.ttest_1samp(campione, popmean=0.1,alternative='greater')
print(f"Statistica t: {risultato.statistic}, p-value: {risultato.pvalue}")

#In questo caso il p-value è certificato perchè super 0.5
#LA PROF CHIEDE SOLO IL P-VALUE
#La t rappresenta
# la differenza tra la media campionaria osservata e la media ipotizzata sotto H0

#t positivo e grande: Evidenza a favore di Ha(mu>0.1)
#t vicino a 0: Poca differenza rispetto a H0
#t negativo: La media campionaria è inferiore a 0.1


#%%

#Dato il seguente test di Hp 

#SRS = 50
rvs = stats.uniform.rvs(size=50,random_state=10)
print(rvs)
tt = stats.ttest_1samp(rvs, popmean=0.5)


#%%

#Si abbia un SRS(50) da una distribuzione normale con media µ deviazione
#standard 1. Supponendo che la media campionaria calcolata sia ¯x = 35,
#e la deviazione standard campionaria sia S = 1.2qual `e l’intervallo di
#confidenza al 95% della media µ?

#Dati
n=50 #Dimensione del campione (SRS(50))
x_bar = 35 #Media campionaria
sigma = 1 #Deviazione standard della popolazione (nota) 
s = 1.2 #Deviazione standard campionaria #dato inutile
confidence = 0.95 #Livello di confidenza

#caso 1a
#Calcolo dell'intervallo di confidenza con distribuzione normale e deviazione standard nota

#trovo il quantile (z alfa/2) per l'intervallo di confidenza 95%
z_critical = stats.norm.ppf(1-(1-confidence)/2)   

#applico la formula per l'intervallo di confidenza 
standard_error = sigma/np.sqrt(n)    #errore standard 
margin_of_error = z_critical * standard_error #margine di errore

#intervallo di confidenza
confidence_interval = (x_bar - margin_of_error, x_bar + margin_of_error)

print(confidence_interval)

#%%


#Si abbia un SRS(100) da una distribuzione normale con media µ deviazione
#standard 2.5. Supponendo che la media campionaria calcolata sia ¯x = 18
#e la deviazione standard campionaria sia S = 2.45, qual `e l’intervallo di
#confidenza al 99% della media µ? E al 90%


#Dati 
#Distribuzione normale 
n=100  #numero di campioni 
sigma = 2.5   #deviazione standard (nota)
x_bar = 18  #media 
s = 2.45   # deviazione standard campionaria
confidenza1 = 0.99 
confidenza2 = 0.90

#caso 1a) abbiamo la deviazione standard

#primo caso 0.99

#trovo il quantile per 0.99
z_alfamezzi1 = stats.norm.ppf(1-(1-confidenza1)/2) 

#errore standard 
errore_stanard = sigma/np.sqrt(n)

#margine di errore
margine_di_errore = z_alfamezzi1*errore_stanard

#intervallo di confidenza
intervalloConf1 = (x_bar-margine_di_errore,x_bar+margine_di_errore)

print(intervalloConf1)

#analogo procedimento per il caso 2
#%%

#Si abbia un SRS(7) da una distribuzione normale con media µ deviazione
#standard non nota. Supponendo che la media campionaria calcolata sia
#x¯ = 18 e la deviazione standard campionaria sia S = 1.8, qual `e l’intervallo
#di confidenza al 99% della media µ? E al 90%

#Dati 
n=7 #numero di elementi
#sigma non nota 
x_bar = 18 #media
s = 1.8 #deviazione standard campionaria
confidenza = 0.99 

#caso 1b) distribuzione normale ma non sappiamo la deviazione standard

#trovo il quantile
#in questo caso devo utilizzare il t di student perchè abbiamo un numero di elementi <40

degree_of_freedom = n-1
t_student = stats.t.ppf(1-(1-confidenza)/2,degree_of_freedom)


errore_standard = s/np.sqrt(n)
margine_errore = t_student*errore_standard
intervallo = (x_bar-margine_errore,x_bar+margine_errore)

print(intervallo)

#analogo per 0.90
#%%

#Si abbia un SRS(80) da una distribuzione di poisson con media µ deviazione standard non nota. 
#Supponendo che la media campionaria calcolata
#sia ¯x = 20 , qual `e l’intervallo di confidenza al 95% della media µ?

#Dati 
#distribuzione non normale 
n=80 #numero elementi
x_bar = 20 #media
confidenza = 0.95
s = 1 #dato dalla prof 

# 1b) ma ci sono n>40 qundi posso usare la z_alfa/2 come quantile

z = stats.norm.ppf(1-(1-confidenza)/2)

#applico la formula per calcolare l'intervallo di confidenza

errore_standard = s/np.sqrt(n)
margine_errore = z*errore_standard
intervallo = (x_bar-margine_errore,x_bar+margine_errore)

print(intervallo)






