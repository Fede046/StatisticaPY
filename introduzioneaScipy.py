import numpy as np
from scipy.stats import norm
import pandas as pd

# Distribuzione Normale: calcola la densità di probabilità (PDF) per x=2 con media 0 e deviazione standard 1
# Esempio: Qual è la probabilità che una variabile casuale normale standard sia esattamente 2?
#Una variabile casuale normale standard è una variabile casuale continua che segue una distribuzione normale con media 
#μ=0 e deviazione standard σ=1. Questa distribuzione è anche chiamata distribuzione normale standard o distribuzione Z.
print(norm.pdf(2, 0, 1))
#Cosa significa questo valore?
#Il valore 0.05399 non rappresenta una probabilità diretta, ma la densità di probabilità nel punto z=2.
#Per una variabile casuale continua, la probabilità di ottenere un valore esatto (ad esempio, esattamente 2) 
#è teoricamente 0, poiché la probabilità è definita su un intervallo. 
#Tuttavia, la densità di probabilità ci dà un'idea di quanto è "probabile" che la variabile casuale assuma valori vicini a 2.
#%%

from scipy.stats import poisson
#La distribuzione di Poisson è una distribuzione di probabilità discreta che descrive la probabilità che un certo 
#numero di eventi si verifichi in un intervallo di tempo o spazio fissato, sapendo che questi eventi si verificano 
#con un tasso medio noto e indipendentemente dal tempo trascorso dall'ultimo evento.
# Distribuzione di Poisson: calcola la probabilità di avere esattamente 3 eventi con un tasso di occorrenza di 10
# Esempio: Se il numero medio di chiamate ricevute da un call center è 10 all'ora,
# qual è la probabilità di ricevere esattamente 3 chiamate in un'ora?
print(poisson.pmf(3, 10))
#Interpretazione
#Se il risultato fosse, ad esempio, 0.0076, significherebbe che c'è una probabilità dello 0.76% di ricevere 
#esattamente 3 chiamate in un'ora.

#%%

from scipy.stats import binom
#La distribuzione binomiale è una distribuzione di probabilità discreta che descrive il 
#numero di successi in un numero fisso di prove indipendenti, ciascuna con la stessa probabilità di successo.

#La distribuzione binomiale è utile quando:

#Ci sono un numero fisso di prove (n).
#Ogni prova ha solo due possibili esiti: successo o fallimento.
#La probabilità di successo (p) è costante per ogni prova.
#Le prove sono indipendenti: il risultato di una prova non influenza le altre.

#La probabilità di ottenere esattamente k successi in n prove è data da = 
#P (X = k) = (n k)*p^k*(1-p)^(n-k)

#k è il numero di successi di cui si vuole calcolare la probabilità.
#(n k) è il coefficiente binomiale, che rappresenta il numero di modi in cui si possono ottenere k successi in n prove. 
#Si calcola come: (n k)= n!/(k!*(n−k)!)
#p è la probabilità di successo in una singola prova.
#1−p è la probabilità di fallimento in una singola prova.



# Distribuzione Binomiale: calcola la probabilità di avere esattamente 4 successi in 5 prove con probabilità di successo 0.51
# Esempio: Se lanci una moneta truccata (con probabilità di testa 0.51) 5 volte,
# qual è la probabilità di ottenere esattamente 4 teste?
print(binom.pmf(4, 5, 0.51))
#Interpretazione
#Se il risultato fosse, ad esempio, 0.166, significherebbe che c'è una probabilità del 16.6% 
#di ottenere esattamente 4 teste in 5 lanci.

#%%

from scipy.stats import binom

# Distribuzione Binomiale: calcola la probabilità cumulativa di avere al massimo 3 successi in 5
# prove con probabilità di successo 0.51
# Esempio: Se lanci una moneta truccata (con probabilità di testa 0.51) 5 volte, 
#qual è la probabilità di ottenere al massimo 3 teste?
#notamo che qua la differenza è la parola al massimo
print(binom.cdf(3.2, 5, 0.51))

#%%

from scipy.stats import binom

# Distribuzione Binomiale: calcola il valore atteso (media) per 5 prove con probabilità di successo 0.51
# Esempio: Se lanci una moneta truccata (con probabilità di testa 0.51) 5 volte, qual è il numero atteso di teste?
print(binom.mean(5, 0.51))

#%%

from scipy.stats import binom

#La deviazione standard è la radice quadrata della varianza. 
#Rappresenta la dispersione dei dati nella stessa unità di misura dei dati originali, 
#rendendola più interpretabile rispetto alla varianza.
#Interpretazione
#Una deviazione standard alta indica che i dati sono molto dispersi.
#Una deviazione standard bassa indica che i dati sono concentrati vicino alla media.

# Distribuzione Binomiale: calcola la deviazione standard per 5 prove con probabilità di successo 0.51
# Esempio: Se lanci una moneta truccata (con probabilità di testa 0.51) 5 volte,
# qual è la deviazione standard del numero di teste?
#attendzione la devizione stadard non la varianza!!!

print(binom.std(5, 0.51))

#%%

from scipy.stats import binom
#La varianza è una misura della dispersione dei dati rispetto alla media.
# Si calcola come la media dei quadrati degli scarti tra ciascun dato e la media.
#Interpretazione
#Una varianza alta indica che i dati sono molto dispersi intorno alla media.
#Una varianza bassa indica che i dati sono concentrati vicino alla media.
#Una varianza è considerata alta se i dati sono molto dispersi rispetto alla media. Ad esempio:
#Se la media è 10 e la varianza è 100, i dati sono molto dispersi.
#Se la media è 10 e la varianza è 1, i dati sono molto concentrati intorno alla media.

# Distribuzione Binomiale: calcola la varianza per 5 prove con probabilità di successo 0.51
# Esempio: Se lanci una moneta truccata (con probabilità di testa 0.51) 5 volte,
# qual è la varianza del numero di teste?
print(binom.var(5, 0.51))

#%%

from scipy.stats import binom

# Distribuzione Binomiale: genera un campione casuale di 5 prove con probabilità di successo 0.51
# Esempio: Simula il risultato di 5 lanci di una moneta truccata (con probabilità di testa 0.51)
print(binom.rvs(5, 0.51))


#%%
### Note:
#- **Distribuzione Normale**: Utilizzata per variabili continue, come altezze, pesi, ecc.
#- **Distribuzione di Poisson**: Utilizzata per contare il numero di eventi che accadono in un intervallo di tempo o spazio.
#- **Distribuzione Binomiale**: Utilizzata per contare il numero di successi in un numero fisso di prove indipendenti.
#%%
import numpy as np
import pandas as pd

v1 = np.array([11,435,332,3,98,798,3])

v2 = np.array([2,425,32,321,928,98,13])
check_vec1 = v1 > v2
check_vec2 = 3 > v2
#%%
confronto1 = check_vec1&check_vec2
confronto1 = check_vec2|check_vec2
#%%
v3 = np.array([-22,332,2,5,-55,332,23,-9,56,6])
v3[v3<0]=0
print(v3)
#%%
#con questo mi crea un sotovettore, elementi selezionati
#in v3
v3[[False,True,False,True,False,True,True,False,False,True]]
#%%
#np.random.normal(mean, std, size)
x= np.random.normal(0,1,(2,2))

print(x)
x= np.random.uniform(0,5,(2,2))
print(x)


#%%
x1 = np.ones((2,2)) # Crea la matrice di tutti 1
print(x1)
x0 = np.zeros((2,2)) # Crea la matrice di tutti 0
print(x0)
#%%
np.reshape(A, (m, n))


#%%
#far attenzione a invertire la matrice
np.inv(A)

#%%
np.linalg.rank(A) 
#restituisce il rango di A
#utilizzato sopratutto per le norme

#%%

b=np.zeros(6) #crea un vettore colonna di e 
A = np.zeros((6, 6)) #crea una matrice 6x6 di 0

colnum= A.shape[0] #numero nigne



for num in range(1, colnum):
    colonna =np.ones(colnum)
    colonna =num *colonna
    A[:, num] = colonna
    
    #%%
#vediamo la potenza del vettore
for num in range(colum):
    A[:,num] = num
    
    
    
#%%
Data = pd.DataFrame({"nome":["Mario","Luca"],
"cognome":["Rossi","Bianchi"],
"eta":[45,38]})
print(Data)

#%%
Data["nome"][1]


#%%
df[[True,False,True,False]]

#%%
import numpy as np
import pandas as pd

# Creazione del DataFrame
dfauto = pd.DataFrame({
    "Modello": ["Ford Focus", "Toyota Corolla", "Audi", "Fiat Panda", "Lancia"],
    "Prezzo (€)": [22000, np.nan, 25000, 35000, np.nan],
    "Anno di Vendita": [2018, 2020, np.nan, 2022, 2014]
})

# Stampa del DataFrame originale
print("DataFrame originale:")
print(dfauto)

# Eliminazione delle righe con valori NaN
dfauto_pulito = dfauto.dropna()

# Stampa del DataFrame pulito
print("\nDataFrame dopo l'eliminazione delle righe con NaN:")
print(dfauto_pulito)

# Aggiunta di una nuova colonna categorica (esempio)
# Supponiamo di voler aggiungere una colonna "Categoria" basata su una logica specifica
# Ad esempio, categorizziamo i modelli in base al prezzo
dfauto["Categoria"] = pd.cut(dfauto["Prezzo (€)"], bins=[0, 20000, 30000, np.inf], labels=["Economico", "Medio", "Costoso"])

# Convertiamo la colonna "Categoria" in tipo categorico
dfauto["Categoria"] = dfauto["Categoria"].astype("category")

# Stampa del DataFrame con la nuova colonna categorica
print("\nDataFrame con la colonna categorica aggiunta:")
print(dfauto)
#%%
import pandas as pd

# Leggere il file CSV
df = pd.read_csv("data.csv")

# Visualizzare le prime righe del DataFrame
print(df.head())
print(df.tail())

#%%
#distinzione da variabili categoriche e numeriche
#le stringhe sono variabili categoriche,
#stringhe che in questo caso non ci dicono troppo 
#di interessante in caso di un analisi,
#tipicamente sono stringhe
#nel caso di dfauto il nome delle macchine, se ti interessano le
#puoi manipolare come un numero







