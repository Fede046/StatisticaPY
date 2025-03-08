import numpy as np
from scipy.stats import norm
import pandas as pd

''' variabile casuale normale standard'''

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

'''distribuzione di Poisson'''

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

'''La distribuzione binomiale'''

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
'''probabilità cumulativa'''
# Distribuzione Binomiale: calcola la probabilità cumulativa di avere al massimo 3 successi in 5
# prove con probabilità di successo 0.51
# Esempio: Se lanci una moneta truccata (con probabilità di testa 0.51) 5 volte, 
#qual è la probabilità di ottenere al massimo 3 teste?
#notamo che qua la differenza è la parola al massimo
print(binom.cdf(3.2, 5, 0.51))

#%%

from scipy.stats import binom
'''media'''
# Distribuzione Binomiale: calcola il valore atteso (media) per 5 prove con probabilità di successo 0.51
# Esempio: Se lanci una moneta truccata (con probabilità di testa 0.51) 5 volte, qual è il numero atteso di teste?
print(binom.mean(5, 0.51))

#%%

from scipy.stats import binom
'''deviazione standard'''
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
'''varianza'''
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
'''genera un campione casuale'''
# Distribuzione Binomiale: genera un campione casuale di 5 prove con probabilità di successo 0.51
# Esempio: Simula il risultato di 5 lanci di una moneta truccata (con probabilità di testa 0.51)
print(binom.rvs(5, 0.51))


#%%
### Note:
#- **Distribuzione Normale**: Utilizzata per variabili continue, come altezze, pesi, ecc.
#- **Distribuzione di Poisson**: Utilizzata per contare il numero di eventi che accadono in un intervallo di tempo o spazio.
#- **Distribuzione Binomiale**: Utilizzata per contare il numero di successi in un numero fisso di prove indipendenti.
#%%
# Importazione delle librerie necessarie
import numpy as np  # Libreria per operazioni numeriche e vettoriali
import pandas as pd  # Libreria per la manipolazione di dati strutturati (es. tabelle)
'''Confronto elemento per elemento tra v1 e v2'''
# Creazione di due array numpy
v1 = np.array([11, 435, 332, 3, 98, 798, 3])
v2 = np.array([2, 425, 32, 321, 928, 98, 13])

# Confronto elemento per elemento tra v1 e v2
check_vec1 = v1 > v2  # Restituisce un array di booleani dove ogni elemento è True se v1[i] > v2[i], altrimenti False
print(check_vec1)
# Confronto tra il numero 3 e ogni elemento di v2
check_vec2 = 3 > v2  # Restituisce un array di booleani dove ogni elemento è True se 3 > v2[i], altrimenti False
print(check_vec2)
#%%
'''Operazioni logiche tra i due array di booleani'''

confronto1 = check_vec1 & check_vec2  # AND logico: True solo se entrambe le condizioni sono vere
print(confronto1)
confronto2 = check_vec2 | check_vec2  # OR logico: True se almeno una delle condizioni è vera (in questo caso, è ridondante)
print(confronto2)
#%%
'''Sostituzione di tutti gli elementi negativi in v3 con 0'''
# Creazione di un nuovo array numpy
v3 = np.array([-22, 332, 2, 5, -55, 332, 23, -9, 56, 6])

# Sostituzione di tutti gli elementi negativi in v3 con 0
v3[v3 < 0] = 0
print(v3)  # Stampa l'array modificato

#%%
'''Selezione di elementi specifici da v3'''
# Selezione di elementi specifici da v3 utilizzando un array di booleani
# Solo gli elementi corrispondenti a True nell'array booleano vengono selezionati
v3[[False, True, False, True, False, True, True, False, False, True]]

#%%
'''Generazione di numeri casuali con distribuzione normale(Gaussiana)'''
#La distribuzione normale è una delle distribuzioni più importanti in statistica. 
#È caratterizzata da una forma a "campana" ed è definita da due parametri:
#Media (μ): Il valore centrale attorno cui si distribuiscono i dati.
#Deviazione Standard (σ): Misura la dispersione dei dati attorno alla media.

# Generazione di numeri casuali con distribuzione normale (media 0, deviazione standard 1)
x = np.random.normal(0, 1, (2, 2))  # Crea una matrice 2x2 con numeri casuali
print(x)

'''Generazione di numeri casuali con distribuzione uniforme'''
#La distribuzione uniforme è una distribuzione in cui tutti i valori hanno la stessa probabilità
# di verificarsi all'interno di un intervallo specificato. È definita da due parametri:
#Limite inferiore (a): Il valore minimo che può essere generato.
#Limite superiore (b): Il valore massimo che può essere generato.

# Generazione di numeri casuali con distribuzione uniforme (tra 0 e 5)
x = np.random.uniform(0, 5, (2, 2))  # Crea una matrice 2x2 con numeri casuali
print(x)

#%%
'''Creazione di matrici con tutti gli elementi uguali a 1 o 0'''
x1 = np.ones((2, 2))  # Crea una matrice 2x2 di tutti 1
print(x1)
x0 = np.zeros((2, 2))  # Crea una matrice 2x2 di tutti 0
print(x0)

#%%
'''Riformattazione di una matrice A in una nuova forma (m, n)'''
A = np.ones((2, 2))
## Riformattazione di una matrice A in una nuova forma (m, n)
# Nota: A deve essere definita prima di usare questa funzione
B = np.reshape(A, (4, 1))
print(A)
print(B)

#%%
# Inversione di una matrice A
# Nota: A deve essere definita e invertibile
# np.inv(A)

#%%
# Calcolo del rango di una matrice A
# Nota: A deve essere definita
# np.linalg.rank(A)  # Restituisce il rango della matrice A

#%%
'''Esercizio Riassuntivo'''
# Creazione di un vettore colonna di zeri e una matrice 6x6 di zeri
b = np.zeros(6)  # Crea un vettore colonna di 6 zeri
A = np.zeros((6, 6))  # Crea una matrice 6x6 di zeri

# Ottenimento del numero di righe della matrice A
colnum = A.shape[0]  # Restituisce il numero di righe (in questo caso 6)

# Riempimento della matrice A con valori crescenti in ogni colonna
for num in range(1, colnum):
    colonna = np.ones(colnum)  # Crea un vettore colonna di 1
    colonna = num * colonna  # Moltiplica ogni elemento del vettore per num
    A[:, num] = colonna  # Assegna il vettore alla colonna num-esima della matrice A

#%%
'''Esercizio Riassuntivo'''
# Riempimento della matrice A con valori crescenti in ogni colonna (versione alternativa)
for num in range(colnum):
    A[:, num] = num  # Assegna il valore num a tutta la colonna num-esima

#%%


'''DATAFRAME CON pandas'''

#%%
# Importazione della libreria pandas, comunemente utilizzata per la manipolazione di dati in formato tabellare (DataFrame)
import pandas as pd

# Creazione di un DataFrame (tabella) utilizzando pandas
'''Cosa è un dataframe??'''
# Un DataFrame è una struttura dati bidimensionale, simile a una tabella in un database o a un foglio di calcolo
'''Creazione DataFrame'''
Data = pd.DataFrame({
    "nome": ["Mario", "Luca"],  # Colonna "nome" con due valori: "Mario" e "Luca"
    "cognome": ["Rossi", "Bianchi"],  # Colonna "cognome" con due valori: "Rossi" e "Bianchi"
    "eta": [45, 38]  # Colonna "eta" con due valori: 45 e 38
})

# Stampa del DataFrame
print(Data)  # Visualizza l'intero DataFrame

#%%
'''Accesso a un elemento specifico del DataFrame'''
# Accesso a un elemento specifico del DataFrame
# La sintassi Data["nome"][1] seleziona la colonna "nome" e poi l'elemento alla riga 1 (indice 1)
# In Python, gli indici partono da 0, quindi [1] si riferisce al secondo elemento
print(Data["nome"][1])  # Restituisce "Luca"

#%%
'''Selezione di righe specifiche del DataFrame utilizzando un array di booleani'''
# Selezione di righe specifiche del DataFrame utilizzando un array di booleani
# Solo le righe corrispondenti a True nell'array booleano vengono selezionate
# Nota: nel codice originale, `df` non è definito. Si presume che l'intenzione fosse di usare `Data`
# Esempio corretto:
selezione = Data[[True, False]]  # Seleziona solo la prima riga (Mario Rossi, 45)
print(selezione)

#%%
'''Esercizio sui DataFrame'''
# Importazione delle librerie numpy e pandas
import numpy as np
import pandas as pd

# Creazione di un nuovo DataFrame chiamato `dfauto`
dfauto = pd.DataFrame({
    "Modello": ["Ford Focus", "Toyota Corolla", "Audi", "Fiat Panda", "Lancia"],  # Colonna "Modello"
    "Prezzo (€)": [22000, np.nan, 25000, 35000, np.nan],  # Colonna "Prezzo (€)" con alcuni valori mancanti (NaN)
    "Anno di Vendita": [2018, 2020, np.nan, 2022, 2014]  # Colonna "Anno di Vendita" con alcuni valori mancanti (NaN)
})

# Stampa del DataFrame originale
print("DataFrame originale:")
print(dfauto)

'''Eliminazione delle righe con valori NaN, col metodo dropna()'''
# Eliminazione delle righe con valori NaN (Not a Number, ovvero valori mancanti)
# Il metodo `dropna()` rimuove tutte le righe che contengono almeno un valore NaN
dfauto_pulito = dfauto.dropna()

# Stampa del DataFrame dopo la rimozione delle righe con NaN
print("\nDataFrame dopo l'eliminazione delle righe con NaN:")
print(dfauto_pulito)



'''Aggiunta di una nuova colonna categorica chiamata "Categoria" con la funzione pd.cut'''
# Aggiunta di una nuova colonna categorica chiamata "Categoria"
# La colonna "Categoria" viene creata utilizzando la funzione `pd.cut()`, che categorizza i valori della colonna "Prezzo (€)"
# in base ai bin specificati: 0-20000 (Economico), 20000-30000 (Medio), 30000-infinito (Costoso)

#pd.cut(): È una funzione di pandas che permette di suddividere i valori di una colonna in intervalli (bin) 
#e assegnare a ciascun intervallo un'etichetta (label).

#dfauto["Prezzo (€)"]: È la colonna del DataFrame da cui vengono presi i valori da categorizzare.

#bins=[0, 20000, 30000, np.inf]: Definisce gli intervalli (bin) in cui suddividere i valori:
#0 - 20000: Tutti i prezzi compresi tra 0 e 20.000 €.
#20000 - 30000: Tutti i prezzi compresi tra 20.000 € e 30.000 €.
#30000 - np.inf: Tutti i prezzi superiori a 30.000 € (dove np.inf rappresenta l'infinito).

#labels=["Economico", "Medio", "Costoso"]: Assegna un'etichetta a ciascun intervallo:
#I prezzi tra 0 e 20.000 € sono categorizzati come "Economico".
#I prezzi tra 20.000 € e 30.000 € sono categorizzati come "Medio".
#I prezzi superiori a 30.000 € sono categorizzati come "Costoso".

dfauto["Categoria"] = pd.cut(dfauto["Prezzo (€)"], bins=[0, 20000, 30000, np.inf], labels=["Economico", "Medio", "Costoso"])

'''Conversione della colonna "Categoria" in tipo categorico con astype("category")'''
# Conversione della colonna "Categoria" in tipo categorico
#astype("category"): Converte la colonna "Categoria" in un tipo di dato categorico.
#Tipo categorico: In pandas, una colonna di tipo categorico è ottimizzata per memorizzare valori ripetuti. 
#Invece di memorizzare ogni valore come una stringa separata, pandas memorizza un elenco unico di categorie 
#e assegna a ciascuna riga un riferimento a una di queste categorie.
# Questo è utile per ottimizzare la memoria e migliorare l'efficienza delle operazioni su colonne con valori ripetuti
dfauto["Categoria"] = dfauto["Categoria"].astype("category")

# Stampa del DataFrame con la nuova colonna categorica
print("\nDataFrame con la colonna categorica aggiunta:")
print(dfauto)


#%%
# Importazione della libreria pandas
import pandas as pd

'''Lettura di un file CSV chiamato "data.csv" in un DataFrame'''
# Lettura di un file CSV chiamato "data.csv" in un DataFrame
# Il file CSV deve essere presente nella directory di lavoro o bisogna specificare il percorso completo
df = pd.read_csv("data.csv")

'''Visualizzazione delle prime righe del DataFrame utilizzando il metodo `head()`'''
# Visualizzazione delle prime righe del DataFrame utilizzando il metodo `head()`
# Di default, `head()` mostra le prime 5 righe
print(df.head())

'''Visualizzazione delle ultime righe del DataFrame utilizzando il metodo `tail()`'''
# Visualizzazione delle ultime righe del DataFrame utilizzando il metodo `tail()`
# Di default, `tail()` mostra le ultime 5 righe
print(dfauto.tail())



