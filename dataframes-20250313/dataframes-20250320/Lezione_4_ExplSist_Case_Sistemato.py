import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import os
from matplotlib import pyplot as plt

#%%
path = r"C:\\Users\\malse\\Desktop\\Secondo Anno Secondo Periodo\\Statistica\\StatisticaPY\\dataframes-20250313\\dataframes-20250320"
path = r"C:/Users/malse/source/repos/StatisticaPY/dataframes-20250313/dataframes-20250320"
data=pd.read_csv(os.path.join(path, 'kc_house_data.csv'))

#%%

data.info()
#data.describe() #informazioni statistiche sulle singole variabili numeriche
print(data.head())

#%%
#Analizziamo le dimensioni del dataset
data.shape
N, d=data.shape
print(f"Il dataset ha {N} righe e {d} colonne")

#%%
#Estraiamo le variabili numeriche
numerical_col=data.select_dtypes(include=['float64','int64'])
print(f"Il dataset ha {numerical_col.shape[1]} colonne numeriche")
print(numerical_col.head())

#%%
#lo variabili categoriche è meglio che siano
#numeri o a cose che posso usare calcoli.
#Ma è possibile in alcuni casi non usare numeri, anche se non è prefereibile
print(numerical_col.isnull().sum())     #Controlliamo la presenza di valori NaN

#%%
# Analisi delle distribuzioni delle variabili numeriche tramite boxplot
# Un boxplot è un grafico che mostra la distribuzione di una variabile numerica, evidenziando:
# - La mediana (linea centrale).
# - I quartili (scatola: Q1 e Q3).
# - I valori minimo e massimo (baffi).
# - Gli outlier (punti al di fuori dei baffi).

# Iteriamo su una lista di colonne numeriche del dataset per creare un boxplot per ciascuna di esse.
for col in ['price', 'sqft_living', 'sqft_above', 'sqft_basement', 'bedrooms', 'bathrooms', 'floors', 'view', 'condition', 'yr_built', 'grade']:
    # Creiamo una nuova figura per ogni boxplot, con dimensioni 8x6 pollici.
    plt.figure(figsize=(8, 6))
    
    # Utilizziamo Seaborn per creare il boxplot della colonna corrente.
    # `x=data[col]` specifica la variabile da visualizzare.
    sns.boxplot(x=data[col])
    
    # Aggiungiamo un titolo al grafico, che include il nome della colonna analizzata.
    plt.title(f'Boxplot di {col}')
    
    # Mostriamo il grafico.
    plt.show()

    # Osservazioni:
    # - Il boxplot ci permette di identificare rapidamente la presenza di outlier, la dispersione dei dati
    #   e la simmetria della distribuzione.
    # - Ad esempio, se i baffi sono molto lunghi, significa che ci sono molti valori estremi (outlier).
    # - Se la mediana non è al centro della scatola, la distribuzione è asimmetrica.
#%%
#Controlliamo quante case non hanno il bagno e le rimuoviamo dal dataset
zero_bagni = data[data['bathrooms'] == 0]   

#controlliamo quante case non hanno una camera da letto
zero_cdl = data[data['bedrooms'] == 0]
numero_zero_cdl = zero_cdl.shape[0]
print(f'Il numero di case senza camere da letto è {numero_zero_cdl}')
print(zero_cdl.head())

data = data[data['bathrooms'] != 0] #rimuoviamo le case senza bagno
#Rimuoviamo le case senza camere da letto ma con una superficie
data=data[~((data['bedrooms'] == 0) & (data['sqft_living'] > 1000))]

data.info()

#%%
#scatter plot tra il prezzo delle case ("price") e la superficie abitabile ("sqft_living")
#molto utili per trovare outiler, e variabili correlate
plt.figure(figsize=(12, 6))
plt.scatter(data['sqft_living'], data['price'], alpha=0.5, color='red')
plt.title('Relazione tra Prezzo e Superficie')
plt.xlabel('sqft_living')
plt.ylabel('Price')

#%%
#Istogrammi per le variabili discrete
# Le variabili discrete sono un tipo di variabile statistica che può assumere solo un numero finito o numerabile di valori distinti.
#  Questi valori sono spesso interi (numeri interi) e rappresentano categorie, conteggi o risultati di processi che non possono essere 
#  suddivisi ulteriormente in modo significativo.

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
variabili=['view', 'condition', 'grade','bedrooms','bathrooms', 'floors']
# Creiamo gli istogrammi
for i, var in enumerate(variabili):
    row = i // 3  # Determiniamo la riga
    col = i % 3   # Determiniamo la colonna
    sns.histplot(data[var], bins=20, kde=False, ax=axes[row, col], color='orange')  # Plot dell'istogramma
    axes[row, col].set_title(f'Distribution of {var}', fontsize=16)
    axes[row, col].set_xlabel(var, fontsize=14)
    axes[row, col].set_ylabel('Frequency', fontsize=14)

# Spaziatura tra i subplots
plt.tight_layout()

plt.show()

#%%
#Istogramma con curva di densità per variabili continue
# Differenza tra variabili discrete e continue:
# Variabili discrete: Assumono valori distinti e separati. Esempi: numero di figli, numero di stanze.

# Variabili continue: Possono assumere qualsiasi valore all'interno di un intervallo. Esempi: altezza, peso, temperatura.
#molto interessante per il prezzo, che possono varire molto
variables = ['price', 'sqft_living', 'sqft_above', 'sqft_basement']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))  

for i, var in enumerate(variables):
    ax = axes[i // 2, i % 2] 
    sns.histplot(data[var], kde=True, color='orange', ax=ax)  
    ax.set_title(f'Histogram and Density Curve of {var}')  
    ax.set_xlabel(var)  
    ax.set_ylabel('Frequency')  

plt.tight_layout()  
plt.show()  

#%%
# Rappresentazione della matrice di correlazione per analizzare le relazioni statistiche tra le variabili numeriche
# La matrice di correlazione è uno strumento utile per identificare relazioni lineari tra coppie di variabili.

# Rimuoviamo le variabili categoriche o non numeriche dal dataset.
# In questo caso, "zipcode" e "view" sono variabili categoriche o non numeriche, quindi le escludiamo.
#Corr_data = numerical_col.drop(["zipcode", "view"], axis=1)  # Usiamo solo colonne numeriche

# Alternativamente, se volessimo usare l'intero dataset (escludendo solo colonne non numeriche):
Corr_data = data.drop(["zipcode", "view","date"], axis=1)  # Rimuoviamo colonne non numeriche
print(data.dtypes)
# Calcoliamo la matrice di correlazione utilizzando il metodo .corr().
# La matrice di correlazione è una tabella quadrata in cui ogni elemento rappresenta il coefficiente di correlazione
# tra due variabili. Il coefficiente di correlazione varia tra -1 e 1:
# - 1: Correlazione positiva perfetta.
# - -1: Correlazione negativa perfetta.
# - 0: Nessuna correlazione lineare.
C = Corr_data.corr()

# Stampiamo la dimensione della matrice di correlazione per verificare quante variabili sono state incluse.
print(f"La dimensione della matrice di correlazione è: {C.shape}")

# Visualizziamo la matrice di correlazione utilizzando una heatmap.
# - `annot=False`: Non mostriamo i valori numerici all'interno delle celle per evitare affollamento.
# - `cmap='coolwarm'`: Usiamo una scala di colori per rappresentare i valori di correlazione.
#   (Colori caldi per correlazioni positive, colori freddi per correlazioni negative).
sns.heatmap(C, annot=False, cmap='coolwarm')
plt.title('Matrice di correlazione')
plt.show()

# Interpretazione della heatmap:
# - Le aree con colori caldi (es. rosso) indicano una forte correlazione positiva tra le variabili.
# - Le aree con colori freddi (es. blu) indicano una forte correlazione negativa.
# - Le aree con colori neutri (es. bianco) indicano una correlazione vicina a zero.
# - La diagonale principale è sempre 1, poiché ogni variabile è perfettamente correlata con sé stessa.

# Esempi di osservazioni:
# - Se due variabili hanno un coefficiente di correlazione vicino a 1 o -1, sono fortemente correlate.
# - Se il coefficiente è vicino a 0, non c'è una relazione lineare significativa.
# - Questa analisi è utile per identificare multicollinearità (correlazione alta tra feature) in modelli di machine learning.

#%%
#°L'analisi di data esplorativa si per dati che utilizzarememo dopo, di
#Di solito di fanno per variabili numeriche
#Le varibili categoriche mi servono successivamente per algoritmi di classificazione
#numerical core utilizzano dati float e int, 
#

# #%%Distribuzioni condizionate
#  Oltre alle informazioni ottenibili dalla matrice di correlazione, e alle statistiche sul dataset
#  visibili tramite la funzione data.describe(), informazioni interessanti possono essere
#  ottenute condizionando su una delle features del dataset (ovvero, filtrando quegli elementi
#  che rispettano una data condizione).
#  Ad esempio, ` e possibile calcolare la media del qualit` a delle arance del dataset, e
#  confrontarla con la media della qualit` a delle arance condizionata al fatto che
#  pH> pHmedio.
#  Analizzando le statistiche delle distribuzioni condizionate rispetto a quelle non
#  condizionate, si possono fare interessanti osservazioni sui dati!
# #scatter plot
#isogrammi 
#bivariati scatter plot
#,multivariate la matrice di correlazione