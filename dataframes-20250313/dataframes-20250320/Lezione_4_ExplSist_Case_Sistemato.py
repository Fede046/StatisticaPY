import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import os
from matplotlib import pyplot as plt

#%%
path = r"C:\\Users\\malse\\Desktop\\Secondo Anno Secondo Periodo\\Statistica\\StatisticaPY\\dataframes-20250313\\dataframes-20250320"
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
for col in ['price','sqft_living', 'sqft_above', 'sqft_basement', 'bedrooms', 'bathrooms', 'floors', 'view', 'condition', 'yr_built', 'grade']:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot di {col}')
    plt.show()
    
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
#rappresentiamo la matrice di correlazione per indagare le relazione statistiche
#
Corr_data=numerical_col.drop(["zipcode","view"],axis=1) #togliamo variabili categoriche

Corr_data=data.drop(["zipcode","view"],axis=1) #togliamo variabili categoriche
C=Corr_data.corr()
print(f"La dimensione della matrice di Correlazione è: {C.shape}")
sns.heatmap(C, annot=False)

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