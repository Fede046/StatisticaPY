import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import os
from matplotlib import pyplot as plt

#%%
path = r"C:/Users/malse/source/repos/StatisticaPY/Esercizi"
data=pd.read_csv(os.path.join(path, 'irish_animals.csv'))

data.info()


#%%
#Analizziamo le dimensioni del dataset
data.shape
N, d=data.shape
print(f"Il dataset ha {N} righe e {d} colonne")

#%%
# Imposta l'opzione per visualizzare tutte le colonne
pd.set_option('display.max_columns', None)

# Mostra una descrizione statistica del dataset
data.describe()

#%%
# Cerchiamo i valori NaN
# data.isnull() restituisce True per le celle che contengono NaN
# Possiamo vedere per ogni colonna quanti NaN ci sono, li cancella anche
# In questo caso non ci sono valori NaN
# Ci potrebbero essere delle procedure per sostituire il NaN, per
# esempio la media,
total_NaN = data.isnull().sum()
print(total_NaN)
data.dropna()

#%%
# E’ spesso una buona idea costruirsi un sub-dataframe del dataset principale contenente
# solo le variabili numeriche di data. Su questo sub-dataframe sarà possibile eseguire le
# indagini statistiche tipiche dell’EDA.
# Extract numeric values
numeric_data = data.select_dtypes(include='number')
print(numeric_data.head())


#%%
# Metodi statistici per la distribuzione dei dati
# Boxplot delle colonne numeriche in matplotlib

# Creiamo una figura con una griglia 2x4 per i grafici
fig, ax = plt.subplots(nrows=2, ncols=4)

# Ciclo per creare istogrammi per ogni colonna numerica
for j in range(len(numeric_data.columns)):
    # Creiamo un istogramma per la colonna corrente con 20 bin e colore arancione
    ax[j//4, j%4].hist(numeric_data.iloc[:,j], bins=20, color='orange')
    # Impostiamo il titolo del grafico con il nome della colonna
    ax[j//4, j%4].set_title(numeric_data.columns[j])
    
# Aggiustiamo il layout per evitare sovrapposizioni
plt.tight_layout() 
# Mostriamo i grafici
plt.show()

#%%
# Boxplot delle colonne con curva di densità
# Utilizziamo seaborn al posto di matplotlib

# Creiamo una figura con una griglia 2x2 per i grafici
fig, ax = plt.subplots(nrows=2, ncols=2)

# Ciclo per creare istogrammi con curva di densità (KDE) per ogni colonna numerica
for j in range(4):
    # Creiamo un istogramma con curva di densità per la colonna corrente
    sns.histplot(numeric_data.iloc[:,j], kde=True, bins=20, color='blue', ax=ax[j//2, j%2])
    
# Aggiustiamo il layout per evitare sovrapposizioni
plt.tight_layout() 
# Mostriamo i grafici
plt.show()



