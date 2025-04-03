import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

#%%
path="C:\\Users\\dario\\Desktop\\DARIO\\Tutorati\\Statistica Numerica"
os.chdir(path)

#Carichiamo il dataset usando pandas 
data=pd.read_csv(os.path.join(path, "Orange Quality Data.csv"))

#%%
#Dimensioni del dataset
N, d = data.shape
print(f"Il Dataset ha {N} righe e {d} colonne.")

#%%
#Stampiamo un po' di informazioni sul dataset
data.info()

#%%
print (data.head())
print (data.tail())

#%%
#Informazioni statistiche sul dataset
pd.set_option('display.max_columns', None)
data.describe()

#%%
#Creiamo un subdataframe con le sole variabili numeriche
num_type=['float64','int64']
numerical_col=data.select_dtypes(include=num_type)
print(numerical_col.head())

#%%
#Impostiamo variabili categoriche quelle che non sono
for col in data.columns:
    print(f"{col} type: {data[col].dtype}.")
    if data[col].dtype not in num_type:
        data[col]=data[col].astype("category")
        print(f"{col} type: {data[col].dtype}.")
    print("-"*45)

#%%
#Cerchiamo i valori NaN
#data.isnull() resitutisce True per le celle che contengono NaN

total_NaN=data.isnull().sum()
print(total_NaN)
data.dropna()


#%%
#Istogramma delle colonne numeriche in matplolib
fig, ax = plt.subplots (nrows=2, ncols=4)

for j in range(len(numerical_col.columns)):
    ax[j//4, j%4].hist(numerical_col.iloc[:,j], bins=20, color='orange')
    ax[j//4, j%4].set_title(numerical_col.columns[j])
    
plt.tight_layout() 
plt.show()
#%%
#Istogramma delle colonne con curva di densità
fig, ax = plt.subplots (nrows=2, ncols=2)

for j in range(4):
    sns.histplot(numerical_col.iloc[:,j], kde=True, bins=20, color='blue', ax=ax[j//2, j%2])
    
plt.tight_layout() 
plt.show()

#%%
fig, ax = plt.subplots(2, 2, figsize=(10, 10))  

# Lista delle colonne da plottare
colonne = ['Softness (1-5)', 'Ripeness (1-5)', 'pH (Acidity)', 'Quality (1-5)']

# Ciclo per creare i grafici a torta
for i, col in enumerate(colonne):
    temp = numerical_col[col].value_counts()  # Conta le occorrenze
    ax[i//2, i%2].pie(temp.values, labels=temp.index, autopct='%1.1f%%', startangle=90)
    ax[i//2, i%2].set_title(f'Distribuzione di {col}')  # Titolo del grafico

# Aggiusta il layout per evitare sovrapposizioni
plt.tight_layout()
plt.show()

#%%
#Diagrammi a torta singoli
for i, col in enumerate(colonne):
    temp = numerical_col[col].value_counts()
    plt.figure(figsize=(8,6))
    plt.pie(temp.values, labels=temp.index, autopct='%1.1f%%', startangle=90)
    plt.title(f'Distribuzione di {col}')
    plt.show()
#%%Boxplot
for colonna in numerical_col.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(numerical_col[colonna])
    plt.title(f'Boxplot di {colonna}')
    plt.show()
#%%
#Calcoliamo la media condizionata della qualità in base al pH e sweet

media_quality=np.mean(numerical_col['Quality (1-5)'])
media_ph=np.mean(numerical_col['pH (Acidity)'])
media_sweet=np.mean(numerical_col['Brix (Sweetness)'])

#Sub-dataframe
filtered_pH=numerical_col[numerical_col['pH (Acidity)']>media_ph]
filtered_sweet=numerical_col[numerical_col['Brix (Sweetness)']>media_sweet]

media_quality_ph=np.mean(filtered_pH['Quality (1-5)'])
media_quality_sweet=np.mean(filtered_sweet['Quality (1-5)'])

print(f"Qualità media: {media_quality:.2f}\n")
print(f"Qualità media per pH>phmedio: {media_quality_ph:.2f} \n")
print(f"Qualità media per sweet>sweet_medio: {media_quality_sweet:.2f} \n")

#%%
plt.figure()
sns.histplot(filtered_pH['Brix (Sweetness)'], bins=20, kde=True)
plt.title('Zuccheri condizionati al valore del pH')
plt.show()
#%%

plt.scatter(data["pH (Acidity)"], data["Quality (1-5)"], alpha=0.5, color='red')
plt.title("Scatter plot tra pH e Qualità")
plt.xlabel('pH')
plt.ylabel('Quality')
plt.show()

#%%
C=numerical_col.corr()
print(f"La dimensione della matrice di Correlazione è: {C.shape}")
sns.heatmap(C, annot=True, cbar=True, cmap='coolwarm', fmt='.2f')


