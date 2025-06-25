import kagglehub
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

# Download latest version
path = kagglehub.dataset_download("adilshamim8/education-and-career-success")

print("Path to dataset files:", path)
#Il file viene salvato in:
#C:\Users\malse\.cache\kagglehub\datasets\adilshamim8\education-and-career-success\versions\1\education_career_success.csv
# List files in the downloaded directory to find the exact filename
print("Files in directory:", os.listdir(path))

# Load the dataset using the correct filename
data = pd.read_csv(os.path.join(path, "education_career_success.csv"))

# Display the first few rows to verify it loaded correctly
print(data.head())

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
# Check the results
data.info()
#%%
#Cerchiamo i valori NaN

total_NaN=data.isnull().sum()
print(total_NaN)
data.dropna()
#%%
#Controlliamo quanti studenti sono other e le rimuoviamo dal dataset
zero_cdl = data[data['Gender'] == 'Other']

#controlliamo quante case non hanno una camera da letto
numero_zero_cdl = zero_cdl.shape[0]
print(f'Il numero di Other è {numero_zero_cdl}')
print(zero_cdl.head())

data = data[data['Gender']!= 'Other'] #rimuoviamo le case senza bagno

#controlliamo quante case non hanno una camera da letto
zero_cdl = data[data['Gender'] == 'Other']
numero_zero_cdl = zero_cdl.shape[0]
print(f'Il numero di Other è {numero_zero_cdl}')
print(zero_cdl.head())

data.info()


#%%

#Domanda: Gli studenti più anziani guadagnano di più?
#scatter plot tra Age e il Starting_Salary

plt.figure(figsize=(12, 6))
plt.scatter(data['Age'], data['Starting_Salary'], alpha=0.5, color='red')
plt.title('Relazione tra Age e Starting_Salary')
plt.xlabel('Starting_Salary')
plt.ylabel('Age')

#%%

#Domanda: Gli studenti più anziani guadagnano di più?
#scatter plot tra Projects_Completed  e Career_Satisfaction

plt.figure(figsize=(12, 6))
plt.scatter(data['Projects_Completed'], data['Career_Satisfaction'], alpha=0.5, color='red')
plt.title('Relazione tra Projects_Completed  e Career_Satisfaction')
plt.xlabel('Career_Satisfaction')
plt.ylabel('Projects_Completed ')

#%%

#Domanda: Gli studenti più anziani guadagnano di più?
#scatter plot tra High_School_GPA  e il University_GPA 

plt.figure(figsize=(12, 6))
plt.scatter(data['Years_to_Promotion'], data['Starting_Salary'], alpha=0.5, color='red')
plt.title('Relazione tra High_School_GPA  e University_GPA ')
plt.xlabel('University_GPA ')
plt.ylabel('High_School_GPA ')

#%%

#%%
#Istogramma con curva di densità per variabili continue
variables = ['High_School_GPA', 'University_GPA', 'SAT_Score', 'Starting_Salary']
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
Corr_data=data.drop(["zipcode","view"],axis=1) #togliamo variabili categoriche
C=Corr_data.corr()
print(f"La dimensione della matrice di Correlazione è: {C.shape}")
sns.heatmap(C, annot=False)
#%%
#rappresentiamo la matrice di correlazione per indagare le relazione statistiche
Corr_data=data.select_dtypes(include=num_type) #togliamo variabili categoriche
C=Corr_data.corr()
print(f"La dimensione della matrice di Correlazione è: {C.shape}")
sns.heatmap(C, annot=False)