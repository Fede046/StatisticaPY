#https://www.kaggle.com/datasets/anubhabswain/brain-weight-in-humans

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

#C:\Users\user\.cache\kagglehub\datasets
# Download latest version
path = kagglehub.dataset_download("anubhabswain/brain-weight-in-humans")

# Load the dataset using the correct filename
data = pd.read_csv(os.path.join(path, "dataset.csv"))

# Display the first few rows to verify it loaded correctly
#print(data.head())

#%%

#Dimensioni del dataset
N, d = data.shape
#print(f"Il Dataset ha {N} righe e {d} colonne.")

#%%
#Stampiamo un po' di informazioni sul dataset
#data.info()

#%%
# =============================================
print('PULIZIA DATASET')
# =============================================
#Impostiamo variabili categoriche quelle che non sono
num_type=['float64','int64']

for col in data.columns:
   # print(f"{col} type: {data[col].dtype}.")
    if data[col].dtype not in num_type:
        data[col]=data[col].astype("category")
  #      print(f"{col} type: {data[col].dtype}.")
 #   print("-"*45)
data.info()
# Cerchiamo i valori NaN
total_NaN = data.isnull().sum()
#print("Valori NaN prima della pulizia:")
#print(total_NaN)

# Rimuovi righe con NaN (sovrascrivi 'data')
data = data.dropna()  # Oppure: data.dropna(inplace=True)


# Verifica i NaN dopo la pulizia
#print("\nValori NaN dopo la pulizia:")
#print(data.isnull().sum())


# Verifica le nuove dimensioni
#print(f"\nNuove dimensioni: {data.shape[0]} righe, {data.shape[1]} colonne")


#%%
#Creiamo un subdataframe con le sole variabili numeriche
num_type=['float64','int64']
numerical_col=data.select_dtypes(include=num_type)
#tolgo la variabile Age Range e Gender perchè variabile categorica
numerical_col=numerical_col.drop(columns=['Gender','Age Range'])
#print(numerical_col.head())
#numerical_col.info()
#numerical_col.head()

#%%
#Vediamo se ci sono dei dati molto diversi da altri dati ed eventualemente togliere (outliers)

for col in numerical_col:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=numerical_col[col], color='skyblue')
    # Calcola e mostra i valori statistici
    valor = numerical_col[col].describe()
    plt.title(f'Boxplot di {col}\n'
              f'Media: {valor["mean"]:.2f} | '
              f'Mediana: {valor["50%"]:.2f}\n'
              f'Min: {valor["min"]:.2f} | '
              f'Max: {valor["max"]:.2f}')

    # Aggiungi linea per la media
    plt.axvline(valor["mean"], color='red', linestyle='--', label='Media')
    plt.axvline(valor["50%"], color='green', linestyle='-', label='Mediana')

    plt.legend()
    plt.tight_layout()
    plt.show()

    # Stampa dettagli sugli outlier (opzionale)
    Q1 = valor["25%"]
    Q3 = valor["75%"]
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = numerical_col[(numerical_col[col] < lower_bound) | (numerical_col[col] > upper_bound)]
    #print(f"Numero di outlier in {col}: {len(outliers)}")
    #print("=" * 50)

#%%
# =============================================
print('Regressione Lineare con Train/Test Split')
# =============================================
np.random.seed(42)
# Variabili indipendente e dipendente
x = numerical_col['Head Size(cm^3)'].values.reshape(-1, 1)
y = numerical_col['Brain Weight(grams)'].values.reshape(-1, 1)

# Split train/test (70% train, 30% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Creo modello e fitto sul training set
reg = LinearRegression().fit(x_train, y_train)

# Previsioni sul test set
y_pred_test = reg.predict(x_test)

# Grafico: dati di training e retta
plt.scatter(x_train, y_train, color='blue', label='Training data')
plt.scatter(x_test, y_test, color='green', label='Test data')
plt.plot(x_test, y_pred_test, color='red', label='Regression line (Test)')
plt.xlabel('Head Size(cm^3)')
plt.ylabel('Brain Weight(grams)')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()
#%%
# =============================================
print("RISULTATI REGRESSIONE LINEARE")
# =============================================

print(f"Intercetta (B0): {reg.intercept_[0]:.2f}")
print(f"Pendenza (B1): {reg.coef_[0][0]:.4f}")

# Coefficiente R^2 sul test set
r2 = r2_score(y_test, y_pred_test)
print(f"Coefficient of determination (R^2) - Test set: {r2:.2f}")

# MSE sul test set
mse = mean_squared_error(y_test, y_pred_test)
print(f"MSE - Test set: {mse:.2f}")
#%%
# =============================================
print("Analisi di Normalità dei Residui (Test set)")
# =============================================

# Residui sul test set
residuals_test = y_test.flatten() - y_pred_test.flatten()

# Istogramma dei residui
plt.figure(figsize=(10, 6))
sns.histplot(residuals_test, kde=True, bins=20)
plt.title('Distribuzione dei Residui (Test set)')
plt.xlabel('Residui')
plt.ylabel('Frequenza')
plt.show()

# QQ-plot dei residui
stats.probplot(residuals_test, plot=plt)
plt.xlabel("Quantili teorici")
plt.ylabel("Residui (Test set)")
plt.title('Q-Q Plot dei Residui (Test set)')
plt.show()

# Test di Shapiro-Wilk sui residui del test set
shapiro_test = stats.shapiro(residuals_test)
print("\n" + "="*50)
print("TEST DI NORMALITÀ DEI RESIDUI (Test set)")
print("="*50)
print(f"Shapiro-Wilk p-value: {shapiro_test.pvalue:.4f}")
if shapiro_test.pvalue > 0.05:
    print("I residui seguono una distribuzione normale (non rifiutiamo H0)")
else:
    print("I residui NON seguono una distribuzione normale (rifiutiamo H0)")

