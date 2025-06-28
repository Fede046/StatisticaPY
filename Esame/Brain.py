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

#C:\Users\malse\.cache\kagglehub\datasets
# Download latest version
path = kagglehub.dataset_download("anubhabswain/brain-weight-in-humans")
#print("Path to dataset files:", path)
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
#Impostiamo variabili categoriche quelle che non sono
num_type=['float64','int64']

for col in data.columns:
   # print(f"{col} type: {data[col].dtype}.")
    if data[col].dtype not in num_type:
        data[col]=data[col].astype("category")
  #      print(f"{col} type: {data[col].dtype}.")
 #   print("-"*45)
# Check the results
#data.info()
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
#Questa parte del codice serve a isolare le variabili numeriche dal
#dataset completo per poter effettuare analisi specifiche che richiedono dati quantitativi.
#numerical_col.head()

#%%
#Vediamo se ci sono dei dati molto diversi da altri dati ed eventualemente togliere
hotel_numerical_features = ['Head Size(cm^3)','Brain Weight(grams)']

for col in hotel_numerical_features:
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
#Regressione Lineare

# Extract input and output variables
x = numerical_col['Head Size(cm^3)'].values.reshape(-1, 1)  # X = variabile indipendente
y = numerical_col['Brain Weight(grams)'].values.reshape(-1, 1)  # Y = variabile dipendente
# Create linear regression object and fit the model
reg = LinearRegression().fit(x, y)

# Predict the y-values using the trained model
y_pred = reg.predict(x)

# Plot the data points and the linear regression line
plt.scatter(x, y, color='blue')
plt.plot(x, y_pred, color='red')

# Add labels and a title to the plot
plt.xlabel('Brain Weight(grams)')
plt.ylabel('Head Size(cm^3)')
plt.title('Simple Linear Regression')
# Display the plot
plt.show()
#Stampo il coefficiente R^2
print("Coefficient of determination (R^2): %.2f" % r2_score(y, y_pred))
#%%
#Tutti i print necessari

#Stampa dei Coefficienti (B0 e B1)
print("RISULTATI REGRESSIONE LINEARE")
print(f"Intercetta (B0): {reg.intercept_[0]:.2f}")
print(f"Pendenza (B1): {reg.coef_[0][0]:.4f}")

#Calcolo del MSE (Mean Squared Error)
mse = mean_squared_error(y, y_pred)
print(f"MSE: {mse:.2f}")
#%%
#Analisi di Normalità dei Residui


# Calcolo dei residui
#Come da slide sulla regressione lineare semplice
#residuals = y - y_pred
# Modello OLS per la tua analisi preso da example_residuals.py
results = smf.ols('Q("Brain Weight(grams)") ~ Q("Head Size(cm^3)")', data=numerical_col).fit()


# Istogramma dei residui
plt.figure(figsize=(10, 6))
sns.histplot(results.resid, kde=True, bins=20)
plt.title('Distribuzione dei Residui')
plt.xlabel('Residui')
plt.ylabel('Frequenza')
plt.show()


#Usata dalla prof
# QQ-plot dei residui (viene male) (retta verticale)
#plt.figure(figsize=(10, 6))
#sm.qqplot(results.resid.values, line='45')
#plt.title('QQ-plot dei Residui')
#plt.show()

# QQ-plot dei residui
stats.probplot(results.resid,plot=plt)
plt.xlabel("Quantili teorici")
plt.ylabel("Residui")
plt.title('Q-Q Plot dei residui')
plt.show()

# 4. Test di Shapiro-Wilk per la normalità
shapiro_test = stats.shapiro(results.resid)
print("\n" + "="*50)
print("TEST DI NORMALITÀ DEI RESIDUI")
print("="*50)
print(f"Shapiro-Wilk p-value: {shapiro_test[1]:.4f}")
if shapiro_test[1] > 0.05:
    print("I residui seguono una distribuzione normale (non rifiutiamo H0)")
else:
    print("I residui NON seguono una distribuzione normale (rifiutiamo H0)")

#%%
# INTERPRETAZIONE FINALE DEI RISULTATI

# 1. Bontà del Modello (R² = 0.64)
# Il modello spiega il 64% della variabilità del peso del cervello in base alla dimensione della testa.
# Si tratta di un valore relativamente buono, soprattutto in ambiti come le scienze biologiche o mediche,
# dove la variabilità individuale è spesso elevata.
# Limite: rimane comunque un 36% di variabilità non spiegata, che potrebbe essere attribuita a variabili non incluse nel modello
# (come età, sesso, caratteristiche genetiche, ecc.).

# 2. Coefficienti di Regressione
# Intercetta (325.57 grammi):
# - Rappresenta il valore teorico del peso cerebrale quando la dimensione della testa è zero.
# - In questo contesto biologico l'interpretazione non è significativa, ma è necessaria per definire la retta di regressione.

# Pendenza (0.2634 grammi/cm³):
# - Indica una relazione positiva tra volume cranico e peso cerebrale.
# - Ogni cm³ in più di volume cranico è associato in media a un aumento di circa 0.26 grammi nel peso del cervello.
# - Questa relazione è plausibile biologicamente: volumi cranici maggiori tendono ad ospitare cervelli più pesanti.

# 3. Errore (MSE = 5201.38)
# - L'errore quadratico medio indica quanto in media si discostano le previsioni dai valori reali.
# - Un MSE relativamente alto segnala che, pur essendo presente una relazione lineare, le previsioni non sono perfette.
# - Questo è coerente con il fatto che il modello spiega solo il 64% della variabilità.

# 4. Problema di Normalità dei Residui (Shapiro-Wilk p-value = 0.0236)
# - Il p-value inferiore a 0.05 suggerisce che i residui **non** seguono una distribuzione normale.
# - Conseguenze:
#   * Le stime dei coefficienti (pendenza e intercetta) sono comunque valide: la regressione lineare è abbastanza robusta.
#   * Tuttavia, gli intervalli di confidenza e i test di significatività (p-value sui coefficienti) potrebbero essere meno affidabili.
#   * Questo potrebbe influire leggermente sulla precisione delle conclusioni statistiche.

# Conclusione generale:
# Il modello è utile e ha una interpretazione coerente con la realtà biologica, ma può essere migliorato includendo altre variabili predittive
# e/o utilizzando modelli più complessi se necessario.
