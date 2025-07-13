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
print('Regressione Lineare')
# =============================================
np.random.seed(6)

# Extract input and output variables
X = numerical_col['Head Size(cm^3)'].values.reshape(-1, 1)  # X = variabile indipendente
y = numerical_col['Brain Weight(grams)'].values.reshape(-1, 1)  # Y = variabile dipendente



# Split the data for train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=100)

# Create new axis for x column
X_train = X_train
X_train=X_train.reshape(-1,1)
X_test = X_test
X_test=X_test.reshape(-1,1)


# Fitting the model
reg = LinearRegression().fit(X_train,y_train)
#farlo su tutto il dataset non solo sul train,
#in questo caso lo facciamo su un dataset grande
#noi lo facciamo su dataset piccoli


# Predicting the Salary for the Training values
y_pred_train = reg.predict(X_train)
plt.scatter(X,y,color='black')
plt.plot(X_train,y_pred_train,color='blue')
plt.xlabel('Years of experience')
plt.ylabel('Salary (in thousand)')

# Predicting the Salary for the Test values
y_pred_test = reg.predict(X_test)


# Plotting the actual and predicted values

c = [i for i in range (1,len(y_test)+1,1)]
plt.plot(c,y_test,color='r',linestyle='-')
plt.plot(c,y_pred_test,color='b',linestyle='-')
plt.xlabel('Salary')
plt.ylabel('index')
plt.title('Prediction')
plt.show()

# Disegna scatterplot e retta di regressione
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_val, color='blue', label='Data')
    plt.xlabel(x_scelta)
    plt.ylabel(y_scelta)
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Retta di regressione')
    plt.title('Retta di regressione')
    plt.show()
#%%
#Tutti i print necessari
# =============================================
print("RISULTATI REGRESSIONE LINEARE")
# =============================================

#Stampa dei Coefficienti (B0 e B1)
print(f"Intercetta (B0): {reg.intercept_[0]:.2f}")
print(f"Pendenza (B1): {reg.coef_[0][0]:.4f}")

#Stampo il coefficiente R^2
print("Coefficient of determination (R^2): %.2f" % r2_score(y_test,y_pred_test))

#Calcolo del MSE (Mean Squared Error)
mse = mean_squared_error(y_test,y_pred_test)
print(f"MSE: {mse:.2f}")
#%%
#
# =============================================
print("Analisi di Normalità dei Residui")
# =============================================

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


# QQ-plot dei residui
#Con questa funzione creo un qqplot con dei residui standardizzati
#processo che ti permette di confrontare i residui su una scala comune,
# indipendentemente dalle unità originali
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







