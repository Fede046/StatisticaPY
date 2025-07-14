#https://www.kaggle.com/datasets/thedevastator/hotel-bookings-analysis

import kagglehub
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats

# =============================================
print('CARICARE DATASET')
# =============================================
# Download latest version
#C:\Users\user\.cache\kagglehub\datasets
path = kagglehub.dataset_download("thedevastator/hotel-bookings-analysis")

#print("Path to dataset files:", path)


# Load the dataset using the correct filename
data = pd.read_csv(os.path.join(path, "hotel_bookings.csv"))

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
print('PRE-PROCESSING')
# =============================================
# =============================================
print('VARIABILI CATEGORICHE')
# =============================================
#Impostiamo variabili categoriche quelle che non sono
num_type=['float64','int64']

for col in data.columns:
    if data[col].dtype not in num_type:
        data[col]=data[col].astype("category")

# Controllo il risultato
#data.info()
# =============================================
print('RIMOZIONE COLONNE NON NECESSARIE E VALORI NAN')
# =============================================
#Le colonne agent e company hanno troppi NAN decido di toglierle
#come variabili temporali
data = data.drop(columns=['required_car_parking_spaces','is_repeated_guest','arrival_date_day_of_month','arrival_date_week_number','index','arrival_date_year','agent', 'company','previous_cancellations',
                          'previous_bookings_not_canceled','booking_changes','days_in_waiting_list'])


# Cerchiamo i valori NaN
total_NaN = data.isnull().sum()
#print("Valori NaN prima della pulizia:")
#print(total_NaN)

# Rimuovi righe con NaN (sovrascrivi 'data')
data = data.dropna()

# Verifica i NaN dopo la pulizia
print("\nValori NaN dopo la pulizia:")
print(data.isnull().sum())

# =============================================
print('RIMOZIONE VALORI NEGATIVI')
# =============================================
# Identifica tutte le colonne numeriche
col_num = data.select_dtypes(include=num_type).columns

# Filtra il dataset mantenendo solo righe con valori positivi in tutte le colonne numeriche
mask = (data[col_num] >= 0).all(axis=1)
data = data[mask]

# Verifica che non ci siano più valori negativi
print("\nControllo valori negativi dopo la pulizia:")
for col in col_num:
    neg_count = (data[col] < 0).sum()
    print(f"{col}: {neg_count} valori negativi")

# =============================================
print('TOLGO VALORI FUORI SOGLIA')
# =============================================

#tolgo un dato che è troppo diverso dagli altri nel caso dell'adr
data = data[data['adr'] < 5000]
data = data[data['adults'] > 0]
data = data[data['adults'] < 5]
data = data[data['children'] < 4]
data = data[data['babies'] < 4]
data = data[data['stays_in_week_nights'] < 24]
#print((data['stays_in_week_nights']>24).sum())


# Verifica le nuove dimensioni
print(f"\nNuove dimensioni: {data.shape[0]} righe, {data.shape[1]} colonne")

#%%
#ELENCO DELLE COLONNE RIMASTE DALLA PULIZIA
# Variabile target - Indica se la prenotazione è stata cancellata (1) o no (0)
#'is_canceled',

# Numero di giorni tra la data di prenotazione e la data di arrivo
#'lead_time',

# Average Daily Rate - Prezzo medio giornaliero della camera
#'adr',

# Numero di notti di soggiorno durante il weekend (Sabato/Domenica)
#'stays_in_weekend_nights',

# Numero di notti di soggiorno durante i giorni lavorativi (Lunedì-Venerdì)
#'stays_in_week_nights',

# Numero di adulti nella prenotazione
#'adults',

# Numero di bambini nella prenotazione
#'children',

# Numero di neonati nella prenotazione
#'babies',

# Numero di richieste speciali (es. lettino per neonato, camere vicine)
#'total_of_special_requests'


#%%
# =============================================
print('EDA (PREPARAZIONE)')
# =============================================
#Creiamo un subdataframe con le sole variabili numeriche
num_type=['float64','int64']
numerical_col=data.select_dtypes(include=num_type)
print("Colonne disponibili:", numerical_col.columns.tolist())
#print(numerical_col.head())
#numerical_col.info()

#%%

# =============================================
print('ANALISI UNIVARIATA (BOX PLOT, DIAGRAMMI A TORTA, ISTOGRAMMI)')
# =============================================

# =============================================
print('BOX PLOT')
# =============================================

for col in numerical_col:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=numerical_col[col], color='skyblue')
 # Calcolo e mostro i valori statistici
    valor = numerical_col[col].describe()
    plt.title(f'Boxplot di {col}\n'
             f'Media: {valor["mean"]:.2f} | '
             f'Mediana: {valor["50%"]:.2f}\n'
             f'Min: {valor["min"]:.2f} | '
             f'Max: {valor["max"]:.2f}')

    # Aggiungo linea per la media e mediana
    plt.axvline(valor["mean"], color='red', linestyle='--', label='Media')
    plt.axvline(valor["50%"], color='green', linestyle='-', label='Mediana')

    plt.legend()
    plt.tight_layout()
    plt.show()

    # Stampo dettagli sugli outlier
    Q1 = valor["25%"]
    Q3 = valor["75%"]
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = numerical_col[(numerical_col[col] < lower_bound) | (numerical_col[col] > upper_bound)]
    print(f"Numero di outlier in {col}: {len(outliers)}")
    print("="*50)


# =============================================
print('DIAGRAMMI A TORTA')
# =============================================
#Diagramma a Torta solo per la variabile target
colonne_torta = ['is_canceled','adults', 'children',
                'babies']
fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 1 riga, 4 colonne

for i, col in enumerate(colonne_torta):
    temp = numerical_col[col].value_counts().sort_index()

    # Percentuali per la legenda
    sizes = temp.values
    labels = [str(int(x)) for x in temp.index]
    percentages = (sizes / sizes.sum()) * 100
    legend_labels = [f'{label}: {perc:.1f}%' for label, perc in zip(labels, percentages)]

    # Pie chart
    wedges, texts = axes[i].pie(sizes, labels=None, startangle=90)

    # Legenda accanto a ogni torta
    axes[i].legend(wedges, legend_labels, title=col, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # Titolo singolo per ogni sottoplot
    axes[i].set_title(f'{col}', fontsize=14)

plt.suptitle('Diagrammi a torta', fontsize=16)
plt.tight_layout()
plt.show()
# =============================================
print('ISTOGRAMMI')
# =============================================
#Istogramma per le altre variabili
colonne_isto = ['adr', 'lead_time', 'stays_in_weekend_nights',
                'stays_in_week_nights', 'total_of_special_requests']

fig, axes = plt.subplots(1, len(colonne_isto), figsize=(18, 5))
axes = axes.ravel()

for i, var in enumerate(colonne_isto):
    sns.histplot(data=numerical_col, x=var, ax=axes[i], bins=30,
                 color='lightgreen', edgecolor='black', alpha=0.7)

    # Calcolo media e mediana
    mean_val = numerical_col[var].mean()
    median_val = numerical_col[var].median()

    # Aggiungo linea per la media e mediana
    axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_val:.1f}')
    axes[i].axvline(median_val, color='blue', linestyle='-', linewidth=2, label=f'Mediana: {median_val:.1f}')

    # Etichette e titolo
    axes[i].set_title(f'{var}', fontsize=12)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Frequenza')

    #Legenda
    axes[i].legend()


plt.suptitle('Istogrammi', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
#%%
# =============================================
print('Analisi multivariata (MATRICE DI CORRELAZIONE)')
# =============================================
#
#Correlazione tra le variabili
C=numerical_col.corr()
print(f"La dimensione della matrice di Correlazione è: {C.shape}")


# Configuro il plot
plt.figure(figsize=(12, 10))

# Creo la heatmap con annotazioni
sns.heatmap(C, annot=True, cbar=True, cmap='coolwarm', fmt='.2f')

# Miglioro la leggibilità
plt.title('Matrice di correlazione', pad=20, fontsize=16)
plt.show()


#%%
# =============================================
print('Analisi bivariata (SCATTER PLOT)')
# =============================================
#
#Scatter plot
#Analizzo se prenotazioni con anticipo (lead_time) hanno prezzi più bassi/alti.
plt.figure(figsize=(12, 6))
plt.scatter(numerical_col['lead_time'], numerical_col['adr'], alpha=0.5, color='red')
plt.title('Relazione tra lead_time e adr')
plt.xlabel('lead_time')
plt.ylabel('adr')
plt.show()


#%%
# =============================================
print('Classificazione')
# =============================================
# =============================================
print('PREPARAZIONE DATASET PER LA CLASSIFICAZIONE (size->less_data)')
# =============================================
np.random.seed(66)


#Creo un dataset con meno campioni perchè faccio fatica a compilare

# Campionamento stratificato -> eventualmente aumentare
size = 0.05 #circa 5924 dati
#less_data = numerical_col.head(500)
less_data = numerical_col.groupby('is_canceled', group_keys=False).apply(
    lambda x: x.sample(frac=size, random_state=100),
    include_groups=False
)
# Aggiungo manualmente la colonna di raggruppamento
less_data['is_canceled'] = numerical_col.loc[less_data.index, 'is_canceled']

print("Colonne disponibili:", less_data.columns.tolist())

# Verifica
print(f"Numero totale di righe nel dataset ridotto: {len(less_data)}")
print("Distribuzione originale:\n", numerical_col['is_canceled'].value_counts(normalize=True))
print("\nDistribuzione campione:\n", less_data['is_canceled'].value_counts(normalize=True))



# 1. Preparazione target binario
y = less_data['is_canceled']

# 2. Selezione features (basata sulla tua analisi EDA)
X=(less_data.drop(columns=['is_canceled'])).values

#%%
# =============================================
print('4-5-6. Splitting Addestramento e Valutazione delle performance')
# =============================================


# Ricerca della migliore proporzione
best_accuracy = 0
best_size = None
results = {}

# Prova diverse dimensioni per train e validation
for train_size in [0.6, 0.7, 0.8]:  # 60%, 70%, 80% train
    remaining_size = 1 - train_size
    val_size = remaining_size * 0.5 #rispetto al rimanente
    test_size = remaining_size - val_size

    accuracies = []

# Loop su diversi random state per mediare le performance e ridurre la varianza dovuta allo split casuale
    for random_state in [0, 42, 100, 200]:
        # Es: Split 70% train, 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            train_size=train_size,
            random_state=random_state
        )

        # Es: Split del 30% in validation e test
        # Es: se val_size=0.15, test_size=0.15 (70-15-15)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=test_size / (test_size + val_size),  # Calcolo proporzione
            random_state=random_state
        )

        # Addestramento e valutazione
        model = SVC(kernel="linear", C=10, random_state=random_state)
        model.fit(X_train, y_train)
        # valutazione dell'accuratezza sul validation set
        val_accuracy = model.score(X_val, y_val)
        # print(f"Random State: {random_state}, Accuracy: {accuracy:.4f}")
        accuracies.append(val_accuracy)

    # Calcolo media
    mean_acc = np.mean(accuracies)
    results[(train_size, val_size, test_size)] = mean_acc

    print(f"\nTrain: {train_size:.0%} | Val: {val_size:.2%} | Test: {test_size:.2%}")
    print(f"Mean accuracy: {mean_acc:.4f}")

    # Aggiorna la migliore configurazione
    if mean_acc > best_accuracy:
        best_accuracy = mean_acc
        best_size = (train_size, val_size, test_size)

# Risultati finali
print("\n" + "=" * 50)
print(f"Miglior configurazione: Train={best_size[0]:.0%}, Val={best_size[1]:.2%}, Test={best_size[2]:.2%}")
print(f"Best mean accuracy: {best_accuracy:.4f}")

#%%
# =============================================
print('7. HYPERPARAMETER TUNING')
# =============================================

# Configurazioni da testare
configurations = [
    {'kernel': 'linear', 'C': 10},
    {'kernel': 'poly', 'C': 10, 'degree': 2},
    {'kernel': 'poly', 'C': 10, 'degree': 3},
    {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'}
]

# Variabili per trovare la configurazione ottimale
best_config = None
best_accuracy = 0

# Test per ogni configurazione
for config in configurations:
    print(f"\n\nTesting configuration: {config}")
    val_accuracies_config = []

    #Dichiaro il modello
    model = SVC(**config)

    for random_state in [0, 42, 100, 200]:
        #############################
        # Split dei dati (70% train, 30% temp)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            train_size=best_size[0],
            random_state=random_state
        )

        # Split ulteriore (con la proporzione ottimale)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=best_size[2] / (best_size[1] + best_size[2]),
            random_state=random_state
        )

        # Addestramento modello
        model = SVC(**config)
        model.fit(X_train, y_train)


        # Predizione sulla validation set
        y_val_pred = model.predict(X_val)
        # Valutazione sul VALIDATION set (come richiesto)
        acc = accuracy_score(y_val, y_val_pred)
        val_accuracies_config.append(acc)

        print(f"Random State {random_state}: Accuracy = {acc:.4f}")


    # Statistiche aggregate
    mean_accuracy = np.mean(val_accuracies_config)

    print(f"\nConfiguration {config['kernel']} (degree {config.get('degree', 'N/A')}):")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")



    # Aggiorna la migliore configurazione
    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_config = config

print(f"\nConfigurazione ottimale: {best_config}")
print(f"Accuracy media: {best_accuracy:.4f}")


#%%
# =============================================
print('8. STUDIO STATISTICO SUI RISULTATI (SOLO ACCURATEZZA) (K)')
# =============================================


# Configurazione
k = 20  # Numero di ripetizioni >= 10 come richiesto
val_accuracies_statistical_study = []

# 1. Ripetizione addestramento e valutazione
for i in range(k):
    # Split dei dati (70% train, 30% temp)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        train_size=best_size[0],
        random_state=i
    )

    # Split ulteriore (col best size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=best_size[2] / (best_size[1] + best_size[2]),
        random_state=i
    )
    # Addestramento modello
    model = SVC(**best_config)
    model.fit(X_train, y_train)

    # Valutazione sul VALIDATION set
    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    val_accuracies_statistical_study.append(acc)

    #Print per vedere l'accuratezza di ogni singolo State (serve per capire se va avanti)
    print(f"Random State {i}: Accuracy = {acc:.4f}")

# Converti in array numpy per calcoli statistici
val_accuracies_statistical_study = np.array(val_accuracies_statistical_study)

# 2. Statistica descrittiva
print("\n=== ANALISI STATISTICA ===")
print(f"Campioni (k): {k}")
print(f"Media accuratezze: {np.mean(val_accuracies_statistical_study):.4f}")
print(f"Deviazione standard: {np.std(val_accuracies_statistical_study):.4f}")
print(f"Minimo: {np.min(val_accuracies_statistical_study):.4f}")
print(f"Massimo: {np.max(val_accuracies_statistical_study):.4f}")
print(f"Mediana: {np.median(val_accuracies_statistical_study):.4f}")
#aggiungere identifica ottimale

#Identificazione configurazione ottimale
best_index = np.argmax(val_accuracies_statistical_study)
best_random_state = best_index
best_accuracy = val_accuracies_statistical_study[best_index]

print("\n=== MIGLIORE CONFIGURAZIONE ===")
print(f"Miglior random_state: {best_random_state}")
print(f"Accuratezza corrispondente: {best_accuracy:.4f}")

# 3. Visualizzazione (Istogramma + Boxplot)
plt.figure(figsize=(12, 5))

# Istogramma
plt.subplot(1, 2, 1)
plt.hist(val_accuracies_statistical_study, bins=10, color='skyblue', edgecolor='black')
plt.title('Distribuzione Accuratezze')
plt.xlabel('Accuracy')
plt.ylabel('Frequenza')
plt.axvline(np.mean(val_accuracies_statistical_study), color='red', linestyle='--', label=f'Media: {np.mean(val_accuracies_statistical_study):.4f}')
plt.legend()

# Boxplot
plt.subplot(1, 2, 2)
plt.boxplot(val_accuracies_statistical_study, vert=False)
plt.title('Boxplot Accuratezze')
plt.xlabel('Accuracy')
plt.tight_layout()
plt.show()

# 4. Inferenza statistica (Intervallo di confidenza)

confidence = 0.95
ci = stats.t.interval(confidence, k-1,
                     loc=np.mean(val_accuracies_statistical_study),
                     scale=stats.sem(val_accuracies_statistical_study))

print("\n=== INFERENZA STATISTICA ===")
print(f"Intervallo di confidenza al {confidence*100}%:")
print(f"({ci[0]:.4f}, {ci[1]:.4f})")




#%%
# =============================================
print('Valutazione finale sul test set')
# =============================================

# Addestramento finale con la migliore configurazione
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    train_size=best_size[0],  # Usa il miglior train_size trovato
    random_state=best_random_state
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=best_size[2] / (best_size[1] + best_size[2]),  # Proporzione ottimale
    random_state=best_random_state
)

# Addestramento modello
model = SVC(**best_config)
model.fit(X_train, y_train)


# =============================================
print('6. Creazione dell heatmap della matrice di confusione')
# =============================================

#Prevediamo i dati e valutiamo le performance della previsione

# Misuriamo l'accuratezza del modello
y_pred_test = model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred_test)

#Accuratezza sul test set
test_accuracy = model.score(X_test, y_test)
print(f"\nAccuracy sul test set: {test_accuracy:.4f}")

# Creazione dell'heatmap della matrice di confusione
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Oranges', cbar=False)
plt.title('Matrice di Confusione - Con kernel Ottimale')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()






