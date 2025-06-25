#https://www.kaggle.com/datasets/thedevastator/hotel-bookings-analysis

import kagglehub
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


# Download latest version
#C:\Users\malse\.cache\kagglehub\datasets\adilshamim8\education-and-career-success\versions\1\education_career_success.csv
path = kagglehub.dataset_download("thedevastator/hotel-bookings-analysis")

print("Path to dataset files:", path)


# Load the dataset using the correct filename
data = pd.read_csv(os.path.join(path, "hotel_bookings.csv"))

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
#Creiamo un subdataframe con le sole variabili numeriche
num_type=['float64','int64']
numerical_col=data.select_dtypes(include=num_type)
print(numerical_col.head())
#Questa parte del codice serve a isolare le variabili numeriche dal 
#dataset completo per poter effettuare analisi specifiche che richiedono dati quantitativi.
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
#Le colonne agent e company hanno troppi NAN decido di toglierle
data = data.drop(columns=['agent', 'company','previous_cancellations',
                          'previous_bookings_not_canceled','booking_changes','days_in_waiting_list'])  # Opzionale


# Cerchiamo i valori NaN
total_NaN = data.isnull().sum()
print("Valori NaN prima della pulizia:")
print(total_NaN)

# Rimuovi righe con NaN (sovrascrivi 'data')
data = data.dropna()  # Oppure: data.dropna(inplace=True)

# Filtra adr > 0
data = data[data['adr'] > 0]
#tolgo un dato che è troppo diverso dagli altri nel caso dell'adr
data = data[data['adr'] < 5000]
# Verifica i NaN dopo la pulizia
print("\nValori NaN dopo la pulizia:")
print(data.isnull().sum())

# Verifica le nuove dimensioni
print(f"\nNuove dimensioni: {data.shape[0]} righe, {data.shape[1]} colonne")

#%%
#Analisi univariata
#Eventualemente togliere box plot e metterici istogrammi o grafici a torta
hotel_numerical_features = ['adr', 'lead_time', 'stays_in_weekend_nights', 
                           'stays_in_week_nights', 'adults', 'children', 
                           'babies', 'total_of_special_requests']

for col in hotel_numerical_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[col], color='skyblue')
 # Calcola e mostra i valori statistici
    stats = data[col].describe()
    plt.title(f'Boxplot di {col}\n'
             f'Media: {stats["mean"]:.2f} | '
             f'Mediana: {stats["50%"]:.2f}\n'
             f'Min: {stats["min"]:.2f} | '
             f'Max: {stats["max"]:.2f}')
    
    # Aggiungi linea per la media
    plt.axvline(stats["mean"], color='red', linestyle='--', label='Media')
    plt.axvline(stats["50%"], color='green', linestyle='-', label='Mediana')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Stampa dettagli sugli outlier (opzionale)
    Q1 = stats["25%"]
    Q3 = stats["75%"]
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
    print(f"Numero di outlier in {col}: {len(outliers)}")
    print("="*50)
    
    
#Diagrammi a torta 
#Dati Discreti
#Variabili che assumono valori interi e spesso contano occorrenze.
colonne_Torta = ['is_canceled','adults', 'children', 
'babies', 'total_of_special_requests', 'stays_in_weekend_nights','stays_in_week_nights']
for i, col in enumerate(colonne_Torta):
    temp = numerical_col[col].value_counts()
    plt.figure(figsize=(8,6))
    plt.pie(temp.values, labels=temp.index, autopct='%1.1f%%', startangle=90)
    plt.title(f'Distribuzione di {col}')
    plt.show()
    
    
#Istogramma con curva di densità per variabili continue
#Dati Continui
#Variabili che possono assumere qualsiasi valore in un intervallo (numeri reali).
colonne_Isto = ['adr', 'lead_time']

fig, axes = plt.subplots(1, 2, figsize=(15, 10))  

# Crea gli istogrammi
for i, var in enumerate(colonne_Isto):
    sns.histplot(data=data, x=var, kde=True, color='green', ax=axes[i], bins=30)
    
    # Aggiungi linee per media e mediana
    mean_val = data[var].mean()
    median_val = data[var].median()
    
    axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Media: {mean_val:.1f}')
    axes[i].axvline(median_val, color='green', linestyle='-', label=f'Mediana: {median_val:.1f}')
    
    # Formattazione
    axes[i].set_title(f'Distribuzione di {var}\n(Skewness: {data[var].skew():.2f})')
    axes[i].set_xlabel('Valore')
    axes[i].set_ylabel('Frequenza')
    axes[i].legend()
    
    
plt.tight_layout()  
plt.show()  
    
# Visualizzare la distribuzione in modo più fluido
#L'istogramma mostra la frequenza dei dati in "bin" discreti (barre), che dipendono dalla scelta del numero di intervalli.

#La curva KDE smussa queste barre, fornendo una stima continua della distribuzione, più facile da interpretare.
    
#%%
#Analisi multivariata
#Eventualmente sistemare colonne
#Eventualemente inserire numeri
#rappresentiamo la matrice di correlazione per indagare le relazione statistiche
#Corr_data=data.select_dtypes(include=num_type) #togliamo variabili categoriche

correlation_columns = [
    # Variabile target (dipendente) - Indica se la prenotazione è stata cancellata (1) o no (0)
    'is_canceled',         
    
    # Numero di giorni tra la data di prenotazione e la data di arrivo
    # (Potenziale predittore forte per cancellazioni)
    'lead_time',           
    
    # Average Daily Rate - Prezzo medio giornaliero della camera
    # (Indicatore del valore economico della prenotazione)
    'adr',                 
    
    # Numero di notti di soggiorno durante il weekend (Sabato/Domenica)
    # (Può influenzare cancellazioni e prezzi)
    'stays_in_weekend_nights',
    
    # Numero di notti di soggiorno durante i giorni lavorativi (Lunedì-Venerdì)
    'stays_in_week_nights',
    
    # Numero di adulti nella prenotazione
    # (Impatto su costi e complessità della prenotazione)
    'adults',
    
    # Numero di bambini nella prenotazione
    # (Potrebbe aumentare probabilità di cancellazione per imprevisti)
    'children',
    
    # Numero di neonati nella prenotazione
    # (Fattore di rischio per cancellazioni last-minute)
    'babies',
    
    
    # Numero di richieste speciali (es. lettino, camere vicine)
    # (Clienti più esigenti potrebbero avere diverso comportamento)
    'total_of_special_requests'
]


Corr_data = data[correlation_columns]
C=Corr_data.corr()
print(f"La dimensione della matrice di Correlazione è: {C.shape}")
sns.heatmap(C, annot=False)
#%%
#Analisi bivariata
#scatter plot 
#Esplora se prenotazioni con anticipo (lead_time) hanno prezzi più bassi/alti.
plt.figure(figsize=(12, 6))
plt.scatter(data['lead_time'], data['adr'], alpha=0.5, color='red')
plt.title('Relazione tra lead_time e adr')
plt.xlabel('lead_time')
plt.ylabel('adr')

#in presenza di bambini si aumenta il prezzo (adr)
plt.figure(figsize=(12, 6))
plt.scatter(data['children'], data['adr'], alpha=0.5, color='blue')
plt.title('Relazione tra children e adr')
plt.xlabel('children')
plt.ylabel('adr')


#%%
#Distribuzioni condizionate
# Calcolo delle medie globali
media_cancellazioni = np.mean(data['is_canceled'])
media_lead_time = np.mean(data['lead_time'])
media_adr = np.mean(data['adr'])

# Sub-dataframe condizionati
filtered_lead_time = data[data['lead_time'] > media_lead_time]  # Prenotazioni con molto anticipo
filtered_adr = data[data['adr'] > media_adr]                    # Prenotazioni costose

# Media condizionata delle cancellazioni
media_canc_lead_time = np.mean(filtered_lead_time['is_canceled'])
media_canc_adr = np.mean(filtered_adr['is_canceled'])

print(f"Tasso globale di cancellazioni: {media_cancellazioni:.2%}\n")
print(f"Tasso cancellazioni per lead_time > {media_lead_time:.0f} giorni: {media_canc_lead_time:.2%}")
print(f"Tasso cancellazioni per adr > {media_adr:.2f}€: {media_canc_adr:.2%}")

# Visualizzazione distribuzioni condizionate
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(filtered_lead_time['adr'], bins=20, kde=True, color='skyblue')
plt.title('Distribuzione del prezzo (ADR)\nper prenotazioni con molto anticipo')
plt.xlabel('Prezzo medio (ADR)')

plt.subplot(1, 2, 2)
sns.histplot(filtered_adr['lead_time'], bins=20, kde=True, color='salmon')
plt.title('Distribuzione dell\'anticipo (lead_time)\nper prenotazioni costose')
plt.xlabel('Giorni di anticipo')

plt.tight_layout()
plt.show()

# ANALISI 
# 1. Tasso Globale di Cancellazioni (37.58%)
# Più di una prenotazione su tre viene cancellata.
# Questo rappresenta un problema operativo serio per gli hotel, che possono perdere revenue
# e avere difficoltà nella pianificazione delle risorse (es. personale, camere libere).
# Strategie raccomandate:
# - Introdurre depositi non rimborsabili
# - Incentivare prenotazioni con clausole di cancellazione più rigide

# 2. Cancellazioni per Prenotazioni con Anticipo Elevato
# Quando il lead_time (giorni tra prenotazione e soggiorno) supera i 105 giorni,
# il tasso di cancellazione sale al 51.30%, cioè un +13.72% rispetto alla media globale.
# Questo significa che le prenotazioni fatte con largo anticipo (> 3 mesi)
# sono il 36.5% più soggette a cancellazione rispetto alla media.
# Possibili cause:
# - Cambi di programma nel tempo
# - Clienti che trovano offerte migliori nel frattempo
# - Minor senso di "impegno" verso la prenotazione
# Azioni consigliate:
# - Politiche più rigide per prenotazioni early-bird (es. penalità maggiori)
# - Sconti per prenotazioni non cancellabili (es. tariffe "non-refundable")

# 3. Cancellazioni per Prenotazioni Costose
# Quando il prezzo medio giornaliero (ADR) è superiore a 103.65€, il tasso di cancellazione
# è del 38.44%, cioè solo un +0.86% rispetto alla media globale (37.58%).
# Questo implica che il prezzo elevato NON è un fattore determinante nelle cancellazioni.
# I clienti disposti a pagare di più non cancellano significativamente di più rispetto ad altri.
# Implicazioni:
# - Le strategie di prezzo aggressivo (alta ADR) non aumentano in modo rilevante il rischio di cancellazione
# - È più efficace concentrarsi sull’anticipo della prenotazione piuttosto che sul prezzo

# CONCLUSIONE:
# Le cancellazioni sono un problema consistente per gli hotel.
# L'anticipo della prenotazione è un fattore molto più rilevante del prezzo nel predire cancellazioni.
# Servono politiche mirate soprattutto per le prenotazioni con lead_time elevato.




