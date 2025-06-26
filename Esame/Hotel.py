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


# Download latest version
#C:\Users\malse\.cache\kagglehub\datasets
path = kagglehub.dataset_download("thedevastator/hotel-bookings-analysis")

#print("Path to dataset files:", path)


# Load the dataset using the correct filename
data = pd.read_csv(os.path.join(path, "hotel_bookings.csv"))

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
 #   print(f"{col} type: {data[col].dtype}.")
    if data[col].dtype not in num_type:
        data[col]=data[col].astype("category")
  #      print(f"{col} type: {data[col].dtype}.")
  #  print("-"*45)
# Check the results
#data.info()
#%%
#Le colonne agent e company hanno troppi NAN decido di toglierle
data = data.drop(columns=['agent', 'company','previous_cancellations',
                          'previous_bookings_not_canceled','booking_changes','days_in_waiting_list'])  # Opzionale


# Cerchiamo i valori NaN
total_NaN = data.isnull().sum()
#print("Valori NaN prima della pulizia:")
#print(total_NaN)

# Rimuovi righe con NaN (sovrascrivi 'data')
data = data.dropna()  # Oppure: data.dropna(inplace=True)

# Filtra adr > 0
data = data[data['adr'] > 0]
#tolgo un dato che è troppo diverso dagli altri nel caso dell'adr
data = data[data['adr'] < 5000]
# Verifica i NaN dopo la pulizia
#print("\nValori NaN dopo la pulizia:")
#print(data.isnull().sum())

# Verifica le nuove dimensioni
#print(f"\nNuove dimensioni: {data.shape[0]} righe, {data.shape[1]} colonne")

#%%
#Creiamo un subdataframe con le sole variabili numeriche
num_type=['float64','int64']
numerical_col=data.select_dtypes(include=num_type)
#print(numerical_col.head())
#numerical_col.info()
#Questa parte del codice serve a isolare le variabili numeriche dal 
#dataset completo per poter effettuare analisi specifiche che richiedono dati quantitativi.
#%%
#Analisi univariata
#Eventualemente togliere box plot e metterici istogrammi o grafici a torta
hotel_numerical_features = ['adr', 'lead_time', 'stays_in_weekend_nights', 
                           'stays_in_week_nights', 'adults', 'children', 
                           'babies', 'total_of_special_requests']

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
   # print(f"Numero di outlier in {col}: {len(outliers)}")
    #print("="*50)
    
    
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
    sns.histplot(data=numerical_col, x=var, kde=True, color='green', ax=axes[i], bins=30)

    # Aggiungi linee per media e mediana
    mean_val = numerical_col[var].mean()
    median_val = numerical_col[var].median()

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


Corr_data = numerical_col[correlation_columns]
C=Corr_data.corr()
print(f"La dimensione della matrice di Correlazione è: {C.shape}")

#sns.heatmap(C, annot=False)

#Aggiunto da GPT eventualemnte togliere
# Configura il plot
plt.figure(figsize=(12, 10))

# Crea la heatmap con annotazioni
heatmap = sns.heatmap(
    C, 
    annot=True,            # Mostra i valori
    fmt=".2f",            # Formato a 2 decimali
    cmap='coolwarm',      # Mappa di colori
    vmin=-1, vmax=1,      # Range valori
    center=0,             # Centro della mappa
    square=True,          # Celle quadrate
    linewidths=.5,        # Linee tra le celle
    cbar_kws={"shrink": .8},  # Dimensione barra colori
    annot_kws={"size": 10}    # Dimensione testo annotazioni
)

# Migliora la leggibilità
plt.title('Matrice di correlazione', pad=20, fontsize=16)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right')
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.show()


#%%
#Analisi bivariata
#scatter plot 
#Esplora se prenotazioni con anticipo (lead_time) hanno prezzi più bassi/alti.
plt.figure(figsize=(12, 6))
plt.scatter(numerical_col['lead_time'], numerical_col['adr'], alpha=0.5, color='red')
plt.title('Relazione tra lead_time e adr')
plt.xlabel('lead_time')
plt.ylabel('adr')

#in presenza di bambini si aumenta il prezzo (adr)
plt.figure(figsize=(12, 6))
plt.scatter(numerical_col['children'], numerical_col['adr'], alpha=0.5, color='blue')
plt.title('Relazione tra children e adr')
plt.xlabel('children')
plt.ylabel('adr')


#%%
#Distribuzioni condizionate
# Calcolo delle medie globali
media_cancellazioni = np.mean(numerical_col['is_canceled'])
media_lead_time = np.mean(numerical_col['lead_time'])
media_tot = np.mean(numerical_col['total_of_special_requests'])

# Sub-dataframe condizionati
filtered_lead_time = numerical_col[numerical_col['lead_time'] > media_lead_time]  # Prenotazioni con molto anticipo
filtered_tot = numerical_col[numerical_col['total_of_special_requests'] > media_tot]                    # Prenotazioni costose

# Media condizionata delle cancellazioni
media_canc_lead_time = np.mean(filtered_lead_time['is_canceled'])
media_canc_tot = np.mean(filtered_tot['is_canceled'])

print(f"Tasso globale di cancellazioni: {media_cancellazioni:.2%}\n")
print(f"Tasso cancellazioni per lead_time > {media_lead_time:.0f} giorni: {media_canc_lead_time:.2%}")
print(f"Tasso cancellazioni per total_of_special_requests > {media_tot:.2f} richieste: {media_canc_tot:.2%}")
# Visualizzazione distribuzioni condizionate
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

sns.histplot(filtered_lead_time['total_of_special_requests'], bins=20, kde=True, color='skyblue')
plt.title('Distribuzione delle richieste speciali (total_of_special_requests)\nper prenotazioni con molto anticipo')
plt.xlabel('Richieste speciali')

plt.subplot(1, 2, 2)
sns.histplot(filtered_tot['lead_time'], bins=20, kde=True, color='salmon')
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


#3.Cancellazioni per molte richieste speciali (>0.57 richieste) - 22.01%
#Paradosso apparente: Meno richieste speciali = più cancellazioni (l'opposto di quanto ci si aspetterebbe)
#Possibili spiegazioni:
#Clienti con richieste specifiche (es. camere per disabili, letti aggiuntivi) hanno maggiore necessità e quindi minore propensione a cancellare
#Le richieste speciali potrebbero indicare viaggi "importanti" (matrimoni, eventi) meno cancellabili
#Implicazioni pratiche:
#Le politiche che limitano le richieste speciali potrebbero aumentare le cancellazioni
#Valorizzare le richieste speciali come strumento di fidelizzazione

# CONCLUSIONE:
# Le cancellazioni sono un problema consistente per gli hotel.
# L'anticipo della prenotazione è un fattore molto più rilevante del prezzo nel predire cancellazioni.
# Servono politiche mirate soprattutto per le prenotazioni con lead_time elevato.


#%%

#Classificazione


#Creo un dataset con meno campioni perchè faccio fatica a compilare
# Campionamento stratificato -> eventualmente aumentare
size = 0.001
less_data = data.groupby('is_canceled', group_keys=False).apply(
    lambda x: x.sample(frac=size, random_state=100),
    include_groups=False
)
# Aggiungi manualmente la colonna di raggruppamento
less_data['is_canceled'] = data.loc[less_data.index, 'is_canceled']

print("Colonne disponibili:", less_data.columns.tolist())

# Verifica
print("Distribuzione originale:\n", data['is_canceled'].value_counts(normalize=True))
print("\nDistribuzione campione:\n", less_data['is_canceled'].value_counts(normalize=True))

#Creiamo un subdataframe con le sole variabili numeriche
numerical_Col_Clf=less_data.select_dtypes(include=num_type)

# 1. Preparazione target binario 
y = less_data['is_canceled']

# 2. Selezione features (basata sulla tua analisi EDA)
X=(numerical_Col_Clf.drop(columns=['is_canceled'])).values

#Dividiamo il modello in train, test, validation 
#sto presendendo il 70% per il traning e 30% per il test, random state è il modo in cui vengono divisi i dati
#train (adddestra) valuation(capire se i parametri che ha ottenuto durante l'addestramento sono buoni o no) 
#test(test finale)
#utilizziamo il parametro train size al posto di trest size
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.5, random_state=100)

print(f"Le dimensioni di (y_val, y_test) sono {y_val.shape[0], y_test.shape[0]}")

#%%

#Definiamo un modello
#C è un iperparametro, controlla il trade-off tra margine ed errori di classificazione.
k='linear'
#Se funziona da documentare bene perchè  class_weight='balanced' l'ho preso da gpt
model=SVC(kernel=k, C=10, class_weight='balanced')

#%%
#Addestriamo il modello sul training set
#stiamo dando al modello delle coppie, in base alle variabili in output sono 0 1
model.fit(X_train, y_train)
#%%

#Valutazione della Performance

#Misuriamo l'accuratezza del modello
#Nella variabile in input mettiamo solo x val, e predice l'output basandosi 
#dalle variabili x val, 
y_pred=model.predict(X_val)
conf_mat = confusion_matrix(y_val, y_pred)
 
# Calcolare l'accuratezza sulla validation set
accuracy_val = accuracy_score(y_val, y_pred)
print(f"Accuracy sul validation set: {accuracy_val:.4f}")


#%%
# Creazione dell'heatmap della matrice di confusione
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Oranges', cbar=False)
plt.title(f'Confusion Matrix - Support Vector Machines with {k} kernel')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#%%
#notiamo che se utilizziamo un random state diverso abbiamo una accuratezza diversa
#cambiamo 4 volte il seme generatre, genero 4 diversi train test e test set, per ogniuno di questi faccio trning e testing
#model. score fa predizione cambio delle metriche e il calcolo dell'acccuratezza
#per vederel la ribustezza el modello devo provare diverse inseimi di train set e test set
#la robustezza la cpisco in base alle metriche, se la devizione standrd è molto alta tra le 
#accuratezza escono il modello non è robusto
#invece se ottengo sempre il risultato
#class_weight='balanced' aggiunto da gpt da verificarePerché class_weight='balanced'?
#Documentazione ufficiale: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#Funzionamento: Assegna pesi automaticamente in modo inversamente proporzionale alle frequenze delle classi. Nel tuo caso:
#Peso classe 0 (Non cancellato): ~1/0.62 ≈ 1.61
#Peso classe 1 (Cancellato): ~1/0.38 ≈ 2.63

for random_state in [0, 42, 100, 200]:
    #ho tolto class_weight='balanced' perchè mi dava errore
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,random_state=random_state)
    model = SVC(kernel="linear", C=10)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Random State: {random_state}, Accuracy: {accuracy:.4f}")


#%%
#7.Hyperparameter Tuning
# Configurazioni da testare
configurations = [
    {'kernel': 'linear', 'C': 10},
    {'kernel': 'poly', 'C': 10, 'degree': 2},
    {'kernel': 'poly', 'C': 10, 'degree': 3},
    {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'}
]

# Scaling dei dati (essenziale per SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Variabili per trovare la configurazione ottimale
best_config = None
best_accuracy = 0

# Test per ogni configurazione
for config in configurations:
    print(f"\n\nTesting configuration: {config}")
    accuracies = []

    for random_state in [0, 42, 100, 200]:
        # Split dei dati
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            train_size=0.7,
            random_state=random_state,
            stratify=y  # Mantiene il bilanciamento delle classi
        )

        # Creazione e addestramento modello
        model = SVC(**config, random_state=100)
        model.fit(X_train, y_train)

        # Valutazione
        # Calcolare l'accuratezza sulla validation set (da valutare se corretto gpt dice)(X_test, y_test)
        accuracy_val = accuracy_score(y_val, y_pred)
        accuracies.append(accuracy_val)
        print(f"Random State {random_state}: Accuracy = {accuracy_val:.4f}")

    # Statistiche aggregate
    mean_accuracy = np.mean(accuracies)

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
# 8. STUDIO STATISTICO SUI RISULTATI (SOLO ACCURATEZZA)
# =============================================
#(non penso si corretta la parte del cilo for 70 15 15 non me lo ricordavo)(molto lento con k=30)
# Configurazione
np.random.seed(100)  # Fisso il seed per riproducibilità
k = 11  # Numero di ripetizioni >= 10 come richiesto
accuracies = []

# 1. Ripetizione addestramento e valutazione
for i in range(k):
    # Split dei dati (70% train, 30% temp)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,
        random_state=i,
        stratify=y
    )

    # Split ulteriore (15% validation, 15% test)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=i,
        stratify=y_temp
    )

    # Addestramento modello
    model = SVC(kernel='linear', C=10, class_weight='balanced', random_state=100)
    model.fit(X_train, y_train)

    # Valutazione sul VALIDATION set (come richiesto)
    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    accuracies.append(acc)

    #Print per vedere l'accuratezza di ogni singolo State (serve per capire se va avanti)
    print(f"Random State {i}: Accuracy = {acc:.4f}")

# Converti in array numpy per calcoli statistici
accuracies = np.array(accuracies)

# 2. Statistica descrittiva
print("\n=== ANALISI STATISTICA ===")
print(f"Campioni (k): {k}")
print(f"Media accuratezze: {np.mean(accuracies):.4f}")
print(f"Deviazione standard: {np.std(accuracies):.4f}")
print(f"Minimo: {np.min(accuracies):.4f}")
print(f"Massimo: {np.max(accuracies):.4f}")
print(f"Mediana: {np.median(accuracies):.4f}")

# 3. Visualizzazione (Istogramma + Boxplot)
plt.figure(figsize=(12, 5))

# Istogramma
plt.subplot(1, 2, 1)
plt.hist(accuracies, bins=10, color='skyblue', edgecolor='black')
plt.title('Distribuzione Accuratezze')
plt.xlabel('Accuracy')
plt.ylabel('Frequenza')
plt.axvline(np.mean(accuracies), color='red', linestyle='--', label=f'Media: {np.mean(accuracies):.4f}')
plt.legend()

# Boxplot
plt.subplot(1, 2, 2)
plt.boxplot(accuracies, vert=False)
plt.title('Boxplot Accuratezze')
plt.xlabel('Accuracy')
plt.tight_layout()
plt.show()

# 4. Inferenza statistica (Intervallo di confidenza)

confidence = 0.95
ci = stats.t.interval(confidence, k-1,
                     loc=np.mean(accuracies),
                     scale=stats.sem(accuracies))

print("\n=== INFERENZA STATISTICA ===")
print(f"Intervallo di confidenza al {confidence*100}%:")
print(f"({ci[0]:.4f}, {ci[1]:.4f})")



#%%





