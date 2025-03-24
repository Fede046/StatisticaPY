
# Lezione 4: Exploratory Data Analysis (EDA)

# EDA: Definizione
"""
In statistica, l'analisi esplorativa dei dati (EDA) è un approccio per analizzare i dataset
al fine di riassumere le loro principali caratteristiche, spesso utilizzando grafici statistici
e altri metodi di visualizzazione dei dati. L'EDA si concentra principalmente sul controllo
delle ipotesi necessarie per il fitting dei modelli e il test delle ipotesi, sulla gestione
dei valori mancanti e sulla trasformazione delle variabili se necessario.
"""

# Gestione di un progetto EDA
"""
Un progetto EDA si articola tipicamente in alcuni step fissati:
1. Scelta del dataset: tramite motori di ricerca come Kaggle o Google Datasets.
2. Esplorazione del dataset: osservare alcuni dati presenti, interpretare le informazioni
   a disposizione, pianificare lo studio che si vuole svolgere.
3. Preparazione dei dati (data cleaning): fondere più datasets (se presenti) per incrementare
   le informazioni disponibili, aggiustare i tipi di dato (Date, Numeri, Stringhe), standardizzare
   i valori numerici, gestire i NaN.
4. Indagine statistica: Utilizzare metodi statistici per estrarre informazioni rilevanti dai dati
   a disposizione.
5. Visualizzazione (strettamente collegata con la precedente): Visualizzare attraverso vari tipi
   di grafici i risultati del punto precedente.
"""

# Esistono motori di ricerca come Kaggle (www.kaggle.com) e Google Datasets
# (https://datasetsearch.research.google.com) in cui è possibile trovare, tramite
# ricerca con parola chiave, una gran quantità di datasets pubblici.
"""
Kaggle è di gran lunga il più utilizzato, possiede centinaia di migliaia di datasets, alcuni
dei quali ben documentati.
Nel seguito andremo ad utilizzare principalmente due datasets di esempio:
1. Orange Quality Analysis Dataset: https://www.kaggle.com/datasets/shruthiiiee/orange-quality?resource=download
2. House Sales in King County, USA: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
"""

# Usabilità è estremamente importante,
# cercatene uno che abbia una usabilità buona,
# vedere come sono fatti i dataset

#%%

# Datasets
"""
Da qui in avanti, consideriamo di avere a disposizione un dataset (che indichiamo con X).
Un dataset è una tabella di valori, in cui le colonne rappresentano le features, mentre le
righe rappresentano le differenti osservazioni. Nel seguito indichiamo con N il numero di
righe di X, mentre con d indichiamo il numero di colonne. Dal punto di vista matematico,
quindi, un dataset è una matrice di dimensione N × d.
Abbiamo già osservato che i dataset sono gestiti in Python tramite le funzioni della
libreria pandas, i cui oggetti (DataFrame) possono essere caricati con la funzione:
> data = pd.read_csv("PATH_TO_CSV.csv")
La libreria seaborn, simile a matplotlib, è molto comoda per visualizzare dati da
DataFrame di pandas.
"""


#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Imposta la directory di lavoro
# path = r"C:\\Users\\malse\\Desktop\\Secondo Anno Secondo Periodo\\Statistica\\StatisticaPY\\dataframes-20250313\\dataframes-20250320"
path = r"C:\Users\malse\source\repos\StatisticaPY\dataframes-20250313\dataframes-20250320"
# os.chdir(path)

os.chdir(path)

# Carica il dataset
try:
    data = pd.read_csv(os.path.join(path, "Orange Quality Data.csv"))
except FileNotFoundError:
    print("File non trovato. Controlla il percorso e il nome del file.")
    exit()

# Mostra le dimensioni del dataset
N, d = data.shape
print(f"Il Dataset ha {N} righe e {d} colonne.")

# Mostra alcune informazioni sul dataset
print("\nInformazioni sul dataset:")
data.info()

# Mostra le prime e le ultime 5 righe del dataset
print("\nPrime 5 righe del dataset:")
print(data.head())

print("\nUltime 5 righe del dataset:")
print(data.tail())

#%%
# Imposta l'opzione per visualizzare tutte le colonne
pd.set_option('display.max_columns', None)

# Mostra una descrizione statistica del dataset
data.describe()

#%%
# Prendiamo in considerazione i dati del file order details.csv, fornito su Virtuale.
# Carichiamolo in memoria:
# > data = pd.read_csv("./Orange Quality Data.csv").
# Visualizzandone alcuni elementi con la funzione:
# > print(data.head())
# osserviamo che alcune colonne possiedono valori numerici, mentre altre sono stringhe.
# Fare particolarmente attenzione quando si lavora con dati che non sono numerici!
# Ricordarsi di visualizzare il numero di righe e di features del dataset.
# > N, d = data.shape
# > print(f"Shape of data: {N, d}.")
# [1] Shape of data: (241, 11).

#%%
# Descrizione dei dati
# E’ possibile ottenere maggiori informazioni sulle features del dataset di riferimento tramite
# il comando data.info().
# Similmente, con il comando data.describe() è possibile accedere rapidamente ad
# informazioni statistiche sul Dataset.
# Possiamo vedere che il dataframe ha 10 colonne (features):
# Dimensione (cm), Peso (g), Brix (Dolcezza), pH (Acidità), Morbidezza (1-5),
# Tempo di raccolta (giorni), Maturazione (1-5), Colore, Varietà, Imperfezioni
# (Sì/No), Qualità (1-5)

#%%
# E’ spesso una buona idea costruirsi un sub-dataframe del dataset principale contenente
# solo le variabili numeriche di data. Su questo sub-dataframe sarà possibile eseguire le
# indagini statistiche tipiche dell’EDA.
# Extract numeric values
numeric_data = data.select_dtypes(include='number')
print(numeric_data.head())

#%%
# Data Cleaning
# La preparazione dei dati, o Data Cleaning, è la fase più lunga e delicata in un progetto
# EDA, e dal suo corretto svolgimento dipende non solo l’analisi statistica successiva, ma
# anche un eventuale addestramento di modelli di predizione.
# In particolare, è di fondamentale importanza imparare a gestire i dati mancanti (indicati
# come Not A Number (NaN)), come vedremo in seguito.

#%%
# Notiamo che, da Kaggle, possiamo avere accesso a datasets contenenti informazioni
# differenti. Tuttavia, se i due dataset sono collegati da una colonna comune (come Order
# ID), questa ci permette di connettere i due datasets.
# Su pandas, i due datasets possono essere uniti mediante la funzione pd.merge(), nella
# quale è necessario specificare il nome della colonna da utilizzare per effettuare la
# connessione tra i dataset.
# L’operazione con merge serve per unire i dataset attraverso la colonna scelta e ordinarli.
# [ref]
# > # Merge the first and the second dataset
# > data2 = pd.read_csv("PATH2_TO_CSV.csv")
# > data = pd.merge(data, data2, on="Order ID")
# Dopo il merging, possiamo eliminare la colonna Order ID, ora inutile, tramite il comando
# data = data.drop("Order ID").

#%%
# Si può controllare la tipologia di dato di una colonna di un DataFrame di pandas con il
# comando data.dtype:
# > print(data["Amount"].dtype)
# [1] int64
# > print(data["Category"].dtype)
# [1] object

#%%
# Creiamo un subdataframe con le sole variabili numeriche
num_type = ['float64', 'int64']
numerical_col = data.select_dtypes(include=num_type)
print(numerical_col.head())

#%%
# Le variabili numeriche possono essere utilizzate per svolgere operazioni matematiche (per
# esempio, su variabili numeriche è ben posta la definizione di correlazione).
# Le variabili categoriche contengono valori non numerici, possono essere usate per
# classificare valori, ma non possono essere utilizzate per funzioni matematiche.
# Nota: il dtype object NON indica variabili categoriche per pandas

#%%
# Data analysis esplorativa variabili numeriche
# Per altro possiamo usare anche variabili categoriche

# Iteriamo attraverso tutte le colonne del DataFrame 'data'
for col in data.columns:
    # Stampiamo il nome della colonna e il suo tipo di dato attuale
    print(f"{col} type: {data[col].dtype}.")
    
    # Verifichiamo se il tipo di dato della colonna non è numerico (es. int, float)
    if data[col].dtype not in num_type:
        # Se il tipo di dato non è numerico, convertiamo la colonna in tipo 'category'
        # Questo è utile per ottimizzare la memoria e migliorare l'efficienza delle operazioni
        data[col] = data[col].astype("category")
        
        # Stampiamo il nuovo tipo di dato della colonna dopo la conversione
        print(f"{col} type: {data[col].dtype}.")
    
    # Stampiamo una linea di separazione per rendere più leggibile l'output
    print("-" * 45)

# Dopo aver convertito le colonne categoriche, possiamo creare più sottoclassi
# e dividere il dataset in base a queste categorie, se necessario.
# Questo può essere utile per analisi più approfondite o per la preparazione dei dati per modelli di machine learning.

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
# Consiglio: se abbiamo tanti dati si possono togliere i dati senza troppi problemi

#%%
# Spesso i dataset contengono uno (o più) elementi mancanti, causati per esempio da
# misurazioni mancate o rimosse poiché irrealistiche, o per via di errori di vario tipo
# collegati alla collezione dei dati.
# Abbiamo già visto che i dati mancanti vengono mostrati da pandas come dei valori NaN
# (Not A Number). Per evitare problemi algoritmici, è necessario rimuovere o sostituire i
# dati mancanti.
# Per controllare la presenza, colonna per colonna, di valori NaN, si può usare il seguente
# comando:
# > # Check the presence of NaN values
# > print(data.isnull().sum())
# Il nostro dataset non ne contiene nessuno.
# Quando i valori NaN appaiono in un numero relativamente basso di righe, la cosa più
# semplice da fare è rimuovere tutte le righe che li contengono. Questo può essere fatto
# semplicemente con la funzione data.dropna()

#%%
# Un metodo molto più efficiente di quello appena descritto, ma molto più complesso da
# utilizzare, è quello di sostituire i dati mancanti.
# L’opzione più semplice è quella di utilizzare la funzione data.fillna(VALORE), che
# sostituisce tutti i valori NaN con il valore inserito. La stessa funzione si può utilizzare per
# sostituire i NaN anche, ad esempio, con il valore del dato precedente.
# Esistono tecniche più avanzate per sostituire i valori NaN, utilizzando informazioni presenti
# nel dataset per prevedere un valore realistico da inserire al posto del NaN

#%%
# Indagine Statistica
# La covarianza mi dice dove si muovono le variabili aleatorie
# Un’operazione comune praticamente a tutti i progetti di EDA, è la computazione della
# Matrice di Correlazione di Pearson.
# Si basa sulla definizione di correlazione tra variabili aleatorie, definita da:
# ρX,Y := Corr(X,Y) = Cov(X,Y)
# Var(X)Var(Y) ,
# dove:
# Cov(X,Y) := E (X −E[X])(Y −E[Y])T .
# Si definisce la matrice di correlazione di Pearson C ∈ Rd×d dove:
# Ci,j = ρXi,Xj
# .

# Ogni entrata è data dalla covarianza
# Si osserva che:
# −1 ≤Ci,j ≤ 1, ∀i,j = 1,...,d.
# Un valore positivo di Ci,j indica una correlazione positiva tra Xi e Xj. Un valore negativo
# di Ci,j indica una correlazione negativa tra Xi e Xj.
# Ci,j = 1 significa che Xi e Xj sono correlate deterministicamente.
# Nota: Ci,i = 1 per ogni i = 1,...,d.

#%%
# Creiamo un subdataframe con le sole variabili numeriche
num_type = ['float64', 'int64']
numerical_col = data.select_dtypes(include=num_type)
print(numerical_col.head())

#%%
# Metodi statistici per la distribuzione dei dati
# Boxplot delle colonne numeriche in matplotlib

# Creiamo una figura con una griglia 2x4 per i grafici
fig, ax = plt.subplots(nrows=2, ncols=4)

# Ciclo per creare istogrammi per ogni colonna numerica
for j in range(len(numerical_col.columns)):
    # Creiamo un istogramma per la colonna corrente con 20 bin e colore arancione
    ax[j//4, j%4].hist(numerical_col.iloc[:,j], bins=20, color='orange')
    # Impostiamo il titolo del grafico con il nome della colonna
    ax[j//4, j%4].set_title(numerical_col.columns[j])
    
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
    sns.histplot(numerical_col.iloc[:,j], kde=True, bins=20, color='blue', ax=ax[j//2, j%2])
    
# Aggiustiamo il layout per evitare sovrapposizioni
plt.tight_layout() 
# Mostriamo i grafici
plt.show()

#%%
# Creiamo una figura con una griglia 2x2 per i grafici a torta
fig, ax = plt.subplots(2, 2, figsize=(10, 10))  

# Lista delle colonne da plottare
colonne = ['Softness (1-5)', 'Ripeness (1-5)', 'pH (Acidity)', 'Quality (1-5)']

# Ciclo per creare i grafici a torta
# Utilizzato per pochi dati non per tanti dati diversi
# Altrimenti diventa poco leggibile
for i, col in enumerate(colonne):
    # Contiamo le occorrenze dei valori nella colonna corrente
    temp = numerical_col[col].value_counts()  
    # Creiamo un grafico a torta con le percentuali e un angolo di partenza di 90 gradi
    ax[i//2, i%2].pie(temp.values, labels=temp.index, autopct='%1.1f%%', startangle=90)
    # Impostiamo il titolo del grafico
    ax[i//2, i%2].set_title(f'Distribuzione di {col}')  

# Aggiustiamo il layout per evitare sovrapposizioni
plt.tight_layout()
# Mostriamo i grafici
plt.show()

#%%
# Ciclo per creare grafici a torta separati per ogni colonna
for i, col in enumerate(colonne):
    # Contiamo le occorrenze dei valori nella colonna corrente
    temp = numerical_col[col].value_counts()
    # Creiamo una nuova figura per ogni grafico a torta
    plt.figure(figsize=(8,6))
    # Creiamo un grafico a torta con le percentuali e un angolo di partenza di 90 gradi
    plt.pie(temp.values, labels=temp.index, autopct='%1.1f%%', startangle=90)
    # Impostiamo il titolo del grafico
    plt.title(f'Distribuzione di {col}')
    # Mostriamo il grafico
    plt.show()

#%%
# Outliers, con il concetto di boxplot

# Ciclo per creare boxplot per ogni colonna numerica
for colonna in numerical_col.columns:
    # Creiamo una nuova figura per ogni boxplot
    plt.figure(figsize=(8, 6))
    # Creiamo un boxplot per la colonna corrente
    sns.boxplot(numerical_col[colonna])
    # Impostiamo il titolo del grafico
    plt.title(f'Boxplot di {colonna}')
    # Mostriamo il grafico
    plt.show()    

#%%
# Analisi delle distribuzioni condizionate
# Oltre alle informazioni ottenibili dalla matrice di correlazione e dalle statistiche descrittive del dataset
# (visibili tramite la funzione `data.describe()`), è possibile ottenere ulteriori informazioni interessanti
# condizionando l'analisi su una delle feature del dataset. 
# Condizionare significa filtrare i dati in base a una specifica condizione (ad esempio, considerare solo i dati
# in cui il pH è superiore al valore medio). Questo approccio permette di confrontare le statistiche delle
# distribuzioni condizionate con quelle non condizionate, fornendo insight utili sui dati.

#%%
# Calcolo delle medie condizionate della qualità in base al pH e alla dolcezza (Brix)
# Calcoliamo la media della qualità, del pH e della dolcezza (Brix) per l'intero dataset.
media_quality = np.mean(numerical_col['Quality (1-5)'])
media_ph = np.mean(numerical_col['pH (Acidity)'])
media_sweet = np.mean(numerical_col['Brix (Sweetness)'])

# Creiamo due sottodataframe:
# 1. `filtered_pH`: contiene solo le righe in cui il pH è maggiore del valore medio.
# 2. `filtered_sweet`: contiene solo le righe in cui la dolcezza (Brix) è maggiore del valore medio.
filtered_pH = numerical_col[numerical_col['pH (Acidity)'] > media_ph]
filtered_sweet = numerical_col[numerical_col['Brix (Sweetness)'] > media_sweet]

# Calcoliamo la media della qualità per i due sottodataframe filtrati.
media_quality_ph = np.mean(filtered_pH['Quality (1-5)'])
media_quality_sweet = np.mean(filtered_sweet['Quality (1-5)'])

# Stampiamo i risultati per confrontare le medie.
print(f"Qualità media: {media_quality:.2f}\n")
print(f"Qualità media per pH > pH medio: {media_quality_ph:.2f} \n")
print(f"Qualità media per dolcezza > dolcezza media: {media_quality_sweet:.2f} \n")

#%%
# Visualizzazione della distribuzione della dolcezza (Brix) condizionata al pH
# Utilizziamo un istogramma con curva di densità (KDE) per visualizzare la distribuzione della dolcezza
# nei casi in cui il pH è superiore al valore medio.
plt.figure()
sns.histplot(filtered_pH['Brix (Sweetness)'], bins=20, kde=True)
plt.title('Distribuzione della dolcezza condizionata al pH > pH medio')
plt.xlabel('Dolcezza (Brix)')
plt.ylabel('Frequenza')
plt.show()

#%%
# Analisi bivariata: Scatter plot tra pH e Qualità
# Uno scatter plot è un utile strumento per visualizzare la relazione tra due variabili continue.
# In questo caso, plottiamo il pH rispetto alla qualità per osservare eventuali pattern o correlazioni.
plt.scatter(data["pH (Acidity)"], data["Quality (1-5)"], alpha=0.5, color='red')
plt.title("Scatter plot tra pH e Qualità")
plt.xlabel('pH')
plt.ylabel('Qualità (1-5)')
plt.show()

#%%
# Matrice di correlazione e heatmap
# La matrice di correlazione è uno strumento utile per analizzare le relazioni lineari tra le variabili numeriche.
# Visualizziamo la matrice di correlazione utilizzando una heatmap, che permette di identificare rapidamente
# le correlazioni positive (colori caldi) e negative (colori freddi).
C = numerical_col.corr()
print(f"La dimensione della matrice di correlazione è: {C.shape}")
sns.heatmap(C, annot=True, cbar=True, cmap='coolwarm', fmt='.2f')
plt.title('Matrice di correlazione')
plt.show()