# -*- coding: utf-8 -*-
#Lezione 4: Exploratory Data Analysis (EDA)
#EDA: Definizione
 ”In statistics, exploratory data analysis (EDA) is an approach of analyzing data sets to
 summarize their main characteristics, often using statistical graphics and other data
 visualization methods. [...] focuses more narrowly on checking assumptions required for
 model fitting and hypothesis testing, and handling missing values and making
 transformations of variables as needed”.
 #Gestione di un progetto EDA
 Un progetto EDA si articola tipicamente in alcuni step fissati:
 Scelta del dataset: tramite motori di ricerca come Kaggle o Google Datasets.
 Esplorazione del dataset: osservare alcuni dati in esso presenti, interpretare le
 informazioni a disposizione, pianificare lo studio che si vuole svolgere.
 Preparazione dei dati (data cleaning): fondere pi` u datasets (se presenti) per
 incrementare le informazioni disponibili, aggiustare i tipi di dato (Date, Numeri,
 Stringhe), standardizzare i valori numerici, gestire i NaN.
 Indagine statistica: Utilizzare metodi statistici per estrarre informazioni rilevanti dai dati
 a disposizione.
 Visualizzazione (strettamente collegata con la precedente): Visualizzare attraverso vari
 tipi di grafici i risultati del punto precedente.
 # Esistono motori di ricerca come Kaggle (www.kaggle.com) e Google Datasets
 (https://datasetsearch.research.google.com) in cui ` e possibile trovare, tramite
 ricerca con parola chiave) una gran quantit` a di datasets pubblici.
 Kaggle ` e di gran lunga il pi` u utilizzato, possiede centinaia di migliaia di datasets, alcuni
 dei quali ben documentati.
 Nel seguito andremo ad utilizzare principalemente due datasets di esempio:
 Orange Quality Analysis Dataset: https:
 //www.kaggle.com/datasets/shruthiiiee/orange-quality?resource=download.
 House Sales in King County, USA:
 https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
 
 #usabilità è estramamente importante,
 #cercatene uno che abbia una usabilità buona, 
 #vedere come sono fatti i dataset
 
 #%%
 
 Datasets
 Da qui in avanti, consideriamo di avere a disposizione un dataset (che indichiamo con X).
 Un dataset ` e una tabella di valori, in cui le colonne rappresentano le features, mentre le
 righe rappresentano le differenti osservazioni. Nel seguito indichiamo con N il numero di
 righe di X, mentre con d indichiamo il numero di colonne. Dal punto di vista
 matematico, quindi, un dataset ` e una matrice di dimensione N × d.
 Abbiamo gi` a osservato che i dataset sono gestiti in Python tramite le funzioni della
 libreria pandas, i cui oggetti (DataFrame) possono essere caricati con la funzione:
 > data = pd.read_csv("PATH_TO_CSV.csv")
 La libreria seaborn, simile a matplotlib, ` e molto comoda per visualizzare dati da
 DataFrame di pandas.
 
 #%%
 #con data.info stampa le informazioni del dataset
 
#per utilizzare un dataset utilizzare bisogna usare 
#il file csv.

#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set the working directory
path = r"C:\\Users\\malse\\Desktop\\Secondo Anno Secondo Periodo\\Statistica\\StatisticaPY\\dataframes-20250313\\dataframes-20250320"
os.chdir(path)

# Load the dataset
try:
    data = pd.read_csv(os.path.join(path, "Orange Quality Data.csv"))
except FileNotFoundError:
    print("File not found. Please check the file path and name.")
    exit()

# Display the dimensions of the dataset
N, d = data.shape
print(f"Il Dataset ha {N} righe e {d} colonne.")

# Display some information about the dataset
print("\nInformazioni sul dataset:")
data.info()

# Display the first and last 5 rows of the dataset
print("\nPrime 5 righe del dataset:")
print(data.head())

print("\nUltime 5 righe del dataset:")
print(data.tail())
#%%
pd.set_option('display.max_columns',None)
data.describe()

#%%
#Prendiamo in considerazione i dati del file order details.csv, fornito su Virtuale.
 Carichiamolo in memoria:
 > data = pd.read_csv("./Orange Quality Data.csv"").
 Visualizzandone alcuni elementi con la funzione:
 > print(data.head())
 osserviamo che alcune colonne possiedono valori numerici, mentre altre sono stringhe.
 Fare particolarmente attenzione quando si lavora con dati che non sono numerici!
 Ricordarsi di visualizzare il numero di righe e di features del dataset.
 > N, d = data.shape
 > print(f"Shape of data: {N, d}.")
 [1] Shape of data: (241, 11).
 
 #%%
 Descrizione dei dati
 E’ possibile ottenere maggiori informazioni sulle features del dataset di riferimento tramite
 il comando data.info().
 Similmente, con il comando data.describe() ` e possibile accedere rapidamente ad
 informazioni statistiche sul Dataset.
 Possiamo vedere che il dataframe ha 10 colonne (features):
 Dimensione (cm), Peso (g), Brix (Dolcezza), pH (Acidit` a), Morbidezza (1-5),
 Tempo di raccolta (giorni), Maturazione (1-5), Colore, Variet`a, Imperfezioni
 (S` ı/No), Qualit` a (1-5)
 
 #%%
 E’ spesso una buona idea costruirsi un sub-dataframe del dataset principale contenente
solo le variabili numeriche di data. Su questo sub-dataframe sar` a possibile eseguire le
indagini statistiche tipiche dell’EDA.
# Extract numeric values
> numeric_data = data.select_dtypes(include=’number’)
> print(numeric_data.head())
Size (cm) Weight (g) ... Ripeness (1-5) Quality (1-5)
0
7.5
1
2
3
4
8.2
6.8
9.0
8.5
[5 rows x 8 columns]

#%%
# Data Cleaning
#  La preparazione dei dati, o Data Cleaning, ` e la fase pi` u lunga e delicata in un progetto
#  EDA, e dal suo corretto svolgimento dipende non solo l’analisi statistica successiva, ma
#  anche un eventuale addestramento di modelli di predizione.
#  In particolare, ` e di fondamentale importanza imparare a gestire i dati mancanti (indicati
#  come Not A Number (NaN)), come vedremo in seguito.
 #%%
 Notiamo che, da Kaggle, possiamo avere accesso a datasets contenenti informazioni
differenti. Tuttavia, se i due dataset sono collegati da una colonna comune (come Order
ID), questa ci permette di connettere i due datasets.
Su pandas, i due datasets possono essere uniti mediante la funzione pd.merge(), nella
quale ` e necessario specificare il nome della colonna da utilizzare per effettuare la
connessione tra i dataset.
L’operazione con merge serve per unire i dataset attraverso la colonna scelta e ordinarli.
[ref]
> # Merge the first and the second dataset
> data2 = pd.read_csv("PATH2_TO_CSV.csv")
> data = pd.merge(data, data2, on="Order ID")
Dopo il merging, possiamo eliminare la colonna Order ID, ora inutile, tramite il comando
data = data.drop("Order ID").
 #%% Si pu` o controllare la tipologia di dato di una colonna di un DataFrame di pandas con il
 comando data.dtype:
 > print(data["Amount"].dtype)
 [1] int64
 > print(data["Category"].dtype)
 [1] object
 
 #%%
 #Creiamo un subdataframe con le sole variabili numeriche
 num_type=['float64','int64']
 numerical_col=data.select_dtypes(include=num_type)
 print(numerical_col.head())

#%%
 # Le variabili numeriche possono essere utilizzate per svolgere operazioni matematiche (per
 # esempio, su variabili numeriche ` e ben posta la definizione di correlazione).
 # Le varabili categoriche contengono valori non numerici, possono essere usate per
 # classificare valori, ma non possono essere utilizzate per funzioni matematiche.
 # Nota: il dtype object NON indica variabili categoriche per pandas

#%%
Data anlisis esplorativa variabili numeriche
Per altro possimoa usare anche variabili categoriche


 #%%
 #Impostiamo variabili categoriche quelle che non sono
 for col in data.columns:
     print(f"{col} type: {data[col].dtype}.")
     if data[col].dtype not in num_type:
         data[col]=data[col].astype("category")
         print(f"{col} type: {data[col].dtype}.")
     print("-"*45)
#possiamo creare più sottclassi e divodere il dataset
 #%%
 #Cerchiamo i valori NaN
 #data.isnull() resitutisce True per le celle che contengono NaN
#possimo vedere per ogni colonna quanti nan ci sono, li cancella anche
#In questo caso non ci sono valori nan
#ci potrebbero essere delle procedure per sistituire il nan, per
#esempio la media,
 total_NaN=data.isnull().sum()
 print(total_NaN)
 data.dropna()
 #%%
 consiglio se abbiamo tanti dati si posso togliere i dati senza troppi probklemi
#%%

 # Spesso i dataset contengono uno (o pi` u) elementi mancanti, causati per esempio da
 # misurazioni mancate o rimosse poich´ e irrealistiche, o per via di errori di vario tipo
 # collegati alla collezione dei dati.
 # Abbiamo gi` a visto che i dati mancanti vengono mostrati da pandas come dei valori NaN
 # (Not A Number). Per evitare problemi algoritmici, ` e necessario rimuovere o sostituire i
 # dati mancanti.
 # Per controllare la presenza, colonna per colonna, di valori NaN, si pu` o usare il seguente
 # comando:
 # > # Check the presence of NaN values
 # > print(data.isnull().sum())
 # Il nostro dataset non ne contiene nessuno.
 # Quando i valori NaN appaiono in un numero relativamente basso di righe, la cosa pi` u
 # semplice da fare ` e rimuovere tutte le righe che li contengono. Questo pu` o essere fatto
 # semplicemente con la funzione data.dropna()

#%%
# Un metodo molto pi` u efficiente di quello appena descritto, ma molto pi` u complesso da
#  utilizzare, ` e quello di sostituire i dati mancanti.
#  L’opzione pi` u semplice ` e quella di utilizzare la funzione data.fillna(VALORE), che
#  sostiuisce tutti i valori NaN con il valore inserito. La stessa funzione si pu` o utilizzare per
#  sostituire i NaN anche, ad esempio, con il valore del dato precedente.
#  Esistono tecniche pi` u avanzate per sostiuire i valori NaN, utilizzando informazioni presenti
#  nel dataset per prevedere un valore realistico da inserire al posto del NaN

 #%%Indagine Statistica
#  La covarianza mi dicono dove si muovo le varibili aleatorie
#  Un’operazione comune praticamente a tutti i progetti di EDA, ` e la computazione della
# Matrice di Correlazione di Pearson.
# Si basa sulla definizione di correlazione tra varabili aleatorie, definita da:
# ρX,Y := Corr(X,Y) = Cov(X,Y)
# Var(X)Var(Y) ,
# dove:
# Cov(X,Y) := E (X −E[X])(Y −E[Y])T .
# Si definisce la matrice di correlazione di Pearson C ∈ Rd×d dove:
# Ci,j = ρXi,Xj
# .

# #Ogni entrata è data dalla covarianza
#   Si osserva che:
#  −1 ≤Ci,j ≤ 1, ∀i,j = 1,...,d.
#  Un valore positivo di Ci,j indica una correlazione positiva tra Xi e Xj. Un valore negativo
#  di Ci,j indica una correlazione negativa tra Xi e Xj.
#  Ci,j = 1 significa che Xi e Xj sono correlate deterministicamente.
#  Nota: Ci,i = 1 per ogni i = 1,...,d.
 
 #%%
 #Creiamo un subdataframe con le sole variabili numeriche
 num_type=['float64','int64']
 numerical_col=data.select_dtypes(include=num_type)
 print(numerical_col.head())
 #%%
 
 #metodi statistici per la distribuaizone dei dati
 #Boxplot delle colonne numeriche in matplolib
 fig, ax = plt.subplots (nrows=2, ncols=4)

 for j in range(len(numerical_col.columns)):
     ax[j//4, j%4].hist(numerical_col.iloc[:,j], bins=20, color='orange')
     ax[j//4, j%4].set_title(numerical_col.columns[j])
     
 plt.tight_layout() 
 plt.show()
 #%%
 #Boxplot delle colonne con curva di densità
 #utilizziamo seaborn al posto di math plot lib
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
 #utilizzato per pochi dati non per tanti dati diversi
 #altrimenti diventa poco leggibile
 for i, col in enumerate(colonne):
     temp = numerical_col[col].value_counts()  # Conta le occorrenze
     ax[i//2, i%2].pie(temp.values, labels=temp.index, autopct='%1.1f%%', startangle=90)
     ax[i//2, i%2].set_title(f'Distribuzione di {col}')  # Titolo del grafico

 # Aggiusta il layout per evitare sovrapposizioni
 plt.tight_layout()
 plt.show()

 #%%
 for i, col in enumerate(colonne):
     temp = numerical_col[col].value_counts()
     plt.figure(figsize=(8,6))
     plt.pie(temp.values, labels=temp.index, autopct='%1.1f%%', startangle=90)
     plt.title(f'Distribuzione di {col}')
     plt.show()
 #%%
 #outliers, con il concetto di boxplot, 
 for colonna in numerical_col.columns:
     plt.figure(figsize=(8, 6))
     sns.boxplot(numerical_col[colonna])
     plt.title(f'Boxplot di {colonna}')
     plt.show()
  #%%
 # Distribuzioni condizionate
 # Oltre alle informazioni ottenibili dalla matrice di correlazione, e alle statistiche sul dataset
 # visibili tramite la funzione data.describe(), informazioni interessanti possono essere
 # ottenute condizionando su una delle features del dataset (ovvero, filtrando quegli elementi
 # che rispettano una data condizione).
 # Ad esempio, ` e possibile calcolare la media del qualit` a delle arance del dataset, e
 # confrontarla con la media della qualit` a delle arance condizionata al fatto che
 # pH> pHmedio.
 # Analizzando le statistiche delle distribuzioni condizionate rispetto a quelle non
 # condizionate, si possono fare interessanti osservazioni sui dati!
 #%%
 #Calcoliamo la media condizionata della qualità in base al pH e sweet
 #tutti nnumerical col mi da il valore di tutti i valori numerici
 #relitivi alla quantità e mi da la media
 #np.mean mi da il phmedio, è una cosa che posso fare con tutte
 #le fetur per un determinato valore. 
 media_quality=np.mean(numerical_col['Quality (1-5)'])
 media_ph=np.mean(numerical_col['pH (Acidity)'])
 media_sweet=np.mean(numerical_col['Brix (Sweetness)'])
 #creiamo un sottodataframe, in cui preso da numerical col in 
 #dove il ph è maggiore di quello medio.
 #
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
 
 #Un esempio di statistica bivariata è l'utilizzo
 #dello scatter plot, potrebbe essere più o meno utile,
 #in base a categorie, e misure con una scala molto alta
 #in questo caso è poco esaustivo perchè il dataest non lo 
 #fa notare bene nel caso ddelle case è meglio
 plt.scatter(data["pH (Acidity)"], data["Quality (1-5)"], alpha=0.5, color='red')
 plt.title("Scatter plot tra pH e Qualità")
 plt.xlabel('pH')
 plt.ylabel('Quality')
 plt.show()

 #%%
 #Standardizzare la colorazione
 C=numerical_col.corr()
 print(f"La dimensione della matrice di Correlazione è: {C.shape}")
 sns.heatmap(C, annot=True, cbar=True, cmap='coolwarm', fmt='.2f')


