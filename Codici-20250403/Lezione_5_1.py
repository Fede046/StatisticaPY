#classificazione binaria
import numpy as np
import os 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Scikit learn è la libreria che serve per gli algoritmi di machine learning
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

#%%
#Caricamento del dataset
path="C:\\Users\\malse\\Documents\\GitHub\\StatisticaPY\\dataframes-20250313\\dataframes-20250320"
data=pd.read_csv(os.path.join(path, "Orange Quality Data.csv"))

#%%
#Trasformiamo la colonna sulla qualità in colonna binaria 

mean_qual=np.mean(data['Quality (1-5)']) #soglia
classes = (data['Quality (1-5)']>mean_qual).astype(int)

#%%
num_type=['float64','int64']
numerical_col=data.select_dtypes(include=num_type)

#%%
#X dati da classificare e y classi tra cui classificare
#harvest time è una variabile categorica (quindi da togliere) 
#Quolity, è la nostra variabile di output non di input
#la prima variabile è un dataframe la seconda variabile è una serie
X=(numerical_col.drop(columns=['HarvestTime (days)', 'Quality (1-5)'])).values

#x rappresenta il sotto dataset che vogliamo raffigurare

y=classes

#%%
#Dividiamo il modello in train, test, validation 
#sto presendendo il 70% per il traning e 30% per il test, random state è il modo in cui vengono divisi i dati
#train (adddestra) valuation(capire se i parametri che ha ottenuto durante l'addestramento sono buoni o no) 
#test(test finale)
#utilizziamo il parametro train size al posto di trest size
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.5, random_state=100)

print(f"Le dimensioni di (y_val, y_test) sono {y_val.shape[0], y_test.shape[0]}")



#%%
#le classi sono ridotte da 5 a 2

# Ripartiamo in modo che la somma sia esatta
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=169, random_state=100)

# Dividiamo il rimanente in test (18) e validation (18)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, train_size=18, random_state=100)

#%%
#Definiamo il modello
#la support vector machine prende il dataset visualizzato in un piano, l'algotimo fa fatica e classifica i
# punti in due d e fa la divisione tra i vari dati
k='linear'
#C è un iperparametro
model=SVC(kernel=k, C=10)

#%%
#Addestriamo il modello sul training set
#stiamo dando al modello delle coppie, in base alle variabili in output sono 0 1
model.fit(X_train, y_train)

#%%
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
for random_state in [0, 42, 100, 200]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=random_state)
    model = SVC(kernel="linear", C=10)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Random State: {random_state}, Accuracy: {accuracy:.4f}")



