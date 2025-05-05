#classificazione in tre classi 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#%%
# Caricamento del dataset Iris
iris = datasets.load_iris()
X = iris.data[:, :2]  # Prendiamo solo due feature per la visualizzazione
y = iris.target  # Tre classi

#%%
# Suddivisione in training e test set
#X -> abbiamo solo le fechure
#y -> abbiamo il targeth
#train size indica a quale parcentuale del dataset è 
#destinata al training
#random state a piacere ma il numero cambia la sequenza
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creazione e addestramento del modello SVM
#classe che contine tutte le unità del modello
#kernel lineare
#cambiano kernel cambi l'accuratezza
clf = SVC(kernel='linear', random_state=42)
#clf = SVC(kernel='poly')
#li abbiamo decise prima, feachur e target del traing e testing
clf.fit(X_train, y_train)

# Predizione sul test set
#test mi predice la classe delle feachure del test set
#quando faccio predizione le faccio solo sulle feachure del test set
# e ottengo le etichette predette
#predizione su dati che non ha mai visto
y_pred = clf.predict(X_test)

# Valutazione del modello
#calcolo accuratezza, con y predette, e y corrette del test set
print("Accuracy:", accuracy_score(y_test, y_pred))
#mi da dei tipi di classificazione della classe shikit learn
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualizzazione dei dati
plt.figure(figsize=(8,6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', edgecolors='k')
plt.title("Classificazione in 3 classi con Support Vector Machine")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# Matrice di confusione
#è una matricec 3x3: dove sulla diagonale per ogni classe, il numero
#di elementi della classe, sono quelli che vanno a contribuire nell'errore di accuratezza 
#nella diagonale la predetta è uguale alla vera
#due fiore sono stati predetti classe virginica invece erano di versicolor
#per ogni classe avete una variabile targhet, che mi dice itipi di iris che abbiamo classificato
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Matrice di Confusione")
plt.show()
