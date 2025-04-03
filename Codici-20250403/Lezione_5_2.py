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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creazione e addestramento del modello SVM
clf = SVC(kernel='linear', random_state=42)
clf.fit(X_train, y_train)

# Predizione sul test set
#predizione su dati che non ha mai visto
y_pred = clf.predict(X_test)

# Valutazione del modello
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
#per ogni classe avete una variabile targhet, che mi dice itipi di iris che abbiamo classificato
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Matrice di Confusione")
plt.show()
