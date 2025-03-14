import numpy as np
import matplotlib.pyplot as plt  # Correzione dell'importazione

# Creazione dell'array x da 1 a 8
x = np.arange(1, 9)

# Calcolo di y come quadrato di x
y = x**2

# Creazione del grafico
plt.plot(x, y, color='green', marker='o', linestyle='dashed')  # Uso di plt.plot e virgolette corrette

# Aggiunta di etichette agli assi e titolo
plt.xlabel('x')  # Etichetta per l'asse x
plt.ylabel('y')  # Etichetta per l'asse y
plt.title('Grafico di x^2')  # Titolo del grafico

# Mostra il grafico
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt

# Creazione dell'array x da 1 a 8
x = np.arange(1, 9)

# Calcolo di y come quadrato di x
y = x**2

# Calcolo di y1 come x + 5
y1 = x + 5

# Creazione del grafico per y (linea nera con cerchi)
plt.plot(x, y, 'ko', label='y = x^2')  # 'ko' indica cerchi neri

# Creazione del grafico per y1 (linea rossa con cerchi)
plt.plot(x, y1, 'ro', label='y = x + 5')  # 'ro' indica cerchi rossi

# Aggiunta di etichette agli assi e titolo
plt.xlabel('x')  # Etichetta per l'asse x
plt.ylabel('y')  # Etichetta per l'asse y
plt.title('Esempio')  # Titolo del grafico

# Aggiunta della legenda
plt.legend()

# Mostra il grafico
plt.show()

#%%

#ozione figure dice la dimensione che do a ogni area di figura, 
#axis sono effettivamente i grafici che vado a fare
import numpy as np
import matplotlib.pyplot as plt

# Creazione dell'array x da 1 a 8
x = np.arange(1, 9)

# Calcolo di y come quadrato di x
y = x**2

'''Creazione di una figura con una griglia 2x2 di Axes'''
# Creazione di una figura con una griglia 2x2 di Axes
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))  # figsize definisce le dimensioni della figura
#fig, ax = plt.subplots(nrows=2, ncols=2)
# Plot 1: Grafico a dispersione con cerchi neri
ax[0, 0].plot(x, y, 'ko')  # 'ko' indica cerchi neri
ax[0, 0].set_title('Grafico a dispersione')  # Titolo del primo plot

# Plot 2: Grafico a linea continua nera
ax[0, 1].plot(x, y, 'k-')  # 'k-' indica linea nera continua
ax[0, 1].set_title('Grafico a linea continua')  # Titolo del secondo plot

# Plot 3: Grafico a linea continua con cerchi neri
ax[1, 0].plot(x, y, 'k-o')  # 'k-o' indica linea nera continua con cerchi
ax[1, 0].set_title('Grafico a linea con cerchi')  # Titolo del terzo plot

# Plot 4: Grafico a linea tratteggiata con cerchi neri
ax[1, 1].plot(x, y, 'k--o')  # 'k--o' indica linea tratteggiata nera con cerchi
ax[1, 1].set_title('Grafico a linea tratteggiata')  # Titolo del quarto plot

# Aggiunta di un titolo generale alla figura
fig.suptitle('Esempio di subplot con 4 grafici', fontsize=16)



#aggiuntiva non necessaria
'''Impostazione della scala per tutti i grafici'''
# Impostazione della scala per tutti i grafici
for a in ax.flat:
   a.set_xlim([min(x), max(x)])  # Imposta i limiti dell'asse x
   a.set_ylim([min(y), max(y)])  # Imposta i limiti dell'asse y

# Aggiunta di un titolo generale alla figura
fig.suptitle('Esempio di subplot con 4 grafici in scala', fontsize=16)


# Regolazione dello spazio tra i subplot
plt.tight_layout()

# Mostra la figura
plt.show()

# %%

'''Introduzione agli istogrammi'''
# Introduzione agli istogrammi
# Questo script mostra come creare istogrammi con diverse impostazioni di bin.
# Più bin si utilizzano, più dettagliato sarà l'istogramma, ma attenzione: troppi bin possono rendere il grafico meno leggibile.

# Importazione delle librerie necessarie
import numpy as np  # Libreria per operazioni numeriche, utile per generare dati casuali
import matplotlib.pyplot as plt  # Libreria per la creazione di grafici

# Definizione del numero di bin (intervalli) da utilizzare negli istogrammi
bin_number = [10, 25, 50, 100]  # Lista che contiene i numeri di bin per i diversi istogrammi

# Generazione di un vettore di numeri casuali con distribuzione normale
randvett = np.random.normal(0, 1, 100000)  # Media = 0, Deviazione standard = 1, 100000 valori
# randvett è un array di 100000 numeri casuali distribuiti normalmente (gaussiana)

# Creazione di una figura con una griglia 2x2 di subplot
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
# fig è la figura principale, ax è una matrice 2x2 di assi (subplot) per disegnare i grafici

# Ciclo per creare gli istogrammi
for k in range(len(bin_number)):
    if k <= 1:
        # Primi due istogrammi nella prima riga
        ax[0, k].hist(randvett, bins=bin_number[k], color='orange', alpha=0.7)
        # hist() crea l'istogramma con il numero di bin specificato
        # color='orange' imposta il colore dell'istogramma
        # alpha=0.7 imposta la trasparenza del colore
        ax[0, k].set_title(f'Istogramma con {bin_number[k]} bin')  # Imposta il titolo del subplot
    else:
        # Ultimi due istogrammi nella seconda riga
        ax[1, k % 2].hist(randvett, bins=bin_number[k], color='orange', alpha=0.7)
        ax[1, k % 2].set_title(f'Istogramma con {bin_number[k]} bin')

# Regolazione dello spazio tra i subplot per evitare sovrapposizioni
plt.tight_layout()

# Mostra la figura con i quattro istogrammi
plt.show()
#%%


#random.choice()
import numpy as np

# Creazione di un array v con valori da 1 a 10
v = np.arange(1, 11)

'''Estrazione casuale di 10 elementi dall'array v CON reinserimento'''
# Estrazione casuale di 10 elementi dall'array v CON reinserimento
# Questo significa che uno stesso elemento può essere estratto più volte
estrazione_con_reinserimento = np.random.choice(v, 10)
print("Estrazione con reinserimento:", estrazione_con_reinserimento)
# Output esempio: [2 9 5 9 1 9 2 6 5 9]

'''Estrazione casuale di 8 elementi dall'array v SENZA reinserimento'''
# Estrazione casuale di 8 elementi dall'array v SENZA reinserimento
# Questo significa che ogni elemento può essere estratto solo una volta
estrazione_senza_reinserimento = np.random.choice(v, 8, replace=False)
print("Estrazione senza reinserimento:", estrazione_senza_reinserimento)
# Output esempio: [2 9 1 7 3 6 8 4]


'''Estrazione di 10 elementi dall'array v CON reinserimento e probabilità specifiche'''
# Estrazione con probabilità personalizzate
# Creazione di un nuovo array v con valori da 1 a 3
v = np.arange(1, 4)

# Estrazione di 10 elementi dall'array v CON reinserimento e probabilità specifiche
# p=(0.7, 0.2, 0.1) significa:
# - Probabilità del 70% di estrarre 1
# - Probabilità del 20% di estrarre 2
# - Probabilità del 10% di estrarre 3
estrazione_con_probabilita = np.random.choice(v, 10, replace=True, p=(0.7, 0.2, 0.1))
print("Estrazione con probabilità personalizzate:", estrazione_con_probabilita)
# Output esempio: [1 1 2 3 1 1 2 3 1 1]

#%%
#rappresentati in due figure diverse

#Barplot & Piechart



import numpy as np
import matplotlib.pyplot as plt

# Creazione di un vettore di tipo qualitativo
x = np.array(("a", "a", "a", "b", "a", "b"))  # Vettore con valori qualitativi

# Conta delle occorrenze di ogni valore unico nel vettore
unique = np.unique(x)  # Trova i valori unici nel vettore
count = [np.sum(x == el) for el in unique]  # Conta le occorrenze di ogni valore unico

'''Rappresentazione con grafico a barre'''
# Rappresentazione con grafico a barre
plt.figure(figsize=(8, 4))  # Crea una nuova figura
plt.bar(unique, count, color=['blue', 'orange'])  # Crea un grafico a barre
plt.title("Grafico a barre delle occorrenze")  # Aggiunge un titolo
plt.xlabel("Valori unici")  # Etichetta per l'asse x
plt.ylabel("Conteggio")  # Etichetta per l'asse y
plt.show()  # Mostra il grafico a barre

'''Rappresentazione con grafico a torta'''
# Rappresentazione con grafico a torta
plt.figure(figsize=(8, 4))  # Crea una nuova figura
plt.pie(count, labels=unique, autopct='%1.1f%%', colors=['blue', 'orange'])  # Crea un grafico a torta
plt.title("Grafico a torta delle occorrenze")  # Aggiunge un titolo
plt.show()  # Mostra il grafico a torta


#%%%
#rappresentati nella stessa figure

import numpy as np
import matplotlib.pyplot as plt

# Creazione di un vettore di tipo qualitativo
x = np.array(("a", "a", "a", "b", "a", "b"))  # Vettore con valori qualitativi

# Conta delle occorrenze di ogni valore unico nel vettore
unique = np.unique(x)  # Trova i valori unici nel vettore
count = [np.sum(x == el) for el in unique]  # Conta le occorrenze di ogni valore unico

'''# Creazione di una figura con due subplot affiancati'''
# Creazione di una figura con due subplot affiancati
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 1 riga, 2 colonne

# Grafico a barre
ax1.bar(unique, count, color=['blue', 'orange'])  # Crea un grafico a barre
ax1.set_title("Grafico a barre delle occorrenze")  # Aggiunge un titolo
ax1.set_xlabel("Valori unici")  # Etichetta per l'asse x
ax1.set_ylabel("Conteggio")  # Etichetta per l'asse y

# Grafico a torta
ax2.pie(count, labels=unique, autopct='%1.1f%%', colors=['blue', 'orange'])  # Crea un grafico a torta
ax2.set_title("Grafico a torta delle occorrenze")  # Aggiunge un titolo

# Regolazione dello spazio tra i subplot
plt.tight_layout()

# Mostra la figura
plt.show()


#%%
#Seaborn

# Importazione delle librerie necessarie
import seaborn as sns  # Seaborn è una libreria di visualizzazione dati basata su matplotlib
import pandas as pd    # Pandas è una libreria per la manipolazione di dati strutturati
import numpy as np     # NumPy è una libreria per il calcolo numerico
import matplotlib.pyplot as plt  # Matplotlib è una libreria per la creazione di grafici

# Seaborn è una libreria che semplifica la creazione di grafici complessi rispetto a Pyplot.
# Offre un'ottima integrazione con i DataFrame di Pandas, rendendo più semplice la visualizzazione
# di dati strutturati. Per una lista completa dei tipi di grafici disponibili, si può consultare
# la Python Graph Gallery.

# Esempio 1: Grafico di una funzione con Seaborn e Pandas
# Creiamo un DataFrame per rappresentare la funzione f(x) = x^2 nell'intervallo [-5, 5]
df = pd.DataFrame({
    'x_axis': np.arange(-5, 6),  # Valori dell'asse x da -5 a 5
    'y_axis': np.arange(-5, 6)**2  # Valori dell'asse y come quadrato di x
})

# Utilizziamo Matplotlib per disegnare il grafico
plt.plot('x_axis', 'y_axis', data=df, linestyle='-', marker='o')  # Linea continua con marcatori a cerchio
plt.title("Grafico di f(x) = x^2")  # Titolo del grafico
plt.xlabel("x")  # Etichetta per l'asse x
plt.ylabel("y")  # Etichetta per l'asse y
plt.grid(True)  # Aggiunge una griglia al grafico
plt.show()  # Mostra il grafico

# Esempio 2: Violin plot con Seaborn
# Carichiamo un dataset di esempio incluso in Seaborn
df_tips = sns.load_dataset('tips')  # Dataset contenente informazioni sui ristoranti

# Creiamo un violin plot per visualizzare la distribuzione dei dati
# Un violin plot combina un boxplot con una rappresentazione della densità di probabilità.
# È utile per mostrare la distribuzione dei dati in un dataset.
sns.violinplot(x=df_tips["day"], y=df_tips["total_bill"], data=df_tips)
plt.title("Violin Plot: Distribuzione del totale del conto per giorno")  # Titolo del grafico
plt.xlabel("Giorno")  # Etichetta per l'asse x
plt.ylabel("Totale conto")  # Etichetta per l'asse y
plt.show()  # Mostra il grafico

# Spiegazione teorica del violin plot:
# - Il violin plot mostra la distribuzione di densità della variabile y (total_bill) in corrispondenza
#   delle categorie della variabile x (day).
# - All'interno del violin plot è presente un boxplot, che rappresenta sinteticamente i quartili,
#   la mediana e i valori estremi della distribuzione.
# - Le distribuzioni a destra e a sinistra del boxplot sono simmetriche e rappresentano la densità
#   di probabilità dei dati. Questo rende più chiara la distribuzione dei dati rispetto a un boxplot tradizionale.

