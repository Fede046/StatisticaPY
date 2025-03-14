#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 11:04:53 2025

@author: elenalolipiccolomini
"""

import pandas as pd
import numpy as np

#sto creando un vettore con diversi elementi, 
#se mi fermassi qua vrei creato un vettore, ovvero
#un oggetto array della liberira py
series_np = pd.Series(np.array([10,20,30,40,50,60]))
print(series_np) 

#questi sono degli array, i valori della serie sono degli array
series_np.values

#questa con questa variabile diventa 
#questa non la posso modificare
series_np[2]

#%%
'''Costruisco un dataframe'''

df = pd.DataFrame([["Fred",80],["Jill",90]],columns=["student", "grade"])

print(df.head())#head è un aproprietà del dataframe che mi ffavedere e prime 5 righe
print(df.tail())#mi fa vedere le utltime 5 righe
print(df.describe)
#la proprietà describe mi fa vedere il dataframe

#ottengo qualcosa che ha dei valori in un array e delle etichette nelle colums;
#Normally Pandas dataframe operations create a new dataframe. 
#But we can use inplace=True in some operations to update the existing dataframe without having to make a new one.


df.set_index("student",inplace=True)

#Add a column to the dataframe
#Vuol dire che io aggiungo al dataframe, la colonna birthdate, e che ha questo valore per la prima righa e 
#questo valreo per la seconda righa
df['birthdate']=['1970-01-12', '1972-05-12']
df.columns

#select a column from the dataframe
#psso selezionare una colonna dal dataframe attraverso la funzione df e il nome della colonna,
#diventan o una series, quindi se vedrò solo la colonna del voto
grade=df['grade']
print(grade)

'''Creo un dataframe'''
#Add rows to the data frame by creating a new one (df2) and the appending it to create df3
df2 = pd.DataFrame([[70,'1980-11-12'],[97, '1984-11-01']],index=["Costas", "Ilya"], columns=["grade", "birthdate"])
df2

#utilizzo la funzione concat per aggiungere i due unire i due dataframe
df3=pd.concat([df,df2])
df3

#select rows from the dataframe

df3.iloc[0:2]
#%%
'''Un dizionario'''
#####################################################################
# Un dizionario è una struttura dati che contiene coppie chiave-valore.
# È definito utilizzando parentesi graffe {}.
# A differenza di un array, un dizionario può contenere dati di tipo diverso.
# Per convertire un dizionario in un DataFrame (una struttura tabellare), 
# si utilizza la libreria Pandas. Un DataFrame è essenzialmente un array 
# bidimensionale che può contenere dati di diverso tipo.

# Creiamo un nuovo DataFrame a partire da un dizionario
mcu_data = {
    'Title': ['Ant-Man and the Wasp', 'Avengers: Infinity War', 'Black Panther', 'Thor: Ragnarok', 
              'Spider-Man: Homecoming', 'Guardians of the Galaxy Vol. 2'],
    'Year': [2018, 2018, 2018, 2017, 2017, 2017],  # Anno di uscita del film
    'Studio': ['Beuna Vista', 'Beuna Vista', 'Beuna Vista', 'Beuna Vista', 'Sony', 'Beuna Vista'],  # Studio di produzione
    'Rating': [np.nan, np.nan, 0.96, 0.92, 0.92, 0.83]  # Valutazione del film (NaN indica valori mancanti)
}

# Convertiamo il dizionario in un DataFrame utilizzando Pandas
df_mcu = pd.DataFrame(mcu_data)

# La proprietà `describe` fornisce statistiche descrittive del DataFrame
df_mcu.describe()

# La proprietà `shape` restituisce le dimensioni del DataFrame (righe, colonne)
df_mcu.shape

# La proprietà `columns` restituisce i nomi delle colonne del DataFrame
df_mcu.columns

# La proprietà `index` restituisce gli indici (etichette delle righe) del DataFrame
df_mcu.index

# La proprietà `values` restituisce il contenuto del DataFrame come un array NumPy
df_mcu.values

# Queste sono alcune delle proprietà più comuni di un DataFrame.

# Estraiamo una singola colonna dal DataFrame
df_mcu['Title']

# Estraiamo più colonne specificandole in una lista
df_mcu[['Title', 'Rating']]

# Operazioni di slicing (selezione di sottoinsiemi di dati)
# Estraiamo i primi due elementi della colonna 'Title'
df_mcu['Title'][:2]

# Estraiamo gli elementi dalla posizione 4 in poi della colonna 'Title'
df_mcu['Title'][4:]

# Estraiamo gli elementi dalla posizione 1 alla 3 della colonna 'Title'
df_mcu['Title'][1:4]

# Utilizziamo `iloc` per accedere a un elemento specifico del DataFrame
# `iloc[riga, colonna]` - qui stiamo accedendo alla prima riga e seconda colonna
df_mcu.iloc[0, 1]

# Utilizziamo `iloc` per selezionare righe specifiche (2, 4, 5)
df_mcu.iloc[[2, 4, 5]]

# Visualizziamo l'intero DataFrame
df_mcu

