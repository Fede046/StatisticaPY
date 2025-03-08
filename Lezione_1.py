import os

#%%
#Stampa la cartella di lavoro corrente, la stessa che vedete in files
os.getcwd()

#%%
#La cartella può essere settata tramite il comando seguente
path="C:\\Users\\dario\\Desktop\\DARIO\\Tutorati\\Statistica Numerica"
os.chdir(path)

#%%
print("Hello World!")
v1='pippo'
v2='Pippo '
5%3


#%%
#Attenzione a scrivere bene il path con le doppie \\ !!
path="C:\Users\dario\Desktop\DARIO\Tutorati"

#%%
#Le tuple sono un tipo di dato immutabile, quindi non potete cambiarne gli
#elementi, una volta che ne avete definita una
t=(1, 2, 3)
#t[0]=100
t[-1]
#%%
s1='Hello'
s2=' '
s3="World!"

print(s1+s2+s3)

#%%
L=[1, 2, 3]
L.append(4)

#%%
for i in range (8):
    print(i)
 
#%%
L=[1, 2, 3]

def mod_list(lista):
    lista[0]=100
    return lista

L2=mod_list(L)

print(L2)
print(L)

#%%
import math 

math.sqrt(2)


#%%
from math import sqrt

sqrt(2)




















