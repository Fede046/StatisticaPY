import os
#coamndo che mi permette di interagire col sistema operativo per creare caretelle ecc

#%%
#miperemtte di esegure la singola cella se si premi Shift + Invio
#con il comando clear mi pulusce la consol
os.getcwd()
print("Hello ")
v2 = "Pippo"
v3 = 'Pippo3'

#%%
#le tuple sono un dato immutabile, quindi non sono modificabili una volta 
#che ne abbiamo definita una 
t = (1,2,3)


#%%
#liste sono mutabili
L = [1,2,3]
L.append(4) #mi aggiunge un elemnto alla fine della lista
print(L[-1])

#%%
a=2
if(a<5):
    print('hi')


#%%

for i in range(8):
    print(i)


#%%
def media(x1, x2):
    m = (x1+x2)/2
    return m


#%%
import math as m
 
print(m.sqrt(2))


#%%
import numpy as np
x=np.array((1,2,3,4))
x.size
x.shape



#%%
import numpy as np
#brodcasting mi permette di sommare array di somma diversa
A = np.array([[1,2,3,4],[5,6,7,8]])
A.shape
print(np.where(A < 3))


