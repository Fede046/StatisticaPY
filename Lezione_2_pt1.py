import numpy as np

#%%
x=np.array((1, 2, 3, 4), dtype=np.int32)
(x.shape, x.ndim)      #è un vettore

#%%
y=np.ones((4,1))
(y.shape, y.ndim)  #è una MATRICE di dimensione 4x1


#%%
#Le matrici sono definite come liste di liste, ma anche come tuple di liste
A=np.array([[100, 200, 300, 400],
           [500, 600, 700, 800]])

B=np.array(([100, 200, 300, 400],
           [500, 600, 700, 800]))

#%%
#Un vettore, cioè un array 1-dim. è utile per fare broadcasting, che permette
#fare operazioni tra array di dimensione diversa 

A=np.array([
    [1, 2, 3, 4], 
    [5, 6, 7, 8]
    ])

#A è una matrice di dimensione (2,4), mentre x è un vettore (4,)
print(A.shape)  

somma=A+x      #x viene sommato ad ogni riga di A, le righe sono quelle minori 
print(somma)



#%%







