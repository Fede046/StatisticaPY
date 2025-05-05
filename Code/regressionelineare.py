import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Generare i dati (leggere i dati)
data = pd.DataFrame(
    {"t": np.array([1, 2, 3, 4, 5]),
     "h": np.array([3.53, 4.76, 7.6, 8.8, 11.25])}
    )

t = np.array(data["t"])
h = np.array(data["h"])

# Visualizzazione
plt.plot(data["t"], data["h"], 'o')
plt.grid()
plt.xlabel("t")
plt.ylabel("h")
plt.show()

# Scelta del modello
model = LinearRegression()

# Addestramento
model.fit(t.reshape((-1, 1)), h)

# Scegliamo un nuovo dato
t_new = np.array([[6]])

# Prediciamo
h_pred = model.predict(t_new)
print(f"Altezza predetta: {h_pred}.")














