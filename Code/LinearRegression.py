import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

t = np.arange(1, 6)
h = 2 * t + 1

h = h + 0.4 * np.random.randn(5)

df = pd.DataFrame({"t": t, "h": h})
df.to_csv("LinearRegression_data.csv")

model = LinearRegression()
model.fit(t.reshape((-1, 1)), h)

tpred = np.linspace(1, 6)
hpred = model.predict(tpred.reshape((-1, 1)))

plt.plot(t, h, "or", mfc="none")
plt.plot(tpred, hpred)
plt.grid()
plt.title("Snow over time")
plt.show()
