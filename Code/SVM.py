import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC

# Set seaborn (and matplotlib) style
sns.set_style("whitegrid")

# Load data
df = pd.read_csv("./data/Classification_data.csv")

# Convert output to "category"
df["class"] = df["class"].astype("category")

# Get sub-DataFrame of each class
df_cross = df[df["class"] == 0]
df_plus = df[df["class"] == 1]

# Define the model

#kernel radiale
model = SVC(kernel="rbf", gamma=10)
#retta (kernel binomiale)
model = SVC(kernel="linear")
#kernel polinomiale, con iperparametro (di defolut di grado 3)
model = SVC(kernel="poly")
#grado 7
model = SVC(kernel="poly",degree=7)

# Train on data
model.fit(df[["x1", "x2"]], df["class"])

#train con .in col test il .fit

# Visualize
xx, yy = np.meshgrid(
    np.linspace(df["x1"].min(), df["x1"].max()),
    np.linspace(df["x2"].min(), df["x2"].max()),
)

#facciamo la predizione sul test set. Prendo un punto a caso in cui faccio il
#set
input_pred = pd.DataFrame({"x1": xx.ravel(), "x2": yy.ravel()})
#prediamo una classe e con questa predizione possiamo avere dei parametri.
#di accuratezza
Z = model.predict(input_pred)
Z = Z.reshape(xx.shape)


plt.plot(df_cross["x1"], df_cross["x2"], "bo")
plt.plot(df_plus["x1"], df_plus["x2"], "ro")
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)

plt.title("SVM Classification example")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
