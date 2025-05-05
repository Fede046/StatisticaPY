import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# Set seaborn (and matplotlib) style
sns.set_style("whitegrid")

# Load data
df = pd.read_csv("./data/Classification_data2.csv")

# Convert output to "category"
df["class"] = df["class"].astype("category")

# Get sub-DataFrame of each class
df_cross = df[df["class"] == 0]
df_plus = df[df["class"] == 1]

# Define the model
model = LogisticRegression()

# Train on data
model.fit(df[["x1", "x2"]], df["class"])

# Visualize
xx, yy = np.meshgrid(
    np.linspace(df["x1"].min(), df["x1"].max()),
    np.linspace(df["x2"].min(), df["x2"].max()),
)
input_pred = pd.DataFrame({"x1": xx.ravel(), "x2": yy.ravel()})
Z = model.predict(input_pred)
Z = Z.reshape(xx.shape)


plt.plot(df_cross["x1"], df_cross["x2"], "bo")
plt.plot(df_plus["x1"], df_plus["x2"], "ro")
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)

plt.title("Logistic Regression Classification example")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
