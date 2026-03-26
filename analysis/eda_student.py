import pandas as pd
from data_loader import load_student_data

df = load_student_data()

print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary:\n", df.describe())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_student_data

df = load_student_data()

print("===== DATASET OVERVIEW =====")
print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nData Types:\n", df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== TARGET SUMMARY (G3) =====")
print(df["G3"].describe())

# -----------------------------
# 1️⃣ Target Distribution
# -----------------------------
plt.figure()
sns.histplot(df["G3"], kde=True)
plt.title("Distribution of Final Grade (G3)")
plt.show()

# -----------------------------
# 2️⃣ Correlation Heatmap
# -----------------------------
plt.figure()
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# -----------------------------
# 3️⃣ Important Feature vs Target
# -----------------------------
plt.figure()
sns.scatterplot(x=df["studytime"], y=df["G3"])
plt.title("Study Time vs Final Grade")
plt.show()

# -----------------------------
# 4️⃣ Outlier Check
# -----------------------------
plt.figure()
sns.boxplot(x=df["G3"])
plt.title("Outliers in Final Grade")
plt.show()