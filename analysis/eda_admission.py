from data_loader import load_admission_data


df = load_admission_data()

print(df.shape)
print(df.isnull().sum())
print(df.describe())


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_admission_data

df = load_admission_data()

print("===== OVERVIEW =====")
print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nMissing Values:\n", df.isnull().sum())

print("\n===== TARGET SUMMARY =====")
print(df["Chance of Admit "].describe())

# -----------------------------
# 1️⃣ Target Distribution
# -----------------------------
plt.figure()
sns.histplot(df["Chance of Admit "], kde=True)
plt.title("Distribution of Admission Chance")
plt.show()

# -----------------------------
# 2️⃣ Correlation Heatmap
# -----------------------------
plt.figure()
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# -----------------------------
# 3️⃣ GRE vs Admission
# -----------------------------
plt.figure()
sns.scatterplot(x=df["GRE Score"], y=df["Chance of Admit "])
plt.title("GRE Score vs Admission Chance")
plt.show()

# -----------------------------
# 4️⃣ CGPA vs Admission
# -----------------------------
plt.figure()
sns.scatterplot(x=df["CGPA"], y=df["Chance of Admit "])
plt.title("CGPA vs Admission Chance")
plt.show()

# -----------------------------
# 5️⃣ Outlier Check
# -----------------------------
plt.figure()
sns.boxplot(x=df["Chance of Admit "])
plt.show()
