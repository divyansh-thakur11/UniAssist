import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_college_data

df = load_college_data()

print("Shape:", df.shape)
print("\nFirst 20 Columns:\n", df.columns[:20])
print("\nMissing Values %:\n", (df.isnull().sum() / len(df)) * 100)

# Select important columns (example)
selected_cols = ["INSTNM", "STABBR", "TUITIONFEE_IN", "ADM_RATE"]

df_small = df[selected_cols].dropna()

print("\nFiltered Shape:", df_small.shape)

# -----------------------------
# 1️⃣ Tuition Distribution
# -----------------------------
plt.figure()
sns.histplot(df_small["TUITIONFEE_IN"], kde=True)
plt.title("In-State Tuition Distribution")
plt.show()

# -----------------------------
# 2️⃣ Admission Rate Distribution
# -----------------------------
plt.figure()
sns.histplot(df_small["ADM_RATE"], kde=True)
plt.title("Admission Rate Distribution")
plt.show()