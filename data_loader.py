import pandas as pd
import requests
import zipfile
import io
import os


# ----------------------------
# 1️⃣ Student Dataset (UCI)
# ----------------------------
def load_student_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
    
    response = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    
    df = pd.read_csv(z.open("student-mat.csv"), sep=';')
    return df


# ----------------------------
# 2️⃣ Admission Dataset (Kaggle)
# ----------------------------
def load_admission_data():
    return None


# ----------------------------
# 3️⃣ College Scorecard Dataset
# ----------------------------
def load_college_data():
    path = os.path.join("data", "college_scorecard.csv")
    df = pd.read_csv(path, low_memory=False)
    return df

from data_loader import load_student_data

