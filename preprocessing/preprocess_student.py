import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_loader import load_student_data


def preprocess_student():

    df = load_student_data()

    # encode categorical columns
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # features and target
    X = df.drop("G3", axis=1)
    y = df["G3"]

    return X, y


if __name__ == "__main__":

    X, y = preprocess_student()

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)