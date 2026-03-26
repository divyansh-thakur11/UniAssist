import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_loader import load_admission_data


def preprocess_admission():

    df = load_admission_data()

    # remove serial number
    if "Serial No." in df.columns:
        df = df.drop("Serial No.", axis=1)

    # target
    y = df["Chance of Admit "]

    # features
    X = df.drop("Chance of Admit ", axis=1)

    return X, y


if __name__ == "__main__":

    X, y = preprocess_admission()

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)