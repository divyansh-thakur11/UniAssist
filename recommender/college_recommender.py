import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_loader import load_college_data


def recommend_colleges(state=None, max_tuition=None):

    df = load_college_data()

    # select useful columns
    cols = ["INSTNM", "STABBR", "TUITIONFEE_IN", "ADM_RATE"]
    df = df[cols]

    df = df.dropna()

    # filter by state
    if state:
        df = df[df["STABBR"] == state]

    # filter by tuition
    if max_tuition:
        df = df[df["TUITIONFEE_IN"] <= max_tuition]

    # sort by admission rate
    df = df.sort_values(by="ADM_RATE", ascending=False)

    return df.head(10)


if __name__ == "__main__":

    results = recommend_colleges(state="CA", max_tuition=20000)

    print(results)