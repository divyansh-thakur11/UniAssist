import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train_intent_classifier():

    df = pd.read_csv("data/intents.csv")

    X = df["text"]
    y = df["intent"]

    vectorizer = TfidfVectorizer()

    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    print("Intent Classifier Accuracy:", accuracy)

    # save model and vectorizer
    joblib.dump(model, "intent_model.pkl")
    joblib.dump(vectorizer, "intent_vectorizer.pkl")

    print("Intent classifier saved!")


if __name__ == "__main__":
    train_intent_classifier()