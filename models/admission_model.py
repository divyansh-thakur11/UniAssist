import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from preprocessing.preprocess_admission import preprocess_admission


def train_admission_model():

    X, y = preprocess_admission()

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # model
    model = LinearRegression()

    # train
    model.fit(X_train, y_train)

    # predictions
    predictions = model.predict(X_test)

    # evaluation
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("Admission Model Performance")
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    # save model
    joblib.dump(model, "admission_model.pkl")

    print("Admission model saved!")


if __name__ == "__main__":
    train_admission_model()