import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from preprocessing.preprocess_student import preprocess_student


def train_student_model():

    X, y = preprocess_student()

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # training
    model.fit(X_train, y_train)

    # predictions
    predictions = model.predict(X_test)

    # evaluation
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("Student Model Performance")
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    # save model
    joblib.dump(model, "student_model.pkl")

    print("Student model saved!")


if __name__ == "__main__":
    train_student_model()