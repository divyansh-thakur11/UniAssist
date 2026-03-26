import joblib
import pandas as pd

from nlp.intent_classifier import train_intent_classifier
from recommender.college_recommender import recommend_colleges
from llm.rag_pipeline import rag_pipeline


# -----------------------------
# Load intent classifier
# -----------------------------
intent_model = joblib.load("intent_model.pkl")
vectorizer = joblib.load("intent_vectorizer.pkl")


# -----------------------------
# Load ML models
# -----------------------------
student_model = joblib.load("student_model.pkl")
admission_model = joblib.load("admission_model.pkl")


# -----------------------------
# Detect intent
# -----------------------------
def detect_intent(query):

    query_vec = vectorizer.transform([query])

    intent = intent_model.predict(query_vec)[0]

    return intent


# -----------------------------
# Handle query
# -----------------------------
def handle_query(query):

    intent = detect_intent(query)

    # Student grade prediction
    if intent == "student_prediction":

        # Example dummy input (replace with UI later)
        sample = [[1] * student_model.n_features_in_]

        prediction = student_model.predict(sample)

        return f"Predicted Final Grade: {prediction[0]:.2f}"


    # Admission prediction
    elif intent == "admission_prediction":

        sample = [[1] * admission_model.n_features_in_]

        prediction = admission_model.predict(sample)

        return f"Predicted Admission Chance: {prediction[0]:.2f}"


    # College recommendation
    elif intent == "college_recommendation":

        results = recommend_colleges()

        return results.head(5).to_string()


    # General knowledge → RAG + LLM
    else:

        answer = rag_pipeline(query)

        return answer


# -----------------------------
# CLI test
# -----------------------------
if __name__ == "__main__":

    print("\nUniAssist AI Assistant\n")

    while True:

        query = input("Ask a question: ")

        if query.lower() in ["exit", "quit"]:
            break

        response = handle_query(query)

        print("\nResponse:\n", response)