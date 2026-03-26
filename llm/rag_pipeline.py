import numpy as np
import faiss
import ollama
from sentence_transformers import SentenceTransformer


# -----------------------------
# Load embedding model once
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Load knowledge base
# -----------------------------
def load_knowledge():

    with open("data/knowledge.txt", "r", encoding="utf-8") as f:
        docs = f.readlines()

    docs = [d.strip() for d in docs if d.strip() != ""]

    return docs


documents = load_knowledge()


# -----------------------------
# Build FAISS index once
# -----------------------------
embeddings = embedding_model.encode(documents)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))


# -----------------------------
# Retrieve context
# -----------------------------
def retrieve_context(query):

    query_embedding = embedding_model.encode([query])

    D, I = index.search(np.array(query_embedding), k=3)

    results = [documents[i] for i in I[0]]

    return results


# -----------------------------
# Generate answer using Mistral
# -----------------------------
def generate_answer(query, context):

    context_text = " ".join(context)

    prompt = f"""
You are UniAssist, an AI academic advisor.

Use the context to answer the question.

Context:
{context_text}

Question:
{query}

Answer clearly in 3-4 sentences.
"""

    response = ollama.chat(
        model="mistral:instruct",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


# -----------------------------
# Main RAG function
# -----------------------------
def rag_pipeline(query):

    context = retrieve_context(query)

    answer = generate_answer(query, context)

    return answer