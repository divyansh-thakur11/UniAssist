import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# -----------------------------
# Initialize Groq Client
# -----------------------------
import streamlit as st
from groq import Groq

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)


# -----------------------------
# Load Embedding Model
# -----------------------------
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------
# Load Knowledge Base
# -----------------------------
def load_knowledge():

    with open(
        "data/knowledge.txt",
        "r",
        encoding="utf-8"
    ) as f:

        docs = f.readlines()

    docs = [
        doc.strip()
        for doc in docs
        if doc.strip()
    ]

    return docs


documents = load_knowledge()


# -----------------------------
# Build FAISS Index
# -----------------------------
embeddings = embedding_model.encode(
    documents,
    convert_to_numpy=True
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


# -----------------------------
# Retrieve Relevant Context
# -----------------------------
def retrieve_context(query):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        query_embedding,
        k=3
    )

    results = [
        documents[i]
        for i in indices[0]
    ]

    return results


# -----------------------------
# Generate Answer Using Groq
# -----------------------------
def generate_answer(query, context):

    context_text = "\n".join(context)

    prompt = f"""
You are UniAssist, an AI academic advisor.

Use ONLY the provided context to answer.

Context:
{context_text}

Question:
{query}

Provide a concise and helpful answer in 3-4 sentences.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=300
    )

    return response.choices[0].message.content


# -----------------------------
# Main RAG Pipeline
# -----------------------------
def rag_pipeline(query):

    context = retrieve_context(query)

    answer = generate_answer(
        query,
        context
    )

    return answer