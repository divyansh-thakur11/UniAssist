import streamlit as st
from app import handle_query


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="UniAssist AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 UniAssist - AI Academic Assistant")

st.write(
    "UniAssist helps students with university admissions, "
    "performance prediction, and academic queries using AI."
)


# -----------------------------
# Sidebar Information
# -----------------------------
st.sidebar.title("About UniAssist")

st.sidebar.write("""
UniAssist is an AI assistant that helps students with:

• Student performance prediction  
• Admission probability prediction  
• College recommendations  
• Academic question answering using RAG + LLM
""")


st.sidebar.title("Example Questions")

st.sidebar.write("""
• What GRE score is good for MS?  
• Recommend universities in California  
• Explain CGPA  
• How does GRE affect admission?  
""")


# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# -----------------------------
# User Input
# -----------------------------
query = st.chat_input("Ask your question")


# -----------------------------
# Handle Query
# -----------------------------
if query:

    # Show user message
    with st.chat_message("user"):
        st.write(query)

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    # Generate response
    with st.spinner("Thinking..."):
        response = handle_query(query)

    # Show assistant message
    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )