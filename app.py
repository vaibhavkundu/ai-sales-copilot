import streamlit as st
import pandas as pd
import os
from groq import Groq

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="AI Sales Copilot", layout="wide")

# ==============================
# PREMIUM UI
# ==============================
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
[data-testid="stChatMessage"] {
    border-radius: 15px;
    padding: 10px;
    margin-bottom: 10px;
}
h1 {
    color: #00d4ff;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Sales Engineer Copilot")

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("products.csv")

df["combined"] = (
    df["product_name"] + " " +
    df["category"] + " " +
    df["use_case"] + " " +
    df["short_description"]
)

# ==============================
# GROQ SETUP
# ==============================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ==============================
# SIMPLE SEARCH
# ==============================
def search_context(query):
    results = df[df["combined"].str.contains(query, case=False, na=False)]
    
    if len(results) == 0:
        results = df.head(3)
    
    return results

# ==============================
# GENERATE ANSWER
# ==============================
def generate_answer(query, results):
    context = ""
    for _, row in results.iterrows():
        context += f"""
        Product: {row['product_name']}
        Category: {row['category']}
        Use Case: {row['use_case']}
        Description: {row['short_description']}
        """

    prompt = f"""
    You are an expert sales engineer.

    Based on the following product knowledge:
    {context}

    Answer this question clearly and professionally:
    {query}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content

# ==============================
# CHAT UI
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
query = st.chat_input("Ask anything about products...")

if query:
    # User message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # AI processing
    results = search_context(query)
    answer = generate_answer(query, results)

    # AI response
    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
