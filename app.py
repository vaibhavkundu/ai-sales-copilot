import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="AI Sales Copilot", layout="wide")

# ==============================
# DARK UI
# ==============================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Sales Engineer Copilot")

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("products.csv")

# ==============================
# PREPARE TEXT
# ==============================
df["combined"] = (
    df["product_name"] + " " +
    df["category"] + " " +
    df["use_case"] + " " +
    df["short_description"]
)

# ==============================
# LOAD MODEL (CACHE)
# ==============================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ==============================
# CREATE FAISS INDEX
# ==============================
@st.cache_resource
def create_index(texts):
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index

index = create_index(df["combined"].tolist())

# ==============================
# GROQ SETUP
# ==============================
import os
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ==============================
# SEARCH FUNCTION (FIXED DUPLICATES)
# ==============================
def search_products(query, k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)

    results = df.iloc[indices[0]]

    # REMOVE DUPLICATES
    results = results.drop_duplicates(subset=["product_name"])

    return results.head(3)

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

    Based on the following products:
    {context}

    Answer this query:
    {query}

    Give clear recommendation.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content

# ==============================
# CHAT UI (ChatGPT STYLE)
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
query = st.chat_input("Ask about products...")

if query:
    # USER MESSAGE
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # AI RESPONSE
    results = search_products(query)
    answer = generate_answer(query, results)

    with st.chat_message("assistant"):
        st.write(answer)

        st.markdown("### 📦 Recommended Products")
        st.dataframe(results[["product_name", "category", "use_case"]])

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ==============================
# PRODUCT COMPARISON
# ==============================
st.divider()
st.header("⚖️ Compare Products")

products = df["product_name"].unique()

col1, col2 = st.columns(2)

p1 = col1.selectbox("Product 1", products)
p2 = col2.selectbox("Product 2", products)

if p1 and p2:
    prod1 = df[df["product_name"] == p1].iloc[0]
    prod2 = df[df["product_name"] == p2].iloc[0]

    comparison = pd.DataFrame({
        "Feature": ["Category", "Use Case", "Description"],
        p1: [prod1["category"], prod1["use_case"], prod1["short_description"]],
        p2: [prod2["category"], prod2["use_case"], prod2["short_description"]]
    })

    st.table(comparison)