import streamlit as st
import pandas as pd
import numpy as np
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
h1, h2, h3 {
    color: #00d4ff;
}
.stButton>button {
    border-radius: 10px;
    background: linear-gradient(90deg, #00d4ff, #007cf0);
    color: white;
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
# SEARCH (NO FAISS - CLOUD SAFE)
# ==============================
def search_products(query):
    results = df[df["combined"].str.contains(query, case=False, na=False)]
    
    if len(results) == 0:
        return df.head(3)
    
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

    Based on:
    {context}

    Answer:
    {query}

    Recommend best product clearly.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content

# ==============================
# RFQ GENERATOR
# ==============================
def generate_rfq(query, results):
    context = results.to_string()

    prompt = f"""
    Generate a professional RFQ response email.

    Requirement:
    {query}

    Products:
    {context}

    Include:
    - Greeting
    - Recommended product
    - Key features
    - MOQ placeholder
    - Delivery timeline placeholder
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content

# ==============================
# FILTERS
# ==============================
st.sidebar.header("🔍 Filters")

category_filter = st.sidebar.multiselect(
    "Category", df["category"].unique()
)

if category_filter:
    df = df[df["category"].isin(category_filter)]

# ==============================
# CHAT UI
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask about products...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    results = search_products(query)
    answer = generate_answer(query, results)

    with st.chat_message("assistant"):
        st.write(answer)

        st.markdown("### 📦 Recommended Products")
        st.dataframe(results[["product_name", "category", "use_case"]])

        # RFQ Button
        if st.button("📄 Generate RFQ Email"):
            rfq = generate_rfq(query, results)
            st.markdown("### ✉️ RFQ Email")
            st.write(rfq)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ==============================
# PRODUCT COMPARISON
# ==============================
st.divider()
st.header("⚖️ Smart Product Comparison")

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

    st.dataframe(comparison)

    # AI Insight
    st.subheader("🧠 AI Insight")

    compare_prompt = f"""
    Compare these two products:

    Product 1:
    {prod1}

    Product 2:
    {prod2}

    Give:
    - Key differences
    - Best use case
    - Recommendation
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": compare_prompt}],
        temperature=0.3
    )

    st.write(response.choices[0].message.content)

# ==============================
# DASHBOARD
# ==============================
st.divider()
st.header("📊 Product Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Category Distribution")
    st.bar_chart(df["category"].value_counts())

with col2:
    st.subheader("Use Case Distribution")
    st.bar_chart(df["use_case"].value_counts())

# ==============================
# POPULAR PRODUCTS
# ==============================
st.subheader("🔥 Top Products")
st.dataframe(df.head(5)[["product_name", "category", "use_case"]])
