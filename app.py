import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModel
from collections import Counter

# =======================================================
# Page Configuration and Title
# =======================================================
st.set_page_config(page_title="Sentiment Analysis (RoBERTa k-NN)", layout="wide")
st.title("🤖 Sentiment Analysis (RoBERTa k-NN)")

st.markdown("""
This application demonstrates sentiment classification using a **Semantic Similarity Search** approach with the **k-Nearest Neighbors (k-NN)** method. Instead of classifying text directly, it finds the most similar examples from an existing dataset to determine the sentiment.
""")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("⚙️ How It Works")
    st.markdown("""
    1.  **Embedding Creation**: The sentence you enter is converted into a vector representation (embedding) to capture its semantic meaning using the **`cardiffnlp/twitter-roberta-base-sentiment-latest`** model.
    2.  **Nearest Neighbor Search**: The application then searches for the **5 most similar text samples** from the reference dataset using a pre-built k-NN index (`roberta_knn_index.joblib`).
    3.  **Majority Vote**: The most common sentiment among these 5 nearest neighbors is chosen as the final prediction.
    """)

with col2:
    st.subheader("💿 Reference Dataset")
    st.markdown("""
    - **Source**: The search index is built using `Sp1786-5000-data.csv`, which is a subset of the `Sp1786/multiclass-sentiment-analysis-dataset` from the Hugging Face Hub.
    - **Content**: This dataset contains **5,000 text samples**, each labeled with one of three sentiment categories: **positive, negative, or neutral**.
    - **Purpose**: This data acts as the "knowledge base" for the k-NN comparison. The quality and diversity of this data directly influence the prediction accuracy.
    """)

st.markdown("---")

# =======================================================
# Load PRE-COMPUTED Models and Data (with Caching)
# =======================================================
@st.cache_resource
def setup_system():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    nn_index = joblib.load('roberta_knn_index.joblib')
    df_labels = pd.read_csv('Sp1786-5000-data.csv')
    df_labels = df_labels[['text', 'sentiment']].rename(columns={'sentiment': 'label'})
    df_labels = df_labels.dropna(subset=['text', 'label'])
    return tokenizer, model, nn_index, df_labels

tokenizer, model, nn_index, df_data = setup_system()

# =======================================================
# Prediction Function
# =======================================================
def predict_sentiment_knn(text_input, k=5):
    def get_roberta_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        return cls_embedding
    input_embedding = get_roberta_embedding(text_input)
    distances, indices = nn_index.kneighbors(input_embedding)
    neighbor_labels = [df_data.iloc[i]['label'] for i in indices[0]]
    most_common_label = Counter(neighbor_labels).most_common(1)[0]
    label = most_common_label[0]
    confidence = most_common_label[1] / k
    return label, confidence

# =======================================================
# User Interface (UI)
# =======================================================
with st.form("sentiment_form"):
    user_input = st.text_area("Enter text to analyze:", "This is a fantastic approach!")
    submit_button = st.form_submit_button("Analyze Now")

# =======================================================
# Backend Logic and Result Display
# =======================================================
if submit_button:
    if user_input:
        label, score = predict_sentiment_knn(user_input)
        st.subheader("Analysis Result:")
        if label.lower() == "negative":
            st.error(f"Predicted Sentiment: **{label.capitalize()}** (Confidence: {score:.2f})")
        elif label.lower() == "neutral":
            st.warning(f"Predicted Sentiment: **{label.capitalize()}** (Confidence: {score:.2f})")
        elif label.lower() == "positive":
            st.success(f"Predicted Sentiment: **{label.capitalize()}** (Confidence: {score:.2f})")
        else:
            st.info(f"Predicted Sentiment: **{label.capitalize()}** (Confidence: {score:.2f})")
    else:
        st.error("Please enter some text first.")