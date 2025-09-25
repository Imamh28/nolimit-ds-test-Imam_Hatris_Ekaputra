import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import NearestNeighbors
from collections import Counter


st.set_page_config(page_title="Sentiment Analysis (Local Verification)")
st.title("Sentiment Analysis with Local Sample Data")
st.write("This app uses RoBERTa embeddings and a small local dataset with k-NN for sentiment classification.")


@st.cache_resource
def setup_system():

    df = pd.read_csv("local_verification.csv")
    df = df[["text", "sentiment"]].rename(columns={"sentiment": "label"})

    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

    def get_roberta_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()

    corpus_embeddings = [get_roberta_embedding(t) for t in df["text"].tolist()]
    corpus_embeddings = np.vstack(corpus_embeddings)

    nn_index = NearestNeighbors(n_neighbors=3, algorithm="ball_tree").fit(corpus_embeddings)

    return tokenizer, model, nn_index, df

tokenizer, model, nn_index, df_data = setup_system()


def predict_sentiment(text_input, k=3):
    inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    input_embedding = outputs.last_hidden_state[:, 0, :].numpy()

    distances, indices = nn_index.kneighbors(input_embedding)
    neighbor_labels = [df_data.iloc[i]["label"] for i in indices[0]]
    most_common_label = Counter(neighbor_labels).most_common(1)[0]
    label = most_common_label[0]
    confidence = most_common_label[1] / k
    return label, confidence

with st.form("sentiment_form"):
    user_input = st.text_area("Enter text to analyze:", "I really enjoyed this product, it's awesome!")
    submit_button = st.form_submit_button("Analyze Now")

if submit_button:
    if user_input:
        label, score = predict_sentiment(user_input)
        st.subheader("Analysis Result:")
        if label.lower() == "negative":
            st.error(f"Predicted Sentiment: **{label}** (Confidence: {score:.2f})")
        elif label.lower() == "neutral":
            st.warning(f"Predicted Sentiment: **{label}** (Confidence: {score:.2f})")
        elif label.lower() == "positive":
            st.success(f"Predicted Sentiment: **{label}** (Confidence: {score:.2f})")
        else:
            st.info(f"Predicted Sentiment: **{label}** (Confidence: {score:.2f})")
    else:
        st.error("Please enter some text first.")
