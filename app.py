import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModel
from collections import Counter


st.set_page_config(page_title="Sentiment Analysis (RoBERTa k-NN)")
st.title("Sentiment Analysis (RoBERTa k-NN)")
st.write("This app uses RoBERTa embeddings and k-NN search to classify sentiment.")


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


with st.form("sentiment_form"):
    user_input = st.text_area("Enter text to analyze:", "This is a fantastic approach!")
    submit_button = st.form_submit_button("Analyze Now")


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