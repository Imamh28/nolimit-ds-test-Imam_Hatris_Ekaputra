import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import NearestNeighbors
from collections import Counter

print("=== Local Verification with Small Sample Dataset ===")

print("1. Loading sample dataset...")
df = pd.read_csv("local_verification.csv")
df = df[["text", "sentiment"]].rename(columns={"sentiment": "label"})

print("2. Loading RoBERTa model...")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

def get_roberta_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return cls_embedding

print("3. Creating embeddings for sample dataset...")
corpus_embeddings = []
for text in df["text"].tolist():
    embedding = get_roberta_embedding(text)
    corpus_embeddings.append(embedding)
corpus_embeddings = np.vstack(corpus_embeddings)

print("4. Building k-NN index...")
nn_index = NearestNeighbors(n_neighbors=3, algorithm="ball_tree").fit(corpus_embeddings)

def predict_sentiment(text_input, k=3):
    input_embedding = get_roberta_embedding(text_input)
    distances, indices = nn_index.kneighbors(input_embedding)
    neighbor_labels = [df.iloc[i]["label"] for i in indices[0]]
    most_common_label = Counter(neighbor_labels).most_common(1)[0]
    label = most_common_label[0]
    confidence = most_common_label[1] / k
    return label, confidence

print("5. Testing local verification...")
test_sentences = [
    "I really enjoyed this product, it's awesome!",
    "This is terrible, I hate it.",
    "The service was okay, nothing too bad."
]

for sent in test_sentences:
    label, conf = predict_sentiment(sent)
    print(f"Text: {sent}\nPredicted: {label} (Confidence: {conf:.2f})\n")
