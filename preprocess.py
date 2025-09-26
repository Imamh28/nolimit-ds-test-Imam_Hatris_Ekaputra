import pandas as pd
import numpy as np
import joblib
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

print("Starting the pre-processing process with RoBERTa...")


print("1. Loading the RoBERTa model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")


print("2. Loading dataset from Hugging Face...")

df = pd.read_csv("Sp1786-5000-data.csv")

df = df[['text', 'sentiment']].rename(columns={'sentiment': 'label'})
df = df.dropna(subset=['text', 'label'])


def get_roberta_embedding(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return cls_embedding


print("3. Generating embeddings from RoBERTa for the entire dataset...")
corpus_embeddings = []
for text in tqdm(df['text'].tolist()):
    embedding = get_roberta_embedding(text)
    corpus_embeddings.append(embedding)


corpus_embeddings = np.vstack(corpus_embeddings)


print("4. Building the k-NN index...")
nn_index = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(corpus_embeddings)


print("5. Save the results to a file...")
np.save('roberta_embeddings.npy', corpus_embeddings)
joblib.dump(nn_index, 'roberta_knn_index.joblib')
df['label'].to_csv('Sp1786-5000-data.csv', index=False)

print("\nPre-processing is complete!")
print("The following files have been created: roberta_embeddings.npy, roberta_knn_index.joblib, Sp1786-5000-data.csv")