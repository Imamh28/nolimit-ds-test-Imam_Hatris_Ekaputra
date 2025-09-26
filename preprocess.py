import pandas as pd
import numpy as np
import joblib
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

print("Memulai proses pre-processing dengan RoBERTa...")

# 1. Muat model dan tokenizer RoBERTa
print("1. Memuat model dan tokenizer RoBERTa...")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

# 2. Muat dataset
print("2. Memuat dataset dari Hugging Face...")
# hf_dataset = load_dataset("500.csv", split='train')
df = pd.read_csv("Sp1786-5000-data.csv")
# df = hf_dataset.to_pandas()
df = df[['text', 'sentiment']].rename(columns={'sentiment': 'label'})
df = df.dropna(subset=['text', 'label'])

# 3. Fungsi untuk mengekstrak embedding [CLS] dari RoBERTa
def get_roberta_embedding(text):
    # Tokenisasi teks
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    # Dapatkan output model tanpa menghitung gradien
    with torch.no_grad():
        outputs = model(**inputs)
    # Ambil last hidden state dan ekstrak embedding untuk token [CLS] (indeks 0)
    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return cls_embedding

# 4. Buat embeddings untuk seluruh dataset (ini bagian yang lambat)
print("3. Membuat embeddings dari RoBERTa untuk seluruh dataset...")
corpus_embeddings = []
for text in tqdm(df['text'].tolist()):
    embedding = get_roberta_embedding(text)
    corpus_embeddings.append(embedding)

# Tumpuk semua embedding menjadi satu array numpy besar
corpus_embeddings = np.vstack(corpus_embeddings)

# 5. Buat dan "fit" indeks k-NN
print("4. Membangun indeks k-NN...")
nn_index = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(corpus_embeddings)

# 6. Simpan semua hasil ke dalam file
print("5. Menyimpan hasil ke file...")
np.save('roberta_embeddings.npy', corpus_embeddings)
joblib.dump(nn_index, 'roberta_knn_index.joblib')
df['label'].to_csv('labels_roberta.csv', index=False) # File ini tetap sama

print("\nPre-processing selesai!")
print("File berikut telah dibuat: roberta_embeddings.npy, roberta_knn_index.joblib, labels_roberta.csv")