# Sentiment Analysis with RoBERTa + k-NN
This repository is created for the NoLimit Data Scientist Technical Test. It demonstrates a sentiment classification system that uses RoBERTa embeddings and a k-Nearest Neighbors (k-NN) search approach.
### 👉 [Live Demo on Streamlit Cloud](https://nolimit-ds-test-imamhatrisekaputra.streamlit.app/)

## 📊 Project Overview
Task: Sentiment Classification (Positive / Negative / Neutral).
Method:
- Convert text into embeddings using RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest).
- Build a k-NN index over the dataset embeddings.
- For a new input, retrieve nearest neighbors and predict sentiment via majority voting.

App: Implemented with Streamlit, providing an interactive interface for real-time sentiment analysis.

## 📂 Dataset

Source: [Sp1786/multiclass-sentiment-analysis-dataset](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset).
Subset Used: Sp1786-5000-data.csv containing ~5,000 samples.
Labels: positive, negative, neutral.
License: Refer to the Hugging Face dataset page for usage rights.

## 🤖 Model
Model Name: [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
Description:
- RoBERTa-base trained on ~124M tweets (2018–2021).
- Finetuned for sentiment analysis using the TweetEval benchmark.
- Labels: 0 → Negative, 1 → Neutral, 2 → Positive.

Reference Paper: [TimeLMs: Diachronic Language Models](https://arxiv.org/abs/2202.03829)
GitHub Repo: [TimeLMs official repository](https://github.com/cardiffnlp/timelms)

## ⚙️ Setup Instructions
#### 1. Clone Repository

```sh
git clone https://github.com/Imamh28/nolimit-ds-test-Imam_Hatris_Ekaputra.git
```
```sh
cd nolimit-ds-test-Imam_Hatris_Ekaputra
```
#### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

#### 🧪 Local Verification (Quick Test)
Use the small dataset local_verification.csv for quick verification without building the full index.
Run: 
```sh
python local_verification.py
```
#### 3. Preprocess Full Dataset (Build Embeddings + k-NN Index)
```sh
python preprocess.py
```
#### 4. Run Streamlit App
```sh
python preprocess.py
```

## 🚀 Example Prediction (Streamlit App)
Input:
```sh
This product is absolutely wonderful, I love it!
```
Output:
```sh
Predicted Sentiment: Positive (Confidence: 0.80)
```

