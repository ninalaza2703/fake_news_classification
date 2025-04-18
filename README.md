# fake_news_classification
This project explores different neural and hybrid methods for **fake news classification** using text data from news articles. The goal is to classify articles as **real or fake** based on their **text and title**.

---

## Overview of Models
### 1️⃣ Word2Vec + LSTM
- Used pretrained Google News Word2Vec (300d)
- Tokenized, lemmatized, and truncated input text and titles
- Dual LSTM model (title and text processed separately)
- Built and trained with PyTorch
  
### 2️⃣ GloVe + LSTM
- Used spaCy's `en_core_web_lg` (GloVe.840B.300d)
- Same architecture and processing as Word2Vec
  
### 3️⃣ BERT + CatBoost
- Combined `title + text` into one input string
- Used `bert-base-uncased` to extract token embeddings
- Fed embeddings into a CatBoostClassifier
- Predictions are probabilistic and thresholded at 0.5

  ###  Requirements

- Python 3.12
- PyTorch
- Hugging Face Transformers (BERT)
- CatBoost
- Gensim (Word2Vec)
- spaCy + `en_core_web_lg` (GloVe)
- scikit-learn, pandas, matplotlib, tqdm


### Dataset

The dataset used in this project was from Kaggle:  
[Fake News Classification Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
