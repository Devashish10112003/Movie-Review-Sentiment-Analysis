# 🎬 Movie Review Sentiment Analysis (IMDB)

This project applies deep learning techniques to analyze and classify movie reviews from the IMDB dataset as either **positive** or **negative**. The system leverages advanced Natural Language Processing (NLP) techniques and neural network architectures—specifically **RNN**, **LSTM**, and a **hybrid RNN + LSTM** model—for robust sentiment analysis.

---

## 📌 Project Objectives

- Build a sentiment analysis system that classifies movie reviews as **positive** or **negative**.
- Compare performance across multiple models:
  - Multinomial Naive Bayes (baseline)
  - LSTM-based deep learning model
  - RNN (stacked)
  - A hybrid model combining LSTM and RNN
- Use the **IMDB 50K Movie Review dataset** for training and evaluation.

---

## 🧠 Models Used

### 1. **Naive Bayes (Baseline)**
- Uses TF-IDF vectorization.
- Fast and interpretable.
- Serves as a performance baseline.

### 2. **LSTM (Long Short-Term Memory)**
- Handles long-range dependencies in text.
- Achieved **~89.2% accuracy**.
- Best performance among all models.

### 3. **Stacked RNN**
- Two-layer SimpleRNN architecture.
- Struggles with long-term dependencies.
- Achieved **~70.6% accuracy**.

### 4. **LSTM + RNN Hybrid**
- Combines LSTM's memory with RNN's sequence modeling.
- Good balance of precision and recall.
- Achieved **~86.4% accuracy**.

---

## 📊 Dataset

- **Source**: [IMDB 50K Movie Reviews - Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size**: 50,000 reviews (25K for training, 25K for testing)
- **Classes**:
  - `1` = Positive
  - `0` = Negative

---

## 🧪 Preprocessing Steps

- Lowercasing
- Removing punctuation and stopwords
- Lemmatization
- Tokenization & padding (for LSTM/RNN)
- TF-IDF vectorization (for Naive Bayes)

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **TensorFlow / Keras**
- **Scikit-learn**
- **NLTK**
- **Jupyter Notebook**

---

## 📈 Results Summary

| Model              | Accuracy  | Notes |
|-------------------|-----------|-------|
| Naive Bayes        | ~83%      | Fast baseline |
| LSTM               | **89.19%**| Best overall |
| Stacked RNN        | 70.59%    | Poor performance on long texts |
| LSTM + RNN Hybrid  | 86.38%    | Good balance, fast convergence |

---

## 📂 Project Structure

