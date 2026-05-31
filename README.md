# IMDb Sentiment Analysis

## Overview

This project performs binary sentiment classification
on IMDb movie reviews using TF-IDF features and
Logistic Regression.

The goal is to classify reviews as positive or negative.

---

## Dataset

Source:
HuggingFace IMDb Dataset

Training Samples: 10,000

Testing Samples: 2,000

Labels:

- 0 = Negative
- 1 = Positive

---

## Model Pipeline

Text Reviews
→
TF-IDF Vectorization
↓
Logistic Regression
↓
Prediction
↓
Evaluation

---

## Feature Engineering

TF-IDF

- max_features=15000
- ngram_range=(1,2)
- stop_words='english'

---

## Model

Logistic Regression

- solver='liblinear'
- max_iter=2000
- C=2.0

---

## Results

Accuracy: 87%

Metrics:

- Precision
- Recall
- F1-score

---

## Visualization

Confusion Matrix

---

## Future Improvements

- Support LSTM
- Support BERT
- Hyperparameter Tuning
- Error Analysis

---

## Tech Stack

Python
Scikit-learn
Pandas
Matplotlib
Seaborn
