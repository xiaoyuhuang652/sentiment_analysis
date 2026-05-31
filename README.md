# Sentiment Analysis

## Overview

This project performs binary sentiment classification
on IMDb movie reviews using TF-IDF features and
multiple machine learning models.

The goal is to classify reviews as positive or negative
and compare the performance of different classifiers.
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
→ TF-IDF Vectorization
→ Model Training (Logistic Regression / SVM / Naive Bayes)
→ Prediction
→ Evaluation
---

## Feature Engineering

TF-IDF

- max_features=15000
- ngram_range=(1,2)
- stop_words='english'

---

## Models

### Logistic Regression

- solver='liblinear'
- max_iter=2000
- C=2.0

### Linear SVM

- LinearSVC()

### Naive Bayes

- MultinomialNB()


## Results

The following models were evaluated on the sampled
IMDb dataset:

| Model | Accuracy |
|------|---------|
| Logistic Regression | 87.0% |
| Linear SVM | 85.6% |
| Naive Bayes | 83.7% |


---

## Experiments

Three traditional machine learning models were compared
using the same TF-IDF feature representation.

Results show that Logistic Regression achieved the best
performance on the sampled IMDb dataset, while Linear SVM
produced competitive results. Naive Bayes achieved lower
accuracy due to its strong independence assumptions.


## Visualization

Confusion Matrix


![Confusion Matrix](results/confusion_matrix.png)
---

## Future Improvements

- Hyperparameter tuning for existing models
- Implement LSTM-based sentiment classification
- Fine-tune BERT for sentiment analysis
- Conduct error analysis on misclassified reviews
- Compare traditional ML methods with deep learning approaches
---

## Project Structure

```text
sentiment_analysis/
│
├── main.py
├── README.md
├── results/
│   └── confusion_matrix.png
└── requirements.txt
```

## Tech Stack

- Python
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```


## Conclusion

Among the tested traditional machine learning models,
Logistic Regression achieved the best performance
(87.0%) on the sampled IMDb dataset.

Linear SVM achieved competitive performance,
while Naive Bayes showed lower accuracy due to
its strong feature independence assumption.