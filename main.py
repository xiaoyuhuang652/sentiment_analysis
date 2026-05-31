# ===== 1. Import Libraries =====

from datasets import load_dataset
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.svm import LinearSVC




# ===== 2. Load IMDb Dataset =====

dataset = load_dataset("imdb")

train_df = pd.DataFrame({
    "text": dataset["train"]["text"],
    "label": dataset["train"]["label"]
})

test_df = pd.DataFrame({
    "text": dataset["test"]["text"],
    "label": dataset["test"]["label"]
})


# ===== 3. Sample Data =====

train_df = train_df.sample(10000, random_state=42)
test_df = test_df.sample(2000, random_state=42)


# ===== 4. Text Vectorization (TF-IDF) =====

vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])

y_train = train_df["label"]
y_test = test_df["label"]


# ===== 5. Train Logistic Regression Model =====

model = LogisticRegression(
    max_iter=2000,
    C=2.0,
    solver="liblinear"
)

model.fit(X_train, y_train)


# ===== 6. Prediction & Evaluation =====

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))




# ===== Train SVM Model =====
svm_model = LinearSVC()

svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)

print(f"SVM Accuracy: {svm_accuracy:.3f}")



# ===== 7. Confusion Matrix Visualization =====

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix - IMDb Sentiment")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# plt.savefig("confusion_matrix.png")


plt.savefig("results/confusion_matrix.png")
plt.show()