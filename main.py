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

from sklearn.naive_bayes import MultinomialNB


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


# ===== Prediction & Evaluation =====

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ⭐ 新增（你要加的）
print("\n===== Per-class Performance =====")    # 这部分之后再仔细斟酌，问题不大
print("This shows precision/recall/F1 for each class")
report_text = classification_report(y_test, y_pred)
print(report_text)

# 简单用正则或字符串解析每行获取每个 class 的 precision/recall/F1



# ===== 6. Train SVM Model =====
svm_model = LinearSVC()

svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)

print(f"SVM Accuracy: {svm_accuracy:.3f}")



# ===== 7. Train Naive Bayes Model =====

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

nb_accuracy = accuracy_score(y_test, nb_pred)

print(f"Naive Bayes Accuracy: {nb_accuracy:.3f}")





# ===== 8. Confusion Matrix Visualization =====

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




# ===== 8. Error Analysis =====

# import numpy as np
#
# print("\n===== Error Analysis (Logistic Regression) =====")
#
# # 找出预测错误的样本（基于 Logistic Regression）
# wrong_indices = np.where(y_pred != y_test)[0]
#
# print(f"Total wrong predictions: {len(wrong_indices)}")
#
# # 查看前10个错误样本
# for i in wrong_indices[:10]:
#
#     print("\n====================")
#     print("Text:", test_df.iloc[i]["text"])
#     print("True Label:", y_test.iloc[i])
#     print("Pred Label:", y_pred[i])


# ===== 8.5 Error Analysis Visualization =====

import numpy as np

print("\n===== Error Analysis Visualization =====")

wrong_indices = np.where(y_pred != y_test)[0]

long_text_errors = 0
negation_errors = 0
other_errors = 0

for i in wrong_indices:

    text = test_df.iloc[i]["text"].lower()

    # 长文本错误
    if len(text.split()) > 200:
        long_text_errors += 1

    # 否定词错误
    elif "not " in text:
        negation_errors += 1

    else:
        other_errors += 1


# ===== 画图 =====

error_types = ["Long Reviews", "Negation", "Others"]
counts = [long_text_errors, negation_errors, other_errors]

plt.figure(figsize=(6,4))

plt.bar(error_types, counts)

plt.title("Error Type Distribution (Logistic Regression)")
plt.ylabel("Number of Errors")

plt.savefig("results/error_analysis.png")

plt.show()


# 打印一下统计
print("\n===== Error Summary =====")
print("Long reviews errors:", long_text_errors)
print("Negation errors:", negation_errors)
print("Other errors:", other_errors)



# ===== 8.6 Misclassified Examples with Interpretation =====
print("\n===== Misclassified Examples with Interpretation =====")

for i in wrong_indices[:5]:

    text = test_df.iloc[i]["text"]
    true_label = y_test.iloc[i]
    pred_label = y_pred[i]

    print("\n----------------------")
    print("Text:", text[:300])
    print("True:", true_label, "Pred:", pred_label)

    # 简单解释（规则型）
    if "not " in text.lower():
        print("Possible reason: negation confusion")
    elif len(text.split()) > 200:
        print("Possible reason: long mixed sentiment text")
    else:
        print("Possible reason: ambiguous sentiment")





# ===== 9. Logistic Regression Hyperparameter Tuning =====

print("\n===== Logistic Regression Hyperparameter Tuning =====")
# 对 Logistic Regression 的 C 做了循环调参
c_values = [0.01, 0.1, 1.0, 2.0, 5.0, 10.0]

tuning_results = []

for c in c_values:

    temp_model = LogisticRegression(
        max_iter=2000,
        C=c,
        solver="liblinear"
    )

    temp_model.fit(X_train, y_train)

    temp_pred = temp_model.predict(X_test)

    temp_acc = accuracy_score(y_test, temp_pred)

    tuning_results.append([c, temp_acc])

    print(f"C={c:<5} Accuracy={temp_acc:.4f}")


# ===== Convert Results to DataFrame =====

tuning_df = pd.DataFrame(
    tuning_results,
    columns=["C", "Accuracy"]
)

print("\n===== Tuning Results Table =====")
print(tuning_df)

plt.figure(figsize=(6,4))

plt.plot(
    tuning_df["C"],
    tuning_df["Accuracy"],
    marker="o"
)

plt.xlabel("C Value")
plt.ylabel("Accuracy")
plt.title("Logistic Regression Hyperparameter Tuning")

plt.savefig("results/lr_tuning.png")

plt.show()






