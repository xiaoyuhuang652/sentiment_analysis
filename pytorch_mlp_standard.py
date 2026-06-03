# PyTorch + MLP（带 Dropout + Batch + Validation）
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# =====================
# 1. 加载 IMDb 数据
# =====================
dataset = load_dataset("imdb")

train_df = pd.DataFrame({
    "text": dataset["train"]["text"],
    "label": dataset["train"]["label"]
})

test_df = pd.DataFrame({
    "text": dataset["test"]["text"],
    "label": dataset["test"]["label"]
})

# =====================
# 2. 抽样，降低规模
# =====================
train_df = train_df.sample(10000, random_state=42)
test_df = test_df.sample(2000, random_state=42)

# =====================
# 3. TF-IDF 向量化
# =====================
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1,2),
    stop_words="english"
)

X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])

y_train = train_df["label"].values
y_test = test_df["label"].values

# =====================
# 4. 划分训练集和验证集
# =====================
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# 转换为 PyTorch Tensor
X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
X_val   = torch.tensor(X_val.toarray(), dtype=torch.float32)
X_test  = torch.tensor(X_test.toarray(), dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val   = torch.tensor(y_val, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)

# =====================
# 5. DataLoader（批训练）
# =====================
batch_size = 64

train_dataset = TensorDataset(X_train, y_train)
val_dataset   = TensorDataset(X_val, y_val)
test_dataset  = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =====================
# 6. 定义 MLP（加入 Dropout）
# =====================
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # 50% dropout
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

input_dim = X_train.shape[1]
model = MLP(input_dim)

# =====================
# 7. 损失函数和优化器
# =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =====================
# 8. 训练（带验证集）
# =====================
train_losses = []
val_accuracies = []


best_val_acc = 0

patience = 3

counter = 0



epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # =====================
    # 验证集
    # =====================
    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.tolist())
            val_labels.extend(y_batch.tolist())

    val_acc = accuracy_score(val_labels, val_preds)


    if val_acc > best_val_acc:

        best_val_acc = val_acc

        counter = 0

        torch.save(
            model.state_dict(),
            "results/best_model.pth"
        )

        print("Best model saved.")

    else:

        counter += 1

        print(
            f"Validation not improved for {counter} epoch(s)"
        )

        if counter >= patience:
            print(
                "\nEarly Stopping Triggered!"
            )

            break


    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    train_losses.append(epoch_loss)
    val_accuracies.append(val_acc)

# =====================
# 9. 测试集最终评估
# =====================
model.eval()
test_preds = []
test_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        test_preds.extend(preds.tolist())
        test_labels.extend(y_batch.tolist())

test_acc = accuracy_score(test_labels, test_preds)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")




### 绘制 Loss & Validation Accuracy 曲线

import matplotlib.pyplot as plt

# epochs_range = range(1, epochs + 1)

epochs_range = range(
    1,
    len(train_losses) + 1
)

plt.figure(figsize=(8,5))

# Loss 曲线
plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')

# Validation Accuracy 曲线
plt.plot(epochs_range, val_accuracies, label='Val Accuracy', marker='s')

plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Loss & Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("results/training_curves.png")
plt.show()