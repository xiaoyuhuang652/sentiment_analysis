# pytorch_mlp_embedding.py
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

# =====================
# 1. 加载 IMDb 数据
# =====================
dataset = load_dataset("imdb")
train_texts = dataset["train"]["text"]
train_labels = dataset["train"]["label"]

test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

# =====================
# 2. 简单抽样
# =====================
import random
random.seed(42)

train_data = list(zip(train_texts, train_labels))
test_data = list(zip(test_texts, test_labels))

train_sample = random.sample(train_data, 10000)
test_sample = random.sample(test_data, 2000)

train_texts, train_labels = zip(*train_sample)
test_texts, test_labels = zip(*test_sample)

# =====================
# 3. 文本 -> Token 索引
# =====================
from collections import Counter

# 最简单的分词: 按空格
def tokenize(text):
    return text.lower().split()

# 构建词表
counter = Counter()
for text in train_texts:
    counter.update(tokenize(text))

vocab_size = 10000  # 取前 10000 个词
most_common = counter.most_common(vocab_size-2)  # 留两个给 <PAD> 和 <UNK>
itos = ["<PAD>", "<UNK>"] + [w for w, _ in most_common]
stoi = {w:i for i,w in enumerate(itos)}

def text_to_indices(text):
    return [stoi.get(w, 1) for w in tokenize(text)]  # 1=<UNK>

# 转换训练/测试文本
train_indices = [torch.tensor(text_to_indices(t), dtype=torch.long) for t in train_texts]
test_indices  = [torch.tensor(text_to_indices(t), dtype=torch.long) for t in test_texts]

train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels  = torch.tensor(test_labels, dtype=torch.long)

# =====================
# 4. 划分训练集和验证集
# =====================
train_idx, val_idx, train_lab, val_lab = train_test_split(
    train_indices, train_labels, test_size=0.2, random_state=42
)

# =====================
# 5. DataLoader
# =====================
batch_size = 64

def collate_batch(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)  # <PAD>=0

    # # 截断到最多50个token
    # texts_padded = texts_padded[:, :50]  #改最大长度max_len

    labels = torch.tensor(labels, dtype=torch.long)
    return texts_padded, labels

train_loader = DataLoader(list(zip(train_idx, train_lab)), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
val_loader   = DataLoader(list(zip(val_idx, val_lab)), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
test_loader  = DataLoader(list(zip(test_indices, test_labels)), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

# =====================
# 6. 定义 MLP（Embedding + Flatten + Dropout）
# =====================
class MLPEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim=50):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        # 平均池化
        pooled = embedded.mean(dim=1)  # [batch, embed_dim]
        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = MLPEmbedding(vocab_size=vocab_size)

# =====================
# 7. 损失和优化器
# =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =====================
# 8. 训练 + Validation + Early Stopping
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

    # 验证集
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
        torch.save(model.state_dict(), "results/best_model_embedding.pth")
        print("Best model saved.")
    else:
        counter += 1
        print(f"Validation not improved for {counter} epoch(s)")
        if counter >= patience:
            print("\nEarly Stopping Triggered!")
            break

    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Val Accuracy: {val_acc:.4f}")
    train_losses.append(epoch_loss)
    val_accuracies.append(val_acc)

# =====================
# 9. 测试集评估
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

# =====================
# 10. 绘制 Loss & Validation Accuracy
# =====================
epochs_range = range(1, len(train_losses)+1)
plt.figure(figsize=(8,5))
plt.plot(epochs_range, train_losses, label="Train Loss", marker='o')
plt.plot(epochs_range, val_accuracies, label="Val Accuracy", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Loss & Validation Accuracy (Embedding MLP)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/training_curves_embedding.png")
plt.show()