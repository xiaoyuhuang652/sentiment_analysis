# cnn_text_classification.py
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
from collections import Counter
import matplotlib.pyplot as plt

# =====================
# 1. 数据加载与抽样
# =====================
dataset = load_dataset("imdb")
train_texts = dataset["train"]["text"]
train_labels = dataset["train"]["label"]
test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

random.seed(42)
train_sample = random.sample(list(zip(train_texts, train_labels)), 5000)
test_sample = random.sample(list(zip(test_texts, test_labels)), 1000)

train_texts, train_labels = zip(*train_sample)
test_texts, test_labels = zip(*test_sample)


# =====================
# 2. 文本 -> 索引
# =====================
def tokenize(text):
    return text.lower().split()


counter = Counter()
for text in train_texts:
    counter.update(tokenize(text))

vocab_size = 5000
most_common = counter.most_common(vocab_size - 2)
itos = ["<PAD>", "<UNK>"] + [w for w, _ in most_common]
stoi = {w: i for i, w in enumerate(itos)}


def text_to_indices(text):
    return [stoi.get(w, 1) for w in tokenize(text)]


train_indices = [torch.tensor(text_to_indices(t), dtype=torch.long) for t in train_texts]
test_indices = [torch.tensor(text_to_indices(t), dtype=torch.long) for t in test_texts]

train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# =====================
# 3. 划分训练集和验证集
# =====================
train_idx, val_idx, train_lab, val_lab = train_test_split(
    train_indices, train_labels, test_size=0.2, random_state=42
)

# =====================
# 4. DataLoader + MAX_LEN 截断
# =====================
MAX_LEN = 200
batch_size = 32


def collate_batch(batch):
    texts, labels = zip(*batch)
    texts = [t[:MAX_LEN] if len(t) > MAX_LEN else t for t in texts]
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return texts_padded, labels


train_loader = DataLoader(list(zip(train_idx, train_lab)), batch_size=batch_size, shuffle=True,
                          collate_fn=collate_batch)
val_loader = DataLoader(list(zip(val_idx, val_lab)), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
test_loader = DataLoader(list(zip(test_indices, test_labels)), batch_size=batch_size, shuffle=False,
                         collate_fn=collate_batch)


# =====================
# 5. 定义 CNN 模型
# =====================
class CNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=2, kernel_sizes=[3, 4, 5], num_filters=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        emb = self.embedding(x)  # [batch, seq, embed_dim]
        emb = emb.permute(0, 2, 1)  # [batch, embed_dim, seq]
        conv_outs = [torch.relu(conv(emb)) for conv in self.convs]  # 每个: [batch, num_filters, L_out]
        pooled = [torch.max(c, dim=2)[0] for c in conv_outs]  # 全局 max pool → [batch, num_filters]
        out = torch.cat(pooled, dim=1)  # 拼接所有卷积核 → [batch, num_filters*len(kernel_sizes)]
        out = self.dropout(out)
        logits = self.fc(out)
        return logits


model = CNNTextClassifier(vocab_size=vocab_size)

# =====================
# 6. 损失与优化器
# =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =====================
# 7. 训练 + 验证 + Early Stopping
# =====================
train_losses = []
val_accuracies = []
best_val_acc = 0
patience = 3
counter = 0
epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # 验证
    model.eval()
    val_preds, val_labels_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.tolist())
            val_labels_list.extend(y_batch.tolist())

    val_acc = accuracy_score(val_labels_list, val_preds)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), "best_cnn.pth")
        print("Best model saved.")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered!")
            break

    print(f"Epoch {epoch + 1} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")
    train_losses.append(epoch_loss)
    val_accuracies.append(val_acc)

# =====================
# 8. 测试集评估
# =====================
model.eval()
test_preds, test_labels_list = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        test_preds.extend(preds.tolist())
        test_labels_list.extend(y_batch.tolist())

test_acc = accuracy_score(test_labels_list, test_preds)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

# =====================
# 9. 绘图
# =====================
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Val Accuracy", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Loss & Val Accuracy (CNN)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()