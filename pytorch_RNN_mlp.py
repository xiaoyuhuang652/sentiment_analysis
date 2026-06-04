# rnn_mlp.py
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import random
import matplotlib.pyplot as plt
from collections import Counter

# =====================
# 1. 数据加载和抽样
# =====================
dataset = load_dataset("imdb")
train_texts = dataset["train"]["text"]
train_labels = dataset["train"]["label"]
test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

random.seed(42)
train_data = list(zip(train_texts, train_labels))
test_data  = list(zip(test_texts, test_labels))

train_sample = random.sample(train_data, 10000)
test_sample  = random.sample(test_data, 2000)

train_texts, train_labels = zip(*train_sample)
test_texts, test_labels   = zip(*test_sample)

# =====================
# 2. 文本->索引
# =====================
def tokenize(text):
    return text.lower().split()

counter = Counter()
for text in train_texts:
    counter.update(tokenize(text))

vocab_size = 10000
most_common = counter.most_common(vocab_size-2)
itos = ["<PAD>", "<UNK>"] + [w for w,_ in most_common]
stoi = {w:i for i,w in enumerate(itos)}

def text_to_indices(text):
    return [stoi.get(w, 1) for w in tokenize(text)]

train_indices = [torch.tensor(text_to_indices(t), dtype=torch.long) for t in train_texts]
test_indices  = [torch.tensor(text_to_indices(t), dtype=torch.long) for t in test_texts]

train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels  = torch.tensor(test_labels, dtype=torch.long)

# =====================
# 3. 划分训练集和验证集
# =====================
train_idx, val_idx, train_lab, val_lab = train_test_split(
    train_indices, train_labels, test_size=0.2, random_state=42
)

# =====================
# 4. DataLoader
# =====================
batch_size = 64

def collate_batch(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return texts_padded, labels

train_loader = DataLoader(list(zip(train_idx, train_lab)), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
val_loader   = DataLoader(list(zip(val_idx, val_lab)), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
test_loader  = DataLoader(list(zip(test_indices, test_labels)), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

# =====================
# 5. 定义模型 (Embedding + RNN + Linear)
# =====================
class RNNMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim=50, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        emb = self.embedding(x)
        out, hidden = self.rnn(emb)
        final = hidden[-1]
        logits = self.fc(final)
        return logits

model = RNNMLP(vocab_size=vocab_size)

# =====================
# 6. 损失与优化器
# =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =====================
# 7. 训练 + 验证 + Early Stopping
# =====================
train_losses, val_accs = [], []
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
        torch.save(model.state_dict(), "results/best_rnn_model.pth")
        print("Best model saved.")
    else:
        counter += 1
        print(f"Validation not improved for {counter} epoch(s)")
        if counter >= patience:
            print("Early Stopping Triggered!")
            break

    print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Val Accuracy: {val_acc:.4f}")
    train_losses.append(epoch_loss)
    val_accs.append(val_acc)

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