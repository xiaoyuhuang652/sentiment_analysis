import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import random
import matplotlib.pyplot as plt
from collections import Counter

# =====================
# 1. 数据加载
# =====================
dataset = load_dataset("imdb")

train_data = list(zip(dataset["train"]["text"], dataset["train"]["label"]))
test_data = list(zip(dataset["test"]["text"], dataset["test"]["label"]))

random.seed(42)
train_data = random.sample(train_data, 10000)
test_data = random.sample(test_data, 2000)

train_texts, train_labels = zip(*train_data)
test_texts, test_labels = zip(*test_data)

# =====================
# 2. tokenizer + vocab
# =====================
def tokenize(text):
    return text.lower().split()

counter = Counter()
for t in train_texts:
    counter.update(tokenize(t))

vocab_size = 20000
most_common = counter.most_common(vocab_size - 2)

itos = ["<PAD>", "<UNK>"] + [w for w, _ in most_common]
stoi = {w: i for i, w in enumerate(itos)}

def encode(text):
    return torch.tensor([stoi.get(w, 1) for w in tokenize(text)], dtype=torch.long)

train_x = [encode(t) for t in train_texts]
test_x = [encode(t) for t in test_texts]

train_y = torch.tensor(train_labels)
test_y = torch.tensor(test_labels)

# =====================
# 3. train/val split
# =====================
train_x, val_x, train_y, val_y = train_test_split(
    train_x, train_y, test_size=0.2, random_state=42
)

# =====================
# 4. DataLoader
# =====================
batch_size = 128

# def collate(batch):
#     x, y = zip(*batch)
#     x = pad_sequence(x, batch_first=True, padding_value=0)
#     y = torch.tensor(y)
#     return x, y
MAX_LEN = 200
def collate(batch):
    x, y = zip(*batch)

    # 截断或填充到 MAX_LEN
    x_padded = []
    for xi in x:
        if len(xi) > MAX_LEN:
            xi = xi[:MAX_LEN]  # 截断
        elif len(xi) < MAX_LEN:
            pad_len = MAX_LEN - len(xi)
            xi = torch.cat([xi, torch.zeros(pad_len, dtype=torch.long)])  # 补零
        x_padded.append(xi)

    x_padded = torch.stack(x_padded)  # [batch, MAX_LEN]
    y = torch.tensor(y)

    return x_padded, y


train_loader = DataLoader(list(zip(train_x, train_y)), batch_size=batch_size, shuffle=True, collate_fn=collate)
val_loader = DataLoader(list(zip(val_x, val_y)), batch_size=batch_size, shuffle=False, collate_fn=collate)
test_loader = DataLoader(list(zip(test_x, test_y)), batch_size=batch_size, shuffle=False, collate_fn=collate)

# =====================
# 5. GRU 模型
# =====================
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*2, 2)  # bidirectional

    def forward(self, x):
        emb = self.embedding(x)

        out, h_n = self.gru(emb)  # GRU 不返回 c_n
        final = out.mean(dim=1)   # mean pooling
       #final = h_n[-1]
        final = self.dropout(final)
        logits = self.fc(final)
        return logits


# =====================
# 6. 初始化
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GRUClassifier(vocab_size=vocab_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =====================
# 7. 训练 + 验证 + Early stopping
# =====================
train_losses = []
val_accs = []

best_acc = 0
patience = 3
counter = 0
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    train_loss = total_loss / len(train_loader.dataset)

    # =====================
    # validation
    # =====================
    model.eval()
    preds, labels_list = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = torch.argmax(out, dim=1)

            preds.extend(pred.cpu().tolist())
            labels_list.extend(y.cpu().tolist())

    val_acc = accuracy_score(labels_list, preds)

    train_losses.append(train_loss)
    val_accs.append(val_acc)

    # early stopping
    if val_acc > best_acc:
        best_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), "best_lstm_model.pth")
        print("Best model saved.")
    else:
        counter += 1
        print(f"No improvement for {counter} epoch(s)")
        if counter >= patience:
            print("Early stopping triggered!")
            break

    print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

# =====================
# 8. test
# =====================
model.eval()
test_preds, test_labels = [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = torch.argmax(out, dim=1)

        test_preds.extend(pred.cpu().tolist())
        test_labels.extend(y.cpu().tolist())

test_acc = accuracy_score(test_labels, test_preds)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

# =====================
# 9. 画图
# =====================
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_accs, label="Val Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.title("GRU Training Curve")
plt.grid()
plt.show()