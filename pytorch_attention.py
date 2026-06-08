# attention_text_classification.py

import torch
import torch.nn as nn
import torch.nn.functional as F
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
# 1. 数据加载
# =====================
dataset = load_dataset("imdb")
texts = dataset["train"]["text"]
labels = dataset["train"]["label"]

random.seed(42)
data = list(zip(texts, labels))
data = random.sample(data, 5000)   # 小规模训练

texts, labels = zip(*data)

# =====================
# 2. tokenizer + vocab
# =====================
def tokenize(text):
    return text.lower().split()

counter = Counter()
for t in texts:
    counter.update(tokenize(t))

vocab_size = 5000
most_common = counter.most_common(vocab_size - 2)

itos = ["<PAD>", "<UNK>"] + [w for w, _ in most_common]
stoi = {w: i for i, w in enumerate(itos)}

def encode(text):
    return torch.tensor([stoi.get(w, 1) for w in tokenize(text)], dtype=torch.long)

X = [encode(t) for t in texts]
y = torch.tensor(labels)

# =====================
# 3. train/val split
# =====================
train_idx, val_idx, train_y, val_y = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================
# 4. DataLoader
# =====================
MAX_LEN = 200
batch_size = 32

def collate(batch):
    x, y = zip(*batch)

    x = [i[:MAX_LEN] for i in x]
    x = pad_sequence(x, batch_first=True, padding_value=0)

    y = torch.tensor(y)
    return x, y

train_loader = DataLoader(list(zip(train_idx, train_y)), batch_size=batch_size, shuffle=True, collate_fn=collate)
val_loader = DataLoader(list(zip(val_idx, val_y)), batch_size=batch_size, shuffle=False, collate_fn=collate)

# =====================
# 5. Self-Attention 模型
# =====================
class AttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        self.ffn = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.ReLU(),
                    nn.Linear(embed_dim * 2, embed_dim)
                    )

        self.fc = nn.Linear(embed_dim, num_classes)


    # def forward(self, x):
    #     # x: [batch, seq_len]
    #     emb = self.embedding(x)  # [batch, seq_len, embed_dim]
    #
    #     Q = self.W_q(emb)
    #     K = self.W_k(emb)
    #     V = self.W_v(emb)
    #
    #     # attention score
    #     scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5) # [batch, seq_len, seq_len]
    #     attn = F.softmax(scores, dim=-1)
    #
    #     out = torch.bmm(attn, V)  # [batch, seq_len, embed_dim]
    #
    #     # pooling
    #     out = out.mean(dim=1)  # [batch, embed_dim]
    #     self.dropout = nn.Dropout(0.3)
    #
    #     logits = self.fc(out)
    #     return logits

    def forward(self, x):
        emb = self.embedding(x)  # [batch, seq_len, embed_dim]

        Q = self.W_q(emb)
        K = self.W_k(emb)
        V = self.W_v(emb)

        # attention score
        scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)

        # ===== 这里加 mask =====
        mask = (x != 0).unsqueeze(1)  # [batch, 1, seq_len]
        scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)

        # out = torch.bmm(attn, V)  # [batch, seq_len, embed_dim]
        #
        # # pooling
        # out = out.mean(dim=1)  # [batch, embed_dim]


        # ===== Attention Pooling =====
        weights = torch.softmax(scores.mean(dim=1), dim=1)  # [batch, seq_len]

        out = torch.sum(weights.unsqueeze(-1) * V, dim=1)  # [batch, embed_dim]

        # ===== 加 FFN =====
        out = self.ffn(out)  # [batch, embed_dim]

        logits = self.fc(out)
        return logits


# =====================
# 6. 模型 / loss / optimizer
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AttentionClassifier(vocab_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =====================
# 7. 训练
# =====================
train_losses = []
val_accs = []

best_acc = 0
patience = 3
counter = 0

for epoch in range(10):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # =====================
    # validation
    # =====================
    model.eval()
    preds, golds = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().tolist())
            golds.extend(y.tolist())

    acc = accuracy_score(golds, preds)

    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

    train_losses.append(avg_loss)
    val_accs.append(acc)

    # early stopping
    if acc > best_acc:
        best_acc = acc
        counter = 0
        torch.save(model.state_dict(), "best_attention.pth")
        print("Best model saved.")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered!")
            break

# =====================
# 8. test（这里用val代替简化）
# =====================
print("\nFinal Best Val Accuracy:", best_acc)

# =====================
# 9. 曲线图
# =====================
plt.plot(train_losses, label="Loss")
plt.plot(val_accs, label="Val Acc")
plt.legend()
plt.title("Self-Attention Training Curve")
plt.savefig("results/Attention training_curves.png")
plt.show()