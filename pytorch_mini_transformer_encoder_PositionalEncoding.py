# mini_transformer_encoder_text_classification.py

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
import os

# =====================
# 1. 数据加载
# =====================
dataset = load_dataset("imdb")
texts = dataset["train"]["text"]
labels = dataset["train"]["label"]

random.seed(42)
data = list(zip(texts, labels))
data = random.sample(data, 5000)
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
MAX_LEN = 400
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
# 5. Multi-head Attention
# =====================
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.size()

        Q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

# =====================
# 6. Transformer Block
# =====================
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1, ffn_expansion=4):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_expansion),
            nn.ReLU(),
            nn.Linear(embed_dim * ffn_expansion, embed_dim)
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.attn(x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x




# =====================
# 5.1 Position Encoding
# =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x


# # =====================
# # 7. Mini Transformer
# # =====================
# class MiniTransformerClassifier(nn.Module):
#     def __init__(self, vocab_size, embed_dim=128, num_layers=2, num_heads=4, num_classes=2, dropout=0.1):
#         super().__init__()
#
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
#
#         self.layers = nn.ModuleList([
#             TransformerEncoderBlock(embed_dim, num_heads, dropout)
#             for _ in range(num_layers)
#         ])
#
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(embed_dim, num_classes)
#
#     def forward(self, x):
#         B, T = x.size()
#
#         # ===== mask（必须在embedding之前做）=====
#         pad_mask = (x != 0)  # [B,T]
#
#         # ===== embedding =====
#         x = self.embedding(x)
#
#         # ===== CLS token =====
#         cls = self.cls_token.expand(B, -1, -1)
#         x = torch.cat([cls, x], dim=1)  # [B,T+1,E]
#
#         # CLS + pad mask
#         cls_mask = torch.ones(B, 1, device=x.device).bool()
#         mask = torch.cat([cls_mask, pad_mask], dim=1)
#
#         # ===== encoder =====
#         for layer in self.layers:
#             x = layer(x, mask)
#
#         out = x[:, 0]  # CLS token
#         out = self.dropout(out)
#         return self.fc(out)

# =====================
# 7. Mini Transformer (修改forward加入PE)
# =====================
class MiniTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_layers=2, num_heads=4, num_classes=2, dropout=0.1, max_len=500):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)  # 新增PE

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, T = x.size()

        # ===== mask（必须在embedding之前做）=====
        pad_mask = (x != 0)  # [B,T]

        # ===== embedding =====
        x = self.embedding(x)

        # ===== CLS token =====
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B,T+1,E]

        # ===== Position Encoding =====
        x = self.pos_encoding(x)

        # ===== mask =====
        cls_mask = torch.ones(B, 1, device=x.device).bool()
        mask = torch.cat([cls_mask, pad_mask], dim=1)  # [B,T+1]

        # ===== encoder =====
        for layer in self.layers:
            x = layer(x, mask)

        out = x[:, 0]  # CLS token
        out = self.dropout(out)
        return self.fc(out)

# =====================
# 8. train setup
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniTransformerClassifier(vocab_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# =====================
# 9. train loop
# =====================
train_losses = []
val_accs = []

best_acc = 0
patience = 3
counter = 0

os.makedirs("results", exist_ok=True)

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

    model.eval()
    preds, golds = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().tolist())
            golds.extend(y.tolist())

    acc = accuracy_score(golds, preds)

    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

    train_losses.append(avg_loss)
    val_accs.append(acc)

    if acc > best_acc:
        best_acc = acc
        counter = 0
        torch.save(model.state_dict(), "results/best_mini_transformer.pth")
        print("Best model saved.")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered!")
            break

print("\nFinal Best Val Accuracy:", best_acc)

# =====================
# 10. curve
# =====================
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_accs, label="Val Acc")
plt.legend()
plt.title("Mini Transformer Training Curve")
plt.savefig("results/MiniTransformer_training_curve.png")
plt.show()