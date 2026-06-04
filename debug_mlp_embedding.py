# Debug 版 Embedding + MLP
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# =====================
# 1. 加载 IMDb 数据
# =====================
dataset = load_dataset("imdb")

texts = dataset['train']['text'][:1000]  # 小规模调试
labels = dataset['train']['label'][:1000]

# 简单的词表构建
from collections import Counter
tokenized_texts = [t.lower().split() for t in texts]
all_tokens = [token for doc in tokenized_texts for token in doc]
vocab = {w:i+1 for i,(w,c) in enumerate(Counter(all_tokens).most_common(1000))}  # 留 0 给 padding
vocab_size = len(vocab) + 1

# 文本转索引
max_len = 20
def text_to_seq(tokens):
    seq = [vocab.get(t,0) for t in tokens][:max_len]
    if len(seq) < max_len:
        seq += [0]*(max_len - len(seq))
    return seq

X = torch.tensor([text_to_seq(doc) for doc in tokenized_texts], dtype=torch.long)
y = torch.tensor(labels, dtype=torch.long)

# =====================
# 2. 划分训练集和验证集
# =====================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=4, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=4)

# =====================
# 3. 定义 Embedding + MLP
# =====================
class EmbeddingMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # x: [batch_size, seq_len]
        emb = self.embedding(x)
        print("Embedding output shape:", emb.shape)
        print("Embedding first sample (first 3 tokens):", emb[0,:3,:])

        # mean pooling
        pooled = emb.mean(dim=1)
        print("Pooled shape:", pooled.shape)
        print("Pooled first sample:", pooled[0])

        out = self.fc1(pooled)
        out = self.relu(out)
        out = self.fc2(out)
        print("MLP output shape:", out.shape)
        print("MLP first sample output:", out[0])
        return out

model = EmbeddingMLP(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# =====================
# 4. 小规模训练循环
# =====================
for epoch in range(2):
    print(f"\nEpoch {epoch+1}")
    model.train()
    for i, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        print(f"Batch {i+1} loss: {loss.item():.4f}")
        if i >= 2:  # 只看前几个 batch
            break