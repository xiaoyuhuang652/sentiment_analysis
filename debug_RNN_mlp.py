# debug_rnn_mlp.py
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import random
from collections import Counter

# =====================
# 1. 小规模数据
# =====================
dataset = load_dataset("imdb")
texts = dataset["train"]["text"][:500]
labels = dataset["train"]["label"][:500]

tokenized = [t.lower().split() for t in texts]
all_tokens = [token for doc in tokenized for token in doc]
vocab = {w:i+2 for i,(w,_) in enumerate(Counter(all_tokens).most_common(1000))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1
vocab_size = len(vocab)

def text_to_seq(doc):
    seq = [vocab.get(t,1) for t in doc]
    return torch.tensor(seq, dtype=torch.long)

X = [text_to_seq(doc) for doc in tokenized]
y = torch.tensor(labels, dtype=torch.long)

# =====================
# 2. DataLoader
# =====================
batch_size = 4
def collate_batch(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return texts_padded, labels

train_loader = DataLoader(list(zip(X, y)), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

# =====================
# 3. 定义 RNN + MLP
# =====================
class DebugRNNMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        print("Input shape:", x.shape)  # [batch, seq_len]

        emb = self.embedding(x)
        print("Embedding shape:", emb.shape)  # [batch, seq_len, embed_dim]

        out, hidden = self.rnn(emb)
        print("RNN output shape:", out.shape)    # [batch, seq_len, hidden_dim]
        print("RNN hidden shape:", hidden.shape) # [1, batch, hidden_dim]

        final = hidden[-1]
        print("Final hidden:", final.shape)      # [batch, hidden_dim]

        logits = self.fc(final)
        print("Logits shape:", logits.shape)      # [batch, 2]
        return logits

model = DebugRNNMLP(vocab_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# =====================
# 4. 小规模训练循环
# =====================
for epoch in range(1):
    print(f"\nEpoch {epoch + 1}")
    model.train()
    for i, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(X_batch)
        print("Outputs shape:", outputs.shape)
        print("First sample output:", outputs[0])

        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        print(f"Batch {i + 1} loss: {loss.item():.4f}")

        if i >= 2:  # 只看前几个 batch
            break