# =====================
# mini_bert_finetune_with_curve.py
# =====================

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
from torch.optim import AdamW

# =====================
# 1. 配置
# =====================
MODEL_NAME = "prajjwal1/bert-mini"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("results", exist_ok=True)

# =====================
# 2. 加载 IMDB 数据
# =====================
dataset = load_dataset("imdb")
texts = dataset["train"]["text"]
labels = dataset["train"]["label"]

# 只取一小部分调试
texts, labels = texts[:15000], labels[:15000]

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# =====================
# 3. Tokenizer
# =====================
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def encode(texts):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

train_enc = encode(train_texts)
val_enc = encode(val_texts)

train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)

# =====================
# 4. Dataset & DataLoader
# =====================
class IMDBDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

train_dataset = IMDBDataset(train_enc, train_labels)
val_dataset = IMDBDataset(val_enc, val_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# =====================
# 5. 加载预训练 Mini-BERT
# =====================
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)

# =====================
# 6. 训练循环 + 记录曲线
# =====================
train_losses = []
val_accs = []
best_acc = 0
patience = 3
counter = 0

for epoch in range(EPOCHS):
    # 训练
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # 验证
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1)
            preds.extend(pred.cpu().tolist())
            golds.extend(labels.cpu().tolist())
    acc = accuracy_score(golds, preds)
    val_accs.append(acc)

    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        counter = 0
        torch.save(model.state_dict(), "results/best_mini_bert.pth")
        print("Best model saved.")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered!")
            break

print("\nFinal Best Val Accuracy:", best_acc)

# =====================
# 7. 绘制训练曲线
# =====================
plt.figure(figsize=(8,5))
plt.plot(range(1,len(train_losses)+1), train_losses, label="Train Loss")
plt.plot(range(1,len(val_accs)+1), val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Mini-BERT Training Curve")
plt.legend()
plt.grid(True)
plt.savefig("results/MiniBERT_training_curve.png")
plt.show()