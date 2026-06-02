# PyTorch + MLP （多层感知器，是一种最基础的神经网络）

### 导入库
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


### 加载 IMDb

dataset = load_dataset("imdb")

train_df = pd.DataFrame({
    "text": dataset["train"]["text"],
    "label": dataset["train"]["label"]
})

test_df = pd.DataFrame({
    "text": dataset["test"]["text"],
    "label": dataset["test"]["label"]
})

train_df = train_df.sample(10000, random_state=42)
test_df = test_df.sample(2000, random_state=42)

### TF-IDF

vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1,2),
    stop_words="english"
)

X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])


### 转换为 PyTorch Tensor

X_train = torch.tensor(
    X_train.toarray(),
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test.toarray(),
    dtype=torch.float32
)

y_train = torch.tensor(
    train_df["label"].values,
    dtype=torch.long
)

y_test = torch.tensor(
    test_df["label"].values,
    dtype=torch.long
)


### 定义 MLP

class MLP(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.fc1 = nn.Linear(
            input_dim,
            128
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            128,
            2
        )

    def forward(self, x):

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        return x


### 创建模型

input_dim = X_train.shape[1]

model = MLP(input_dim)


### 损失函数和优化器

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


### 训练
epochs = 10

for epoch in range(epochs):

    outputs = model(X_train)

    loss = criterion(
        outputs,
        y_train
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch+1}, Loss={loss.item():.4f}"
    )


    ### 测试
    with torch.no_grad():
        outputs = model(X_test)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

    accuracy = accuracy_score(
        y_test.numpy(),
        predictions.numpy()
    )

    print(
        f"MLP Accuracy: {accuracy:.4f}"
    )