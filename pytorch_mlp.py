# PyTorch + MLP （多层感知器，是一种最基础的神经网络）
# IMDb 文本首先经过 TF-IDF 转换为 15000 维特征向量，然后输入两层 MLP。第一层 Linear 将特征映射到 128 维隐藏层，通过 ReLU 引入非线性，第二层 Linear 输出两个类别得分。使用 CrossEntropyLoss 计算预测与真实标签的误差，通过 loss.backward() 计算梯度，再利用 Adam 优化器的 optimizer.step() 更新网络参数，经过多个 epoch 迭代后完成情感分类训练。


### 导入库
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


### 加载 IMDb 做DataFrame

dataset = load_dataset("imdb")

train_df = pd.DataFrame({
    "text": dataset["train"]["text"],
    "label": dataset["train"]["label"]
})

test_df = pd.DataFrame({
    "text": dataset["test"]["text"],
    "label": dataset["test"]["label"]
})

### 抽样，降低规模

train_df = train_df.sample(10000, random_state=42)
test_df = test_df.sample(2000, random_state=42)

### TF-IDF 向量化

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

        self.relu = nn.ReLU()  # 激活函数   增加非线性能力

        self.fc2 = nn.Linear(
            128,
            2
        )

    def forward(self, x):     #数据怎么流动

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

    loss.backward() # 计算梯度

    optimizer.step() # 根据梯度更新参数

    print(
        f"Epoch {epoch+1}, Loss={loss.item():.4f}"
    )
    # print(model)

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