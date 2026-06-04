import torch
import torch.nn as nn


# ==================================
# 1. 构造一个假的 Batch
# ==================================

X_batch = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [4.0, 3.0, 2.0, 1.0]
])

print("\n========== Input ==========")
print(X_batch)

print("\nShape:")
print(X_batch.shape)


# ==================================
# 2. 定义 MLP
# ==================================

class DebugMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(
            4,
            3
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            0.5
        )

        self.fc2 = nn.Linear(
            3,
            2
        )

    def forward(self, x):

        print("\n========== FC1 ==========")

        x = self.fc1(x)

        print(x)
        print("Shape:", x.shape)

        print("\n========== ReLU ==========")

        x = self.relu(x)

        print(x)
        print("Shape:", x.shape)

        print("\n========== Dropout ==========")

        x = self.dropout(x)

        print(x)
        print("Shape:", x.shape)

        print("\n========== FC2 ==========")

        x = self.fc2(x)

        print(x)
        print("Shape:", x.shape)

        return x


# ==================================
# 3. 创建模型
# ==================================

model = DebugMLP()

print("\n========== Model ==========")
print(model)


# ==================================
# 4. Forward
# ==================================

output = model(X_batch)


# ==================================
# 5. 预测
# ==================================

print("\n========== Final Output ==========")
print(output)

preds = torch.argmax(
    output,
    dim=1
)

print("\n========== Prediction ==========")
print(preds)