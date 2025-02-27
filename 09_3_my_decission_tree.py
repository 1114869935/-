import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np


class DecisionTree(nn.Module):
    def __init__(self, input_size, num_classes, depth=3):  # 输入大小，类型数，树深度
        super(DecisionTree, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.depth = depth

        self.layers = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                self.layers.append(nn.Linear(input_size, 2 * input_size))
            else:
                self.layers.append(nn.Linear(input_size * 2, input_size * 2))
        self.output_layer = nn.Linear(2 * input_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)
        return x


# 生成分类数据 make_classification是一个自己生成分类数据的函数
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

input_size=X_train.shape[1]
num_classes=len(np.unique(y_train))
model=DecisionTree(input_size,num_classes,depth=3)

critetion=nn.CrossEntropyLoss()#
optimizer=optim.Adam(model.parameters(),lr=0.01)

num_epochs=50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs=model(X_train)
    loss=critetion(outputs,y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    outputs=model(X_test)
    _,predicted=torch.max(outputs.data,1)
    accuracy= predicted.sum().item() / len(y_test)
    print(f'Accuracy on test set: {accuracy:.4f}')