import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


# 定义决策树节点
class TreeNode:
    # feature_idx 特征索引 threshold 划分阈值 left左子树，为None表示无左子树 class_label叶节点的标签类型
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, class_label=None):
        self.feature_idx = feature_idx  # 用于划分的特征索引
        self.threshold = threshold  # 划分的阈值
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.class_label = class_label  # 叶节点的类别标签


# 定义决策树
class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.n_classes = len(torch.unique(y))
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_classes, counts = torch.unique(y, return_counts=True)  # 返回y中只出现一次的元素+其真实数量

        # 如果只有一个类别，返回叶节点
        if len(unique_classes) == 1:
            return TreeNode(class_label=unique_classes[0])

        # 如果达到最大深度，返回叶节点
        if depth >= self.max_depth:
            majority_class = unique_classes[counts.argmax()]  # 找到张量上最大的索引以代表这个叶节点
            return TreeNode(class_label=majority_class)

        # 随机选择特征
        feature_idx = np.random.randint(0, n_features)
        # 随机选择阈值
        threshold = np.random.choice(X[:, feature_idx])

        # 划分数据集
        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        # 递归构建左右子树
        left = self._build_tree(X_left, y_left, depth + 1)
        right = self._build_tree(X_right, y_right, depth + 1)

        return TreeNode(feature_idx, threshold, left, right)

    def predict(self, X):
        return torch.tensor([self._predict(x, self.tree) for x in X])

    def _predict(self, x, tree):
        if tree.class_label is not None:  # 叶节点
            return tree.class_label
        if x[tree.feature_idx] <= tree.threshold:
            return self._predict(x, tree.left)
        else:
            return self._predict(x, tree.right)


# 定义随机森林
class RandomForest(nn.Module):
    def __init__(self, n_estimators, max_depth):
        super(RandomForest, self).__init__()
        self.n_estimators = n_estimators  # n_estimators指的是随机森林中决策树的数量
        self.max_depth = max_depth
        #  [DecisionTree(max_depth) for _ in range(n_estimators)]  ：
        #  • 这是一个列表推导式，用于生成一个包含多个决策树实例的列表。
        #  •   n_estimators   是随机森林中决策树的数量。
        #  for _ in range(n_estimators)   表示循环   n_estimators   次，每次创建一个决策树实例。
        #  •   _   是一个占位符变量，表示循环变量在逻辑上没有被使用。
        self.trees = nn.ModuleList([DecisionTree(max_depth) for _ in range(n_estimators)])

    def fit(self, X, y):
        for tree in self.trees:
            # 随机选择样本子集
            indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            X_sample, y_sample = X[indices], y[indices]
            tree.fit(X_sample, y_sample)

    def predict(self, X):
        predictions = torch.stack([tree.predict(X) for tree in self.trees])
        # 通过多数投票确定最终预测
        return torch.mode(predictions, dim=0).values


# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 创建随机森林模型
model = RandomForest(n_estimators=10, max_depth=3)  # n_estimators指的是随机森林中决策树的数量

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test.numpy(), y_pred.numpy())
print(f'Accuracy on test set: {accuracy:.4f}')

# 上边是自己实现的决策树，实际上scikit-learn已经为我们提供了直接的接口


# 生成随机分类数据集
# 参数可以根据需要调整
# X, y = make_classification(n_samples=1000,  # 数据集中的样本数量
#                            n_features=20,   # 每个样本的特征数量
#                            n_informative=2, # 有信息量的特征数量
#                            n_redundant=10,  # 冗余特征的数量
#                            n_classes=2,     # 类别数量
#                            random_state=42) # 随机种子，保证结果可复现
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 创建随机森林分类器
# # 参数可以根据需要调整
# clf = RandomForestClassifier(n_estimators=100,  # 决策树的数量
#                              max_depth=10,     # 每棵树的最大深度
#                              random_state=42)  # 随机种子
#
# # 训练模型
# clf.fit(X_train, y_train)
#
# # 在测试集上进行预测
# y_pred = clf.predict(X_test)
#
# # 评估模型性能
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy on test set: {accuracy:.4f}')
#
# # 打印分类报告和混淆矩阵
# print("Classification Report:")
# print(classification_report(y_test, y_pred))
#
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
