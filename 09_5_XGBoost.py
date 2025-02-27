import xgboost as xgb
import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 生成随机分类数据集
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=3, n_classes=2, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据转换为 DMatrix 格式，这是 XGBoost 的内部数据格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置 XGBoost 模型的超参数
params = {
    'objective': 'binary:logistic',  # 二分类任务
    'max_depth': 3,                  # 树的最大深度
    'eta': 0.1,                      # 学习率
    'subsample': 0.8,                # 样本采样比例
    'colsample_bytree': 0.8,         # 特征采样比例
    'eval_metric': 'logloss'         # 评估指标
}

# 训练模型
num_round = 100  # 迭代次数
bst = xgb.train(
    params,
    dtrain,
    num_round,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=10  # 早停机制，防止过拟合
)

# 进行预测
y_pred = bst.predict(dtest)
y_pred_binary = [round(value) for value in y_pred]  # 将概率转换为二分类结果

# 评估模型
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy:.4f}")

# 打印分类报告
print("Classification Report:")
print(classification_report(y_test, y_pred_binary))

# 打印混淆矩阵
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_binary))
