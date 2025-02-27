import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 生成随机数据集
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)


# 定义孤立树（Isolation Tree）
class IsolationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth  # 树的最大深度
        self.tree = None  # 孤立树的结构

    def fit(self, X, depth=0):
        """
        递归构建孤立树
        :param X: 当前节点的数据
        :param depth: 当前深度
        :return: 构建好的孤立树节点
        """
        if len(X) <= 1 or depth >= self.max_depth:
            # 如果数据点少于等于1个，或者达到最大深度，返回叶子节点
            return {'size': len(X)}

        # 随机选择一个特征
        feature_index = np.random.randint(0, X.shape[1])
        # 随机选择一个分裂点
        feature_value = np.random.uniform(X[:, feature_index].min(), X[:, feature_index].max())

        # 根据分裂点将数据分为左右子树
        left_indices = X[:, feature_index] < feature_value
        right_indices = X[:, feature_index] >= feature_value

        # 递归构建左右子树
        left_tree = self.fit(X[left_indices], depth + 1)
        right_tree = self.fit(X[right_indices], depth + 1)

        # 返回当前节点的信息
        return {
            'feature_index': feature_index,
            'feature_value': feature_value,
            'left': left_tree,
            'right': right_tree,
            'size': len(X)
        }

    def path_length(self, x, depth=0):
        """
        计算数据点 x 在孤立树中的路径长度
        :param x: 数据点
        :param depth: 当前深度
        :return: 路径长度
        """
        if 'size' in self.tree and 'feature_index' not in self.tree:
            # 如果是叶子节点，返回路径长度
            return depth + c(self.tree['size'])

        feature_index = self.tree['feature_index']
        feature_value = self.tree['feature_value']

        if x[feature_index] < feature_value:
            # 如果数据点小于分裂点，进入左子树
            return self.path_length(x, depth + 1) if 'left' in self.tree else depth + c(self.tree['size'])
        else:
            # 如果数据点大于等于分裂点，进入右子树
            return self.path_length(x, depth + 1) if 'right' in self.tree else depth + c(self.tree['size'])


# 定义孤立森林（Isolation Forest）
class IsolationForest:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators  # 孤立树的数量
        self.max_depth = max_depth  # 每棵树的最大深度
        self.trees = []  # 存储所有孤立树

    def fit(self, X):
        """
        训练孤立森林模型
        :param X: 训练数据
        """
        for _ in range(self.n_estimators):
            # 随机选择样本子集
            sample_indices = np.random.choice(len(X), size=len(X), replace=True)
            sample_X = X[sample_indices]
            # 构建孤立树
            tree = IsolationTree(max_depth=self.max_depth)
            tree.fit(sample_X)
            self.trees.append(tree)

    def predict(self, X):
        """
        预测数据点的异常分数
        :param X: 待预测数据
        :return: 异常分数
        """
        scores = []
        for x in X:
            # 计算每个数据点在所有孤立树中的平均路径长度
            path_lengths = [tree.path_length(x) for tree in self.trees]
            avg_path_length = np.mean(path_lengths)
            # 计算异常分数
            score = 2 ** (-avg_path_length / c(len(X)))
            scores.append(score)
        return scores


# 辅助函数：计算路径长度的期望值
def c(n):
    if n > 2:
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
    elif n == 2:
        return 1
    else:
        return 0


# 使用孤立森林进行异常检测
if __name__ == "__main__":
    # 生成随机数据集
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # 创建孤立森林模型
    clf = IsolationForest(n_estimators=100, max_depth=10)
    clf.fit(X)

    # 预测异常分数
    scores = clf.predict(X)

    # 可视化结果
    plt.scatter(X[:, 0], X[:, 1], c=scores, cmap='viridis')
    plt.colorbar(label='Anomaly Score')
    plt.title('Isolation Forest Anomaly Detection')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
