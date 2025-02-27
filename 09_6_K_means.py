import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 以下是基于scikit-lean的K-means算法
# # 生成随机数据集
# X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
#
# # 数据标准化
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # 应用 K-Means 算法
# kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
# kmeans.fit(X_scaled)
#
# # 获取簇标签和簇中心
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_
#
# # 可视化结果
# plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
# plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X')
# plt.title('K-Means Clustering')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.show()

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)


class KMeans:

    def __init__(self, n_clusters, max_iter=100, random_state=None):
        self.n_clusters = n_clusters  # 簇的数量
        self.max_iter = max_iter  # 最大迭代次数
        self.random_state = random_state  # 随机选取中心
        self.centroids = None  # 中心位置
        self.labels = None  # 每个点的属于哪个簇

    def fit(self, X):
        np.random.seed(self.random_state)  # 生成一个随机种子
        # 随机在X中选择n_clusters个不重复（replace=False）的索引（可理解为数组标号），X可理解唯一个二维数组
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]  # 直接将这些数据作为样本的几个中心
        for _ in range(self.max_iter):  # 迭代次数
            self.labels = self._assign_clusters(X)  # 分配到簇
            old_centroids = self.centroids.copy()  # 保留旧的簇
            self.centroids = self._compute_centroids(X, self.labels)  # 计算新的簇中心
            if np.all(self.centroids == old_centroids):  # 如果所有老的簇和新的簇都一样，证明完美了，和冒泡排序类似
                break

    def _assign_clusters(self, X):  # 计算新的簇
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)  # 算距离
        # 分配到最近的簇
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X, labels):
        # 重新计算簇中心
        return np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])

    def predict(self, X):
        # 预测每个数据点的簇标签
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)



# 使用 K-Means 算法
kmeans = KMeans(n_clusters=4, max_iter=100, random_state=42)
kmeans.fit(X)

# 获取簇标签和簇中心
labels = kmeans.labels
centroids = kmeans.centroids

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')  # 数据点按簇标签着色
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X')  # 簇中心
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
