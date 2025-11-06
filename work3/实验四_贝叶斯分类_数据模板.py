"""
实验四：贝叶斯分类器

实验目标：
在数据集上应用贝叶斯规则进行分类，计算分类错误率，分析实验结果

数据已生成，请自行实现贝叶斯分类算法
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'STHeiti', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('out', exist_ok=True)

# ============================================================
# 数据生成
# ============================================================

print('='*60)
print('实验四：贝叶斯分类器')
print('='*60)

# 生成数据集1：高分离度
print('\n生成数据集1 (高分离度)...')
X1, y1 = make_classification(
    n_samples=500,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_classes=2,
    n_clusters_per_class=1,
    class_sep=2.0,
    random_state=42
)

# 生成数据集2：低分离度  
print('生成数据集2 (低分离度)...')
X2, y2 = make_classification(
    n_samples=500,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_classes=2,
    n_clusters_per_class=1,
    class_sep=0.5,
    random_state=42
)

# 划分训练集和测试集
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)

print(f'\n数据集1: 训练集{X1_train.shape[0]}样本, 测试集{X1_test.shape[0]}样本')
print(f'数据集2: 训练集{X2_train.shape[0]}样本, 测试集{X2_test.shape[0]}样本')

# ============================================================
# 实现贝叶斯分类器
# ============================================================

from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

class GaussianNaiveBayes:
    def __init__(self):
        self.means_ = None
        self.vars_ = None
        self.priors_ = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.means_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[idx] = X_c.mean(axis=0)
            self.vars_[idx] = X_c.var(axis=0)
            self.priors_[idx] = X_c.shape[0] / n_samples

    def predict(self, X):
        log_priors = np.log(self.priors_)
        log_likelihoods = []

        for idx in range(len(self.classes_)):
            mean = self.means_[idx]
            var = self.vars_[idx]
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var))
            log_likelihood -= 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            log_likelihoods.append(log_likelihood)

        log_likelihoods = np.array(log_likelihoods).T
        log_posteriors = log_likelihoods + log_priors
        return np.argmax(log_posteriors, axis=1)


def plot_decision_boundary(model, X_train, y_train, X_test, y_test, title, filename):
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']), alpha=0.6)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
               cmap=ListedColormap(['#FF0000', '#0000FF']),
               edgecolor='k', s=30, marker='o', label='训练集', alpha=0.7)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
               cmap=ListedColormap(['#FF0000', '#0000FF']),
               edgecolor='k', s=50, marker='s', label='测试集', alpha=0.9)

    y_pred = model.predict(X_test)
    errors = (y_test != y_pred)
    if np.any(errors):
        ax.scatter(X_test[errors, 0], X_test[errors, 1], c='yellow',
                   marker='x', s=150, linewidths=3, label='错误分类', zorder=5)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_error = 1 - accuracy_score(y_train, train_pred)
    test_error = 1 - accuracy_score(y_test, test_pred)

    ax.set_title(f'{title}\n训练错误率: {train_error:.4f}, 测试错误率: {test_error:.4f}')
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')
    ax.legend()

    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  图片已保存: {filename}')

    return train_error, test_error


# 运行实验
print('\n' + '='*60)
print('开始训练和评估')
print('='*60)

print('\n[数据集1 - 高分离度]')
model1 = GaussianNaiveBayes()
model1.fit(X1_train, y1_train)
train_err1, test_err1 = plot_decision_boundary(
    model1, X1_train, y1_train, X1_test, y1_test,
    '数据集1 (高分离度)', 'out/实验四_数据集1_决策边界.png'
)
print(f'  训练错误率: {train_err1:.4f}')
print(f'  测试错误率: {test_err1:.4f}')

print('\n[数据集2 - 低分离度]')
model2 = GaussianNaiveBayes()
model2.fit(X2_train, y2_train)
train_err2, test_err2 = plot_decision_boundary(
    model2, X2_train, y2_train, X2_test, y2_test,
    '数据集2 (低分离度)', 'out/实验四_数据集2_决策边界.png'
)
print(f'  训练错误率: {train_err2:.4f}')
print(f'  测试错误率: {test_err2:.4f}')

print('\n' + '='*60)
print('实验总结')
print('='*60)
print(f'数据集1: 训练错误率={train_err1:.4f}, 测试错误率={test_err1:.4f}')
print(f'数据集2: 训练错误率={train_err2:.4f}, 测试错误率={test_err2:.4f}')
print('\n实验四完成！所有图片已保存到 out/ 目录')

print('提示：可参考实验一的代码结构')

# ============================================================
# 示例：可视化数据分布
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 数据集1
axes[0].scatter(X1[y1==0, 0], X1[y1==0, 1], c='red', label='类别0', alpha=0.6, edgecolors='k')
axes[0].scatter(X1[y1==1, 0], X1[y1==1, 1], c='blue', label='类别1', alpha=0.6, edgecolors='k')
axes[0].set_title('数据集1 (高分离度)')
axes[0].set_xlabel('特征1')
axes[0].set_ylabel('特征2')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 数据集2
axes[1].scatter(X2[y2==0, 0], X2[y2==0, 1], c='red', label='类别0', alpha=0.6, edgecolors='k')
axes[1].scatter(X2[y2==1, 0], X2[y2==1, 1], c='blue', label='类别1', alpha=0.6, edgecolors='k')
axes[1].set_title('数据集2 (低分离度)')
axes[1].set_xlabel('特征1')
axes[1].set_ylabel('特征2')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('out/实验四_数据分布.png', dpi=100, bbox_inches='tight')
plt.close()
print('\n数据分布图已保存: out/实验四_数据分布.png')

print('\n'+'='*60)
print('数据准备完成！请实现分类算法。')
print('='*60)
