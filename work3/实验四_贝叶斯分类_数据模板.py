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
plt.rcParams['font.sans-serif'] = ['SimHei']
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
# TODO: 请在此处实现贝叶斯分类器
# ============================================================

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
