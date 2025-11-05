"""
实验三：k-近邻概率密度估计

实验目标：
1. 理解k-NN密度估计方法（非参数方法）
2. 实现k-NN密度估计公式
3. 分析k值对密度估计的影响

你只需要实现：
  ✓ predict() - k-NN密度估计公式（核心）

已提供：fit()、可视化等辅助函数
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class MyKNNDensityEstimator:
    """
    k-近邻密度估计器
    """
    def __init__(self):
        self.X_train_ = None
        self.N_ = 0
        self._epsilon = 1e-9

    def fit(self, X):
        """
        保存训练数据
        
        输入:
            X: 训练数据, shape=(n_samples, n_features)
        """
        self.X_train_ = X
        self.N_ = X.shape[0]

    def predict(self, X_new, k):
        """
        k-NN密度估计
        
        公式: p(x) = k / (N × V)
        其中:
            k: 近邻数
            N: 训练样本数
            V: 包含k个近邻的体积
            在2D情况下: V = π × r_k²
            r_k: 第k个最近邻的距离
        
        输入:
            X_new: 待估计的点, shape=(n_test, n_features)
            k: 近邻数
        
        输出:
            密度值, shape=(n_test,)
        
        TODO: 请完成k-NN密度估计
        提示:
        1. 对每个新样本，计算到所有训练样本的欧氏距离
        2. 排序，找到第k个最近邻的距离r_k
        3. 计算体积 V = π × r_k²
        4. 计算密度 p(x) = k / (N × V)
        """
        densities = []
        for x in X_new:
            # TODO: 1. 计算距离
            distances = None  # np.linalg.norm(self.X_train_ - x, axis=1)
            
            # TODO: 2. 排序并找到第k个距离
            distances_sorted = None  # np.sort(distances)
            r_k = None  # distances_sorted[k-1]  # 注意: k=1时索引是0
            
            # TODO: 3. 计算体积 (2D情况)
            volume = None  # np.pi * r_k**2 + self._epsilon
            
            # TODO: 4. 计算密度
            density = None  # k / (self.N_ * volume)
            
            densities.append(density)
        
        return np.array(densities)


def plot_density_heatmap(X, densities, xx, yy, k, dataset_name, output_dir):
    """
    绘制密度热力图
    
    输入:
        X: 原始数据
        densities: 网格点的密度值
        xx, yy: 网格坐标
        k: 近邻数
        dataset_name: 数据集名称
        output_dir: 输出目录
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    Z = densities.reshape(xx.shape)
    c = ax.contourf(xx, yy, Z, levels=20, cmap='viridis')
    fig.colorbar(c, ax=ax, label='概率密度 p(x)')
    
    ax.scatter(X[:, 0], X[:, 1], c='red', edgecolor='k',
               s=10, alpha=0.6, label='训练数据')
    ax.set_title(f'{dataset_name}\nk-NN密度估计 (k={k})')
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')
    ax.legend()
    
    safe_name = dataset_name.replace(' ', '_')
    filename = os.path.join(output_dir, f'{safe_name}_knn_k{k}.png')
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  图片已保存: {filename}')


def run_experiment(X, dataset_name, k_values, output_dir):
    """
    运行实验三
    
    步骤:
    1. 训练k-NN密度估计器
    2. 对不同k值进行密度估计
    3. 绘制密度热力图
    """
    print(f'\n{"="*60}')
    print(f'{dataset_name}')
    print(f'{"="*60}')
    
    # 训练模型
    model = MyKNNDensityEstimator()
    model.fit(X)
    
    # 创建网格用于可视化
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # 对每个k值进行密度估计
    for k in k_values:
        print(f'\n[k={k}] 计算密度估计...')
        densities = model.predict(grid_points, k=k)
        plot_density_heatmap(X, densities, xx, yy, k, dataset_name, output_dir)


if __name__ == '__main__':
    output_dir = 'out'
    os.makedirs(output_dir, exist_ok=True)
    
    print('='*60)
    print('实验三：k-近邻概率密度估计')
    print('='*60)
    
    # 生成数据集
    X1, _ = make_classification(
        n_samples=200, n_features=2, n_redundant=0, n_informative=2,
        n_classes=2, n_clusters_per_class=1, class_sep=2.0, random_state=42
    )
    X2, _ = make_classification(
        n_samples=200, n_features=2, n_redundant=0, n_informative=2,
        n_classes=2, n_clusters_per_class=1, class_sep=0.5, random_state=42
    )
    
    k_values = [1, 3, 5]
    
    try:
        run_experiment(X1, '数据集1 (高分离度)', k_values, output_dir)
        run_experiment(X2, '数据集2 (低分离度)', k_values, output_dir)
        
        print('\n' + '='*60)
        print('实验总结')
        print('='*60)
        print('分析:')
        print('  - k=1: 密度估计尖锐，在数据点处形成尖峰 (过拟合)')
        print('  - k=3: 密度估计开始平滑')
        print('  - k=5: 密度估计更加平滑 (可能欠拟合)')
        print('\n实验三完成！')
        
    except (TypeError, AttributeError, ValueError) as e:
        print('\n⚠️  检测到未完成的TODO!')
        print('请完成所有标记为 TODO 的部分')
        print(f'错误信息: {e}')
