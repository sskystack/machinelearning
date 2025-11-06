"""
实验二：高斯核函数密度估计 + LRT分类

实验目标：
1. 理解核密度估计方法（非参数方法）
2. 实现高斯核函数密度估计
3. 使用LRT规则进行分类

你只需要实现：
  ✓ predict_log_density() - 核密度估计公式（核心！）
  ✓ predict() - LRT分类规则（核心！）

已提供：fit()、交叉验证、可视化等辅助函数
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import os

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'STHeiti', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class MyGaussianKernelDensity:
    """
    高斯核密度估计器
    用于估计概率密度 p(x)
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

    def predict_log_density(self, X_new, h):
        """
        计算对数概率密度 log p(x)
        
        核密度估计公式: 
            p(x) = (1/N) × Σ K((x-xi)/h)
            K(u) = exp(-0.5 × ||u||²)  [高斯核]
        
        输入:
            X_new: 待估计的点, shape=(n_test, n_features)
            h: 带宽参数 (控制平滑度)
        
        输出:
            对数密度, shape=(n_test,)
        
        TODO: 请完成核密度估计
        提示:
        1. 对每个新样本x计算到所有训练样本的欧氏距离
        2. 应用高斯核: K(u) = exp(-0.5 * (distance/h)²)
        3. 求和并除以N，再取对数
        """
        log_densities = []
        for x in X_new:
            # TODO: 1. 计算距离
            distances = np.linalg.norm(self.X_train_ - x, axis=1)
            
            # TODO: 2. 应用高斯核
            kernel_vals = np.exp(-0.5 * (distances / h)**2)
            
            # TODO: 3. 求和并取对数
            density_sum = np.sum(kernel_vals) + self._epsilon
            log_density = np.log(density_sum)
            
            log_densities.append(log_density)
        
        return np.array(log_densities)


class MyKernelClassifier:
    """
    基于核密度估计的分类器
    使用似然率测试规则(LRT)进行分类
    """
    def __init__(self):
        self.kde_c0_ = MyGaussianKernelDensity()
        self.kde_c1_ = MyGaussianKernelDensity()
        self.log_prior_c0_ = 0
        self.log_prior_c1_ = 0

    def fit(self, X, y):
        """
        训练分类器
        
        步骤:
        1. 分离两个类别的数据
        2. 为每个类别训练核密度估计器
        3. 计算先验概率
        
        输入:
            X: 训练数据, shape=(n_samples, n_features)
            y: 标签, shape=(n_samples,)
        """
        # 1. 分离类别数据（已实现）
        X_c0 = X[y == 0]
        X_c1 = X[y == 1]
        
        # 2. 训练核密度估计器（已实现）
        self.kde_c0_.fit(X_c0)
        self.kde_c1_.fit(X_c1)
        
        # 3. 计算对数先验概率（已实现）
        self.log_prior_c0_ = np.log(len(X_c0) / len(X))
        self.log_prior_c1_ = np.log(len(X_c1) / len(X))

    def predict(self, X_new, h):
        """
        使用似然率测试规则(LRT)分类
        
        决策规则:
            若 p(x|C1)/p(x|C0) > P(C0)/P(C1), 则预测为 C1
            
        对数形式:
            若 log p(x|C1) - log p(x|C0) > log P(C0) - log P(C1), 则预测为 C1
        
        输入:
            X_new: 待分类样本, shape=(n_test, n_features)
            h: 带宽参数
        
        输出:
            预测类别, shape=(n_test,)
        
        TODO: 请完成LRT分类
        """
        # TODO: 1. 计算两个类别的对数似然
        log_like_c0 = self.kde_c0_.predict_log_density(X_new, h)
        log_like_c1 = self.kde_c1_.predict_log_density(X_new, h)
        
        # TODO: 2. 计算对数似然率
        log_lr = log_like_c1 - log_like_c0
        
        # TODO: 3. 计算阈值
        log_threshold = self.log_prior_c0_ - self.log_prior_c1_
        
        # TODO: 4. 应用LRT规则
        predictions = (log_lr > log_threshold).astype(int)
        
        return predictions


def find_best_h(X, y, h_values, n_splits=5):
    """
    使用K折交叉验证寻找最优带宽h
    
    输入:
        X: 特征数据
        y: 标签
        h_values: 候选h值列表 [0.1, 0.5, 1, 1.5, 2]
        n_splits: 交叉验证折数 (默认5折)
    
    输出:
        best_h: 最优h值
    
    TODO: 请完成K折交叉验证
    提示:
    1. 对每个h值，进行K折交叉验证
    2. 在每折上训练和验证
    3. 计算平均准确率
    4. 选择准确率最高的h
    """
    print('  [交叉验证] 开始寻找最优h...')
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    best_h = None
    best_accuracy = -1.0
    
    for h in h_values:
        fold_accuracies = []
        
        # TODO: K折交叉验证循环
        for train_idx, val_idx in kf.split(X):
            # TODO: 1. 分割数据
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # TODO: 2. 训练模型
            model_cv = MyKernelClassifier()
            model_cv.fit(X_train, y_train)
            
            # TODO: 3. 预测并计算准确率
            y_pred = model_cv.predict(X_val, h=h)
            acc = accuracy_score(y_val, y_pred)
            
            fold_accuracies.append(acc)
        
        # 计算平均准确率
        mean_accuracy = np.mean(fold_accuracies)
        print(f'    h={h:.1f}: 平均准确率={mean_accuracy:.4f}')
        
        # 更新最优h
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_h = h
    
    print(f'  [交叉验证] 完成! 最优h={best_h}, 准确率={best_accuracy:.4f}')
    return best_h


def plot_decision_boundary(model, X, y, h, title, filename):
    """绘制决策边界"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()], h=h)
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']), alpha=0.6)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#0000FF']),
               edgecolor='k', s=20, label='训练数据')
    
    predictions = model.predict(X, h=h)
    errors = (y != predictions)
    if np.any(errors):
        ax.scatter(X[errors, 0], X[errors, 1], c='yellow', marker='x',
                   s=100, linewidths=3, label='错误分类')
    
    error_rate = 1 - accuracy_score(y, predictions)
    ax.set_title(f'{title}\n核密度估计 (h={h}), 错误率={error_rate:.4f}')
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')
    ax.legend()
    
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  图片已保存: {filename}')


def run_experiment(X, y, dataset_name, h_values, output_dir):
    """运行实验二"""
    print(f'\n{"="*60}')
    print(f'{dataset_name}')
    print(f'{"="*60}')
    
    # 步骤1: 交叉验证找最优h
    best_h = find_best_h(X, y, h_values)
    
    # 步骤2: 使用最优h在全数据上训练
    print(f'\n[使用最优h={best_h}训练模型]')
    model = MyKernelClassifier()
    model.fit(X, y)
    
    # 步骤3: 分类并评估
    predictions = model.predict(X, h=best_h)
    error_rate = 1 - accuracy_score(y, predictions)
    print(f'  错误率: {error_rate:.4f}')
    
    # 步骤4: 可视化
    safe_name = dataset_name.replace(' ', '_')
    plot_decision_boundary(model, X, y, best_h, dataset_name,
                          os.path.join(output_dir, f'{safe_name}_kernel.png'))
    
    return best_h, error_rate


if __name__ == '__main__':
    output_dir = 'out'
    os.makedirs(output_dir, exist_ok=True)
    
    print('='*60)
    print('实验二：高斯核函数密度估计 + LRT分类')
    print('='*60)
    
    # 生成数据集
    X1, y1 = make_classification(
        n_samples=200, n_features=2, n_redundant=0, n_informative=2,
        n_classes=2, n_clusters_per_class=1, class_sep=2.0, random_state=42
    )
    X2, y2 = make_classification(
        n_samples=200, n_features=2, n_redundant=0, n_informative=2,
        n_classes=2, n_clusters_per_class=1, class_sep=0.5, random_state=42
    )
    
    h_values = [0.1, 0.5, 1, 1.5, 2]
    
    try:
        result1 = run_experiment(X1, y1, '数据集1 (高分离度)', h_values, output_dir)
        result2 = run_experiment(X2, y2, '数据集2 (低分离度)', h_values, output_dir)
        
        print('\n' + '='*60)
        print('实验总结')
        print('='*60)
        print(f'数据集1: 最优h={result1[0]}, 错误率={result1[1]:.4f}')
        print(f'数据集2: 最优h={result2[0]}, 错误率={result2[1]:.4f}')
        print('\n实验二完成！')
        
    except (TypeError, AttributeError, ValueError) as e:
        print('\n⚠️  检测到未完成的TODO!')
        print('请完成所有标记为 TODO 的部分')
        print(f'错误信息: {e}')
