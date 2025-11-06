"""
实验目标：
1. 理解并实现最大后验概率规则(MAP)
2. 理解并实现似然率测试规则(LRT)  
3. 验证两种规则的等价性

你只需要实现：
  ✓ predict() - MAP分类规则（核心！）
  ✓ apply_LRT_rule() - LRT分类规则（核心！）

已提供：参数估计、高斯PDF计算、对数似然计算等辅助函数
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import os

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'STHeiti', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class MyGaussianNB:
    """
    高斯朴素贝叶斯分类器
    用于实现似然率测试规则(LRT)和最大后验概率规则(MAP)
    """
    def __init__(self):
        self.priors_ = None  # 先验概率 P(C_i)
        self.means_ = None   # 均值 μ
        self.vars_ = None    # 方差 σ²
        self.classes_ = None # 类别标签 [0, 1]
        self._epsilon = 1e-9 # 防止除零和log(0)

    def fit(self, X, y):
        """
        参数估计：计算每个类别的均值、方差和先验概率
        （本实验中参数已直接提供，此函数仅用于保存参数）
        
        输入:
            X: 训练数据特征, shape=(n_samples, n_features)
            y: 训练数据标签, shape=(n_samples,)
        """
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.vars_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors_ = np.zeros(n_classes, dtype=np.float64)
        
        # 参数估计（已实现）
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[idx, :] = np.mean(X_c, axis=0)
            self.vars_[idx, :] = np.var(X_c, axis=0)
            self.priors_[idx] = X_c.shape[0] / float(n_samples)

    def _gaussian_pdf(self, X, mean, var):
        """
        计算高斯概率密度函数 (PDF)
        公式: f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
        （已实现）
        """
        numerator = np.exp(-((X - mean)**2) / (2 * (var + self._epsilon)))
        denominator = np.sqrt(2 * np.pi * (var + self._epsilon))
        return numerator / denominator

    def _calculate_log_likelihood(self, X):
        """
        计算对数似然 log P(X|C_i)
        朴素贝叶斯假设: 特征条件独立
        log P(X|C_i) = Σ log P(xj|C_i)
        （已实现）
        """
        log_likelihoods = []
        for i in range(len(self.classes_)):
            mean = self.means_[i, :]
            var = self.vars_[i, :]
            pdfs = self._gaussian_pdf(X, mean, var)
            pdfs[pdfs == 0] = self._epsilon
            log_pdfs = np.log(pdfs)
            log_likelihood_per_sample = np.sum(log_pdfs, axis=1)
            log_likelihoods.append(log_likelihood_per_sample)
        return np.array(log_likelihoods).T

    def predict(self, X):
        """
        最大后验概率规则 (MAP) 进行分类
        
        决策规则: C* = argmax P(C_i|X) = argmax [P(X|C_i) × P(C_i)]
        对数形式: C* = argmax [log P(X|C_i) + log P(C_i)]
        
        输入:
            X: 待分类样本, shape=(n_samples, n_features)
        
        输出:
            预测类别, shape=(n_samples,)
        
        TODO: 请完成MAP预测
        """
        # TODO: 1. 计算对数似然
        log_likelihoods = self._calculate_log_likelihood(X)
        
        # TODO: 2. 计算对数先验
        log_priors = np.log(self.priors_)
        
        # TODO: 3. 计算对数后验 = 对数似然 + 对数先验
        log_joint = log_likelihoods + log_priors
        
        # TODO: 4. 选择后验概率最大的类别
        return np.argmax(log_joint, axis=1)


def apply_LRT_rule(model, X):
    """
    似然率测试规则 (LRT) 进行分类
    
    决策规则: 
        若 P(X|C1)/P(X|C0) > P(C0)/P(C1), 则预测为 C1
        
    对数形式:
        若 log P(X|C1) - log P(X|C0) > log P(C0) - log P(C1), 则预测为 C1
    
    输入:
        model: 已训练的 MyGaussianNB 模型
        X: 待分类样本, shape=(n_samples, n_features)
    
    输出:
        预测类别, shape=(n_samples,)
    
    TODO: 请完成LRT规则实现
    """
    # TODO: 1. 计算对数似然
    log_likelihoods = model._calculate_log_likelihood(X)
    
    # TODO: 2. 提取两个类别的对数似然
    log_like_c0 = log_likelihoods[:, 0]
    log_like_c1 = log_likelihoods[:, 1]
    
    # TODO: 3. 计算对数似然率 (左边)
    log_likelihood_ratio = log_like_c1 - log_like_c0
    
    # TODO: 4. 计算阈值 (右边)
    log_priors = np.log(model.priors_)
    log_threshold = log_priors[0] - log_priors[1]
    
    # TODO: 5. 应用LRT规则
    predictions =(log_likelihood_ratio > log_threshold).astype(int)
    
    return predictions


def plot_decision_boundary(model, X, y, title, filename, method='MAP'):
    """
    绘制决策边界和数据分布
    
    输入:
        model: 训练好的模型
        X: 数据特征
        y: 真实标签
        title: 图表标题
        filename: 保存文件名
        method: 'MAP' 或 'LRT'
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # 根据方法选择预测函数
    if method == 'MAP':
        Z = model.predict(grid_points)
    else:  # LRT
        Z = apply_LRT_rule(model, grid_points)
    
    Z = Z.reshape(xx.shape)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']), alpha=0.6)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#0000FF']),
               edgecolor='k', s=20, label='训练数据')
    
    # 标记错误分类的点
    if method == 'MAP':
        predictions = model.predict(X)
    else:
        predictions = apply_LRT_rule(model, X)
    
    errors = (y != predictions)
    if np.any(errors):
        ax.scatter(X[errors, 0], X[errors, 1], c='yellow', marker='x', 
                   s=100, linewidths=3, label='错误分类')
    
    # 计算错误率
    error_rate = 1 - accuracy_score(y, predictions)
    
    ax.set_title(f'{title}\n错误率: {error_rate:.4f}')
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')
    ax.legend()
    
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  图片已保存: {filename}')


def run_experiment(X, y, dataset_name, output_dir):
    """
    运行完整的实验一
    
    步骤:
    1. 创建模型并提供已估计的参数
    2. 使用MAP规则分类
    3. 使用LRT规则分类
    4. 可视化结果
    """
    print(f'\n{"="*60}')
    print(f'{dataset_name}')
    print(f'{"="*60}')
    
    # 步骤1: 创建模型并估计参数（已实现）
    print('\n[步骤1] 准备模型参数...')
    model = MyGaussianNB()
    model.fit(X, y)  # 自动估计参数
    
    print(f'  类别0: 均值={model.means_[0]}, 方差={model.vars_[0]}, 先验={model.priors_[0]:.3f}')
    print(f'  类别1: 均值={model.means_[1]}, 方差={model.vars_[1]}, 先验={model.priors_[1]:.3f}')
    print('  参数已准备好，请实现分类规则！')
    
    # 步骤2: MAP规则分类
    print('\n[步骤2] 最大后验概率规则 (MAP)...')
    map_predictions = model.predict(X)
    map_error_rate = 1 - accuracy_score(y, map_predictions)
    print(f'  错误率: {map_error_rate:.4f}')
    
    # 步骤3: LRT规则分类
    print('\n[步骤3] 似然率测试规则 (LRT)...')
    lrt_predictions = apply_LRT_rule(model, X)
    lrt_error_rate = 1 - accuracy_score(y, lrt_predictions)
    print(f'  错误率: {lrt_error_rate:.4f}')
    
    # 步骤4: 可视化
    print('\n[步骤5] 生成决策边界图...')
    safe_name = dataset_name.replace(' ', '_')
    plot_decision_boundary(model, X, y, f'{dataset_name} - MAP规则',
                          os.path.join(output_dir, f'{safe_name}_MAP.png'), 'MAP')
    plot_decision_boundary(model, X, y, f'{dataset_name} - LRT规则',
                          os.path.join(output_dir, f'{safe_name}_LRT.png'), 'LRT')
    
    return map_error_rate, lrt_error_rate


if __name__ == '__main__':
    # 创建输出目录
    output_dir = 'out'
    os.makedirs(output_dir, exist_ok=True)
    
    print('='*60)
    print('实验一：似然率测试规则(LRT) 与 最大后验概率规则(MAP) 分类')
    print('='*60)
    
    # 生成数据集1: 高分离度 (容易分类)
    print('\n生成数据集1 (高分离度)...')
    X1, y1 = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=2.0,  # 高分离度
        random_state=42
    )
    
    # 生成数据集2: 低分离度 (较难分类)
    print('生成数据集2 (低分离度)...')
    X2, y2 = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=0.5,  # 低分离度
        random_state=42
    )
    
    # 运行实验
    try:
        results1 = run_experiment(X1, y1, '数据集1 (高分离度)', output_dir)
        results2 = run_experiment(X2, y2, '数据集2 (低分离度)', output_dir)
        
        # 总结
        print('\n' + '='*60)
        print('实验总结')
        print('='*60)
        print(f'数据集1: MAP错误率={results1[0]:.4f}, LRT错误率={results1[1]:.4f}')
        print(f'数据集2: MAP错误率={results2[0]:.4f}, LRT错误率={results2[1]:.4f}')
        print('\n实验一完成！所有图片已保存到 out/ 目录')
        
    except (TypeError, AttributeError) as e:
        print('\n⚠️  检测到未完成的TODO!')
        print('请完成所有标记为 TODO 的部分')
        print(f'错误信息: {e}')
