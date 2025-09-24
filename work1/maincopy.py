import numpy as np
from scipy.spatial.distance import cdist
## import... 安装所需要的依赖库 

def loo_eval(X, y, k):
    # 这部分为实验的核心代码
    n_samples = X.shape[0]
    correct_predictions = 0
    
    # 留一法交叉验证：每次取一个样本作为测试样本，其余作为训练样本
    for i in range(n_samples):
        # 当前测试样本
        test_sample = X[i:i+1]  # 保持二维形状
        test_label = y[i]
        
        # 训练样本（除了当前测试样本）
        train_X = np.concatenate([X[:i], X[i+1:]], axis=0)
        train_y = np.concatenate([y[:i], y[i+1:]], axis=0)
        
        # 计算测试样本与所有训练样本的欧氏距离
        distances = cdist(test_sample, train_X, metric='euclidean')[0]
        
        # 找到k个最近邻的索引
        k_nearest_indices = np.argsort(distances)[:k]
        
        # 获取k个最近邻的标签
        k_nearest_labels = train_y[k_nearest_indices]
        
        # 投票预测：选择最常见的标签
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]
        
        # 检查预测是否正确
        if predicted_label == test_label:
            correct_predictions += 1
    
    # 计算准确率
    acc = correct_predictions / n_samples
    return acc

# 主流程
raw = np.loadtxt('semeion.data.txt')
X, y = raw[:, :256], np.argmax(raw[:, 256:], 1)

for k in [1, 3, 5]:
    acc = loo_eval(X, y, k)
    print(f'k={k}  LOO 准确率 = {acc:.4f}')