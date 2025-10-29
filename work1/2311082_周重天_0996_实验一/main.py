import numpy as np
from math import sqrt

## import... 安装所需要的依赖库 

def euclidean_distance(test_sample, train_X):
    """计算欧氏距离 - 根据公式 √(∑(xi - xj)²) 实现"""
    # test_sample: (1, 256), train_X: (n-1, 256)
    # 计算每个训练样本与测试样本的欧氏距离
    distances = np.sqrt(np.sum((train_X - test_sample) ** 2, axis=1))
    return distances

def manhattan_distance(test_sample, train_X):
    """计算曼哈顿距离"""
    # test_sample: (1, 256), train_X: (n-1, 256)
    # 计算每个训练样本与测试样本的曼哈顿距离
    distances = np.sum(np.abs(train_X - test_sample), axis=1)
    return distances

def knn_eval(train_X, train_y, test_X, test_y, k):
    """使用训练集训练，测试集评估的kNN算法"""
    n_test_samples = test_X.shape[0]
    correct_predictions = 0
    
    # 对每个测试样本进行预测
    for i in range(n_test_samples):
        # 当前测试样本
        test_sample = test_X[i:i+1]  # 保持二维形状 (1, 256)
        test_label = test_y[i]
        
        # 计算测试样本与所有训练样本的欧氏距离 √(∑(xi - xj)²)
        distances = euclidean_distance(test_sample, train_X)  # 使用欧氏距离
        # distances = manhattan_distance(test_sample, train_X)  # 使用曼哈顿距离
        
        # 找到k个最近邻的索引
        k_nearest_indices = np.argsort(distances)[:k]
        
        # 获取k个最近邻的标签
        k_nearest_labels = train_y[k_nearest_indices]
        
        # 投票：标签多数表决（可平分时随机）
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        max_count = np.max(counts)
        # 找出得票最多的所有标签
        max_labels = unique_labels[counts == max_count]
        
        # 如果有平分情况，随机选择一个
        if len(max_labels) > 1:
            predicted_label = np.random.choice(max_labels)
        else:
            predicted_label = max_labels[0]
        
        # 检查预测是否正确
        if predicted_label == test_label:
            correct_predictions += 1
    
    # 计算准确率
    acc = correct_predictions / n_test_samples
    return acc

def loo_eval(X, y, k):
    n_samples = X.shape[0]  # 应该是1593
    correct_predictions = 0
    
    # 留一法交叉验证：每次取一个样本作为测试样本，其余作为训练样本
    for i in range(n_samples):
        # 当前测试样本
        test_sample = X[i:i+1]  # 保持二维形状 (1, 256)
        test_label = y[i]
        
        # 训练样本（除了当前测试样本），共1592个样本
        train_X = np.concatenate([X[:i], X[i+1:]], axis=0)
        train_y = np.concatenate([y[:i], y[i+1:]], axis=0)
        
        # 计算测试样本与所有训练样本的欧氏距离 √(∑(xi - xj)²)
        distances = euclidean_distance(test_sample, train_X)  # 使用欧氏距离
        # distances = manhattan_distance(test_sample, train_X)  # 使用曼哈顿距离
        
        # 找到k个最近邻的索引
        k_nearest_indices = np.argsort(distances)[:k]
        
        # 获取k个最近邻的标签
        k_nearest_labels = train_y[k_nearest_indices]
        
        # 投票：标签多数表决（可平分时随机）
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        max_count = np.max(counts)
        # 找出得票最多的所有标签
        max_labels = unique_labels[counts == max_count]
        
        # 如果有平分情况，随机选择一个
        if len(max_labels) > 1:
            predicted_label = np.random.choice(max_labels)
        else:
            predicted_label = max_labels[0]
        
        # 检查预测是否正确
        if predicted_label == test_label:
            correct_predictions += 1
    
    # 计算准确率
    acc = correct_predictions / n_samples
    return acc

# Helper function to generate prime numbers
def generate_primes(n):
    """生成小于等于 n 的所有质数"""
    primes = []
    for num in range(2, n + 1):
        is_prime = True
        for i in range(2, int(sqrt(num)) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes

# 主流程
# 加载训练集
train_raw = np.loadtxt('semeion_train.txt')
train_X, train_y = train_raw[:, :256], np.argmax(train_raw[:, 256:], 1)

# 加载测试集
test_raw = np.loadtxt('semeion_test.txt')
test_X, test_y = test_raw[:, :256], np.argmax(test_raw[:, 256:], 1)

# 生成质数作为 k 值
k_values = generate_primes(20)  # 生成小于等于 20 的质数
k_values = [1] + k_values
# 在训练集上使用 LOO 找到最佳 k 值
best_k = None
best_accuracy = 0
for k in k_values:
    acc = loo_eval(train_X, train_y, k)
    print(f'k={k} 训练集 LOO 准确率 = {acc:.4f}')
    if acc > best_accuracy:
        best_accuracy = acc
        best_k = k

# 使用最佳 k 值在测试集上评估
final_accuracy = knn_eval(train_X, train_y, test_X, test_y, best_k)
print(f'最佳 k 值: {best_k}')
print(f'测试集准确率: {final_accuracy:.4f}')