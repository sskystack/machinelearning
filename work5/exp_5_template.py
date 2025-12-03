"""
实验五：层次聚类分析（学生版）

实验内容：
1. 基本要求(4分): 实现single-linkage和complete-linkage层次聚类
2. 中级要求(1分): 实现average-linkage层次聚类
3. 提高要求(1分): 对比三种算法，给出结论
4. 拓展要求: 变换聚类簇个数，测试性能

你需要实现：
  ✓ _single_linkage_distance() - 最小距离（核心！）
  ✓ _complete_linkage_distance() - 最大距离（核心！）
  ✓ _average_linkage_distance() - 平均距离（核心！）

已提供：距离矩阵计算、聚类主循环、可视化等辅助函数
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score, silhouette_score
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class HierarchicalClustering:
    """
    层次聚类算法实现
    支持三种链接方式: single, complete, average
    """
    
    def __init__(self, n_clusters=3, linkage='single'):
        """
        初始化
        
        参数:
            n_clusters: 目标聚类数
            linkage: 链接方式 ('single', 'complete', 'average')
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
    
    def _compute_distance_matrix(self, X):
        """
        计算样本间的距离矩阵（欧氏距离）
        （已实现）
        """
        n = X.shape[0]
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(X[i] - X[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return dist_matrix
    
    def _single_linkage_distance(self, cluster1, cluster2, dist_matrix):
        """
        Single-linkage: 两个簇之间的【最小】距离
        
        公式: d(C1, C2) = min{d(a,b) | a∈C1, b∈C2}
        
        输入:
            cluster1: 第一个簇的样本索引列表, 如 [0, 3, 5]
            cluster2: 第二个簇的样本索引列表, 如 [1, 2]
            dist_matrix: 距离矩阵, shape=(n_samples, n_samples)
        
        输出:
            两个簇之间的最小距离
        
        TODO: 请完成single-linkage距离计算
        提示:
        1. 遍历cluster1中的每个样本i
        2. 遍历cluster2中的每个样本j
        3. 使用dist_matrix[i, j]获取i和j的距离
        4. 返回所有距离中的最小值
        """
        min_dist = np.inf
        
        # TODO: 遍历两个簇中的所有样本对，找最小距离
        for i in cluster1:
            for j in cluster2:
                # TODO: 比较并更新最小距离
                if dist_matrix[i, j] < min_dist:
                    min_dist = dist_matrix[i, j]
        
        return min_dist
    
    def _complete_linkage_distance(self, cluster1, cluster2, dist_matrix):
        """
        Complete-linkage: 两个簇之间的【最大】距离
        
        公式: d(C1, C2) = max{d(a,b) | a∈C1, b∈C2}
        
        输入:
            cluster1: 第一个簇的样本索引列表
            cluster2: 第二个簇的样本索引列表
            dist_matrix: 距离矩阵
        
        输出:
            两个簇之间的最大距离
        
        TODO: 请完成complete-linkage距离计算
        提示: 与single-linkage类似，但找最大值
        """
        max_dist = 0
        
        # TODO: 遍历两个簇中的所有样本对，找最大距离
        for i in cluster1:
            for j in cluster2:
                # TODO: 比较并更新最大距离
                if dist_matrix[i, j] > max_dist:
                    max_dist = dist_matrix[i, j]
        
        return max_dist
    
    def _average_linkage_distance(self, cluster1, cluster2, dist_matrix):
        """
        Average-linkage: 两个簇之间的【平均】距离
        
        公式: d(C1, C2) = (1/(|C1|×|C2|)) × Σ Σ d(a,b)
        
        输入:
            cluster1: 第一个簇的样本索引列表
            cluster2: 第二个簇的样本索引列表
            dist_matrix: 距离矩阵
        
        输出:
            两个簇之间的平均距离
        
        TODO: 请完成average-linkage距离计算
        提示:
        1. 累加所有样本对的距离
        2. 除以样本对的数量 (|C1| × |C2|)
        """
        total_dist = 0
        count = 0
        
        # TODO: 遍历两个簇中的所有样本对，累加距离
        for i in cluster1:
            for j in cluster2:
                # TODO: 累加距离并计数
                total_dist += dist_matrix[i, j]
                count += 1
        
        # TODO: 返回平均距离
        return total_dist / count
    
    def _get_cluster_distance(self, cluster1, cluster2, dist_matrix):
        """根据链接方式计算簇间距离（已实现）"""
        if self.linkage == 'single':
            return self._single_linkage_distance(cluster1, cluster2, dist_matrix)
        elif self.linkage == 'complete':
            return self._complete_linkage_distance(cluster1, cluster2, dist_matrix)
        elif self.linkage == 'average':
            return self._average_linkage_distance(cluster1, cluster2, dist_matrix)
        else:
            raise ValueError(f"Unknown linkage: {self.linkage}")
    
    def fit(self, X):
        """
        执行层次聚类（已实现）
        
        算法步骤:
        1. 初始化：每个样本为一个簇
        2. 重复直到簇数量达到目标:
           a. 计算所有簇对之间的距离
           b. 找到距离最小的两个簇
           c. 合并这两个簇
        3. 分配标签
        """
        n_samples = X.shape[0]
        
        # 1. 计算距离矩阵
        dist_matrix = self._compute_distance_matrix(X)
        
        # 2. 初始化：每个样本是一个簇
        clusters = [[i] for i in range(n_samples)]
        
        # 3. 迭代合并，直到达到目标簇数
        while len(clusters) > self.n_clusters:
            # 找到距离最小的两个簇
            min_dist = np.inf
            merge_i, merge_j = 0, 1
            
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    dist = self._get_cluster_distance(clusters[i], clusters[j], dist_matrix)
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            # 合并两个簇
            new_cluster = clusters[merge_i] + clusters[merge_j]
            
            # 删除旧簇，添加新簇
            clusters = [c for k, c in enumerate(clusters) if k != merge_i and k != merge_j]
            clusters.append(new_cluster)
        
        # 4. 分配标签
        self.labels_ = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for sample_idx in cluster:
                self.labels_[sample_idx] = cluster_id
        
        return self
    
    def fit_predict(self, X):
        """执行聚类并返回标签"""
        self.fit(X)
        return self.labels_


# ============================================================
# 以下为辅助函数（已实现）
# ============================================================

def generate_data(n_samples=300, random_state=42):
    """生成人工数据集（已实现，高斯簇）
    - 三个簇之间有适中的间隔，存在少量重叠
    """
    centers = np.array([
        [-3.0, 0.0],
        [ 3.0, 0.0],
        [ 0.0, 4.0],
    ])
    X, y_true = make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=2,
        cluster_std=1.3,
        random_state=random_state,
    )
    return X, y_true


def generate_moons_data(n_samples=300, noise=0.08, random_state=0):
    """
    生成 make_moons 数据集（已实现，用于非凸形状簇的聚类实验）
    """
    X, y_true = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    upper_mask = y_true == 0
    lower_mask = ~upper_mask
    X[upper_mask, 1] += 0.3
    X[lower_mask, 1] -= 0.3

    return X, y_true


def plot_clustering_result(X, labels, title, filename, true_labels=None):
    """绘制聚类结果（已实现）"""
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(X[mask, 0], X[mask, 1], c=[color], 
                   label=f'Cluster {label}', s=50, edgecolors='k', alpha=0.7)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if true_labels is not None:
        ari = adjusted_rand_score(true_labels, labels)
        silhouette = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0
        plt.text(0.02, 0.98, f'ARI: {ari:.4f}\nSilhouette: {silhouette:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  图片已保存: {filename}')


def plot_original_data(X, y_true, filename):
    """绘制原始数据分布（已实现）"""
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(y_true)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = y_true == label
        plt.scatter(X[mask, 0], X[mask, 1], c=[color], 
                   label=f'Class {label}', s=50, edgecolors='k', alpha=0.7)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Original data (true labels)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  图片已保存: {filename}')


def compare_algorithms(X, y_true, n_clusters, output_dir, plot=True):
    """对比三种链接方式（已实现）

    """
    linkages = ['single', 'complete', 'average']
    results = {}
    
    print('\n[算法对比]')
    print('-' * 60)
    print(f'{"linkage":<15} {"ARI":<12} {"Silhouette":<12}')
    print('-' * 60)
    
    for linkage in linkages:
        model = HierarchicalClustering(n_clusters=n_clusters, linkage=linkage)
        labels = model.fit_predict(X)
        
        ari = adjusted_rand_score(y_true, labels)
        silhouette = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0
        
        results[linkage] = {'ari': ari, 'silhouette': silhouette, 'labels': labels}
        print(f'{linkage:<15} {ari:<12.4f} {silhouette:<12.4f}')
        
        if plot:
            plot_clustering_result(X, labels, f'{linkage.capitalize()}-Linkage clustering result',
                                  os.path.join(output_dir, f'clustering_{linkage}.png'), y_true)
    
    print('-' * 60)
    return results


def test_different_k(X, y_true, k_values, output_dir):
    """测试不同聚类数k（已实现）"""
    linkages = ['single', 'complete', 'average']
    results = {linkage: {'k': [], 'ari': [], 'silhouette': []} for linkage in linkages}
    
    print('\n[不同聚类数k的性能测试]')
    
    for k in k_values:
        print(f'\nk={k}:')
        for linkage in linkages:
            model = HierarchicalClustering(n_clusters=k, linkage=linkage)
            labels = model.fit_predict(X)
            
            ari = adjusted_rand_score(y_true, labels)
            silhouette = silhouette_score(X, labels) if k > 1 and len(np.unique(labels)) > 1 else 0
            
            results[linkage]['k'].append(k)
            results[linkage]['ari'].append(ari)
            results[linkage]['silhouette'].append(silhouette)
            
            print(f'  {linkage:<10}: ARI={ari:.4f}, Silhouette={silhouette:.4f}')
    
    # 绘制性能对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for linkage in linkages:
        axes[0].plot(results[linkage]['k'], results[linkage]['ari'], 
                    'o-', label=linkage, linewidth=2, markersize=8)
        axes[1].plot(results[linkage]['k'], results[linkage]['silhouette'], 
                    'o-', label=linkage, linewidth=2, markersize=8)
    
    axes[0].set_xlabel('Number of clusters k')
    axes[0].set_ylabel('ARI')
    axes[0].set_title('ARI vs k')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Number of clusters k')
    axes[1].set_ylabel('Silhouette score')
    axes[1].set_title('Silhouette vs k')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print(f'\n  性能对比图已保存')
    
    return results


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    output_dir = 'out_exp5'
    os.makedirs(output_dir, exist_ok=True)
    
    print('='*60)
    print('实验五：层次聚类分析')
    print('='*60)
    
    # 1. 生成 make_blobs 数据
    print('\n[步骤1] 生成人工数据集 (make_blobs)...')
    X, y_true = generate_data(n_samples=150, random_state=42)
    print(f'  样本数: {X.shape[0]}, 特征数: {X.shape[1]}')
    print(f'  真实类别数: {len(np.unique(y_true))}')
    plot_original_data(X, y_true, os.path.join(output_dir, 'original_data.png'))
    
    try:
        # 2. 基本要求：Single-linkage 和 Complete-linkage
        print('\n[步骤2] 基本要求: Single-linkage 和 Complete-linkage')
        
        print('\n  Single-linkage聚类...')
        model_single = HierarchicalClustering(n_clusters=3, linkage='single')
        labels_single = model_single.fit_predict(X)
        ari_single = adjusted_rand_score(y_true, labels_single)
        print(f'    ARI: {ari_single:.4f}')
        plot_clustering_result(X, labels_single, 'Single-Linkage 聚类结果',
                              os.path.join(output_dir, 'single_linkage.png'), y_true)
        
        print('\n  Complete-linkage聚类...')
        model_complete = HierarchicalClustering(n_clusters=3, linkage='complete')
        labels_complete = model_complete.fit_predict(X)
        ari_complete = adjusted_rand_score(y_true, labels_complete)
        print(f'    ARI: {ari_complete:.4f}')
        plot_clustering_result(X, labels_complete, 'Complete-Linkage 聚类结果',
                              os.path.join(output_dir, 'complete_linkage.png'), y_true)
        
        # 3. 中级要求：Average-linkage
        print('\n[步骤3] 中级要求: Average-linkage')
        model_average = HierarchicalClustering(n_clusters=3, linkage='average')
        labels_average = model_average.fit_predict(X)
        ari_average = adjusted_rand_score(y_true, labels_average)
        print(f'    ARI: {ari_average:.4f}')
        plot_clustering_result(X, labels_average, 'Average-Linkage 聚类结果',
                              os.path.join(output_dir, 'average_linkage.png'), y_true)
        
        # 4. 提高要求：算法对比
        print('\n[步骤4] 提高要求: 算法对比 (make_blobs 数据集)')
        # 这里只打印指标，不再重复绘图，避免与上面的三张图片重复
        results = compare_algorithms(X, y_true, n_clusters=3, output_dir=output_dir, plot=False)
        
        print('\n结论 (make_blobs 数据集):')
        best_linkage = max(results.keys(), key=lambda k: results[k]['ari'])
        print(f'  - 在当前数据集上，{best_linkage}-linkage 表现最好')
        print(f'  - Single-linkage: 容易产生链式效应')
        print(f'  - Complete-linkage: 倾向于产生紧凑的簇')
        print(f'  - Average-linkage: 两者的折中')
        
        # 5. 拓展要求：不同k
        print('\n[步骤5] 拓展要求: 测试不同聚类数k (make_blobs 数据集)')
        k_values = [2, 3, 4, 5, 6]
        test_different_k(X, y_true, k_values, output_dir)

        # 6. 使用 make_moons 数据集进行额外实验
        output_dir_moons = 'out_exp5_moons'
        os.makedirs(output_dir_moons, exist_ok=True)

        print('\n' + '='*60)
        print('使用 make_moons 数据集进行层次聚类分析')
        print('='*60)

        print('\n[moons-1] 生成 make_moons 数据集...')
        X_moons, y_moons = generate_moons_data(n_samples=300, noise=0.1, random_state=0)
        print(f'  样本数: {X_moons.shape[0]}, 特征数: {X_moons.shape[1]}')
        print(f'  真实类别数: {len(np.unique(y_moons))}')
        plot_original_data(X_moons, y_moons, os.path.join(output_dir_moons, 'original_data_moons.png'))

        print('\n[moons-2] 三种链接方式算法对比 (k=2, make_moons 数据集)')
        # 在 make_moons 数据集上保留聚类结果图片
        results_moons = compare_algorithms(X_moons, y_moons, n_clusters=2, output_dir=output_dir_moons, plot=True)

        print('\n[moons-3] 拓展: 测试不同聚类数k (make_moons 数据集)')
        k_values_moons = [2, 3, 4, 5, 6]
        test_different_k(X_moons, y_moons, k_values_moons, output_dir_moons)

        print('\n分析 (make_moons 数据集):')
        best_linkage_moons = max(results_moons.keys(), key=lambda k: results_moons[k]['ari'])
        print('  - make_moons 数据集为非凸形状簇，更考验聚类算法的鲁棒性')
        print(f'  - 在 make_moons 数据集上，{best_linkage_moons}-linkage 的 ARI 最高')
        print('  - 可以观察不同链接方式在非球形簇上的优劣')

        print('\n' + '='*60)
        print('实验完成!')
        print('='*60)
        
    except (TypeError, AttributeError) as e:
        print('\n⚠️  检测到未完成的TODO!')
        print('请完成以下函数:')
        print('  1. _single_linkage_distance() - 最小距离')
        print('  2. _complete_linkage_distance() - 最大距离')
        print('  3. _average_linkage_distance() - 平均距离')
        print(f'\n错误信息: {e}')
