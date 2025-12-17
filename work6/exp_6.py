import pandas as pd
import numpy as np
from collections import Counter
import os


# ==================== 决策树节点类 ====================
class TreeNode:
    """
    决策树节点类
    
    属性：
      - is_leaf: 是否为叶子节点
      - prediction: 叶子节点的预测值（类别）
      - feature: 分裂属性名称
      - threshold: 分裂阈值（仅连续属性使用）
      - is_continuous: 该分裂属性是否连续
      - children: 子节点字典 {值 -> 子节点} 或 {">threshold" -> 右子树, "<=threshold" -> 左子树}
      - majority_class: 该节点训练子集的多数类
      - samples: 该节点包含的样本数
      - class_distribution: 该节点的类别分布 {类 -> 数量}
    """
    def __init__(self):
        self.is_leaf = False
        self.prediction = None  # 叶子节点预测
        self.feature = None  # 分裂特征名
        self.threshold = None  # 连续属性分裂阈值
        self.is_continuous = False  # 是否为连续属性分裂
        self.children = {}  # 子节点
        self.majority_class = None  # 多数类
        self.samples = 0  # 样本数
        self.class_distribution = {}  # 类分布


# ==================== ID3 决策树 ====================
class ID3DecisionTree:
    """
    ID3 决策树（处理离散属性）
    
    使用信息增益 (Information Gain) 选择分裂属性
    """
    
    def __init__(self, random_state=42):
        self.tree = None
        self.feature_names = None
        self.class_label = None
        self.random_state = random_state
        np.random.seed(random_state)
    
    def entropy(self, labels):
        """计算信息熵 H(S) = -∑ p_i * log2(p_i)"""
        if len(labels) == 0:
            return 0.0
        value_counts = Counter(labels)
        entropy_val = 0.0
        total = len(labels)
        for count in value_counts.values():
            if count > 0:
                p = count / total
                entropy_val -= p * np.log2(p)
        return entropy_val
    
    def information_gain(self, parent_labels, child_labels_list):
        """
        计算信息增益
        Gain(S, A) = Entropy(S) - ∑ |S_v|/|S| * Entropy(S_v)
        """
        parent_entropy = self.entropy(parent_labels)
        
        total_samples = len(parent_labels)
        if total_samples == 0:
            return 0.0
        
        weighted_child_entropy = 0.0
        for child_labels in child_labels_list:
            if len(child_labels) > 0:
                weight = len(child_labels) / total_samples
                weighted_child_entropy += weight * self.entropy(child_labels)
        
        return parent_entropy - weighted_child_entropy
    
    def build_tree(self, X, y, available_features):
        """
        递归构建 ID3 决策树
        
        停止条件：
          1. 子集标签全相同
          2. 没有可用属性
          3. 子集为空
        """
        node = TreeNode()
        node.samples = len(y)
        node.class_distribution = dict(Counter(y))
        node.majority_class = Counter(y).most_common(1)[0][0] if len(y) > 0 else None
        
        # 停止条件 1：标签全相同
        if len(np.unique(y)) <= 1:
            node.is_leaf = True
            node.prediction = y.iloc[0] if len(y) > 0 else None
            return node
        
        # 停止条件 2 和 3：没有属性或子集为空
        if len(available_features) == 0 or len(X) == 0:
            node.is_leaf = True
            node.prediction = node.majority_class
            return node
        
        # 选择最优分裂属性（信息增益最大）
        best_feature = None
        best_gain = -1
        
        for feature in available_features:
            # 按特征值分组
            feature_values = X[feature].unique()
            child_labels_list = []
            
            for value in feature_values:
                mask = X[feature] == value
                child_labels = y[mask]
                child_labels_list.append(child_labels.values)
            
            gain = self.information_gain(y.values, child_labels_list)
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        
        # 如果没有属性能提供增益，设为叶子
        if best_feature is None:
            node.is_leaf = True
            node.prediction = node.majority_class
            return node
        
        # 设置分裂属性
        node.feature = best_feature
        node.children = {}
        
        # 递归构建子树
        remaining_features = [f for f in available_features if f != best_feature]
        
        for value in X[best_feature].unique():
            mask = X[best_feature] == value
            X_subset = X[mask].reset_index(drop=True)
            y_subset = y[mask].reset_index(drop=True)
            
            if len(X_subset) == 0:
                # 该分支无数据，创建叶子节点（多数类）
                leaf = TreeNode()
                leaf.is_leaf = True
                leaf.prediction = node.majority_class
                leaf.samples = 0
                node.children[value] = leaf
            else:
                node.children[value] = self.build_tree(X_subset, y_subset, remaining_features)
        
        return node
    
    def fit(self, X, y):
        """训练 ID3 决策树"""
        self.feature_names = X.columns.tolist()
        self.class_label = y.name if y.name else "label"
        available_features = [f for f in self.feature_names]
        self.tree = self.build_tree(X.reset_index(drop=True), y.reset_index(drop=True), available_features)
        return self
    
    def predict_one(self, x):
        """预测单个样本"""
        node = self.tree
        
        while not node.is_leaf:
            feature = node.feature
            feature_value = x.get(feature, None)
            
            # 如果特征值不在子节点中，使用多数类回退
            if feature_value not in node.children:
                return node.majority_class
            
            node = node.children[feature_value]
        
        return node.prediction
    
    def predict(self, X):
        """预测数据集"""
        predictions = []
        for idx, row in X.iterrows():
            pred = self.predict_one(row)
            predictions.append(pred)
        return np.array(predictions)
    
    def get_tree_stats(self, node=None):
        """获取树的节点数和深度"""
        if node is None:
            node = self.tree
        
        if node.is_leaf:
            return 1, 0
        
        total_nodes = 1
        max_depth = 0
        
        for child in node.children.values():
            child_nodes, child_depth = self.get_tree_stats(child)
            total_nodes += child_nodes
            max_depth = max(max_depth, child_depth + 1)
        
        return total_nodes, max_depth


# ==================== C4.5 决策树 ====================
class C45DecisionTree:
    """
    C4.5 决策树（处理离散 + 连续属性）
    
    使用信息增益率 (Information Gain Ratio) 选择分裂属性
    对连续属性自动寻找最优分裂阈值
    """
    
    def __init__(self, random_state=42):
        self.tree = None
        self.feature_names = None
        self.class_label = None
        self.random_state = random_state
        self.continuous_features = set()
        np.random.seed(random_state)
    
    def entropy(self, labels):
        """计算信息熵"""
        if len(labels) == 0:
            return 0.0
        value_counts = Counter(labels)
        entropy_val = 0.0
        total = len(labels)
        for count in value_counts.values():
            if count > 0:
                p = count / total
                entropy_val -= p * np.log2(p)
        return entropy_val
    
    def split_info(self, sizes):
        """计算分裂信息 SplitInfo(S, A) = -∑ |S_v|/|S| * log2(|S_v|/|S|)"""
        total = sum(sizes)
        if total == 0:
            return 0.0
        split_info_val = 0.0
        for size in sizes:
            if size > 0:
                p = size / total
                split_info_val -= p * np.log2(p)
        return split_info_val
    
    def information_gain_ratio(self, parent_labels, child_labels_list):
        """
        计算信息增益率
        GainRatio(S, A) = Gain(S, A) / SplitInfo(S, A)
        """
        parent_entropy = self.entropy(parent_labels)
        total_samples = len(parent_labels)
        
        if total_samples == 0:
            return 0.0
        
        weighted_child_entropy = 0.0
        child_sizes = []
        
        for child_labels in child_labels_list:
            child_sizes.append(len(child_labels))
            if len(child_labels) > 0:
                weight = len(child_labels) / total_samples
                weighted_child_entropy += weight * self.entropy(child_labels)
        
        gain = parent_entropy - weighted_child_entropy
        split_info = self.split_info(child_sizes)
        
        if split_info == 0:
            return 0.0
        
        gain_ratio = gain / split_info
        return gain_ratio
    
    def find_best_continuous_split(self, X_col, y, feature_name):
        """
        为连续属性寻找最优分裂阈值
        候选阈值：排序后相邻不同值的中点
        """
        X_col_sorted = sorted(set(X_col.values))
        
        if len(X_col_sorted) <= 1:
            return None, -1
        
        thresholds = []
        for i in range(len(X_col_sorted) - 1):
            threshold = (X_col_sorted[i] + X_col_sorted[i + 1]) / 2
            thresholds.append(threshold)
        
        best_threshold = None
        best_gain_ratio = -1
        
        for threshold in thresholds:
            # 分裂：X <= threshold 和 X > threshold
            left_mask = X_col <= threshold
            right_mask = X_col > threshold
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            # 如果某一侧为空，跳过
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            
            child_labels_list = [y_left.values, y_right.values]
            gain_ratio = self.information_gain_ratio(y.values, child_labels_list)
            
            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_threshold = threshold
        
        return best_threshold, best_gain_ratio
    
    def build_tree(self, X, y, available_features):
        """递归构建 C4.5 决策树"""
        node = TreeNode()
        node.samples = len(y)
        node.class_distribution = dict(Counter(y))
        node.majority_class = Counter(y).most_common(1)[0][0] if len(y) > 0 else None
        
        # 停止条件 1：标签全相同
        if len(np.unique(y)) <= 1:
            node.is_leaf = True
            node.prediction = y.iloc[0] if len(y) > 0 else None
            return node
        
        # 停止条件 2 和 3：没有属性或子集为空
        if len(available_features) == 0 or len(X) == 0:
            node.is_leaf = True
            node.prediction = node.majority_class
            return node
        
        # 选择最优分裂属性
        best_feature = None
        best_gain_ratio = -1
        best_threshold = None
        best_is_continuous = False
        
        for feature in available_features:
            if feature in self.continuous_features:
                # 连续属性：寻找最优阈值
                threshold, gain_ratio = self.find_best_continuous_split(X[feature], y, feature)
                if threshold is not None and gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature = feature
                    best_threshold = threshold
                    best_is_continuous = True
            else:
                # 离散属性：枚举所有值
                feature_values = X[feature].unique()
                child_labels_list = []
                
                for value in feature_values:
                    mask = X[feature] == value
                    child_labels = y[mask]
                    child_labels_list.append(child_labels.values)
                
                gain_ratio = self.information_gain_ratio(y.values, child_labels_list)
                
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature = feature
                    best_is_continuous = False
        
        # 如果没有属性能提供增益，设为叶子
        if best_feature is None:
            node.is_leaf = True
            node.prediction = node.majority_class
            return node
        
        # 设置分裂属性
        node.feature = best_feature
        node.is_continuous = best_is_continuous
        node.threshold = best_threshold
        node.children = {}
        
        # 递归构建子树
        remaining_features = [f for f in available_features if f != best_feature]
        
        if best_is_continuous:
            # 连续属性分裂：二分
            left_mask = X[best_feature] <= best_threshold
            right_mask = X[best_feature] > best_threshold
            
            X_left = X[left_mask].reset_index(drop=True)
            y_left = y[left_mask].reset_index(drop=True)
            
            X_right = X[right_mask].reset_index(drop=True)
            y_right = y[right_mask].reset_index(drop=True)
            
            if len(X_left) > 0:
                node.children["<="] = self.build_tree(X_left, y_left, remaining_features)
            else:
                leaf = TreeNode()
                leaf.is_leaf = True
                leaf.prediction = node.majority_class
                node.children["<="] = leaf
            
            if len(X_right) > 0:
                node.children[">"] = self.build_tree(X_right, y_right, remaining_features)
            else:
                leaf = TreeNode()
                leaf.is_leaf = True
                leaf.prediction = node.majority_class
                node.children[">"] = leaf
        else:
            # 离散属性分裂：多值
            for value in X[best_feature].unique():
                mask = X[best_feature] == value
                X_subset = X[mask].reset_index(drop=True)
                y_subset = y[mask].reset_index(drop=True)
                
                if len(X_subset) == 0:
                    leaf = TreeNode()
                    leaf.is_leaf = True
                    leaf.prediction = node.majority_class
                    node.children[value] = leaf
                else:
                    node.children[value] = self.build_tree(X_subset, y_subset, remaining_features)
        
        return node
    
    def fit(self, X, y):
        """训练 C4.5 决策树"""
        self.feature_names = X.columns.tolist()
        self.class_label = y.name if y.name else "label"
        
        # 自动识别连续属性
        from pandas.api.types import is_numeric_dtype
        for col in self.feature_names:
            if is_numeric_dtype(X[col]):
                self.continuous_features.add(col)
        
        available_features = [f for f in self.feature_names]
        self.tree = self.build_tree(X.reset_index(drop=True), y.reset_index(drop=True), available_features)
        return self
    
    def predict_one(self, x):
        """预测单个样本"""
        node = self.tree
        
        while not node.is_leaf:
            feature = node.feature
            feature_value = x.get(feature, None)
            
            if feature_value is None:
                return node.majority_class
            
            if node.is_continuous:
                # 连续属性
                if feature_value <= node.threshold:
                    key = "<="
                else:
                    key = ">"
                
                if key not in node.children:
                    return node.majority_class
                
                node = node.children[key]
            else:
                # 离散属性
                if feature_value not in node.children:
                    return node.majority_class
                
                node = node.children[feature_value]
        
        return node.prediction
    
    def predict(self, X):
        """预测数据集"""
        predictions = []
        for idx, row in X.iterrows():
            pred = self.predict_one(row)
            predictions.append(pred)
        return np.array(predictions)
    
    def get_tree_stats(self, node=None):
        """获取树的节点数和深度"""
        if node is None:
            node = self.tree
        
        if node.is_leaf:
            return 1, 0
        
        total_nodes = 1
        max_depth = 0
        
        for child in node.children.values():
            child_nodes, child_depth = self.get_tree_stats(child)
            total_nodes += child_nodes
            max_depth = max(max_depth, child_depth + 1)
        
        return total_nodes, max_depth


# ==================== CART 决策树 (使用基尼指数) ====================
class CARTDecisionTree:
    """
    CART 决策树 (Classification and Regression Tree)
    
    特点：
      - 总是产生二叉树（即使离散属性也二分）
      - 使用基尼指数 (Gini Index) 作为分裂准则
      - 同时支持离散和连续属性
    
    基尼指数公式：
      Gini(D) = 1 - ∑ (|C_k| / |D|)^2
      其中 C_k 是类 k 的样本集
    """
    
    def __init__(self, random_state=42):
        self.tree = None
        self.feature_names = None
        self.class_label = None
        self.random_state = random_state
        self.continuous_features = set()
        np.random.seed(random_state)
    
    def gini_index(self, labels):
        """计算基尼指数 Gini(D) = 1 - ∑ p_k^2"""
        if len(labels) == 0:
            return 0.0
        value_counts = Counter(labels)
        gini = 1.0
        total = len(labels)
        for count in value_counts.values():
            if count > 0:
                p = count / total
                gini -= p * p
        return gini
    
    def weighted_gini(self, left_labels, right_labels):
        """计算二分后的加权基尼指数"""
        total = len(left_labels) + len(right_labels)
        if total == 0:
            return 0.0
        left_weight = len(left_labels) / total
        right_weight = len(right_labels) / total
        return left_weight * self.gini_index(left_labels) + \
               right_weight * self.gini_index(right_labels)
    
    def find_best_split(self, X, y, available_features):
        """
        找到最优二分分裂
        返回: (最优特征, 最优阈值, 最小加权基尼指数, 是否为连续属性)
        """
        best_feature = None
        best_threshold = None
        best_gini = float('inf')
        best_is_continuous = False
        
        for feature in available_features:
            if feature in self.continuous_features:
                # 连续属性：枚举阈值
                X_col_sorted = sorted(set(X[feature].values))
                if len(X_col_sorted) <= 1:
                    continue
                
                thresholds = [(X_col_sorted[i] + X_col_sorted[i + 1]) / 2 
                             for i in range(len(X_col_sorted) - 1)]
                
                for threshold in thresholds:
                    left_mask = X[feature] <= threshold
                    right_mask = X[feature] > threshold
                    
                    if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                        continue
                    
                    weighted_gini_val = self.weighted_gini(
                        y[left_mask].values,
                        y[right_mask].values
                    )
                    
                    if weighted_gini_val < best_gini:
                        best_gini = weighted_gini_val
                        best_feature = feature
                        best_threshold = threshold
                        best_is_continuous = True
            else:
                # 离散属性：尝试所有可能的二分
                unique_values = X[feature].unique()
                if len(unique_values) <= 1:
                    continue
                
                # 对于离散属性，我们尝试每个值作为"左分支"，其他作为"右分支"
                for split_value in unique_values:
                    left_mask = X[feature] == split_value
                    right_mask = X[feature] != split_value
                    
                    if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                        continue
                    
                    weighted_gini_val = self.weighted_gini(
                        y[left_mask].values,
                        y[right_mask].values
                    )
                    
                    if weighted_gini_val < best_gini:
                        best_gini = weighted_gini_val
                        best_feature = feature
                        best_threshold = split_value  # 离散属性的"阈值"是具体的取值
                        best_is_continuous = False
        
        return best_feature, best_threshold, best_gini, best_is_continuous
    
    def build_tree(self, X, y, available_features):
        """递归构建 CART 二叉树"""
        node = TreeNode()
        node.samples = len(y)
        node.class_distribution = dict(Counter(y))
        node.majority_class = Counter(y).most_common(1)[0][0] if len(y) > 0 else None
        
        # 停止条件 1：标签全相同
        if len(np.unique(y)) <= 1:
            node.is_leaf = True
            node.prediction = y.iloc[0] if len(y) > 0 else None
            return node
        
        # 停止条件 2 和 3：没有属性或子集为空
        if len(available_features) == 0 or len(X) == 0:
            node.is_leaf = True
            node.prediction = node.majority_class
            return node
        
        # 找最优分裂
        best_feature, best_threshold, best_gini, best_is_continuous = \
            self.find_best_split(X, y, available_features)
        
        # 如果没有找到好的分裂，设为叶子
        if best_feature is None:
            node.is_leaf = True
            node.prediction = node.majority_class
            return node
        
        # 设置分裂属性
        node.feature = best_feature
        node.threshold = best_threshold
        node.is_continuous = best_is_continuous
        node.children = {}
        
        # 二分裂
        remaining_features = [f for f in available_features if f != best_feature]
        
        if best_is_continuous:
            # 连续属性二分
            left_mask = X[best_feature] <= best_threshold
            right_mask = X[best_feature] > best_threshold
        else:
            # 离散属性二分：一个值 vs 其他值
            left_mask = X[best_feature] == best_threshold
            right_mask = X[best_feature] != best_threshold
        
        X_left = X[left_mask].reset_index(drop=True)
        y_left = y[left_mask].reset_index(drop=True)
        
        X_right = X[right_mask].reset_index(drop=True)
        y_right = y[right_mask].reset_index(drop=True)
        
        # 递归构建左右子树
        if len(X_left) > 0:
            node.children["0_left"] = self.build_tree(X_left, y_left, remaining_features)
        else:
            leaf = TreeNode()
            leaf.is_leaf = True
            leaf.prediction = node.majority_class
            node.children["0_left"] = leaf
        
        if len(X_right) > 0:
            node.children["1_right"] = self.build_tree(X_right, y_right, remaining_features)
        else:
            leaf = TreeNode()
            leaf.is_leaf = True
            leaf.prediction = node.majority_class
            node.children["1_right"] = leaf
        
        return node
    
    def fit(self, X, y):
        """训练 CART 决策树"""
        self.feature_names = X.columns.tolist()
        self.class_label = y.name if y.name else "label"
        
        # 自动识别连续属性
        from pandas.api.types import is_numeric_dtype
        for col in self.feature_names:
            if is_numeric_dtype(X[col]):
                self.continuous_features.add(col)
        
        available_features = [f for f in self.feature_names]
        self.tree = self.build_tree(X.reset_index(drop=True), y.reset_index(drop=True), available_features)
        return self
    
    def predict_one(self, x):
        """预测单个样本"""
        node = self.tree
        
        while not node.is_leaf:
            feature = node.feature
            feature_value = x.get(feature, None)
            
            if feature_value is None:
                return node.majority_class
            
            if node.is_continuous:
                # 连续属性
                if feature_value <= node.threshold:
                    key = "0_left"
                else:
                    key = "1_right"
            else:
                # 离散属性
                if feature_value == node.threshold:
                    key = "0_left"
                else:
                    key = "1_right"
            
            if key not in node.children:
                return node.majority_class
            
            node = node.children[key]
        
        return node.prediction
    
    def predict(self, X):
        """预测数据集"""
        predictions = []
        for idx, row in X.iterrows():
            pred = self.predict_one(row)
            predictions.append(pred)
        return np.array(predictions)
    
    def get_tree_stats(self, node=None):
        """获取树的节点数和深度"""
        if node is None:
            node = self.tree
        
        if node.is_leaf:
            return 1, 0
        
        total_nodes = 1
        max_depth = 0
        
        for child in node.children.values():
            child_nodes, child_depth = self.get_tree_stats(child)
            total_nodes += child_nodes
            max_depth = max(max_depth, child_depth + 1)
        
        return total_nodes, max_depth


# ==================== 剪枝函数 ====================
def reduced_error_pruning(tree_model, X_valid, y_valid):
    """
    简化后剪枝 (Reduced Error Pruning, REP)
    
    策略：
      1. 在验证集上计算当前树的精度
      2. 对每个内部节点尝试将其替换为叶子
      3. 若替换后验证集精度不下降，则保留剪枝
      4. 自底向上重复直到无法继续剪枝
    """
    def calculate_accuracy(model, X, y):
        predictions = model.predict(X)
        return np.mean(predictions == y.values)
    
    def prune_node(node, X_valid, y_valid, model):
        """递归剪枝"""
        if node.is_leaf:
            return False
        
        # 先对所有子节点尝试剪枝
        any_pruned = False
        for child in node.children.values():
            if prune_node(child, X_valid, y_valid, model):
                any_pruned = True
        
        # 计算剪枝前的精度
        acc_before = calculate_accuracy(model, X_valid, y_valid)
        
        # 尝试将当前节点替换为叶子
        old_is_leaf = node.is_leaf
        old_prediction = node.prediction
        old_feature = node.feature
        old_children = node.children
        
        node.is_leaf = True
        node.prediction = node.majority_class
        node.feature = None
        node.children = {}
        
        # 计算剪枝后的精度
        acc_after = calculate_accuracy(model, X_valid, y_valid)
        
        # 如果精度没有下降，保留剪枝
        if acc_after >= acc_before:
            return True
        else:
            # 恢复原状
            node.is_leaf = old_is_leaf
            node.prediction = old_prediction
            node.feature = old_feature
            node.children = old_children
            return any_pruned
    
    # 重复剪枝直到无法继续
    for _ in range(100):  # 最多迭代 100 次
        if not prune_node(tree_model.tree, X_valid, y_valid, tree_model):
            break
    
    return tree_model


# ==================== 工具函数 ====================
def load_data(filename, encoding='gbk'):
    """
    加载 CSV 数据
    
    参数：
      - filename: 文件名（支持相对路径）
      - encoding: 编码，默认 GBK
    
    返回：
      - X: 特征数据（不包括标签列）
      - y: 标签数据
    """
    try:
        # 如果是相对路径，拼接到脚本目录
        if not os.path.isabs(filename):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(script_dir, filename)
        
        # 尝试以 GBK 编码加载
        df = pd.read_csv(filename, encoding=encoding)
        
        # 最后一列为标签
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        return X, y
    
    except FileNotFoundError:
        print(f"[错误] 文件不存在: {filename}")
        print(f"      请检查文件名（大小写敏感）和路径")
        print(f"      脚本期望的位置: {os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)}")
        raise
    
    except UnicodeDecodeError as e:
        print(f"[错误] 编码错误: {filename}")
        print(f"      请确保文件是 GBK 编码")
        print(f"      错误信息: {e}")
        raise
    
    except Exception as e:
        print(f"[错误] 加载数据失败: {e}")
        raise


def accuracy(y_true, y_pred):
    """计算分类精度"""
    return np.mean(y_true == y_pred)


def main():
    print("=" * 70)
    print("决策树实验 (Decision Tree Experiment)")
    print("=" * 70)
    
    # ==================== 任务 1: ID3 决策树 ====================
    print("\n[任务 1] ID3 决策树（离散属性）")
    print("-" * 70)
    
    try:
        # 加载训练和测试数据
        X_train1, y_train1 = load_data("Watermelon-train1.csv")
        X_test1, y_test1 = load_data("Watermelon-test1.csv")
        
        print(f"训练集: {X_train1.shape[0]} 样本, {X_train1.shape[1]} 特征")
        print(f"测试集: {X_test1.shape[0]} 样本, {X_test1.shape[1]} 特征")
        
        # 训练 ID3
        id3_model = ID3DecisionTree(random_state=42)
        id3_model.fit(X_train1, y_train1)
        
        # 预测
        y_pred1 = id3_model.predict(X_test1)
        acc1 = accuracy(y_test1.values, y_pred1)
        
        print(f"ID3 test1 accuracy: {acc1:.2f}")
        
        # 记录剪枝前的统计
        nodes1_before, depth1_before = id3_model.get_tree_stats()
        
        # ==================== 任务 1.5: ID3 剪枝 ====================
        print("\n  [子任务] ID3 剪枝")
        
        # 从训练集分割出验证集 (20%)
        np.random.seed(42)
        valid_indices = np.random.choice(len(X_train1), size=int(0.2 * len(X_train1)), replace=False)
        train_indices = np.array([i for i in range(len(X_train1)) if i not in valid_indices])
        
        X_train1_train = X_train1.iloc[train_indices].reset_index(drop=True)
        y_train1_train = y_train1.iloc[train_indices].reset_index(drop=True)
        
        X_train1_valid = X_train1.iloc[valid_indices].reset_index(drop=True)
        y_train1_valid = y_train1.iloc[valid_indices].reset_index(drop=True)
        
        # 重新在较小训练集上训练
        id3_model_prune = ID3DecisionTree(random_state=42)
        id3_model_prune.fit(X_train1_train, y_train1_train)
        
        # 剪枝前精度（在测试集上）
        y_pred1_before = id3_model_prune.predict(X_test1)
        acc1_before = accuracy(y_test1.values, y_pred1_before)
        nodes1_before_prune, depth1_before_prune = id3_model_prune.get_tree_stats()
        
        # 执行剪枝
        id3_model_prune = reduced_error_pruning(id3_model_prune, X_train1_valid, y_train1_valid)
        
        # 剪枝后精度
        y_pred1_after = id3_model_prune.predict(X_test1)
        acc1_after = accuracy(y_test1.values, y_pred1_after)
        nodes1_after, depth1_after = id3_model_prune.get_tree_stats()
        
        print(f"ID3 before prune: acc={acc1_before:.2f}, nodes={nodes1_before_prune}, depth={depth1_before_prune}")
        print(f"ID3 after prune: acc={acc1_after:.2f}, nodes={nodes1_after}, depth={depth1_after}")
        
    except Exception as e:
        print(f"[任务 1 失败] {e}")
        return
    
    # ==================== 任务 2: C4.5 决策树 ====================
    print("\n[任务 2] C4.5 决策树")
    print("-" * 70)
    
    try:
        # 加载训练和测试数据
        X_train2, y_train2 = load_data("Watermelon-train2.csv")
        X_test2, y_test2 = load_data("Watermelon-test2.csv")
        
        print(f"训练集: {X_train2.shape[0]} 样本, {X_train2.shape[1]} 特征")
        print(f"测试集: {X_test2.shape[0]} 样本, {X_test2.shape[1]} 特征")
        
        # 训练 C4.5
        c45_model = C45DecisionTree(random_state=42)
        c45_model.fit(X_train2, y_train2)
        
        # 预测
        y_pred2 = c45_model.predict(X_test2)
        acc2 = accuracy(y_test2.values, y_pred2)
        
        print(f"C45 test2 accuracy: {acc2:.2f}")
        
        # ==================== 任务 2.5: C4.5 剪枝 ====================
        print("\n  [子任务] C4.5 剪枝")
        
        # 从训练集分割出验证集 (20%)
        np.random.seed(42)
        valid_indices = np.random.choice(len(X_train2), size=int(0.2 * len(X_train2)), replace=False)
        train_indices = np.array([i for i in range(len(X_train2)) if i not in valid_indices])
        
        X_train2_train = X_train2.iloc[train_indices].reset_index(drop=True)
        y_train2_train = y_train2.iloc[train_indices].reset_index(drop=True)
        
        X_train2_valid = X_train2.iloc[valid_indices].reset_index(drop=True)
        y_train2_valid = y_train2.iloc[valid_indices].reset_index(drop=True)
        
        # 重新在较小训练集上训练
        c45_model_prune = C45DecisionTree(random_state=42)
        c45_model_prune.fit(X_train2_train, y_train2_train)
        
        # 剪枝前精度（在测试集上）
        y_pred2_before = c45_model_prune.predict(X_test2)
        acc2_before = accuracy(y_test2.values, y_pred2_before)
        nodes2_before, depth2_before = c45_model_prune.get_tree_stats()
        
        # 执行剪枝
        c45_model_prune = reduced_error_pruning(c45_model_prune, X_train2_valid, y_train2_valid)
        
        # 剪枝后精度
        y_pred2_after = c45_model_prune.predict(X_test2)
        acc2_after = accuracy(y_test2.values, y_pred2_after)
        nodes2_after, depth2_after = c45_model_prune.get_tree_stats()
        
        print(f"C45 before prune: acc={acc2_before:.2f}, nodes={nodes2_before}, depth={depth2_before}")
        print(f"C45 after prune: acc={acc2_after:.2f}, nodes={nodes2_after}, depth={depth2_after}")
        
    except Exception as e:
        print(f"[任务 2 失败] {e}")
        return
    
    # ==================== 任务 3: CART 决策树 ====================
    print("\n[任务 3] CART 决策树")
    print("-" * 70)
    
    try:
        # 使用与 C4.5 相同的数据
        print(f"训练集: {X_train2.shape[0]} 样本, {X_train2.shape[1]} 特征")
        print(f"测试集: {X_test2.shape[0]} 样本, {X_test2.shape[1]} 特征")
        
        # 训练 CART
        cart_model = CARTDecisionTree(random_state=42)
        cart_model.fit(X_train2, y_train2)
        
        # 预测
        y_pred_cart = cart_model.predict(X_test2)
        acc_cart = accuracy(y_test2.values, y_pred_cart)
        
        print(f"CART test2 accuracy: {acc_cart:.2f}")
        
        # ==================== 任务 3.5: CART 剪枝 ====================
        print("\n  [子任务] CART 剪枝")
        
        # 从训练集分割出验证集 (20%)
        np.random.seed(42)
        valid_indices = np.random.choice(len(X_train2), size=int(0.2 * len(X_train2)), replace=False)
        train_indices = np.array([i for i in range(len(X_train2)) if i not in valid_indices])
        
        X_train2_train = X_train2.iloc[train_indices].reset_index(drop=True)
        y_train2_train = y_train2.iloc[train_indices].reset_index(drop=True)
        
        X_train2_valid = X_train2.iloc[valid_indices].reset_index(drop=True)
        y_train2_valid = y_train2.iloc[valid_indices].reset_index(drop=True)
        
        # 重新在较小训练集上训练
        cart_model_prune = CARTDecisionTree(random_state=42)
        cart_model_prune.fit(X_train2_train, y_train2_train)
        
        # 剪枝前精度（在测试集上）
        y_pred_cart_before = cart_model_prune.predict(X_test2)
        acc_cart_before = accuracy(y_test2.values, y_pred_cart_before)
        nodes_cart_before, depth_cart_before = cart_model_prune.get_tree_stats()
        
        # 执行剪枝
        cart_model_prune = reduced_error_pruning(cart_model_prune, X_train2_valid, y_train2_valid)
        
        # 剪枝后精度
        y_pred_cart_after = cart_model_prune.predict(X_test2)
        acc_cart_after = accuracy(y_test2.values, y_pred_cart_after)
        nodes_cart_after, depth_cart_after = cart_model_prune.get_tree_stats()
        
        print(f"CART before prune: acc={acc_cart_before:.2f}, nodes={nodes_cart_before}, depth={depth_cart_before}")
        print(f"CART after prune: acc={acc_cart_after:.2f}, nodes={nodes_cart_after}, depth={depth_cart_after}")
        
    except Exception as e:
        print(f"[任务 3 失败] {e}")
        return
    
    print("\n" + "=" * 70)
    print("实验完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
