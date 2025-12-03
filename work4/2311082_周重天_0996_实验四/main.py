import numpy as np

# ================================================================================
# 【基础要求】5.1 算法与流程
# ================================================================================

# ============ 分层采样 ============
def stratified_split(X, y, train_ratio=0.7):
    """按类别比例划分训练集与测试集 (7:3)"""
    train_X, train_y = [], []
    test_X, test_y = [], []

    for class_label in np.unique(y):
        indices = np.where(y == class_label)[0]
        np.random.shuffle(indices)
        n_train = int(len(indices) * train_ratio)

        train_X.extend(X[indices[:n_train]])
        train_y.extend(y[indices[:n_train]])
        test_X.extend(X[indices[n_train:]])
        test_y.extend(y[indices[n_train:]])

    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)


# ============ 高斯朴素贝叶斯 ============
class GaussianNaiveBayes:
    def fit(self, X, y):
        """训练：计算每个类别的先验概率和高斯分布参数"""
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0)
            self.priors[c] = len(X_c) / len(X)

    def predict(self, X):
        """预测：计算后验概率，选择最大的类别"""
        def gaussian_pdf(x, mean, var):
            var = np.clip(var, 1e-10, None)  # 防止方差过小
            return np.exp(-((x - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)

        predictions = []
        for x in X:
            posteriors = {}
            for c in self.classes:
                prior = np.log(self.priors[c])
                posterior = prior + np.sum(np.log(gaussian_pdf(x, self.mean[c], self.var[c]) + 1e-10))
                posteriors[c] = posterior
            predictions.append(max(posteriors, key=posteriors.get))

        return np.array(predictions)

    def predict_proba(self, X):
        """返回各类别的预测概率"""
        def gaussian_pdf(x, mean, var):
            var = np.clip(var, 1e-10, None)
            return np.exp(-((x - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)

        proba = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                posterior = prior + np.sum(np.log(gaussian_pdf(x, self.mean[c], self.var[c]) + 1e-10))
                posteriors.append(posterior)
            # 转换为概率（softmax）
            posteriors = np.array(posteriors)
            posteriors = np.exp(posteriors - np.max(posteriors))  # 数值稳定性
            posteriors = posteriors / posteriors.sum()
            proba.append(posteriors)

        return np.array(proba)


# ============ 基础要求的主流程 ============
print("\n" + "=" * 80)
print("【基础要求】5.1 算法与流程")
print("=" * 80)

data = np.loadtxt('semeion.data.txt')
X, y = data[:, :256], np.argmax(data[:, 256:], axis=1)

# 分层采样
train_X, train_y, test_X, test_y = stratified_split(X, y, train_ratio=0.7)

print(f"\n① 分层采样结果:")
print(f"   训练集大小: {len(train_X)} (占比: {len(train_X)/len(X)*100:.1f}%)")
print(f"   测试集大小: {len(test_X)} (占比: {len(test_X)/len(X)*100:.1f}%)")

# 训练朴素贝叶斯分类器
gnb = GaussianNaiveBayes()
gnb.fit(train_X, train_y)

print(f"\n② 高斯朴素贝叶斯分类器:")
print(f"   - 核心假设: 特征条件独立，特征服从高斯分布")
print(f"   - 先验概率 P(y): 已计算 {len(gnb.classes)} 个类别")
print(f"   - 高斯参数: 均值 μ 和方差 σ² (每个特征 {X.shape[1]} 维)")

# 预测
train_pred = gnb.predict(train_X)
test_pred = gnb.predict(test_X)

# 基础要求的评估
train_acc = np.mean(train_pred == train_y)
test_acc = np.mean(test_pred == test_y)

print(f"\n③ 核心步骤:")
print(f"   - 训练阶段: 计算先验概率和高斯分布参数 ✓")
print(f"   - 预测阶段: 计算后验概率，选择最大的类别 ✓")
print(f"\n   训练准确率: {train_acc:.4f}")
print(f"   测试准确率: {test_acc:.4f}")


# ================================================================================
# 【中级要求】6 混淆矩阵 + 精度/召回率/F值
# ================================================================================

print("\n" + "=" * 80)
print("【中级要求】6 混淆矩阵 + 精度/召回率/F值")
print("=" * 80)

# ============ 混淆矩阵计算 ============
def confusion_matrix(y_true, y_pred, n_classes=10):
    """计算混淆矩阵"""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[int(true)][int(pred)] += 1
    return cm


# ============ 精度、召回率、F1值计算 ============
def precision_recall_f1(cm):
    """计算精度、召回率、F1值"""
    n_classes = cm.shape[0]
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)

    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision[i] + recall[i] > 0:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        else:
            f1[i] = 0

    return precision, recall, f1


# ============ 中级要求的评估 ============
cm = confusion_matrix(test_y, test_pred)

print("\n1. 混淆矩阵 (行=真实类别, 列=预测类别):")
print("-" * 80)
print(cm)

precision, recall, f1 = precision_recall_f1(cm)

print("\n2. 各类别的精度、召回率、F1值:")
print("-" * 80)
print(f"{'类别':<6} {'精度':<12} {'召回率':<12} {'F1值':<12}")
print("-" * 80)
for i in range(10):
    print(f"{i:<6} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f}")

print("-" * 80)
print(f"{'平均':<6} {precision.mean():<12.4f} {recall.mean():<12.4f} {f1.mean():<12.4f}")
print("=" * 80)


# ================================================================================
# 【高级要求】7 多分类 ROC 曲线绘制与 AUC 值计算
# ================================================================================

print("\n" + "=" * 80)
print("【高级要求】7 多分类 ROC 曲线绘制与 AUC 值计算")
print("=" * 80)

# ============ ROC曲线和AUC计算 ============
def calculate_roc_auc(y_true, y_score, n_classes=10):
    """
    计算多分类问题的 ROC 曲线和 AUC 值
    使用 One-vs-Rest 方法
    """
    fpr_dict = {}  # False Positive Rate
    tpr_dict = {}  # True Positive Rate
    auc_dict = {}
    thresholds_dict = {}

    for i in range(n_classes):
        # 转换为二分类问题 (class_i vs rest)
        y_binary = (y_true == i).astype(int)
        y_score_binary = y_score[:, i]

        # 按得分从高到低排序
        sorted_indices = np.argsort(y_score_binary)[::-1]
        y_binary_sorted = y_binary[sorted_indices]
        y_score_sorted = y_score_binary[sorted_indices]

        # 计算 TPR 和 FPR
        n_pos = np.sum(y_binary)
        n_neg = len(y_binary) - n_pos

        if n_pos == 0 or n_neg == 0:
            continue

        tpr = []
        fpr = []
        thresholds = []

        tp = 0
        fp = 0

        # 添加起点 (0, 0)
        fpr.append(0)
        tpr.append(0)
        thresholds.append(y_score_sorted[0] + 1)

        for j in range(len(y_binary_sorted)):
            if y_binary_sorted[j] == 1:
                tp += 1
            else:
                fp += 1

            tpr.append(tp / n_pos)
            fpr.append(fp / n_neg)
            thresholds.append(y_score_sorted[j])

        fpr_dict[i] = np.array(fpr)
        tpr_dict[i] = np.array(tpr)
        thresholds_dict[i] = np.array(thresholds)

        # 计算 AUC (梯形法则)
        auc = 0
        for j in range(1, len(fpr)):
            auc += (fpr[j] - fpr[j-1]) * (tpr[j] + tpr[j-1]) / 2

        auc_dict[i] = auc

    return fpr_dict, tpr_dict, auc_dict, thresholds_dict


# 获取预测概率
y_proba = gnb.predict_proba(test_X)

# 计算 ROC 曲线和 AUC
fpr_dict, tpr_dict, auc_dict, _ = calculate_roc_auc(test_y, y_proba)

print("\n1. AUC 值统计:")
print("-" * 80)
print(f"{'类别':<6} {'AUC值':<12} {'性能评价':<15}")
print("-" * 80)

auc_values = []
for i in range(10):
    if i in auc_dict:
        auc = auc_dict[i]
        auc_values.append(auc)
        if auc >= 0.9:
            rating = "优秀"
        elif auc >= 0.8:
            rating = "好"
        elif auc >= 0.7:
            rating = "一般"
        else:
            rating = "较差"
        print(f"{i:<6} {auc:<12.4f} {rating:<15}")

print("-" * 80)
print(f"{'平均':<6} {np.mean(auc_values):<12.4f}")
print("=" * 80)

# ============ 绘制 ROC 曲线 ============
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端

    # 设置中文字体（macOS）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti SC', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图表（2x5 网格显示10个类别）
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    fig.suptitle('多分类 ROC 曲线 (One-vs-Rest)', fontsize=16, fontweight='bold')

    for i in range(10):
        ax = axes[i // 5, i % 5]

        if i in fpr_dict:
            fpr = fpr_dict[i]
            tpr = tpr_dict[i]
            auc = auc_dict[i]

            ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC={auc:.4f})')
            ax.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'Class {i} vs Rest')
            ax.legend(loc='lower right')
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'Class {i}\nNo positive samples',
                   ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=100, bbox_inches='tight')
    print("\n✓ ROC 曲线已保存到 'roc_curves.png'")

    # 绘制平均 ROC 曲线
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.Set3(np.linspace(0, 1, 10))
    for i in range(10):
        if i in fpr_dict:
            fpr = fpr_dict[i]
            tpr = tpr_dict[i]
            auc = auc_dict[i]
            ax.plot(fpr, tpr, color=colors[i], lw=2, label=f'Class {i} (AUC={auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('多分类 ROC 曲线 (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    plt.savefig('roc_curves_combined.png', dpi=100, bbox_inches='tight')
    print("✓ 组合 ROC 曲线已保存到 'roc_curves_combined.png'")

except ImportError:
    print("\n⚠ matplotlib 未安装，跳过 ROC 曲线绘制")
    print("可使用命令: pip install matplotlib")

print("\n" + "=" * 80)