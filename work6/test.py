import pandas as pd
test1 = pd.read_csv ("Watermelon-train2.csv", encoding="gbk") # 以 test1 为例，其他数据集同理
# 3. 快速验证：看数据是否加载成功
print ("1. 数据集维度（样本数 × 属性数）：", test1.shape)
print ("2. 前 3 行数据（快速了解格式）：\n", test1.head (3))