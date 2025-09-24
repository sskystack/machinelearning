import numpy as np
import matplotlib.pyplot as plt #注意依赖库的安装

raw = np.loadtxt('semeion.data.txt')
X, y = raw[:, :256], raw[:, 256:]
y = np.argmax(y, axis=1)          # 独热 → 0~9 整数
print('样本数:', len(X), '像素数:', X.shape[1])

# 随机画 6 张图
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(X[i].reshape(16,16), cmap='gray')
    plt.title(y[i]); plt.axis('off')
plt.tight_layout(); plt.show()