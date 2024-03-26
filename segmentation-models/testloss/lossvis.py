import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 设置绘图上下文和风格
sns.set(style="whitegrid", context="talk", palette="colorblind")

# 损失函数文件路径
file_paths = {
    "ACELoss": "loss_log_ace.txt",
    "Cross-Entropy Loss": "loss_log_cross.txt",
    "NLLLoss": "loss_log_null.txt",
    "MultiMarginLoss": "loss_log_MML.txt",
}

# 加载数据并进行预处理
dfs = {}
for label, path in file_paths.items():
    data = pd.read_csv(path, sep=": ", engine='python', header=None, names=["Epoch", "Loss"])
    data["Epoch"] = data["Epoch"].apply(lambda x: int(x.split(" ")[1]))
    data['Loss'] = data['Loss'].str.extract('([-+]?\d*\.\d+|\d+)').astype(float)
    # 仅保留Epoch 30及以下的数据
    data = data[data["Epoch"] <= 30]
    dfs[label] = data

# 标准化损失函数值
standard_scaler = StandardScaler()
for label in dfs.keys():
    dfs[label]["Standard Scaled Loss"] = standard_scaler.fit_transform(dfs[label][["Loss"]])

# 绘制标准化后的损失函数比较图
plt.figure(figsize=(10, 8))
markers = ['o-', 's--', '^-.', '*:']  # 组合标记符号和线型
for (label, df), marker in zip(dfs.items(), markers):
    plt.plot(df["Epoch"], df["Standard Scaled Loss"], marker, label=label, linewidth=2, markersize=8)

plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.legend(loc='best', frameon=True, framealpha=0.9, fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=1)
plt.tight_layout()

# 高亮特定的数据点或趋势
plt.annotate('Important Trend', xy=(16, -0.23), xytext=(16, 0.3),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=18, color='red')
plt.savefig('loss trend.png')
plt.show()
