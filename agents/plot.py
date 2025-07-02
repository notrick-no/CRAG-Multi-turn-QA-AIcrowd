import matplotlib.pyplot as plt
import numpy as np

# 数据准备
methods = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G"
]
accuracy = [0.25, 0.25, 0.2917, 0.375, 0.4167, 0.4583, 0.5317]
missing_rate = [0, 0, 0.01, 0.03, 0.03, 0.02, 0.1528]
hallucination_rate = [0.75, 0.75, 0.5833, 0.5, 0.4583, 0.4583, 0.3155]
true_score = [-0.5, -0.5, -0.29, -0.12, -0.0417, 0, 0.21]

x = np.arange(len(methods))  # x轴位置
width = 0.2  # 柱状图宽度

# 创建图表
plt.figure(figsize=(12, 6))

# 绘制折线
plt.plot(x, accuracy, 'o-', label='accuracy', color='blue')
plt.plot(x, missing_rate, 's-', label='missing rate', color='green')
plt.plot(x, hallucination_rate, 'D-', label='hallucination rate', color='red')
plt.plot(x, true_score, '*-', label='true score', color='purple')

# 添加标签和标题
plt.title('Multi-source RAG Performance', fontsize=14)
plt.xlabel('Method', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.xticks(x, methods, rotation=45, ha='right')
plt.legend()

# 调整布局
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.6)

# 显示图表
plt.show()