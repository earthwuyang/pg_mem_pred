import matplotlib.pyplot as plt
import numpy as np

# 示例数据
category = 'TPC-DS'  # 仅显示 tpcds 组
values = [1.1411, 1.3235, 1.0729, 1.0586, 1.0948, 1.0663, 1.0437]  # 各种方法在 tpcds 数据集上的 Q-error
labels = ['XGBoost', 'ZeroShot', 'queryformer', 'Hetero\nGraphRGCN', 'Hetero\nGraphConv', 'ZeroShot\n(more data)', 'GIN']

# 设置柱的宽度和位置
bar_width = 0.4
x = np.arange(len(values))  # 生成 x 轴的位置

# 定义截断阈值
cutoff_value = 1.15

# 将数据进行截断
clipped_values = [min(value, cutoff_value) for value in values]

# 绘制柱状图
plt.bar(x, clipped_values, width=bar_width, color='orange')

# 添加截断的真实值标注
for i, value in enumerate(values):
    if value > cutoff_value:
        plt.text(i, 1.14, f"{value:.4f}", ha='center', color='black')

# 设置 X 轴的标签为各个方法的名称
plt.xticks(x, labels, rotation=45, ha='right')  # X 轴标签旋转45度，以便更易读
plt.xlabel('Methods')
plt.ylabel('Median Q-error')
plt.title(f'Prediction Error on {category}')

# 设置 Y 轴范围
plt.ylim(1.0, 1.15)  # 设置Y轴范围，以突出截断效果

# 显示图形
plt.tight_layout()  # 自动调整布局以防止标签重叠
plt.savefig('chart_tpcds.png', dpi=300)
plt.show()
