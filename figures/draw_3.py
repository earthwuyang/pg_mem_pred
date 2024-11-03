import matplotlib.pyplot as plt
import numpy as np

# 示例数据
categories = ['tpcds', 'hepatitis', 'airline']  # X轴的分类
values1 = [1.1411, 1.0269, 1.0209]         # 第一组数据
values2 = [1.3235, 0, 1.0088]         # 第二组数据
values3 = [1.07296740616429, 0, 0]          # 第三组数据
values4 = [1.05863642692565, 0, 0]
values5 = [1.09486818313598, 1.01485121250152, 1.00740426778793]
values6 = [1.0663, 1.0229, 1.045]
values7 = [1.04370266199111, 1.01466882228851, 1.00857973098754]

# 设置柱的宽度和位置
bar_width = 0.1
x = np.arange(len(categories))

# 定义截断阈值
cutoff_value = 1.15

# 将数据进行截断
clipped_values1 = [min(value, cutoff_value) for value in values1]
clipped_values2 = [min(value, cutoff_value) for value in values2]
clipped_values3 = [min(value, cutoff_value) for value in values3]
clipped_values4 = [min(value, cutoff_value) for value in values4]
clipped_values5 = [min(value, cutoff_value) for value in values5]
clipped_values6 = [min(value, cutoff_value) for value in values6]
clipped_values7 = [min(value, cutoff_value) for value in values7]

# 绘制簇状柱形图
plt.bar(x - 3*bar_width, clipped_values1, width=bar_width, label='XGBoost')
plt.bar(x - 2*bar_width, clipped_values2, width=bar_width, label='ZeroShot')
plt.bar(x - bar_width, clipped_values3, width=bar_width, label='queryformer')
plt.bar(x, clipped_values4, width=bar_width, label='HeteroGraphRGCN')
plt.bar(x + bar_width, clipped_values5, width=bar_width, label='HeteroGraphConv')
plt.bar(x + 2*bar_width, clipped_values6, width=bar_width, label='ZeroShot (more data)')
plt.bar(x + 3*bar_width, clipped_values7, width=bar_width, label='GIN')

# 添加截断的真实值标注
for i, value in enumerate(values2):
    if value > cutoff_value:
        plt.text(x[i] - 2*bar_width, cutoff_value + 0.02, f"{value:.4f}", ha='center', color='red')

# 添加标签和标题
plt.xlabel('Datasets')
plt.ylabel('Median Q-error')
plt.title('Prediction Accuracy with Clipped Values')
plt.xticks(x, categories)
plt.ylim(1.0, 1.15)  # 设置Y轴范围，以突出截断效果
plt.legend(loc='upper right')

# 显示图形
plt.savefig('clipped_bar_chart.png', dpi=300)