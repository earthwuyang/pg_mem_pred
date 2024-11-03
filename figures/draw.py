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
bar_width = 0.1                    # 柱的宽度
x = np.arange(len(categories))     # X轴每个类别的位置

# 绘制簇状柱形图
plt.bar(x - 3*bar_width, values1, width=bar_width, label='XGBoost')
plt.bar(x - 2*bar_width, values2, width=bar_width, label='ZeroShot')
plt.bar(x - bar_width, values3, width=bar_width, label='queryformer')
plt.bar(x , values4, width=bar_width, label='HeteroGraphRGCN')
plt.bar(x + bar_width, values5, width=bar_width, label='HeteroGraphConv')
plt.bar(x + 2*bar_width, values6, width=bar_width, label='ZeroShot\n(more training data)')
plt.bar(x + 3*bar_width, values7, width=bar_width, label='GIN')


# 添加标签和标题
plt.xlabel('Datasets')
plt.ylabel('Median Q-error ')
plt.title('Prediction Accuracy')
plt.xticks(x, categories)          # 设置X轴的标签为分类名称
plt.ylim(1, 1.33) 
plt.legend()                       # 添加图例

# 显示图形
plt.savefig('clustered_bar_chart.png')