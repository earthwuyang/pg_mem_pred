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

# 创建两个子图共享 x 轴
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
fig.subplots_adjust(hspace=0.1)  # 调整两个子图之间的间距

# 绘制上方图表 (y轴从1.32到1.325)
ax1.bar(x - 3*bar_width, values1, width=bar_width, label='XGBoost')
ax1.bar(x - 2*bar_width, values2, width=bar_width, label='ZeroShot')
ax1.bar(x - bar_width, values3, width=bar_width, label='queryformer')
ax1.bar(x, values4, width=bar_width, label='HeteroGraphRGCN')
ax1.bar(x + bar_width, values5, width=bar_width, label='HeteroGraphConv')
ax1.bar(x + 2*bar_width, values6, width=bar_width, label='ZeroShot (more data)')
ax1.bar(x + 3*bar_width, values7, width=bar_width, label='GIN')
ax1.set_ylim(1.32, 1.325)  # 设置上方图的 y 轴范围
ax1.yaxis.set_ticks(np.arange(1.32, 1.326, 0.01))  # 设置更密集的y轴刻度

# 绘制下方图表 (y轴从1到1.15)
ax2.bar(x - 3*bar_width, values1, width=bar_width, label='XGBoost')
ax2.bar(x - 2*bar_width, values2, width=bar_width, label='ZeroShot')
ax2.bar(x - bar_width, values3, width=bar_width, label='queryformer')
ax2.bar(x, values4, width=bar_width, label='HeteroGraphRGCN')
ax2.bar(x + bar_width, values5, width=bar_width, label='HeteroGraphConv')
ax2.bar(x + 2*bar_width, values6, width=bar_width, label='ZeroShot (more data)')
ax2.bar(x + 3*bar_width, values7, width=bar_width, label='GIN')
ax2.set_ylim(1.0, 1.15)  # 设置下方图的 y 轴范围

# 添加断轴效果
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.tick_params(labeltop=False)  # 关闭上方图的顶部刻度
ax2.xaxis.tick_bottom()

# 在两个图表之间添加断线符号
d = .005  # 断线的大小
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)        # 左侧断线
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # 右侧断线

kwargs.update(transform=ax2.transAxes)  # 更新坐标系为下方图表
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # 左侧断线
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # 右侧断线

# 添加标签和标题
ax2.set_xlabel('Datasets')
ax1.set_ylabel('Median Q-error')
ax2.set_ylabel('Median Q-error')
fig.suptitle('Prediction Accuracy')
plt.xticks(x, categories)
ax1.legend(loc='upper right')

# 显示图形
plt.savefig('draw_2.png', dpi=300)