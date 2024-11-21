import matplotlib.pyplot as plt

# Data
models = ['XGBoost', 'ZeroShot', 'QueryFormer', 'GIN', 'HeteroGraphConv']
median_qerrors = [1.2996, 1.1406, 1.17437216, 1.17993289, 1.249329]
colors = ['blue', 'orange', 'gray', 'red', 'brown'] 
# Plot
plt.figure(figsize=(8, 6))
plt.bar(models, median_qerrors, color=colors, width=0.5)
plt.xlabel('Models', fontsize=16)
plt.ylabel('Median Q-Error', fontsize=16)
# plt.title('Median Q-Error of Models on TPC-DS Memory Prediction', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(1, max(median_qerrors) + 0.03) 
plt.tight_layout()

# Show the chart
plt.savefig('queryformer_comparison.png')
