import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm
# for font in fm.fontManager.ttflist:
#     print(font.name)
# exit()
from matplotlib import font_manager

# # Set font for Chinese characters (SimHei is a commonly used Chinese font)
# font = font_manager.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')  # or any other available font
# plt.rcParams['font.family'] = font.get_name()
font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

# Plot the memory metrics
def plot_memory_metrics(metrics, result_dir, num_queries):
    plt.figure(figsize=(10, 6))
    
    # Convert time to relative
    start_time = min(
        metrics['naive']['time'][0] if metrics['naive']['time'] else float('inf'),
        metrics['memory_based']['time'][0] if metrics['memory_based']['time'] else float('inf')
    )
    naive_time = [t - start_time for t in metrics['naive']['time']]
    memory_based_time = [t - start_time for t in metrics['memory_based']['time']]
    
    # Plot naive
    if naive_time:
        plt.plot(naive_time, metrics['naive']['swap_mem'], label="默认策略的交换内存 (KB)", linestyle='--')
        plt.plot(naive_time, metrics['naive']['total_mem'], label="默认策略的总内存 (KB)")

    # Plot memory-based
    if memory_based_time:
        plt.plot(memory_based_time, metrics['memory_based']['swap_mem'], label="FFD策略的交换内存 (KB)", linestyle='--')
        plt.plot(memory_based_time, metrics['memory_based']['total_mem'], label="FFD策略的总内存 (KB)")
    
    plt.xlabel("时间（秒）", fontsize=16)
    plt.ylabel("内存 (KB)", fontsize=16)
    # plt.title("Swap and Total Memory Usage During Execution", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    # adjust x and y axis ticks font size
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(os.path.join(result_dir,f'{num_queries}_queries_memory_usage_chinese.png'))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_queries', type=int, help='number of queries', required=True)
args = parser.parse_args()

result_dir = './'
with open(f'{args.num_queries}_metrics.pkl', 'rb') as f:
    import pickle
    metrics = pickle.load(f)
plot_memory_metrics(metrics, result_dir, args.num_queries)