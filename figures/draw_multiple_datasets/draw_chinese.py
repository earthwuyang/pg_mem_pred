import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

# Load the uploaded Excel file
file_path = '../multiple_datasets_pretrain.xlsx'
excel_data = pd.ExcelFile(file_path)

# Function to create a single big figure with 6 subfigures
def plot_all_combined_figures(data_dict, clip_value_list):
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    axs = axs.flatten()

    for idx, (sheet_name, data) in enumerate(data_dict.items()):
        settings = data.columns[1:]  # Assume first column is labels, the rest are settings
        models = data.iloc[:, 0]  # First column as model names
        n_clusters = len(settings)  # Number of clusters
        n_models = len(models)  # Number of models
        bar_width = 0.2
        x_positions = range(n_clusters)

        # Check if clip value is -1 and set y-axis to logarithmic if so
        use_log_scale = clip_value_list[idx] == -1

        if use_log_scale:
            # Use logarithmic values for the y-axis
            clipped_data = data.iloc[:, 1:].applymap(lambda x: np.log1p(x))
        else:
            # Clip large values for this subfigure
            clipped_data = data.iloc[:, 1:].apply(lambda x: np.clip(x, None, clip_value_list[idx]), axis=0)

        ax = axs[idx]
        for j, model in enumerate(models):
            # Offset positions for clusters
            bar_positions = [x + bar_width * j for x in x_positions]
            ax.bar(bar_positions, clipped_data.iloc[j, :], bar_width, label=model)

        # Customize subplot
        ax.set_title(sheet_name, fontsize=22)
        ax.set_xlabel("模型", fontsize=18)
        ax.set_ylabel(f"{sheet_name}值", fontsize=18)
        ax.set_xticks([x + bar_width * 1.5 for x in x_positions])
        ax.set_xticklabels(settings, rotation=0, fontsize=14)
        ax.tick_params(axis='y', labelsize=16)

        # Set y-axis scale to logarithmic if needed
        if use_log_scale:
            ax.set_yscale('log')

        ax.legend(title="组", loc="upper right", fontsize=14)

    # Adjust layout and save
    plt.tight_layout()
    output_path = "all_metrics_combined_chinese.png"
    plt.savefig(output_path)
    plt.close()
    return output_path

# Load data for all sheets
data_dict = {sheet: excel_data.parse(sheet) for sheet in excel_data.sheet_names}

clip_value_list = [-1, -1, -1, -1, -1, -1]
# Generate the big combined figure
big_figure_path = plot_all_combined_figures(data_dict, clip_value_list)

print(f"Combined figure saved at: {big_figure_path}")
