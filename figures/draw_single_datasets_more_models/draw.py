import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample data preparation (replace this part with your dataset loading process)
file_path = '../single_datasets_more_models.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

# Extract dataset names and models
datasets = ['train_dataset', 'val_dataset', 'test_dataset']
models = data['model'].unique()[:-1]

dataset_names = data[datasets[2]].unique()

# Filter out NaN or invalid dataset names
dataset_names = [dataset for dataset in dataset_names if isinstance(dataset, str)]


# Create a dictionary for qerror_50 values grouped by model and dataset
qerror_50_data = {dataset: data.loc[data[datasets[2]] == dataset, 'qerror_50'].values for dataset in dataset_names}

# Align models and datasets properly to avoid indexing errors
aligned_qerror_50 = {model: [] for model in models}
for dataset in dataset_names:
    values = qerror_50_data.get(dataset, [])
    for i, model in enumerate(models):
        if i < len(values):  # Ensure no out-of-bound indexing
            aligned_qerror_50[model].append(values[i])
        else:
            aligned_qerror_50[model].append(0)  # Fill missing values with 0

# Define bar chart positions with closer spacing between clusters
x = np.arange(len(dataset_names)) * 1.5  # Adjust spacing between clusters
width = 0.15  # Bar width

# Plot clustered bar chart
plt.figure(figsize=(12, 8))
for i, model in enumerate(models):
    plt.bar(
        x + i * width,
        aligned_qerror_50[model],
        width=width,
        label=model
    )

# Customize chart
plt.xlabel('Datasets', fontsize=18)
plt.ylabel('qerror_50', fontsize=18)
plt.ylim(1.0, 1.16)  # Set y-axis range
plt.yticks(fontsize=14)
plt.title('Median QError Comparison between 8 Models on 4 Datasets', fontweight='bold', fontsize=20)
plt.xticks(x + width * (len(models) - 1) / 2, [dn.split('_')[0] for dn in dataset_names], rotation=0, ha='right', fontsize=18)
# plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.legend(title='Models', bbox_to_anchor=(0.02, 0.98), loc='upper left', fontsize=14, title_fontsize=16)
plt.tight_layout()
plt.savefig('qerror_50_comparison_single_datasets_more_models.png')
