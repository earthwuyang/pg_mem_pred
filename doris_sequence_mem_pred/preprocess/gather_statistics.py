import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
data_dir = '/home/wuy/DB/doris_mem_pred/tpch_data/'
data_file = os.path.join(data_dir, 'query_mem_data_tpch_sf100.csv')
plan_dir = os.path.join(data_dir, 'plans')
df=pd.read_csv(data_file, sep=';')


time_labels = []
mem_labels = []
for row in tqdm(df.iterrows(), total=len(df)):
    row = row[1]
    query_id = row['queryid']
    runtime = row['time'] # int
    mem_list = eval(row['mem_list'].strip()) # list of int
    time_labels.append(runtime)
    mem_labels.extend(mem_list)

time_scaler = RobustScaler()
mem_scaler = RobustScaler()
time_labels = time_scaler.fit_transform(np.array(time_labels).reshape(-1, 1)).flatten()
mem_labels = mem_scaler.fit_transform(np.array(mem_labels).reshape(-1, 1)).flatten()

statistics={'time':{},'mem':{}}
statistics['time']['center'] = time_scaler.center_.item()
statistics['time']['scale'] = time_scaler.scale_.item()
statistics['mem']['center'] = mem_scaler.center_.item()
statistics['mem']['scale'] = mem_scaler.scale_.item()
print(statistics)

with open(os.path.join("", 'statistics.json'), 'w') as f:
    json.dump(statistics, f)