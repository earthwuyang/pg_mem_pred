
# Memory Prediction and Workload Scheduling

This repository provides tools for memory prediction and workload scheduling in analytical database systems. The code includes various predictive models and supports training, validation, and testing across multiple datasets.

---

## Setup

### Conda Environment
To set up the required environment, execute:
```bash
conda create -n zsce python=3.8.13
conda activate zsce
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt


---

## Usage

### Database Setup
#### Configure Database Connection
Write a `conn.json` file in the root directory with the database connection details.

Example:
```json
{
  "user": "wuy",
  "password": "wuy",
  "host": "localhost",
  "port": 5432
}
```

#### Create Databases
Use the following command to create and populate the databases:
```bash
python src/preprocessing/export_import_data.py
```

#### TPC-H and TPC-DS Datasets
- Place TPC-H CSV files in `/data/datasets/tpch_sf1`.
- Place TPC-DS CSV files in `/data/datasets/tpcds_sf1`.
- Import the data into PostgreSQL using:
  ```bash
  /data/datasets/tpch-kit/import_data.sh
  /data/datasets/tpcds-kit/import_data.sh
  ```

#### Analyze Datasets
Run:
```bash
python src/preprocessing/analyze_datasets.py --port 5432
python src/preprocessing/get_column_type_for_databases.py
python src/preprocessing/generate_column_string_stats.py
```

#### Generate Queries
```bash
python zsce/generate_zsce_queries.py
```

---

### Memory Usage Extraction
1. **Execute workloads and collect memory/time information**:
   ```bash
   python src/preprocessing/execute_all_workloads.py
   ```

2. **Extract memory usage from logs**:
   ```bash
   python src/preprocessing/extract_mem_time_info.py --dataset tpch_sf1 tpcds_sf1
   ```

3. **Combine extracted statistics**:
   ```bash
   python src/dataset/combine_stats.py
   ```

---

### Training

#### Zero-Shot Cost Estimation (ZSCE)
1. Generate raw plans:
   ```bash
   python zsce/combine_stats.py
   ```

2. Train the ZSCE method:
   ```bash
   python zsce/train.py
   ```

#### Cross-Dataset Training
Train models across datasets:
```
python train.py --model GIN --train_dataset tpch_sf1 tpcds_sf1 airline --val_dataset credit --test_dataset geneea
```

<!-- #### Heterogeneous Graph
Run:
```bash
python train.py --train_dataset 'tpch_sf1' --test_dataset 'tpcds_sf1'
``` -->

<!-- ---

### Testing Across Machines
1. **Start Docker for PostgreSQL**:
   ```bash
   cd cross_machines/1
   sh start_docker.sh
   ```

2. **Import and execute workloads**:
   ```bash
   python src/preprocessing/import_data.py --port 5422
   python src/preprocessing/execute_all_workloads.py --port 5422 --docker_name my_postgres_2
   ```

--- -->

## Code Structure

- `src`: Contains implementation for GIN, GAT, GraphTransformer, and TreeTransformer models.
- `zsce`: Code for Zero-Shot Cost Estimation (ZSCE) method.
- `workload_scheduling`: Code for workload scheduling.
<!-- - `xgboost`: Code for XGBoost-based predictions. -->
<!-- - `treelstm`: Code for TreeLSTM model. -->
<!-- - `heterogeneous_graph`: Code for the heterogeneous graph-based method. -->

---

## Additional Notes

### QueryFormer Integration
1. Run the example notebook:
   - `Example Tool to collect sample data.ipynb`.
2. Train QueryFormer:
   ```bash
   python train.py
   ```

### pg bastch execution
```bash
cd workload_scheduling
python scheduling_docker.py --num_queries 100
```

### Sequential Execution
Unset proxy settings if using sequential execution:
```bash
unset https_proxy
unset http_proxy
```
Run memory-based strategy:
```
python proxy.py
python client.py --num_queries 100
```

Run naive strategy:
```
python proxy_FCFS.py
python client.py --num_queries 100
```