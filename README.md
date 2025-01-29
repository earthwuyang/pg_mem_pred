
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

#### Fast Forward
Directly Download our precessed datasets from https://cloud.tsinghua.edu.cn/d/9ad34a4caafe405ebcc7/, otherwise follow below steps.



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

## revision related
### generate queries with more predicates and joins
First, we modify `zsce/generate_zsce_queries.py` to generate queries with more predicates and joins.
Then, `cd zsce/cross_db_benchmark/datasets && cp -r tpcds_sf1 tpcds_sf100` 
Then, run `python zsce/generate_column_stats.py --dataset tpcds_sf100` to regenerate column statistics for tpcds_sf100
Then, run `python zsce/generate_string_stats.py --dataset tpcds_sf100` to regenerate string statistics for tpcds_sf100
Then, run `python zsce/generate_zsce_queries.py --dataset tpcds_sf100` to generate queries for tpcds_sf100 with more predicates and joins.



### training and test on tpch_sf10
`python src/preprocessing/execute_workload.py --dataset tpch_sf10` to execute workloads on tpch_sf10.
First cp pg logs to pg_mem_data/pg_log/tpch_sf10, and chmod +r for these files
Then, run  `python src/preprocessing/extract_mem_time_info.py --dataset tpch_sf10` to extract memory usage from logs.
Then, run `python train.py --model GIN --train_dataset airline carcinogenesis employee hepatitis financial geneea tpch_sf1 tpcds_sf1 --val_dataset credit --test_dataset tpch_sf10` to train models across datasets and test on tpch_sf10.

### restrict postgresql's memory usage by cgroup
#### create a cgroup
`sudo cgcreate -g memory:postgresql`
#### set memory limit for the cgroup
echo 2G | sudo tee /sys/fs/cgroup/memory/postgresql/memory.limit_in_bytes
#### start postgresql within the cgroup
sudo cgexec -g memory:postgresql systemctl start postgresql

### grant read permission on /proc
sudo sysctl -w kernel.yama.ptrace_scope=0
echo "kernel.yama.ptrace_scope=0" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

sudo setcap cap_sys_ptrace+ep $(which python3)


### set swap memmory
sudo swapoff /www/swapfile
sudo rm /www/swapfile
sudo fallocate -l 1G /www/swapfile
sudo chmod 600 /www/swapfile
sudo mkswap /www/swapfile
sudo swapon /www/swapfile
echo "/www/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab

#### grant `mlock` permission to python
sudo setcap cap_ipc_lock+ep $(which python)

#### grant read /proc permission to python
sudo setcap cap_sys_ptrace+ep $(realpath $(which python))

#### edit postgresql's systemd service
sudo systemctl edit postgresql
add these lines:
```
[Service]
MemoryMax=2G
MemorySwapMax=1G
```
sudo systemctl daemon-reexec
sudo systemctl restart postgresql
systemctl show postgresql | grep Memory

alternative:
`sudo systemctl set-property postgresql.service MemoryMax=2G MemorySwapMax=1G`

unset using:
sudo systemctl set-property postgresql.service MemoryMax=infinity MemorySwapMax=infinity


#### another way to lock memory
sudo mount -o remount,size=12G /dev/shm
sudo mkdir -p /dev/shm/mem_holder  # Use shared memory for fast access
sudo dd if=/dev/zero of=/dev/shm/mem_holder/ramfile bs=1M count=11264     # 11GB
sudo prlimit --memlock=unlimited -- sudo python3 lock_shm.py

##### clean up
sudo munlock /dev/shm/mem_holder/ramfile
rm -f /dev/shm/mem_holder/ramfile
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches


