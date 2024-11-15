import pandas as pd
import os
import json
import torch
from transformers import RobertaTokenizer, RobertaModel
import multiprocessing
from tqdm import tqdm
from functools import partial

# Define the function outside of __main__ to make it pickleable
def process_row(row, data_dir, dataset, tokenizer, model, device):
    queryid = int(row['queryid'])
    # Read the SQL query file
    with open(os.path.join(data_dir, dataset, 'raw_data', 'query_dir', f"{queryid}.sql")) as f:
        query = f.read().strip().replace('"', '').replace('\\', '')
    
    # Tokenize the query and move input tensors to GPU
    inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True).to(device)
    
    # Forward pass through the model
    with torch.no_grad():
        query_feature = model(**inputs)
    
    # Collect the feature as a list (moving tensor back to CPU)
    query_feature = query_feature.pooler_output.cpu().numpy().tolist()[0]
    
    # Return the processed data
    return {
        'query': query,
        'queryid': queryid,
        'query_mem': row['peakmem'],
        'query_time': row['time'],
        'query_feature': query_feature
    }

# Define the function to parallelize the process
def parallel_process(df, data_dir, dataset, tokenizer, model, device):
    # Create a partial function that binds the additional arguments
    process_row_partial = partial(process_row, data_dir=data_dir, dataset=dataset, tokenizer=tokenizer, model=model, device=device)
    
    # Set up the multiprocessing pool
    pool = multiprocessing.Pool(processes=10)
    
    # Use tqdm for progress tracking
    data = list(tqdm(pool.imap(process_row_partial, [row for _, row in df.iterrows()], chunksize=10), total=len(df)))
    
    # Close and join the pool
    pool.close()
    pool.join()
    
    return data

if __name__ == "__main__":
    # Set the start method for multiprocessing to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/wuy/DB/pg_mem_data')
    parser.add_argument('--dataset', type=str, default='tpch_sf1')
    args = parser.parse_args()
    data_dir = args.data_dir
    dataset = args.dataset


    # Load the dataframe
    df = pd.read_csv(f'{data_dir}/{dataset}/raw_data/mem_info.csv')

    # Initialize the tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    model = RobertaModel.from_pretrained('microsoft/codebert-base')

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Process the dataset in parallel
    data = parallel_process(df, data_dir, dataset, tokenizer, model, device)

    # Save the processed data to a JSON file
    with open(f'{args.dataset}_data.json', 'w') as f:
        json.dump(data, f)
