import numpy as np
import pandas as pd
import h5py
import torch
from tqdm import tqdm
import sys, os

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

try:
    from pfhc.datasets.cloudbrain_dataset import CloudBrainDataset, SEQUENCE_LENGTH
    from pfhc.utils.mode_rules import K_MODES
except ImportError as e:
    print(f"Error: Cannot find cloudbrain_dataset.py or mode_rules.py. {e}")
    sys.exit(1)

H5_FILE_PATH = "data/processed/cloudbrain_task_centric_RAW.h5"

def assemble_dynamic_hypergraph_v2(packet, max_neighbors=None):
    
    target_data = packet['target_data_chunk']
    target_labels = packet['target_labels_chunk']
    
    chunk_start_abs, chunk_end_abs = packet['target_chunk_meta']
    
    neighbor_count = packet['neighbor_count']
    neighbor_data_full = packet['neighbor_data_full']
    neighbor_meta_full = packet['neighbor_meta_full']
    
    T_A = target_data.shape[0]
    C = target_data.shape[1]
    
    if max_neighbors is None:
        N_max = neighbor_count
    else:
        N_max = max_neighbors
        
    if N_max == 0:
        empty_neighbors = np.zeros((T_A, 0, C), dtype=np.float32)
        empty_mask = np.zeros((T_A, 0), dtype=np.bool_)
        return target_data, target_labels, empty_neighbors, empty_mask

    neighbor_data_aligned = np.zeros((T_A, N_max, C), dtype=np.float32)
    neighbor_mask_aligned = np.zeros((T_A, N_max), dtype=np.bool_)

    for t_a in range(T_A):
        
        current_real_time = chunk_start_abs + t_a * 15
        
        for i in range(min(neighbor_count, N_max)):
            
            n_start, n_end = neighbor_meta_full[i]
            
            if current_real_time >= n_start and current_real_time < n_end:
                
                neighbor_t = (current_real_time - n_start) // 15
                
                n_data_array = neighbor_data_full[i]
                
                if neighbor_t < len(n_data_array):
                    neighbor_data_aligned[t_a, i, :] = n_data_array[neighbor_t, :]
                    neighbor_mask_aligned[t_a, i] = True

    return target_data, target_labels, neighbor_data_aligned, neighbor_mask_aligned


if __name__ == "__main__":
    
    print('--- Starting Step 3 (v2 "chunk" version) test ---')
    
    print('\n[Step 2] Initializing Dataset (v2 "chunk" version)...')
    try:
        dataset = CloudBrainDataset(h5_file_path=H5_FILE_PATH, seq_len=SEQUENCE_LENGTH)
    except Exception as e:
        print(f"Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit()
        
    print(f"Dataset initialized successfully. Total *chunks*: {len(dataset)}")

    if len(dataset) > 0:
        print(f"\n[Step 2] Extracting sample 500 (chunk) 'packet'...")
        packet = dataset[500] 
        
        if not packet:
            print("[Failed] Failed to get 'packet' (returned None).")
            sys.exit()
            
        print(f"[Success] Got 'packet'! Target Task: {packet['target_task_name']}")
        print(f"  > Target shape (T_chunk, C): {packet['target_data_chunk'].shape}")
        print(f"  > Number of neighbors found: {packet['neighbor_count']}")

        print("\n[Step 3] 'Cooking' packet, building dynamic graph (matrix+mask)...")
        
        N_MAX_TEST = 150 
        
        target_seq, label_seq, neighbor_seq, hypergraph_mask = \
            assemble_dynamic_hypergraph_v2(packet, max_neighbors=N_MAX_TEST)
        
        print("[Success] 'Cooking' complete!")
        print(f"\n--- Step 3 (v2) Output (Final Result) ---")
        print(f"  1. Target sequence (T_chunk, C):        {target_seq.shape}")
        print(f"  2. Target *labels* (T_chunk,):      {label_seq.shape}")
        print(f"  3. Neighbor sequence (T_chunk, N_max, C): {neighbor_seq.shape}")
        print(f"  4. [Dynamic Hypergraph] Mask (T_chunk, N_max): {hypergraph_mask.shape}")
        print("---------------------------------")
        
        if target_seq.shape[0] == label_seq.shape[0] == neighbor_seq.shape[0] == hypergraph_mask.shape[0]:
            T_chunk_len = target_seq.shape[0]
            print(f"  [✅] Time dimension T_chunk (={T_chunk_len}) fully aligned.")
            if T_chunk_len > SEQUENCE_LENGTH:
                print(f"  [!! Warning !!] Chunk length {T_chunk_len} exceeds {SEQUENCE_LENGTH}!")
        else:
            print(f"  [!! Failed !!] Time dimension T_chunk does not match!")
            
        if neighbor_seq.shape[1] == hypergraph_mask.shape[1] == N_MAX_TEST:
             print(f"  [✅] Neighbor dimension N_max (={N_MAX_TEST}) fully aligned.")
        else:
            print(f"  [!! Failed !!] Neighbor dimension N_max does not match!")

        print("\n--- Step 3 (v2) Test Complete ---")
        
    else:
        print("Dataset is empty, cannot test.")
