import torch
import numpy as np
import sys, os
from torch.nn.utils.rnn import pad_sequence
#Note for this file: The file path on line 65 must be modified.
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

try:
    from pfhc.graphs.assembler import assemble_dynamic_hypergraph_v2
except ImportError:
    print("Error: Cannot find pfhc.graphs.assembler")
    print("Please ensure you have created pfhc/graphs/assembler.py (v2) file")
    sys.exit(1)

MAX_NEIGHBORS = 394 

def dynamic_graph_collate_fn_v2(batch):
    
    target_seqs = []
    label_seqs = []
    neighbor_seqs = []
    mask_seqs = []
    lengths = []

    for packet in batch:
        if packet is None:
            continue
            
        target, labels, neighbors, mask = \
            assemble_dynamic_hypergraph_v2(packet, max_neighbors=MAX_NEIGHBORS)
        
        target_seqs.append(torch.tensor(target, dtype=torch.float32))
        label_seqs.append(torch.tensor(labels, dtype=torch.long))
        neighbor_seqs.append(torch.tensor(neighbors, dtype=torch.float32))
        mask_seqs.append(torch.tensor(mask, dtype=torch.bool))
        lengths.append(target.shape[0])

    if not lengths:
        return None, None, None, None, None

    targets_padded = pad_sequence(target_seqs, batch_first=True, padding_value=0.0)
    
    labels_padded = pad_sequence(label_seqs, batch_first=True, padding_value=-1)
    
    neighbors_padded = pad_sequence(neighbor_seqs, batch_first=True, padding_value=0.0)
    
    masks_padded = pad_sequence(mask_seqs, batch_first=True, padding_value=0)
    
    lengths_tensor = torch.tensor(lengths, dtype=torch.int64)
    
    return targets_padded, labels_padded, neighbors_padded, masks_padded, lengths_tensor


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    try:
        from pfhc.datasets.cloudbrain_dataset import CloudBrainDataset, SEQUENCE_LENGTH
    except ImportError:
        print("Error: Cannot find pfhc.datasets.cloudbrain_dataset (v2)")
        sys.exit(1)

    H5_FILE_PATH_TEST = "../cloudbrain_task_centric_RAW.h5"#Note: File path must be changed here.

    print('--- Starting Step 4 (v2 "chunk" version) test ---')
    
    if not os.path.exists(H5_FILE_PATH_TEST):
        print(f"[!! Failed !!] HDF5 file not found: {H5_FILE_PATH_TEST}")
        sys.exit()

    print('\n[Step 2] Initializing Dataset (v2 "chunk" version)...')
    try:
        dataset = CloudBrainDataset(h5_file_path=H5_FILE_PATH_TEST, seq_len=SEQUENCE_LENGTH)
        print(f"Dataset initialized successfully. Total *chunks*: {len(dataset)}")
    except Exception as e:
        print(f"Initialization failed: {e}")
        sys.exit()
        
    BATCH_SIZE = 4
    
    print(f"\n[Step 4] Initializing DataLoader (Batch Size = {BATCH_SIZE})...")
    
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,          
        collate_fn=dynamic_graph_collate_fn_v2,
        num_workers=0          
    )

    print("\n[Step 4] Extracting 1 batch from DataLoader...")
    
    try:
        targets, labels, neighbors, masks, lengths = next(iter(data_loader))
        
        print("[Success] Successfully obtained a batch!")
        
        print("\n--- Step 4 (v2) Output (Final Result) ---")
        print(f"  1. Target *data* (B, T_max, C):        {targets.shape}")
        print(f"  2. Target *labels* (B, T_max):          {labels.shape}")
        print(f"  3. Neighbor *data* (B, T_max, N_max, C): {neighbors.shape}")
        print(f"  4. [Dynamic Hypergraph] Mask (B, T_max, N_max): {masks.shape}")
        print(f"  5. True lengths (B,):                    {lengths.shape}")
        print("---------------------------------")
        
        print(f"\n[Deep Inspection] True lengths in batch (T_chunk_A, T_chunk_B, ...):")
        print(f"  > {lengths.numpy()}")
        
        T_max_in_batch = torch.max(lengths).item()
        print(f"  > T_max (longest chunk) should be: {T_max_in_batch}")
        
        if T_max_in_batch > SEQUENCE_LENGTH:
             print(f"  [!! Failed !!] T_max (={T_max_in_batch}) exceeds SEQUENCE_LENGTH (={SEQUENCE_LENGTH})!")
        else:
             print(f"  [✅] T_max (={T_max_in_batch}) <= {SEQUENCE_LENGTH}. OOM error fixed!")

        print("\n--- Step 4 (v2) Test Complete ---")

    except Exception as e:
        print(f"\n[!! Failed !!] DataLoader execution failed: {e}")
        import traceback
        traceback.print_exc()
