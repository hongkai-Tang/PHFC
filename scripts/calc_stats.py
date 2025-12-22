import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys, os
import numpy as np
import h5py

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

try:
    from pfhc.datasets.cloudbrain_dataset import CloudBrainDataset, SEQUENCE_LENGTH
except ImportError as e:
    print(f"Error: Cannot find dependency files: {e}")
    sys.exit(1)

H5_FILE_PATH = "data/processed/cloudbrain_task_centric_RAW.h5"
STATS_SAVE_PATH = "data/baked_v10/normalization_stats.pt"
SAMPLE_RATE = 0.9

class FastStatsDataset(CloudBrainDataset):
    def __getitem__(self, index):
        task_name, chunk_start, chunk_end = self.chunk_index[index]
        
        if self.hf is None:
            self.hf = h5py.File(self.h5_path, 'r')
            
        try:
            raw_data = self.hf[f'task_timeseries_data/{task_name}'][chunk_start:chunk_end]
            
            N_cols = raw_data.shape[1]
            
            if N_cols >= 4:
                target_data = raw_data[:, 1:4]
            else:
                target_data = raw_data[:, :3]

            return torch.from_numpy(target_data).float()
        except Exception:
            return None

def fast_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    return torch.cat(batch, dim=0)

def calculate_stats():
    print(f"--- [Step 1] Starting ultra-fast statistics calculation (Sampling rate {SAMPLE_RATE*100}%) ---")
    
    dataset = FastStatsDataset(
        h5_file_path=H5_FILE_PATH, 
        seq_len=SEQUENCE_LENGTH, 
        mode="train"
    )
    
    BATCH_SIZE = 1024 
    NUM_WORKERS = 8 
    
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=fast_collate_fn,
        num_workers=NUM_WORKERS,
        shuffle=True
    )
    
    all_features = []
    samples_to_collect = int(len(dataset) * SAMPLE_RATE)
    collected_count = 0
    
    print(f"Collecting approximately {samples_to_collect} samples...")
    print(f"Using accelerated configuration: Batch Size={BATCH_SIZE}, Workers={NUM_WORKERS}")
    
    for batch_tensor in tqdm(loader, desc="Scanning (Fast Mode)"):
        if batch_tensor is None: continue
        
        approx_chunks = batch_tensor.shape[0] / 512.0
        collected_count += approx_chunks
        
        all_features.append(batch_tensor)
        
        if collected_count >= samples_to_collect:
            break
            
    if len(all_features) == 0:
        print("Error: No data collected!")
        return

    print("Merging data...")
    full_tensor = torch.cat(all_features, dim=0)
    
    print(f"\nCollection complete. Total feature points: {full_tensor.shape[0]}")
    
    MAX_SAMPLES_FOR_STATS = 10000000
    if full_tensor.shape[0] > MAX_SAMPLES_FOR_STATS:
        print(f"Data volume ({full_tensor.shape[0]}) exceeds processing limit, randomly downsampling to {MAX_SAMPLES_FOR_STATS} points...")
        indices = torch.randperm(full_tensor.shape[0])[:MAX_SAMPLES_FOR_STATS]
        full_tensor = full_tensor[indices]
    
    print("Calculating statistics (Force CPU)...")
    
    full_tensor = full_tensor.cpu()
    
    q25 = torch.quantile(full_tensor, 0.25, dim=0)
    median = torch.quantile(full_tensor, 0.50, dim=0)
    q75 = torch.quantile(full_tensor, 0.75, dim=0)
    
    iqr = q75 - q25
    iqr = torch.clamp(iqr, min=1e-5)
    scale = iqr / 1.35
    
    stats = {
        "median": median,
        "scale": scale,
        "q25": q25,
        "q75": q75
    }
    
    print("\n--- Statistics Results (Robust Version) ---")
    print(f"Median: {stats['median']}")
    print(f"Scale:  {stats['scale']}")
    
    os.makedirs(os.path.dirname(STATS_SAVE_PATH), exist_ok=True)
    torch.save(stats, STATS_SAVE_PATH)
    print(f"Statistics saved to: {STATS_SAVE_PATH}")

if __name__ == "__main__":
    calculate_stats()
