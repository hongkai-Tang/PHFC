import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys, os
import h5py
import io
import pandas as pd
import numpy as np
import random
import warnings
import contextlib

warnings.filterwarnings("ignore")
torch.set_num_threads(1) 

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

try:
    from pfhc.datasets.cloudbrain_dataset import CloudBrainDataset, SEQUENCE_LENGTH
    from pfhc.datasets.collate import dynamic_graph_collate_fn_v2
except ImportError as e:
    print(f"Error: Cannot find old v2 dataset/collate files: {e}")
    sys.exit(1)

H5_FILE_PATH = "data/processed/cloudbrain_task_centric_RAW.h5"
JOBS_INFO_PATH = "data/jobs_info.csv" 

VALID_SPLIT = 0.2
RANDOM_SEED = 42

BAKE_BATCH_SIZE = 1
NUM_WORKERS = 16

BAKE_DIR = "data/baked_v10"
TRAIN_DIR = os.path.join(BAKE_DIR, "train")
VALID_DIR = os.path.join(BAKE_DIR, "val")
STATS_PATH = os.path.join(BAKE_DIR, "normalization_stats.pt")

def load_task_start_times(jobs_csv_path):
    print(f"Reading {jobs_csv_path} to build time index...")
    try:
        df = pd.read_csv(jobs_csv_path, header=None, usecols=[0, 5])
        task_map = dict(zip(df[0].astype(str), df[5]))
        print(f"Successfully loaded time information for {len(task_map)} tasks.")
        return task_map
    except Exception as e:
        print(f"[Critical Error] Cannot read jobs_info.csv: {e}")
        sys.exit(1)

def load_stats():
    if not os.path.exists(STATS_PATH):
        print(f"[Error] Cannot find statistics file: {STATS_PATH}")
        sys.exit(1)
    print(f"Loading normalization statistics: {STATS_PATH}")
    return torch.load(STATS_PATH)

def robust_normalize_tensor(tensor, stats):
    median = stats['median'].to(tensor.device)
    scale = stats['scale'].to(tensor.device)
    normalized = (tensor - median) / scale
    clamped = torch.clamp(normalized, min=-5.0, max=5.0)
    return clamped

def generate_time_features(start_ts, seq_len, step_seconds=15):
    offsets = np.arange(seq_len) * step_seconds
    timestamps = start_ts + offsets
    dt = pd.to_datetime(timestamps, unit='s')
    hours = dt.hour + dt.minute / 60.0
    hour_rad = 2 * np.pi * hours / 24.0
    sin_hour = np.sin(hour_rad)
    cos_hour = np.cos(hour_rad)
    weekday_norm = dt.dayofweek / 6.0
    time_feats = np.stack([sin_hour, cos_hour, weekday_norm], axis=-1)
    return torch.tensor(time_feats, dtype=torch.float32)

def bake_dataset(dataset, output_dir, stats, task_start_map, desc=""):
    os.makedirs(output_dir, exist_ok=True)
    
    loader = DataLoader(
        dataset, 
        batch_size=BAKE_BATCH_SIZE, 
        collate_fn=dynamic_graph_collate_fn_v2,
        num_workers=NUM_WORKERS,
        shuffle=False,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True
    )
    
    total_samples = len(dataset)
    saved_count = 0
    
    print(f"\n--- Starting baking {desc} (Total: {total_samples} | Workers: {NUM_WORKERS}) ---")
    
    chunk_info_list = dataset.chunk_index
    
    pbar = tqdm(loader, desc=f"Baking {desc}", total=total_samples, mininterval=1.0)
    
    for i, batch in enumerate(pbar):
        if batch[0] is None: 
            continue
            
        save_path = os.path.join(output_dir, f"{i:08d}.pt")
        
        task_name, chunk_start, chunk_end = chunk_info_list[i]
        
        targets_raw = batch[0].squeeze(0) 
        labels_true = batch[1].squeeze(0) 
        neighbors_raw = batch[2].squeeze(0) 
        masks = batch[3].squeeze(0)       
        lengths = batch[4]                
        
        targets_norm = robust_normalize_tensor(targets_raw, stats)
        neighbors_norm = robust_normalize_tensor(neighbors_raw, stats)
        
        job_start_time = task_start_map.get(str(task_name))
        if job_start_time is None:
            current_seq_start_time = 0
        else:
            current_seq_start_time = job_start_time + (chunk_start * 15)
            
        seq_len = targets_raw.shape[0]
        time_features = generate_time_features(current_seq_start_time, seq_len).to(targets_raw.device)
        
        targets_final = torch.cat([targets_norm, time_features], dim=-1)
        
        N_neighbors = neighbors_norm.shape[1]
        time_features_expanded = time_features.unsqueeze(1).repeat(1, N_neighbors, 1)
        neighbors_final = torch.cat([neighbors_norm, time_features_expanded], dim=-1)
        
        data_to_save = [targets_final, labels_true, neighbors_final, masks, lengths]
        torch.save(data_to_save, save_path)
        saved_count += 1
        
    skipped_count = total_samples - saved_count
    drop_rate = (skipped_count / total_samples) * 100 if total_samples > 0 else 0.0
    
    print(f"--- {desc} Baking Complete ---")
    print(f"📊 Statistics: Success {saved_count} / Skipped {skipped_count} (Drop rate {drop_rate:.2f}%)")

def get_all_task_names(h5_file_path):
    try:
        with h5py.File(h5_file_path, 'r') as hf:
            available_tasks = list(hf['task_timeseries_data'].keys())
        return available_tasks
    except Exception as e:
        print(f"H5 read error: {e}")
        sys.exit(1)

def main():
    print(f"--- Starting v10 Data Baking Script (Ultra-fast Parallel Version) ---")
    
    global JOBS_INFO_PATH
    possible_paths = ["data/cloudbrain-datasets/jobs_info.csv", "data/jobs_info.csv", "jobs_info.csv", "../cloudbrain-datasets/jobs_info.csv"]
    if not os.path.exists(JOBS_INFO_PATH):
        for p in possible_paths:
            if os.path.exists(p):
                JOBS_INFO_PATH = p
                print(f"[Auto-corrected] Found jobs_info.csv at: {p}")
                break
    
    stats = load_stats()
    task_start_map = load_task_start_times(JOBS_INFO_PATH)
    
    all_task_names = get_all_task_names(H5_FILE_PATH)
    random.seed(RANDOM_SEED)
    random.shuffle(all_task_names)
    
    valid_size = int(len(all_task_names) * VALID_SPLIT)
    train_names = all_task_names[valid_size:]
    val_names = all_task_names[:valid_size]
    
    print("\n[Step 2] Initializing Dataset (this may take a few minutes)...")
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        train_dataset = CloudBrainDataset(h5_file_path=H5_FILE_PATH, seq_len=SEQUENCE_LENGTH, task_list=train_names, mode="train")
        valid_dataset = CloudBrainDataset(h5_file_path=H5_FILE_PATH, seq_len=SEQUENCE_LENGTH, task_list=val_names, mode="val")
    
    bake_dataset(train_dataset, TRAIN_DIR, stats, task_start_map, desc="Train")
    bake_dataset(valid_dataset, VALID_DIR, stats, task_start_map, desc="Valid")
    
    print("\n--- Success! ---")
    print("Data baking complete. Input Dim = 6")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
