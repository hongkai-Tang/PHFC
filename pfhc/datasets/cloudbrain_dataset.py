import h5py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import io
import sys, os
from tqdm import tqdm
import pickle 

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

try:
    from pfhc.utils.mode_rules import apply_rules_to_batch, K_MODES
except ImportError:
    print("Error: Cannot find pfhc.utils.mode_rules.py")
    sys.exit(1)

H5_FILE_PATH = "data/processed/cloudbrain_task_centric_RAW.h5"
SEQUENCE_LENGTH = 512 

def load_meta_from_h5(h5_file_path):
    print(f"Loading metadata blueprint from {h5_file_path}...")
    try:
        with h5py.File(h5_file_path, 'r') as hf:
            meta_csv_string = hf['task_graph_meta'][()].decode('utf-8')
            meta_df = pd.read_csv(io.StringIO(meta_csv_string))
        
        if 'TaskName' in meta_df.columns:
            meta_df['TaskName'] = meta_df['TaskName'].astype(str).str.strip()
            
        if 'NodePrefix' in meta_df.columns:
            meta_df['NodePrefix'] = meta_df['NodePrefix'].astype(str).str.strip()
            
        meta_df['StartTime'] = meta_df['StartTime'].astype(np.int64)
        meta_df['EndTime'] = meta_df['EndTime'].astype(np.int64)
        print(f"Blueprint loaded. Found {len(meta_df)} total tasks.")
        return meta_df
    except Exception as e:
        print(f"Fatal: Could not load metadata from HDF5! Error: {e}")
        return pd.DataFrame()

def create_index_of_chunks(h5_file_path, meta_df, seq_len=SEQUENCE_LENGTH):
    print("Pre-calculating all possible chunks (this may take a minute)...")
    
    index = []
    
    with h5py.File(h5_file_path, 'r') as hf:
        available_tasks = list(hf['task_timeseries_data'].keys())
    
    print(f"Found {len(available_tasks)} tasks with data. Now chunking...")
    
    meta_map = {name: i for i, name in enumerate(meta_df['TaskName'])}
    
    for task_name_raw in tqdm(available_tasks, desc="Chunking Tasks"):
        try:
            if isinstance(task_name_raw, bytes):
                task_name = task_name_raw.decode('utf-8').strip()
            else:
                task_name = str(task_name_raw).strip()
            
            if task_name not in meta_map:
                continue
                
            with h5py.File(h5_file_path, 'r') as hf:
                T_k = hf[f'task_timeseries_data/{task_name_raw}'].shape[0]
            
            if T_k < 2:
                continue
                
            for i in range(0, T_k - 1, seq_len):
                start = i
                end = min(i + seq_len, T_k)
                if (end - start) >= 2:
                    index.append((task_name, start, end))
                    
        except Exception:
            pass
            
    print(f"Chunking complete. Total samples (chunks) found: {len(index)}")
    return index, meta_df

class CloudBrainDataset(Dataset):
    def __init__(self, h5_file_path=H5_FILE_PATH, seq_len=SEQUENCE_LENGTH, task_list=None, mode="full"):
        super().__init__()
        
        print(f"\n[Step 2] Initializing Dataset (v3 fixed version, mode: {mode})...")
        
        self.h5_path = h5_file_path
        self.hf = None
        self.seq_len = seq_len
        self.mode = mode
        
        meta_df_full = load_meta_from_h5(h5_file_path)
        if meta_df_full.empty:
            raise ValueError("Metadata (Blueprint) is empty. Cannot proceed.")
            
        if task_list:
            clean_task_list = [str(t).strip() for t in task_list]
            self.meta_df = meta_df_full[meta_df_full['TaskName'].isin(clean_task_list)].copy()
        else:
            self.meta_df = meta_df_full

        self.meta_df.set_index('TaskName', inplace=True, drop=False)

        base_name = os.path.splitext(h5_file_path)[0]
        CACHE_PATH = f"{base_name}_{self.mode}_v2_chunks.pkl"

        if os.path.exists(CACHE_PATH):
            print(f"--- Loading 'chunk' data from cache ({self.mode}): {CACHE_PATH} ---")
            try:
                with open(CACHE_PATH, 'rb') as f:
                    self.chunk_index = pickle.load(f)
                print(f"--- Cache loaded. Total {len(self.chunk_index)} chunks ---")
            except Exception as e:
                print(f"*** Warning: Cache corrupted: {e}. Will recalculate... ***")
                self.chunk_index = None
        else:
            print(f"--- Cache not found ({self.mode}), will start calculation... ---")
            self.chunk_index = None
        
        if self.chunk_index is None:
            self.chunk_index, _ = create_index_of_chunks(
                h5_file_path, 
                self.meta_df,
                seq_len
            )
            
            if len(self.chunk_index) > 0:
                print(f"\n--- Saving cache ({self.mode}): {CACHE_PATH} ---")
                try:
                    with open(CACHE_PATH, 'wb') as f:
                        pickle.dump(self.chunk_index, f)
                except Exception:
                    pass
        
        print(f"Dataset initialized ({self.mode}). Ready to serve {len(self.chunk_index)} chunks.")

    def __len__(self):
        return len(self.chunk_index)

    def __getitem__(self, index):
        task_name, chunk_start, chunk_end = self.chunk_index[index]
        
        try:
            if isinstance(task_name, bytes):
                search_key = task_name.decode('utf-8').strip()
            else:
                search_key = str(task_name).strip()
                
            target_meta = self.meta_df.loc[search_key]
            if isinstance(target_meta, pd.DataFrame):
                target_meta = target_meta.iloc[0]
                
        except KeyError:
            print(f"Warning: Metadata missing for {task_name}. Please delete .pkl cache and retry.")
            return None 

        target_node = target_meta['NodePrefix']
        target_start_abs = target_meta['StartTime']
        target_end_abs = target_meta['EndTime']

        all_tasks_on_node = self.meta_df[self.meta_df['NodePrefix'] == target_node]
        
        chunk_start_time_abs = target_start_abs + chunk_start * 15
        chunk_end_time_abs = target_start_abs + chunk_end * 15
        
        neighbors_df = all_tasks_on_node[
            (all_tasks_on_node['StartTime'] < chunk_end_time_abs) & 
            (all_tasks_on_node['EndTime'] > chunk_start_time_abs) & 
            (all_tasks_on_node.index != task_name)
        ]
        
        if self.hf is None:
            self.hf = h5py.File(self.h5_path, 'r')
            
        try:
            X_A_full = self.hf[f'task_timeseries_data/{task_name}']
            X_A_chunk = X_A_full[chunk_start:chunk_end]
            
            Y_A_chunk_labels = apply_rules_to_batch(
                torch.tensor(X_A_chunk)
            ).numpy()
            
            X_neighbors = []
            neighbor_metas = []
            
            for n_name, neighbor in neighbors_df.iterrows():
                try:
                    X_neighbor = self.hf[f'task_timeseries_data/{n_name}'][:]
                    X_neighbors.append(X_neighbor.astype(np.float32))
                    neighbor_metas.append((neighbor.StartTime, neighbor.EndTime))
                except KeyError:
                    pass
            
            packet = {
                'target_task_name': task_name,
                'target_data_chunk': X_A_chunk.astype(np.float32), 
                'target_labels_chunk': Y_A_chunk_labels,          
                'target_chunk_meta': (chunk_start_time_abs, chunk_end_time_abs), 
                'neighbor_count': len(X_neighbors),
                'neighbor_data_full': X_neighbors,
                'neighbor_meta_full': neighbor_metas
            }
            return packet

        except Exception as e:
            print(f"Error extraction for {task_name}: {e}")
            return None

if __name__ == "__main__":
    print('--- Starting Step 2 (v3 test) ---')
    try:
        dataset = CloudBrainDataset(h5_file_path=H5_FILE_PATH, mode="test")
        print(f"Dataset initialized successfully. Total samples: {len(dataset)}")
    except Exception as e:
        print(f"Initialization failed: {e}")
