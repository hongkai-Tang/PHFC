import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import h5py
import concurrent.futures
#Note for this file: The file path on line 65 must be modified.
BASE_DATA_PATH = "../CloudBrain-datasets-master/cloudbrain-datasets"
CPU_DIR = os.path.join(BASE_DATA_PATH, "cpu_usage")
GUTIL_DIR = os.path.join(BASE_DATA_PATH, "dcgm-gpu-utilization")
GMEM_DIR = os.path.join(BASE_DATA_PATH, "dcgm_fb_used")
CSV_PATH = "../CloudBrain-datasets-master/cloudbrain-datasets"
JOBS_INFO_FILE = os.path.join(CSV_PATH, "jobs_info.csv")
OUTPUT_H5_FILE = "../data/processed/cloudbrain_task_centric_RAW.h5"
MAX_WORKERS = os.cpu_count()

def get_node_prefix(node_name_full: str):
    try:
        return node_name_full.split('-')[0]
    except Exception:
        return None

def parse_15s_data_line(line: str):
    try:
        return np.fromstring(line, dtype=np.float32, sep=' ')
    except Exception:
        return np.array([], dtype=np.float32)

def read_index_file(file_path):
    try:
        with open(file_path, 'r') as f:
            header_line = f.readline()
        if header_line:
            task_name = header_line.strip().split()[0]
            return task_name, file_path
    except Exception:
        pass
    return None, None

def build_index_map_parallel(data_dir, desc):
    print(f"Building index for {desc} (using {MAX_WORKERS} cores)...")
    index_map = {}
    file_paths = glob.glob(os.path.join(data_dir, '*'))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = executor.map(read_index_file, file_paths)
        for task_name, file_path in tqdm(results, total=len(file_paths), desc=f"Indexing {desc}"):
            if task_name and file_path:
                index_map[task_name] = file_path
                
    print(f"Index for {desc} built. Found {len(index_map)} entries.")
    return index_map

def process_task_wrapper(task_info_tuple):
    task_name, cpu_path, gutil_path, gmem_path = task_info_tuple
    
    try:
        if not (cpu_path and gutil_path and gmem_path):
            return task_name, None

        with open(cpu_path, 'r') as f:
            lines = f.readlines()
        cpu_values = parse_15s_data_line(lines[1])
        
        with open(gutil_path, 'r') as f:
            lines = f.readlines()
        header_parts = lines[0].strip().split()
        n_gpu = int(header_parts[2])
        
        gutil_values_sum = np.zeros_like(cpu_values, dtype=np.float32)
        gutil_values_count = np.zeros_like(cpu_values, dtype=np.float32)
        for i in range(n_gpu):
            gpu_data_line = lines[2 + i*2]
            gpu_values = parse_15s_data_line(gpu_data_line)
            max_len = min(len(gutil_values_sum), len(gpu_values))
            gutil_values_sum[:max_len] += gpu_values[:max_len]
            gutil_values_count[:max_len] += 1
        gutil_values_count[gutil_values_count == 0] = 1
        gutil_values_avg = gutil_values_sum / gutil_values_count

        with open(gmem_path, 'r') as f:
            lines = f.readlines()
        header_parts = lines[0].strip().split()
        n_gpu = int(header_parts[2])
        gmem_values_sum = np.zeros_like(cpu_values, dtype=np.float32)
        gmem_values_count = np.zeros_like(cpu_values, dtype=np.float32)
        for i in range(n_gpu):
            gpu_data_line = lines[2 + i*2]
            gpu_values = parse_15s_data_line(gpu_data_line)
            max_len = min(len(gmem_values_sum), len(gpu_values))
            gmem_values_sum[:max_len] += gpu_values[:max_len]
            gmem_values_count[:max_len] += 1
        gmem_values_count[gmem_values_count == 0] = 1
        gmem_values_avg = gmem_values_sum / gmem_values_count

        T_k = min(len(cpu_values), len(gutil_values_avg), len(gmem_values_avg))
        if T_k == 0:
            return task_name, None
        
        task_data = np.zeros((T_k, 3), dtype=np.float32)
        task_data[:, 0] = cpu_values[:T_k]
        task_data[:, 1] = gutil_values_avg[:T_k]
        task_data[:, 2] = gmem_values_avg[:T_k]
        task_data = np.nan_to_num(task_data, nan=0.0)
        
        return task_name, task_data.astype(np.float32)
        
    except Exception as e:
        return task_name, None

def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
    
    output_dir = os.path.dirname(OUTPUT_H5_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("--- Pass 1: Building Index Maps (Parallel) ---")
    cpu_map = build_index_map_parallel(CPU_DIR, "CPU")
    gutil_map = build_index_map_parallel(GUTIL_DIR, "GPU Util")
    gmem_map = build_index_map_parallel(GMEM_DIR, "GPU Mem")
    print("\nAll indexes built.")

    print("--- Loading Metadata (Jobs) ---")
    try:
        df_jobs = pd.read_csv(JOBS_INFO_FILE, header=None,
                              names=['TaskName', 'ReqGpu', 'ReqCpu', 'ReqMem',
                                     'SubmitTime', 'StartTime', 'EndTime',
                                     'JobType', 'ScheduledNode', 'AvgGpuUtil',
                                     'AvgGpuMem', 'AvgCpuUtil', 'AvgHostMem'])
        df_jobs['JobId'] = df_jobs['TaskName'].apply(lambda x: str(x).split('_')[1])
        df_jobs['NodePrefix'] = df_jobs['ScheduledNode'].apply(get_node_prefix)
        print(f"Loaded {len(df_jobs)} tasks from jobs_info.csv.")
    except Exception as e:
        print(f"Fatal: Could not load {JOBS_INFO_FILE}: {e}")
        return

    print(f"\n--- Pass 2: Processing Tasks in Parallel (using {MAX_WORKERS} cores) ---")
    
    tasks_to_process = [] 
    print("Preparing task list for parallel execution...")
    for row in tqdm(df_jobs.itertuples(), total=len(df_jobs), desc="Finding data paths"):
        task_name = row.TaskName
        cpu_path = cpu_map.get(task_name)
        gutil_path = gutil_map.get(task_name)
        gmem_path = gmem_map.get(task_name)
        tasks_to_process.append((task_name, cpu_path, gutil_path, gmem_path))

    print("Starting parallel processing pool (Stage 1/2)...")
    results_in_memory = []
    success_count = 0
    missing_count = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results_iterable = executor.map(process_task_wrapper, tasks_to_process)
        
        print("Collecting results from workers (this may take time)...")
        for task_name, task_data in tqdm(results_iterable, total=len(tasks_to_process), desc="Processing Tasks"):
            if task_data is not None:
                results_in_memory.append((task_name, task_data))
                success_count += 1
            else:
                missing_count += 1
    
    print("\nAll processing complete. Now writing to HDF5 (Stage 2/2)...")
    with h5py.File(OUTPUT_H5_FILE, 'w') as hf:
        meta_df_to_save = df_jobs[['TaskName', 'JobId', 'NodePrefix', 'StartTime', 'EndTime']]
        meta_csv_string = meta_df_to_save.to_csv(index=False)
        hf.create_dataset('task_graph_meta', data=meta_csv_string, dtype=h5py.string_dtype(encoding='utf-8'))
        
        data_group = hf.create_group('task_timeseries_data')
        for task_name, task_data in tqdm(results_in_memory, desc="Writing to HDF5"):
            data_group.create_dataset(task_name, data=task_data, compression="gzip")

    print("\n--- Preprocessing Complete! (RAW Values) ---")
    print(f"Successfully processed and saved {success_count} tasks.")
    print(f"Skipped {missing_count} tasks (due to missing data files).")
    print(f"Final dataset saved to: {OUTPUT_H5_FILE}")

if __name__ == "__main__":
    main()
