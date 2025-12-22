import torch
from torch.utils.data import Dataset
import os
import glob
from torch.nn.utils.rnn import pad_sequence

BAKE_DIR = "data/baked_v10"

class BakedDataset_v10(Dataset):
    def __init__(self, mode="train"):
        self.mode = mode
        self.data_dir = os.path.join(BAKE_DIR, mode)
        
        if not os.path.exists(self.data_dir):
            print(f"--- [!! FATAL ERROR !!] ---")
            print(f"Baked data directory does not exist: {self.data_dir}")
            print(f"Please run the 'scripts/bake_dataset_v10.py' script first!")
            raise FileNotFoundError(self.data_dir)
            
        # (Load all .pt file paths)
        self.file_list = sorted(glob.glob(os.path.join(self.data_dir, "*.pt")))
        
        if len(self.file_list) == 0:
            print(f"--- [!! FATAL ERROR !!] ---")
            print(f"No .pt files found in {self.data_dir}!")
            print(f"Please run the 'scripts/bake_dataset_v10.py' script first!")
            raise FileNotFoundError("No .pt files found in baked directory")
        try:
            sample_0 = torch.load(self.file_list[0])
            feat_dim = sample_0[0].shape[-1]
            
            print(f"BakedDataset_v10 ({mode}) initialized successfully.")
            print(f"   > Num samples: {len(self.file_list)}")
            print(f"   > Detected feature dim (Input Dim): {feat_dim} (Expected: 6)")
            
            if feat_dim != 6:
                print(f"\n   > [!!! WARNING !!!] Feature dimension detected as {feat_dim}, not 6!")
                print(f"   > This implies you likely haven't run the new 'bake_dataset_v10.py' script.")
                print(f"   > To aim for 85% accuracy, please make sure to re-bake the data!\n")
                
        except Exception as e:
            print(f"   > [WARNING] Could not preload first sample for inspection: {e}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        try:
            return torch.load(self.file_list[index])
        except Exception as e:
            print(f"Warning: Failed to load baked file {self.file_list[index]}: {e}")
            return None

def baked_collate_fn_v10(batch):

    batch = [sample for sample in batch if sample is not None]
    if not batch:
        return None, None, None, None, None

    targets_raw_list = [item[0] for item in batch]
    labels_true_list = [item[1] for item in batch]
    neighbors_list   = [item[2] for item in batch]
    masks_list       = [item[3] for item in batch]
    lengths_list     = [item[4] for item in batch] 

    targets_padded = pad_sequence(targets_raw_list, batch_first=True, padding_value=0.0)
    
    labels_padded = pad_sequence(labels_true_list, batch_first=True, padding_value=-1)
    
    neighbors_padded = pad_sequence(neighbors_list, batch_first=True, padding_value=0.0)
    
    masks_padded = pad_sequence(masks_list, batch_first=True, padding_value=False)
    
    return targets_padded, labels_padded, neighbors_padded, masks_padded, lengths_list