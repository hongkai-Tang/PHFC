import torch

K_MODES = 12

MODE_IDLE = 0
MODE_CPU_HEAVY = 1
MODE_GPU_HEAVY = 2
MODE_CPU_GPU_MIXED = 3
MODE_MEM_HEAVY = 4
MODE_CPU_MEM = 5
MODE_GPU_MEM = 6
MODE_OTHER = 7

def apply_rules_to_batch(targets_batch):
    
    if targets_batch.dim() == 2:
        targets_batch = targets_batch.unsqueeze(1)
        
    B, T_max, C = targets_batch.shape
    device = targets_batch.device
    
    cpu = targets_batch[..., 0]
    gpu_util = targets_batch[..., 1]
    gpu_mem = targets_batch[..., 2]

    labels = torch.full((B, T_max), MODE_OTHER, dtype=torch.long, device=device)
    
    labels = torch.where(
        (gpu_util > 80) & (gpu_mem > 4000), 
        MODE_GPU_MEM, labels
    )
    
    labels = torch.where(
        (cpu > 80) & (gpu_mem > 4000), 
        MODE_CPU_MEM, labels
    )
    
    labels = torch.where(
        (gpu_mem > 4000) & (cpu < 20) & (gpu_util < 20), 
        MODE_MEM_HEAVY, labels
    )
    
    labels = torch.where(
        (cpu > 50) & (gpu_util > 50), 
        MODE_CPU_GPU_MIXED, labels
    )
    
    labels = torch.where(
        (cpu < 20) & (gpu_util > 80), 
        MODE_GPU_HEAVY, labels
    )

    labels = torch.where(
        (cpu > 80) & (gpu_util < 20), 
        MODE_CPU_HEAVY, labels
    )
    
    labels = torch.where(
        (cpu < 10) & (gpu_util < 10), 
        MODE_IDLE, labels
    )
    
    return labels

if __name__ == "__main__":
    print("--- Starting Step K (Rule Engine) Test ---")
    
    test_batch = torch.tensor([
        [  5.0,   3.0,    100.0],
        [ 90.0,  10.0,    100.0],
        [ 10.0,  90.0,    100.0],
        [ 60.0,  60.0,    100.0],
        [ 10.0,  10.0,   5000.0],
        [ 90.0,  10.0,   5000.0],
        [ 10.0,  90.0,   5000.0],
        [ 40.0,  30.0,   1000.0],
        [ 90.0,  90.0,   5000.0]
    ]).unsqueeze(0)

    print(f"Created test batch (B=1, T=9, C=3):\n {test_batch.squeeze(0)}")

    labels = apply_rules_to_batch(test_batch)
    
    labels = labels.squeeze(0)
    
    print(f"\n[Success] 'K Process' API call completed.")
    print(f"  > Output shape: {labels.shape}")
    print(f"  > Output labels: {labels}")

    expected_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 6])
    
    if torch.equal(labels, expected_labels):
        print("  [✅] Output matches expected rules 100%!")
    else:
        print(f"  [!! Failed !!] Output does not match expected!")
        print(f"     Expected: {expected_labels}")
        print(f"     Got: {labels}")
        
    print("\n--- Step K (Rule Engine) Test Complete ---")
