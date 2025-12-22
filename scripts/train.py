import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys, os
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, precision_score, f1_score
from torch.cuda.amp import autocast, GradScaler 

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

try:
    from pfhc.datasets.baked_dataset_v10 import BakedDataset_v10, baked_collate_fn_v10
    from pfhc.models.model import PFHC_Model, SpatialHNNModule 
    from pfhc.utils.mode_rules import K_MODES 
except ImportError as e:
    print("--- [!! Failed !!] ---")
    print(f"Error: {e}")
    sys.exit(1)

MODEL_SAVE_PATH = "checkpoints/pfhc_model_v10_final.pth" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32

ACCUMULATION_STEPS = 8 

LEARNING_RATE = 0.001
EPOCHS = 50

INPUT_DIM = 6 
HNN_HIDDEN = 256
HNN_LAYERS = 3
NUM_FUZZY_RULES = K_MODES

TCN_HIDDEN = 256      
TCN_LAYERS = 4       
TCN_KERNEL_SIZE = 3  
DROPOUT = 0.20

VOLATILITY_WINDOW_SIZE = 10 
TAU_0 = 1                   
DELTA_TAU = 5               
TEMPERATURE = 0.7
GRAD_CLIP_MAX_NORM = 0.8

CACHE_DIR = "data/baked_v10"
CLASS_WEIGHTS_CACHE = os.path.join(CACHE_DIR, "cache_class_weights.pt")
KMEANS_CENTERS_CACHE = os.path.join(CACHE_DIR, "cache_kmeans_centers.pt")
KMEANS_BANDWIDTHS_CACHE = os.path.join(CACHE_DIR, "cache_kmeans_bandwidths.pt")

def get_class_weights(train_dataset, num_classes, ignore_index=-1):
    if os.path.exists(CLASS_WEIGHTS_CACHE):
        print(f"Loading class weights from cache...")
        return torch.load(CLASS_WEIGHTS_CACHE).to(DEVICE)
    print("Calculating class weights...")
    label_counts = Counter()
    workers = 6 if sys.platform == "linux" else 0
    temp_loader = DataLoader(train_dataset, batch_size=64, collate_fn=baked_collate_fn_v10, num_workers=workers)
    
    MAX_BATCHES = 500
    curr = 0
    for batch in tqdm(temp_loader, desc="Counting Labels"):
        _, labels_true, _, _, _ = batch
        if labels_true is not None:
            labels_flat = labels_true.view(-1)
            valid_labels = labels_flat[labels_flat != ignore_index]
            label_counts.update(valid_labels.numpy())
        curr += 1
        if curr >= MAX_BATCHES: break

    total_samples = sum(label_counts.values())
    weights = torch.zeros(num_classes)
    for i in range(num_classes):
        count = label_counts.get(i, 0)
        weights[i] = (total_samples / (count + 1e-6)) ** 0.5
    weights = weights / weights.mean()
    torch.save(weights, CLASS_WEIGHTS_CACHE)
    return weights.to(DEVICE)

def initialize_fuzzification_layer(train_loader_v10, hnn_hidden_dim, input_dim_val, k=NUM_FUZZY_RULES):
    if os.path.exists(KMEANS_CENTERS_CACHE) and os.path.exists(KMEANS_BANDWIDTHS_CACHE):
        cached_centers = torch.load(KMEANS_CENTERS_CACHE)
        if cached_centers.shape[0] == k and cached_centers.shape[1] == hnn_hidden_dim:
            print(f"Loading K-Means from cache (K={k}, H={hnn_hidden_dim})...")
            return cached_centers.to(DEVICE), torch.load(KMEANS_BANDWIDTHS_CACHE).to(DEVICE)
        else:
            print(f"⚠️ Cache mismatch! Cached: {cached_centers.shape}, Expected: ({k}, {hnn_hidden_dim})")
            print(f"   Deleting old cache and re-initializing...")
            os.remove(KMEANS_CENTERS_CACHE)
            os.remove(KMEANS_BANDWIDTHS_CACHE)

    print(f"\n--- Running K-Means (K={k}) Initialization (Multiple runs for best result) ---")
    base_model = SpatialHNNModule(input_dim=input_dim_val, hnn_hidden_dim=hnn_hidden_dim, hnn_layers=HNN_LAYERS).to(DEVICE)
    base_model.eval()
    
    best_kmeans = None
    best_inertia = float('inf')
    num_trials = 3
    
    for trial in range(num_trials):
        print(f"  Trial {trial+1}/{num_trials}...")
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42+trial, n_init=10, batch_size=1024, max_iter=100)
        
        processed = 0
        with torch.no_grad():
            for batch in tqdm(train_loader_v10, desc=f"K-Means Fitting (Trial {trial+1})"):
                targets_raw, _, neighbors, masks, lengths = batch
                if targets_raw is None: continue
                
                hnn_features = base_model(targets_raw.to(DEVICE), neighbors.to(DEVICE), masks.to(DEVICE), lengths)
                mask = (torch.arange(targets_raw.size(1)).unsqueeze(0).to(DEVICE) < torch.tensor(lengths).unsqueeze(1).to(DEVICE))
                valid_features = hnn_features[mask]
                if valid_features.shape[0] > 0:
                    kmeans.partial_fit(valid_features.cpu().numpy())
                
                processed += 1
                if processed > 200: break
        
        if kmeans.inertia_ < best_inertia:
            best_inertia = kmeans.inertia_
            best_kmeans = kmeans
            print(f"    New best inertia: {best_inertia:.4f}")
    
    centers = torch.tensor(best_kmeans.cluster_centers_, dtype=torch.float32)
    bandwidths = torch.ones(k) * 5.0
    torch.save(centers, KMEANS_CENTERS_CACHE)
    torch.save(bandwidths, KMEANS_BANDWIDTHS_CACHE)
    return centers.to(DEVICE), bandwidths.to(DEVICE)

def main():
    try:
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')
    except: pass

    torch.backends.cudnn.benchmark = True 
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, "training_log.txt")
    
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("Epoch,Stage,Loss,Accuracy,Precision,F1,LR\n")
    
    print(f"--- Starting Training (4090 Fix - Batch {BATCH_SIZE}x{ACCUMULATION_STEPS}) ---")
    
    IGNORE_INDEX = -1

    try:
        train_dataset = BakedDataset_v10(mode="train")
        valid_dataset = BakedDataset_v10(mode="val")
    except FileNotFoundError:
        print("Please run 'scripts/bake_dataset_v10.py' first!")
        sys.exit(1)
    
    workers = 6 if sys.platform == "linux" else 0 
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=baked_collate_fn_v10, num_workers=workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=baked_collate_fn_v10, num_workers=workers, pin_memory=True)
    
    class_weights = get_class_weights(train_dataset, NUM_FUZZY_RULES, ignore_index=IGNORE_INDEX)

    print(f"\n[Init] PFHC Model...")
    model = PFHC_Model(
        input_dim=INPUT_DIM, 
        hnn_hidden_dim=HNN_HIDDEN, 
        hnn_layers=HNN_LAYERS,
        num_fuzzy_rules=NUM_FUZZY_RULES,
        tcn_hidden_dim=TCN_HIDDEN, 
        tcn_layers=TCN_LAYERS, 
        dropout=DROPOUT,
        kernel_size=TCN_KERNEL_SIZE,
        volatility_window_size=VOLATILITY_WINDOW_SIZE, 
        tau_0=TAU_0, 
        delta_tau=DELTA_TAU, 
        temperature=TEMPERATURE
    ).to(DEVICE)
    
    centers, bandwidths = initialize_fuzzification_layer(train_loader, HNN_HIDDEN, INPUT_DIM)
    model.fuzzification.centers.data = centers
    model.fuzzification.bandwidths.data = bandwidths

    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights, 
        ignore_index=IGNORE_INDEX
    )
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    
    scaler = GradScaler()
    best_valid_loss = float('inf')
    
    best_valid_acc = 0.0
    patience_counter = 0
    early_stop_patience = 10
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        model.train() 
        total_loss = 0.0
        all_targets = []
        all_preds = []
        
        optimizer.zero_grad()
        
        for idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            targets_raw, labels_true, neighbors, masks, lengths = batch
            if targets_raw is None: continue

            targets_raw, labels_true = targets_raw.to(DEVICE), labels_true.to(DEVICE)
            neighbors, masks = neighbors.to(DEVICE), masks.to(DEVICE)
            
            with autocast():
                pred_mu_seq = model(targets_raw, neighbors, masks, lengths)
                preds_loss = pred_mu_seq[:, :-1, :].reshape(-1, NUM_FUZZY_RULES)
                targets_loss = labels_true[:, 1:].reshape(-1)
                
                loss = loss_fn(preds_loss, targets_loss)
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            
            if (idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                with torch.no_grad():
                    model.fuzzification.bandwidths.data.clamp_(min=0.01)
            
            total_loss += loss.item() * ACCUMULATION_STEPS
            
            mask = (targets_loss != IGNORE_INDEX)
            all_targets.extend(targets_loss[mask].cpu().numpy())
            all_preds.extend(preds_loss.argmax(dim=-1)[mask].cpu().numpy())
        
        if len(train_loader) % ACCUMULATION_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        avg_loss = total_loss / (len(train_loader) + 1e-9)
        acc = accuracy_score(all_targets, all_preds)
        prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        curr_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {avg_loss:.8f} | Acc: {acc:.4f} | Precision: {prec:.4f} | F1: {f1:.4f} | LR: {curr_lr:.6f}")
        with open(LOG_FILE, "a") as f:
            f.write(f"{epoch+1},Train,{avg_loss:.6f},{acc:.4f},{prec:.4f},{f1:.4f},{curr_lr:.6f}\n")

        model.eval()
        total_loss = 0.0
        all_targets = []
        all_preds = []
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validating"):
                targets_raw, labels_true, neighbors, masks, lengths = batch
                if targets_raw is None: continue
                
                targets_raw, labels_true = targets_raw.to(DEVICE), labels_true.to(DEVICE)
                neighbors, masks = neighbors.to(DEVICE), masks.to(DEVICE)
                
                with autocast():
                    pred_mu_seq = model(targets_raw, neighbors, masks, lengths)
                    preds_loss = pred_mu_seq[:, :-1, :].reshape(-1, NUM_FUZZY_RULES)
                    targets_loss = labels_true[:, 1:].reshape(-1)
                    loss = loss_fn(preds_loss, targets_loss)
                
                total_loss += loss.item()
                mask = (targets_loss != IGNORE_INDEX)
                all_targets.extend(targets_loss[mask].cpu().numpy())
                all_preds.extend(preds_loss.argmax(dim=-1)[mask].cpu().numpy())

        avg_loss = total_loss / (len(valid_loader) + 1e-9)
        acc = accuracy_score(all_targets, all_preds)
        prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        scheduler.step()
        
        print(f"Valid Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Precision: {prec:.4f} | F1: {f1:.4f}")
        with open(LOG_FILE, "a") as f:
            f.write(f"{epoch+1},Valid,{avg_loss:.6f},{acc:.4f},{prec:.4f},{f1:.4f},{curr_lr:.6f}\n")

        if avg_loss < best_valid_loss:
            best_valid_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✨ Saved Best Model (Loss: {best_valid_loss:.4f})")
        
        if acc > best_valid_acc:
            best_valid_acc = acc
            patience_counter = 0
            print(f"✨ New Best Accuracy: {best_valid_acc:.4f}")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement for {patience_counter}/{early_stop_patience} epochs")
            if patience_counter >= early_stop_patience:
                print(f"🛑 Early Stopping triggered! Best Acc: {best_valid_acc:.4f}")
                break

if __name__ == "__main__":
    main()
