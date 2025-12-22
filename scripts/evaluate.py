import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
import os
import argparse
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix

# Add project root to path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

try:
    from pfhc.datasets.baked_dataset_v10 import BakedDataset_v10, baked_collate_fn_v10
    from pfhc.models.model import PFHC_Model
    from pfhc.utils.mode_rules import K_MODES
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate PFHC Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, default='data/baked_v10/val/',
                        help='Path to test data directory')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model architecture parameters
    parser.add_argument('--input_dim', type=int, default=6)
    parser.add_argument('--hnn_hidden', type=int, default=256)
    parser.add_argument('--hnn_layers', type=int, default=3)
    parser.add_argument('--num_fuzzy_rules', type=int, default=12)
    parser.add_argument('--tcn_hidden', type=int, default=256)
    parser.add_argument('--tcn_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.20)
    parser.add_argument('--tcn_kernel_size', type=int, default=3)
    parser.add_argument('--volatility_window_size', type=int, default=10)
    parser.add_argument('--tau_0', type=int, default=1)
    parser.add_argument('--delta_tau', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.7)
    
    return parser.parse_args()


def evaluate(model, test_loader, device, ignore_index=-1):
    """Evaluate model on test set"""
    model.eval()
    
    all_targets = []
    all_preds = []
    total_loss = 0.0
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            targets_raw, labels_true, neighbors, masks, lengths = batch
            
            if targets_raw is None:
                continue
            
            targets_raw = targets_raw.to(device)
            labels_true = labels_true.to(device)
            neighbors = neighbors.to(device)
            masks = masks.to(device)
            
            # Forward pass
            pred_mu_seq = model(targets_raw, neighbors, masks, lengths)
            
            # Compute loss
            preds_loss = pred_mu_seq[:, :-1, :].reshape(-1, model.fusion_head.K)
            targets_loss = labels_true[:, 1:].reshape(-1)
            
            loss = loss_fn(preds_loss, targets_loss)
            total_loss += loss.item()
            
            # Collect predictions
            mask = (targets_loss != ignore_index)
            all_targets.extend(targets_loss[mask].cpu().numpy())
            all_preds.extend(preds_loss.argmax(dim=-1)[mask].cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    # Calculate per-class metrics
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }


def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test dataset
    print(f"Loading test data from {args.test_data}...")
    try:
        test_dataset = BakedDataset_v10(mode="val")
    except FileNotFoundError:
        print("Test dataset not found. Please run data preprocessing first.")
        sys.exit(1)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=baked_collate_fn_v10,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    print("Initializing model...")
    model = PFHC_Model(
        input_dim=args.input_dim,
        hnn_hidden_dim=args.hnn_hidden,
        hnn_layers=args.hnn_layers,
        num_fuzzy_rules=args.num_fuzzy_rules,
        tcn_hidden_dim=args.tcn_hidden,
        tcn_layers=args.tcn_layers,
        dropout=args.dropout,
        kernel_size=args.tcn_kernel_size,
        volatility_window_size=args.volatility_window_size,
        tau_0=args.tau_0,
        delta_tau=args.delta_tau,
        temperature=args.temperature
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Evaluate
    print("\n" + "="*50)
    print("Starting Evaluation")
    print("="*50 + "\n")
    
    results = evaluate(model, test_loader, device)
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Loss:      {results['loss']:.6f}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"F1-Score:  {results['f1_score']:.4f}")
    print("="*50)
    
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    # Save results
    output_file = os.path.join(os.path.dirname(args.checkpoint), 'evaluation_results.txt')
    with open(output_file, 'w') as f:
        f.write("="*50 + "\n")
        f.write("PFHC Model Evaluation Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test Data: {args.test_data}\n\n")
        f.write(f"Loss:      {results['loss']:.6f}\n")
        f.write(f"Accuracy:  {results['accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"F1-Score:  {results['f1_score']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(results['confusion_matrix']))
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
