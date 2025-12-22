import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, num_classes=None, alpha=None, gamma=2.0, reduction='mean', ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, preds, targets):
        p = preds.clamp(min=1e-9, max=1.0 - 1e-9)
        
        mask = (targets != self.ignore_index)
        targets_safe = targets.clone()
        targets_safe[~mask] = 0
        
        p_t = p.gather(1, targets_safe.unsqueeze(1)).squeeze(1)
        
        log_p_t = p_t.log()
        focal_term = (1 - p_t).pow(self.gamma)
        loss = -1 * log_p_t * focal_term
        
        if self.alpha is not None:
            alpha_t = self.alpha.to(preds.device)[targets_safe]
            loss = loss * alpha_t
            
        loss = loss * mask.float()
        
        if self.reduction == 'mean':
            return loss.sum() / (mask.sum() + 1e-9)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

WeightedLabelSmoothingCrossEntropyLoss = FocalLoss
