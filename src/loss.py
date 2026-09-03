import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import get_class_weights

class DiceLoss(nn.Module):
    def __init__(self, num_classes=5, weight=None, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B, C, H, W), targets: (B, H, W)
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score
        
        if self.weight is not None:
            w = self.weight / torch.sum(self.weight)
            return torch.sum(w * dice_loss)
        else:
            return torch.mean(dice_loss)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        # logits: (B, C, H, W), targets: (B, H, W)
        log_pt = F.log_softmax(logits, dim=1)
        pt = torch.exp(log_pt)
        
        targets_unsqueezed = targets.unsqueeze(1)
        log_pt_target = log_pt.gather(1, targets_unsqueezed).squeeze(1)
        pt_target = pt.gather(1, targets_unsqueezed).squeeze(1)
        
        focal_weight = (1.0 - pt_target) ** self.gamma
        loss = -focal_weight * log_pt_target
        
        if self.weight is not None:
            alpha = self.weight[targets]
            loss = alpha * loss
            
        return loss.mean()

class SRLoss(nn.Module):
    def __init__(self, class_weights=None, consistency_lambda=0.3, use_focal=False, focal_gamma=2.0, use_dice=False):
        super(SRLoss, self).__init__()
        self.use_focal = use_focal
        self.use_dice = use_dice
        self.class_weights = class_weights
        
        if use_focal:
            self.clf_criterion = FocalLoss(weight=class_weights, gamma=focal_gamma)
        else:
            self.clf_criterion = nn.CrossEntropyLoss(weight=class_weights)
            
        if use_dice:
            self.dice_criterion = DiceLoss(num_classes=len(class_weights) if class_weights is not None else 5, weight=class_weights)
            
        self.consistency_lambda = consistency_lambda
        self.num_classes = len(class_weights) if class_weights is not None else 5

    def forward(self, logits_10m, labels_10m):
        """
        logits_10m: (B, C, H, W) - 10m resolution (e.g. 96x96)
        labels_10m: (B, H, W) - 10m resolution (e.g. 96x96)
        """
        # 1. Classification Loss (Cross Entropy / Focal / Combined Dice)
        if self.use_dice:
            ce_loss = self.clf_criterion(logits_10m, labels_10m)
            d_loss = self.dice_criterion(logits_10m, labels_10m)
            clf_loss = 0.5 * ce_loss + 0.5 * d_loss
        else:
            clf_loss = self.clf_criterion(logits_10m, labels_10m)

        # 2. Consistency Loss
        probs_10m = F.softmax(logits_10m, dim=1) # (B, C, H, W)
        
        # Average-pool predicted probabilities to 30m
        pred_30m_composition = F.avg_pool2d(probs_10m, kernel_size=3, stride=3) # (B, C, H/3, W/3)
        
        # Convert true labels to one-hot and average-pool to 30m
        labels_one_hot = F.one_hot(labels_10m, num_classes=self.num_classes).permute(0, 3, 1, 2).float() # (B, C, H, W)
        true_30m_composition = F.avg_pool2d(labels_one_hot, kernel_size=3, stride=3) # (B, C, H/3, W/3)
        
        # Mean Absolute Error between predicted and true composition
        cons_loss = F.l1_loss(pred_30m_composition, true_30m_composition)

        total_loss = clf_loss + self.consistency_lambda * cons_loss
        
        return total_loss, clf_loss, cons_loss
