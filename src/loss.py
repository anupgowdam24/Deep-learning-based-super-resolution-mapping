"""
loss.py

Combined loss function for Super-Resolution Land-Cover Mapping:
  1. Classification Loss (Weighted Cross-Entropy, Focal Loss, or Dice Loss)
  2. Physical Consistency Loss (3x3 average-pooled 10m prediction matches 30m ground-truth composition)

The physical consistency requirement enforces that the predicted 10m class
probabilities, when aggregated over 3x3 pixel windows, match the true land-cover
fractions at the 30m Sentinel-2 pixel level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class Soft Dice Loss with optional class weights."""
    def __init__(self, num_classes=5, weight=None, smooth=1e-6):
        super().__init__()
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
            w = self.weight / (torch.sum(self.weight) + 1e-6)
            return torch.sum(w * dice_loss)
        return torch.mean(dice_loss)


class FocalLoss(nn.Module):
    """Multi-class Focal Loss for addressing class imbalance."""
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        # logits: (B, C, H, W), targets: (B, H, W)
        log_pt = F.log_softmax(logits, dim=1)
        pt = torch.exp(log_pt)

        targets_sq = targets.unsqueeze(1)
        log_pt_target = log_pt.gather(1, targets_sq).squeeze(1)
        pt_target = pt.gather(1, targets_sq).squeeze(1)

        focal_weight = (1.0 - pt_target) ** self.gamma
        loss = -focal_weight * log_pt_target

        if self.weight is not None:
            alpha = self.weight[targets]
            loss = alpha * loss

        return loss.mean()


class SRLoss(nn.Module):
    """
    Dual-objective Super-Resolution Loss:
      L_total = L_clf + lambda * L_consistency
    """
    def __init__(self, class_weights=None, consistency_lambda=0.3,
                 use_focal=False, focal_gamma=2.0, use_dice=False,
                 logit_adjustment=None):
        super().__init__()
        self.consistency_lambda = consistency_lambda
        self.use_focal = use_focal
        self.use_dice = use_dice
        self.class_weights = class_weights
        self.num_classes = len(class_weights) if class_weights is not None else 5
        self.logit_adjustment = logit_adjustment

        if use_focal:
            self.clf_criterion = FocalLoss(weight=class_weights, gamma=focal_gamma)
        else:
            self.clf_criterion = nn.CrossEntropyLoss(weight=class_weights)

        if use_dice:
            self.dice_criterion = DiceLoss(num_classes=self.num_classes, weight=class_weights)

    def forward(self, logits_10m, labels_10m):
        """
        Args:
            logits_10m: (B, 5, 96, 96) model logits at 10m
            labels_10m: (B, 96, 96) integer class labels (0..4)
        Returns:
            total_loss, clf_loss, cons_loss
        """
        # Apply logit adjustment to classification loss if present
        if self.logit_adjustment is not None:
            adj = self.logit_adjustment.view(1, -1, 1, 1).to(logits_10m.device)
            clf_logits = logits_10m + adj
        else:
            clf_logits = logits_10m

        # 1. Classification Loss
        if self.use_dice:
            ce_loss = self.clf_criterion(clf_logits, labels_10m)
            d_loss = self.dice_criterion(clf_logits, labels_10m)
            clf_loss = 0.5 * ce_loss + 0.5 * d_loss
        else:
            clf_loss = self.clf_criterion(clf_logits, labels_10m)

        # 2. Physical Consistency Loss
        probs_10m = F.softmax(logits_10m, dim=1)  # (B, 5, 96, 96)

        # Average predicted sub-pixel probabilities across 3x3 windows to 30m
        pred_30m_composition = F.avg_pool2d(probs_10m, kernel_size=3, stride=3)  # (B, 5, 32, 32)

        # Average true one-hot sub-pixel labels across 3x3 windows to true 30m composition
        labels_one_hot = F.one_hot(labels_10m, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        true_30m_composition = F.avg_pool2d(labels_one_hot, kernel_size=3, stride=3)  # (B, 5, 32, 32)

        # Mean Absolute Error (L1 loss) between predicted and true composition
        cons_loss = F.l1_loss(pred_30m_composition, true_30m_composition)

        total_loss = clf_loss + self.consistency_lambda * cons_loss
        return total_loss, clf_loss, cons_loss
