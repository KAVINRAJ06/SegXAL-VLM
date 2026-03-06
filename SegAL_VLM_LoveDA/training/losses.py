import torch
import torch.nn as nn
import torch.nn.functional as F

class SegLoss(nn.Module):
    def __init__(self, num_classes=7, ignore_index=255, label_smoothing=0.0, class_weights=None, focal_gamma=0.0, dice_weight=1.0):
        super().__init__()
        self.focal_gamma = float(focal_gamma) if focal_gamma is not None else 0.0
        self.label_smoothing = float(label_smoothing) if label_smoothing is not None else 0.0
        self.dice_weight = float(dice_weight) if dice_weight is not None else 1.0
        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=torch.float)
            if w.numel() != int(num_classes):
                raise ValueError(f"class_weights must have length {num_classes}, got {int(w.numel())}")
            self.register_buffer("class_weights", w)
        else:
            self.class_weights = None

        try:
            self.ce = nn.CrossEntropyLoss(
                ignore_index=ignore_index,
                label_smoothing=self.label_smoothing,
                weight=self.class_weights
            )
        except TypeError:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=self.class_weights)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
    def dice_loss(self, logits, target):
        # target: (B, H, W)
        # logits: (B, C, H, W)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target.clamp(0, self.num_classes-1), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Softmax logits
        probs = F.softmax(logits, dim=1)
        
        # Mask out ignore index
        mask = (target != self.ignore_index).unsqueeze(1).float()
        
        intersection = torch.sum(probs * target_one_hot * mask, dim=(2, 3))
        union = torch.sum(probs * mask, dim=(2, 3)) + torch.sum(target_one_hot * mask, dim=(2, 3))
        
        dice = 2.0 * intersection / (union + 1e-6)
        return 1.0 - dice.mean()
        
    def forward(self, logits, target):
        if self.focal_gamma > 0:
            try:
                ce_map = F.cross_entropy(
                    logits,
                    target,
                    ignore_index=self.ignore_index,
                    weight=self.class_weights,
                    reduction="none",
                    label_smoothing=self.label_smoothing
                )
            except TypeError:
                ce_map = F.cross_entropy(
                    logits,
                    target,
                    ignore_index=self.ignore_index,
                    weight=self.class_weights,
                    reduction="none"
                )
            valid = (target != self.ignore_index)
            if valid.any():
                ce = ce_map[valid]
                pt = torch.exp(-ce)
                loss_ce = (((1.0 - pt) ** self.focal_gamma) * ce).mean()
            else:
                loss_ce = ce_map.mean() * 0.0
        else:
            loss_ce = self.ce(logits, target)
        if self.dice_weight <= 0:
            return loss_ce
        loss_dice = self.dice_loss(logits, target)
        return loss_ce + (self.dice_weight * loss_dice)
