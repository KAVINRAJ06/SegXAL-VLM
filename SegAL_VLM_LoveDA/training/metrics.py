import torch
import numpy as np
import os
import json

class Evaluator:
    def __init__(self, num_classes, device='cpu'):
        self.num_classes = num_classes
        self.device = device
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    def update(self, preds, targets):
        """
        preds: [B, H, W] or [B, C, H, W]
        targets: [B, H, W]
        """
        if preds.dim() == 4:
            preds = torch.argmax(preds, dim=1)
            
        preds = preds.flatten()
        targets = targets.flatten()
        
        # Filter out ignore_index (255)
        mask = (targets >= 0) & (targets < self.num_classes)
        preds = preds[mask]
        targets = targets[mask]
        
        # Update confusion matrix
        # indices = self.num_classes * targets + preds
        # self.confusion_matrix += torch.bincount(indices, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        
        # Efficient confusion matrix computation on GPU
        # y_true * num_classes + y_pred
        if preds.numel() > 0:
            indices = targets * self.num_classes + preds
            counts = torch.bincount(indices, minlength=self.num_classes**2)
            self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def get_scores(self):
        """
        Returns dictionary of scores:
        - Pixel Accuracy
        - mIoU
        - Class-wise IoU
        """
        cm = self.confusion_matrix.float()
        
        # Pixel Accuracy
        pixel_acc = torch.diag(cm).sum() / (cm.sum() + 1e-10)
        
        # IoU per class
        intersection = torch.diag(cm)
        union = cm.sum(dim=1) + cm.sum(dim=0) - intersection
        iou = intersection / (union + 1e-10)
        
        miou = torch.nanmean(iou)
        
        return {
            "pixel_acc": pixel_acc.item(),
            "miou": miou.item(),
            "class_iou": iou.cpu().numpy().tolist() # Convert to list for JSON serialization
        }

    def reset(self):
        self.confusion_matrix.zero_()

def update_metrics_history(history_path, epoch_metrics, class_names=None):
    """
    Append epoch metrics to metrics_history.json in the required format.
    history_path: path to experiments/logs/metrics_history.json
    epoch_metrics: dict with keys:
      - epoch
      - train_loss
      - train_miou
      - train_pixel_acc
      - val_loss
      - val_miou
      - val_pixel_acc
      - val_class_iou (list of length 6)
    class_names: optional list of class names in the expected order
                 ['Building','Road','Water','Barren','Forest','Agriculture']
    """
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    
    if class_names is None:
        class_names = ['Building', 'Road', 'Water', 'Barren', 'Forest', 'Agriculture']
    
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            hist = json.load(f)
    else:
        hist = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "train_miou": [],
            "val_miou": [],
            "classwise_iou": {name: [] for name in class_names}
        }
    
    hist["epochs"].append(epoch_metrics["epoch"])
    hist["train_loss"].append(float(epoch_metrics["train_loss"]))
    hist["val_loss"].append(float(epoch_metrics["val_loss"]))
    hist["train_accuracy"].append(float(epoch_metrics["train_pixel_acc"]))
    hist["val_accuracy"].append(float(epoch_metrics["val_pixel_acc"]))
    hist["train_miou"].append(float(epoch_metrics["train_miou"]))
    hist["val_miou"].append(float(epoch_metrics["val_miou"]))
    
    class_iou = epoch_metrics.get("val_class_iou", [])
    # Ensure length matches class_names
    if isinstance(class_iou, list):
        for i, name in enumerate(class_names):
            val = float(class_iou[i]) if i < len(class_iou) else float('nan')
            hist["classwise_iou"][name].append(val)
    else:
        for name in class_names:
            hist["classwise_iou"][name].append(float('nan'))
    
    with open(history_path, 'w') as f:
        json.dump(hist, f, indent=4)
