import torch
import torch.nn.functional as F

def compute_entropy(logits):
    """
    Compute pixel-wise entropy from logits.
    Args:
        logits: (B, C, H, W)
    Returns:
        entropy: (B, H, W)
    """
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy
