import torch

class DiceSelector:
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        
    def select_high_confidence(self, probs):
        """
        Select high confidence pixels/regions for pseudo-labeling.
        """
        max_probs, _ = torch.max(probs, dim=1)
        mask = max_probs > self.threshold
        return mask
