import torch
import numpy as np

class ALSampler:
    def __init__(self, budget_per_round):
        self.budget = budget_per_round
        
    def rank_samples(self, mem_scores):
        """
        Rank samples based on aggregate MEM score.
        Args:
            mem_scores: List of (B, H, W) tensors or a single tensor.
                        Wait, typically we process the whole pool.
                        Input: dictionary {img_id: mem_score_scalar}
        Returns:
            selected_ids: List of selected image IDs.
        """
        # Assuming mem_scores is a dict of {img_name: score}
        # Score could be mean MEM over the image.
        
        sorted_samples = sorted(mem_scores.items(), key=lambda item: item[1], reverse=True)
        selected = sorted_samples[:self.budget]
        return [s[0] for s in selected]

    def compute_score(self, mem_map):
        # Aggregate pixel-wise MEM to image-level score
        return mem_map.mean().item()
