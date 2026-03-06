import numpy as np
from torch.utils.data import Subset

def create_initial_splits(dataset, initial_ratio=0.1, seed=42):
    """
    Creates initial labeled and unlabeled splits.
    
    Args:
        dataset: The full training dataset.
        initial_ratio: Percentage of data to start with.
        seed: Random seed.
        
    Returns:
        labeled_indices (list): Indices for the labeled pool.
        unlabeled_indices (list): Indices for the unlabeled pool.
    """
    np.random.seed(seed)
    total_size = len(dataset)
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    
    split_point = int(total_size * initial_ratio)
    labeled_indices = indices[:split_point].tolist()
    unlabeled_indices = indices[split_point:].tolist()
    
    return labeled_indices, unlabeled_indices

def get_subsets(dataset, labeled_indices, unlabeled_indices):
    labeled_subset = Subset(dataset, labeled_indices)
    unlabeled_subset = Subset(dataset, unlabeled_indices)
    return labeled_subset, unlabeled_subset
