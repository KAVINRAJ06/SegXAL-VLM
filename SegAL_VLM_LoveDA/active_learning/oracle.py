import torch
import numpy as np

class Oracle:
    def __init__(self, mode='machine', error_rate=0.05, ignore_index=255):
        self.mode = mode
        self.error_rate = error_rate
        self.ignore_index = int(ignore_index)
        
    def query(self, image, gt_mask, query_mask):
        """
        Simulate annotation.
        Args:
            image: input image
            gt_mask: Ground truth mask
            query_mask: Boolean mask of regions to annotate (from MEM)
        Returns:
            annotated_mask: Mask with new labels filled in for query regions.
                            Pixels outside query regions are ignored (or -1/255).
        """
        # In a real scenario, we might return a sparse mask.
        # Here we simulate returning the GT for the queried regions.
        
        # If machine oracle with error
        if self.mode == 'machine' and self.error_rate > 0:
            # Add noise? For segmentation, maybe flip some pixels.
            # Keeping it simple: just return GT.
            pass
            
        if isinstance(query_mask, torch.Tensor):
            qm = query_mask.to(dtype=torch.bool, device=gt_mask.device)
        else:
            qm = torch.as_tensor(query_mask, dtype=torch.bool, device=gt_mask.device)

        out = torch.full_like(gt_mask, fill_value=self.ignore_index)
        out[qm] = gt_mask[qm]
        return out
