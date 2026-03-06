import torch

def process_attention_maps(attn_weights, original_size, feature_hw=None):
    """
    Convert attention weights from token space to spatial maps.
    Args:
        attn_weights: (B, L, K) or (B, heads, L, K)
        original_size: (H, W) tuple.
        feature_hw: Optional (H_f, W_f) for reshaping L into spatial grid.
    Returns:
        spatial_attn: (B, K, H, W)
    """
    if attn_weights.dim() == 4:
        attn_weights = attn_weights.mean(dim=1)
        
    B, L, K = attn_weights.shape
    H_orig, W_orig = original_size
    
    if feature_hw is not None:
        H_f, W_f = int(feature_hw[0]), int(feature_hw[1])
    else:
        side = int(L**0.5)
        H_f = W_f = side
        
    spatial_attn = attn_weights.permute(0, 2, 1).view(B, K, H_f, W_f)
    
    spatial_attn = torch.nn.functional.interpolate(
        spatial_attn, size=original_size, mode='bilinear', align_corners=False
    )
    
    return spatial_attn
