import torch
import torch.nn.functional as F
from .entropy import compute_entropy
from .attention_maps import process_attention_maps

class MEMGenerator:
    def __init__(self, alpha=0.5, beta=None, adaptive=True, smooth_kernel=7, smooth_sigma=2.0):
        self.alpha = float(alpha)
        self.beta = float(1.0 - self.alpha) if beta is None else float(beta)
        self.adaptive = bool(adaptive)
        self.smooth_kernel = int(smooth_kernel)
        self.smooth_sigma = float(smooth_sigma)
        
    def generate(self, logits, attn_weights, original_size, feature_hw=None):
        """
        Generate Multi-modal Error Mask (MEM).
        MEM = alpha * TextGuidedAttention + beta * Entropy
        
        Args:
            logits: (B, C, H, W)
            attn_weights: (B, L, K) or similar
            original_size: (H, W)
            feature_hw: Optional (H_f, W_f) to reshape attention tokens.
            
        Returns:
            mem_score: (B, H, W) - scalar score map per pixel
        """
        # 1. Entropy (Where)
        entropy = compute_entropy(logits)  # (B, H, W)
        
        # Normalize entropy to [0, 1] per image
        entropy = self._normalize(entropy)
        
        # 2. Text Attention (What)
        if attn_weights is None:
            text_uncertainty = torch.zeros_like(entropy)
        else:
            spatial_attn = process_attention_maps(attn_weights, original_size, feature_hw=feature_hw)  # (B, K, H, W)
            max_attn = torch.max(spatial_attn, dim=1).values  # (B, H, W)
            text_uncertainty = 1.0 - self._normalize(max_attn)

        if self.adaptive:
            ent_w = entropy.mean(dim=(1, 2), keepdim=True)
            txt_w = text_uncertainty.mean(dim=(1, 2), keepdim=True)
            denom = (ent_w + txt_w).clamp_min(1e-6)
            alpha = (txt_w / denom).clamp(0.0, 1.0)
            beta = (ent_w / denom).clamp(0.0, 1.0)
        else:
            alpha = torch.tensor(self.alpha, device=logits.device, dtype=logits.dtype)
            beta = torch.tensor(self.beta, device=logits.device, dtype=logits.dtype)

        mem = alpha * text_uncertainty + beta * entropy

        if self.smooth_kernel and self.smooth_kernel > 1:
            mem = self._gaussian_blur(mem, kernel_size=self.smooth_kernel, sigma=self.smooth_sigma)
        
        return mem

    def _gaussian_blur(self, x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
        radius = kernel_size // 2
        coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        kernel_1d = torch.exp(-(coords ** 2) / (2 * (sigma ** 2)))
        kernel_1d = (kernel_1d / kernel_1d.sum()).view(1, 1, -1)
        kernel_2d = kernel_1d.transpose(1, 2) @ kernel_1d
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

        out = []
        for i in range(x.shape[0]):
            xi = x[i:i+1].unsqueeze(1)  # (1, 1, H, W)
            yi = F.conv2d(xi, kernel_2d, padding=radius)
            out.append(yi.squeeze(1))
        return torch.cat(out, dim=0)
        
    def _normalize(self, tensor):
        # Min-max normalization per batch item
        B = tensor.shape[0]
        normalized = []
        for i in range(B):
            t = tensor[i]
            min_val = t.min()
            max_val = t.max()
            if max_val - min_val > 1e-6:
                norm_t = (t - min_val) / (max_val - min_val)
            else:
                norm_t = torch.zeros_like(t)
            normalized.append(norm_t)
        return torch.stack(normalized)
