import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.visual_proj = nn.Identity() if int(visual_dim) == int(hidden_dim) else nn.Linear(int(visual_dim), int(hidden_dim))
        self.text_proj = nn.Identity() if int(text_dim) == int(hidden_dim) else nn.Linear(int(text_dim), int(hidden_dim))
        
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_features, text_features):
        """
        Args:
            visual_features: (B, H*W, visual_dim) - flattened spatial features
            text_features: (B, num_classes, text_dim) - text embeddings per class
        """
        B, L, _ = visual_features.shape
        
        v = self.visual_proj(visual_features) # (B, L, hidden_dim)
        t = self.text_proj(text_features)     # (B, num_classes, hidden_dim)

        attn_output, attn_weights = self.multihead_attn(query=v, key=t, value=t)
        
        output = self.norm(v + self.dropout(attn_output))
        
        return output, attn_weights
