import torch
import torch.nn as nn


class PromptEncoder(nn.Module):
    def __init__(
        self,
        text_dim: int,
        hidden_dim: int,
        visual_dim: int = None,
        dropout: float = 0.0,
        context_exchange: bool = False,
        num_heads: int = 8,
        context_dropout: float = 0.0,
    ):
        super().__init__()
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_exchange = bool(context_exchange)
        self.norm = nn.LayerNorm(self.text_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        visual_dim = self.hidden_dim if visual_dim is None else int(visual_dim)
        self.visual_proj = nn.Identity() if visual_dim == self.hidden_dim else nn.Linear(visual_dim, self.hidden_dim)
        self.ctx_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=int(num_heads),
            dropout=float(context_dropout),
            batch_first=True,
        )
        self.ctx_norm = nn.LayerNorm(self.hidden_dim)
        self.ctx_dropout = nn.Dropout(float(context_dropout))

    def forward(self, text_features: torch.Tensor, visual_features: torch.Tensor = None) -> torch.Tensor:
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(0)
        x = self.mlp(self.norm(text_features))

        if self.context_exchange and visual_features is not None:
            v = self.visual_proj(visual_features)
            ctx, _ = self.ctx_attn(query=x, key=v, value=v)
            x = self.ctx_norm(x + self.ctx_dropout(ctx))

        return x
