import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class TextEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch16", freeze=True):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.embed_dim = int(getattr(getattr(self.model, "config", None), "projection_dim", 512))
        self._cache_key = None
        self._cache_value = None
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
                
    def forward(self, text_prompts, device):
        """
        Args:
            text_prompts: List of strings.
            device: torch device.
        Returns:
            text_features: Tensor of shape (num_prompts, embed_dim)
        """
        key = (tuple(text_prompts), str(device))
        if self._cache_key == key and self._cache_value is not None:
            return self._cache_value

        inputs = self.processor(text=list(text_prompts), return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask", "position_ids")}
            text_features = self.model.get_text_features(**text_inputs)
            if not isinstance(text_features, torch.Tensor):
                if hasattr(text_features, "pooler_output") and text_features.pooler_output is not None:
                    pooled = text_features.pooler_output
                elif hasattr(text_features, "last_hidden_state") and text_features.last_hidden_state is not None:
                    pooled = text_features.last_hidden_state[:, 0, :]
                else:
                    raise TypeError(f"Unexpected text feature output type: {type(text_features)}")

                if hasattr(self.model, "text_projection") and self.model.text_projection is not None:
                    pooled = self.model.text_projection(pooled)
                text_features = pooled

            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        text_features = text_features.detach()
        self._cache_key = key
        self._cache_value = text_features
        return text_features
