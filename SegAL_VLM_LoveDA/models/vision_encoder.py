import torch
import torch.nn as nn
import timm

class VisionEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, freeze=False, drop_rate=0.0, drop_path_rate=0.0, unfreeze_last_n_blocks: int = 0):
        super().__init__()
        self.model_name = str(model_name)
        try:
            self.model = timm.create_model(
                self.model_name,
                pretrained=pretrained,
                features_only=True,
                dynamic_img_size=True,
                drop_rate=float(drop_rate),
                drop_path_rate=float(drop_path_rate)
            )
        except Exception:
            fallback = self.model_name.split(".", 1)[0] if "." in self.model_name else self.model_name
            self.model = timm.create_model(
                fallback,
                pretrained=pretrained,
                features_only=True,
                dynamic_img_size=True,
                drop_rate=float(drop_rate),
                drop_path_rate=float(drop_path_rate)
            )
        self.feature_info = self.model.feature_info
        self.out_dim = int(self.feature_info.channels()[-1]) if hasattr(self.feature_info, "channels") else None
        
        self.set_trainable(freeze=bool(freeze), unfreeze_last_n_blocks=int(unfreeze_last_n_blocks))

    def set_trainable(self, freeze: bool, unfreeze_last_n_blocks: int = 0):
        for param in self.model.parameters():
            param.requires_grad = not freeze

        if freeze and unfreeze_last_n_blocks > 0:
            backbone = getattr(self.model, "model", None)
            if backbone is None:
                backbone = getattr(self.model, "backbone", None)
            if backbone is None:
                backbone = self.model

            blocks = getattr(backbone, "blocks", None)
            if blocks is None:
                blocks = getattr(getattr(backbone, "model", None), "blocks", None)
            if blocks is None:
                return

            last_n = max(0, min(int(unfreeze_last_n_blocks), len(blocks)))
            if last_n == 0:
                return

            for b in blocks[-last_n:]:
                for p in b.parameters():
                    p.requires_grad = True
                
    def forward(self, x):
        # Returns list of features from different stages
        features = self.model(x)
        return features
