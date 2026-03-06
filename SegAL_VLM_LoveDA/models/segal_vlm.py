import torch
import torch.nn as nn
import torch.nn.functional as F
from .vision_encoder import VisionEncoder
from .text_encoder import TextEncoder
from .prompt_encoder import PromptEncoder
from .cross_attention import CrossModalAttention
from .decoder import build_decoder

class SegAL_VLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Vision Encoder
        unfreeze_last_n_blocks = int(config.get('vision_encoder', {}).get('unfreeze_last_n_blocks', 0))
        self.vision_encoder = VisionEncoder(
            model_name=config['vision_encoder']['type'],
            pretrained=config['vision_encoder']['pretrained'],
            freeze=config['vision_encoder']['freeze_backbone'],
            drop_rate=float(config.get('vision_encoder', {}).get('drop_rate', 0.0)),
            drop_path_rate=float(config.get('vision_encoder', {}).get('drop_path_rate', 0.0)),
            unfreeze_last_n_blocks=unfreeze_last_n_blocks
        )
        
        # 2. Text Encoder
        self.text_encoder = TextEncoder(
            model_name=config['text_encoder']['model_name'],
            freeze=config['text_encoder']['freeze_text_encoder']
        )
        
        # 3. Prompt Encoder + Cross-Modal Attention
        visual_dim = int(self.vision_encoder.out_dim or 768)
        hidden_dim = config['decoder']['hidden_dim']
        text_dim = int(getattr(self.text_encoder, "embed_dim", 512))

        self.prompt_encoder = PromptEncoder(
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            visual_dim=visual_dim,
            dropout=float((config.get('prompt_encoder', {}) or {}).get('dropout', 0.0)),
            context_exchange=bool((config.get('prompt_encoder', {}) or {}).get('context_exchange', False)),
            num_heads=int((config.get('prompt_encoder', {}) or {}).get('num_heads', config['cross_attention']['num_heads'])),
            context_dropout=float((config.get('prompt_encoder', {}) or {}).get('context_dropout', config['cross_attention']['dropout']))
        )
        
        self.cross_attention = CrossModalAttention(
            visual_dim=visual_dim,
            text_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=config['cross_attention']['num_heads'],
            dropout=config['cross_attention']['dropout']
        )
        
        # 4. Decoder
        decoder_cfg = config.get('decoder', {}) or {}
        skip_channels = None
        if hasattr(self.vision_encoder, "feature_info") and hasattr(self.vision_encoder.feature_info, "channels"):
            chans = self.vision_encoder.feature_info.channels()
            if isinstance(chans, (list, tuple)) and len(chans) > 1:
                skip_channels = list(chans[:-1])
        self.decoder = build_decoder(
            decoder_cfg=decoder_cfg,
            hidden_dim=hidden_dim,
            num_classes=int(decoder_cfg.get('num_classes', 7)),
            skip_channels=skip_channels
        )
        
    def forward(self, images, text_prompts, guidance_prompts=None):
        B, C, H, W = images.shape
        
        # --- Vision ---
        # features is a list. Take the last one for simplicity.
        visual_features_list = self.vision_encoder(images)
        visual_feat = visual_features_list[-1] # (B, C_v, H_f, W_f)
        B, C_v, H_f, W_f = visual_feat.shape
        
        # Flatten for attention: (B, H_f*W_f, C_v)
        visual_feat_flat = visual_feat.flatten(2).transpose(1, 2)
        
        # --- Text ---
        # text_prompts is a list of strings [class1, class2, ...]
        all_prompts = list(text_prompts)
        if guidance_prompts:
            all_prompts = all_prompts + list(guidance_prompts)

        text_feat = self.text_encoder(all_prompts, images.device)  # (K, text_dim)
        
        # Expand text feat to batch: (B, num_classes, text_dim)
        text_feat_batch = text_feat.unsqueeze(0).expand(B, -1, -1)
        prompt_tokens = self.prompt_encoder(text_feat_batch, visual_feat_flat)  # (B, K, hidden_dim)
        
        # --- Cross Attention ---
        # fused_feat: (B, L, hidden_dim)
        # attn_weights: (B, num_heads, L, num_classes) or similar depending on implementation
        fused_feat, attn_weights = self.cross_attention(visual_feat_flat, prompt_tokens)
        
        # Reshape back to spatial
        fused_feat_spatial = fused_feat.transpose(1, 2).view(B, -1, H_f, W_f)
        attn_logits = None
        if attn_weights is not None:
            attn = attn_weights
            if attn.dim() == 4:
                attn = attn.mean(dim=1)
            if attn.dim() == 3 and attn.shape[1] == int(H_f * W_f):
                attn_spatial = attn.view(B, int(H_f), int(W_f), -1).permute(0, 3, 1, 2).contiguous()
                attn_up = F.interpolate(attn_spatial, size=(int(H), int(W)), mode="bilinear", align_corners=False)
                attn_logits = torch.log(attn_up.clamp_min(1e-8))
        
        # --- Decoder ---
        skips = visual_features_list[:-1] if isinstance(visual_features_list, (list, tuple)) else None
        try:
            logits = self.decoder(fused_feat_spatial, (H, W), skips=skips)
        except TypeError:
            logits = self.decoder(fused_feat_spatial, (H, W))
        
        return {
            "logits": logits,
            "attn_weights": attn_weights,
            "attn_logits": attn_logits,
            "visual_features": visual_feat,
            "feature_hw": (int(H_f), int(W_f)),
            "original_size": (int(H), int(W))
        }
