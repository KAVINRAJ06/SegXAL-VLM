import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SegmentationDecoder(nn.Module):
    def __init__(self, hidden_dim, num_classes, dropout=0.0):
        super().__init__()
        
        # Simple decoder: 
        # 1. Project to num_classes
        # 2. Upsample
        
        # A more complex one would involve upsampling blocks.
        # For simplicity in this research framework, we assume the feature map is reasonably sized
        # or we use a few ConvTranspose layers.
        
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim // 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity()
        
        self.conv2 = nn.Conv2d(hidden_dim // 2, num_classes, kernel_size=1)
        
    def forward(self, x, original_size):
        """
        Args:
            x: (B, hidden_dim, H, W) - feature map
            original_size: (H_orig, W_orig)
        """
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)
        
        # Upsample to original size
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        return x

class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        groups = 32 if out_ch % 32 == 0 else (16 if out_ch % 16 == 0 else (8 if out_ch % 8 == 0 else 1))
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class _UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.block = _ConvBlock(out_ch, out_ch, dropout=dropout)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.proj(x)
        return self.block(x)

class StrongUNetDecoder(nn.Module):
    def __init__(self, hidden_dim, num_classes, width=256, depth=4, dropout=0.0):
        super().__init__()
        width = int(width)
        depth = int(depth)
        width = max(32, width)
        depth = max(1, depth)
        self.depth = depth
        self.num_classes = int(num_classes)

        self.stem = nn.Conv2d(int(hidden_dim), width, kernel_size=1, bias=False)

        chs = [width]
        for _ in range(depth):
            chs.append(max(32, chs[-1] // 2))
        self.up = nn.ModuleList([_UpBlock(chs[i], chs[i + 1], dropout=dropout) for i in range(depth)])
        self.head = nn.Conv2d(chs[min(depth, len(chs) - 1)], self.num_classes, kernel_size=1)

    def forward(self, x, original_size):
        target_h, target_w = int(original_size[0]), int(original_size[1])
        h, w = int(x.shape[-2]), int(x.shape[-1])
        ratio = max(target_h / max(1, h), target_w / max(1, w))
        if ratio <= 1.0:
            steps = 0
        else:
            steps = int(min(self.depth, max(0, math.ceil(math.log2(ratio)))))

        x = self.stem(x)
        for i in range(steps):
            x = self.up[i](x)
        x = self.head(x)
        x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return x

class StrongUNetDecoderWithSkips(nn.Module):
    def __init__(self, hidden_dim, num_classes, width=256, depth=4, dropout=0.0, skip_channels=None):
        super().__init__()
        width = int(width)
        depth = int(depth)
        width = max(32, width)
        depth = max(1, depth)
        self.depth = depth
        self.num_classes = int(num_classes)

        self.stem = nn.Conv2d(int(hidden_dim), width, kernel_size=1, bias=False)

        chs = [width]
        for _ in range(depth):
            chs.append(max(32, chs[-1] // 2))
        self.up = nn.ModuleList([_UpBlock(chs[i], chs[i + 1], dropout=dropout) for i in range(depth)])
        self.head = nn.Conv2d(chs[min(depth, len(chs) - 1)], self.num_classes, kernel_size=1)

        self.skip_projs = nn.ModuleList()
        skip_channels = list(skip_channels) if isinstance(skip_channels, (list, tuple)) else []
        skip_channels = list(reversed(skip_channels))[:depth]
        for i in range(depth):
            out_ch = chs[i + 1]
            if i < len(skip_channels) and skip_channels[i]:
                self.skip_projs.append(nn.Conv2d(int(skip_channels[i]), int(out_ch), kernel_size=1, bias=False))
            else:
                self.skip_projs.append(nn.Identity())

        self.skip_fuse = nn.ModuleList([_ConvBlock(chs[i + 1], chs[i + 1], dropout=dropout) for i in range(depth)])

    def forward(self, x, original_size, skips=None):
        target_h, target_w = int(original_size[0]), int(original_size[1])
        h, w = int(x.shape[-2]), int(x.shape[-1])
        ratio = max(target_h / max(1, h), target_w / max(1, w))
        if ratio <= 1.0:
            steps = 0
        else:
            steps = int(min(self.depth, max(0, math.ceil(math.log2(ratio)))))

        x = self.stem(x)

        skips = list(skips) if isinstance(skips, (list, tuple)) else []
        skips = list(reversed(skips))[:steps]

        for i in range(steps):
            x = self.up[i](x)
            if i < len(skips) and skips[i] is not None:
                s = skips[i]
                if (int(s.shape[-2]) != int(x.shape[-2])) or (int(s.shape[-1]) != int(x.shape[-1])):
                    s = F.interpolate(s, size=(int(x.shape[-2]), int(x.shape[-1])), mode='bilinear', align_corners=False)
                proj = self.skip_projs[i]
                s = proj(s) if not isinstance(proj, nn.Identity) else s
                x = x + s
                x = self.skip_fuse[i](x)

        x = self.head(x)
        x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return x

class _ASPPConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, dilation):
        out_ch = int(out_ch)
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(num_groups=(32 if out_ch % 32 == 0 else (16 if out_ch % 16 == 0 else (8 if out_ch % 8 == 0 else 1))), num_channels=out_ch),
            nn.ReLU(inplace=True),
        )

class _ASPPPooling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        out_ch = int(out_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=(32 if out_ch % 32 == 0 else (16 if out_ch % 16 == 0 else (8 if out_ch % 8 == 0 else 1))), num_channels=out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.pool(x)
        x = self.proj(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, atrous_rates=(6, 12, 18), dropout=0.0):
        super().__init__()
        out_ch = int(out_ch)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=(32 if out_ch % 32 == 0 else (16 if out_ch % 16 == 0 else (8 if out_ch % 8 == 0 else 1))), num_channels=out_ch),
            nn.ReLU(inplace=True),
        )
        self.branches = nn.ModuleList([_ASPPConv(in_ch, out_ch, d) for d in atrous_rates])
        self.pool = _ASPPPooling(in_ch, out_ch)
        proj_in = out_ch * (2 + len(atrous_rates))
        self.project = nn.Sequential(
            nn.Conv2d(proj_in, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=(32 if out_ch % 32 == 0 else (16 if out_ch % 16 == 0 else (8 if out_ch % 8 == 0 else 1))), num_channels=out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity(),
        )

    def forward(self, x):
        feats = [self.branch1(x)]
        feats.extend([b(x) for b in self.branches])
        feats.append(self.pool(x))
        x = torch.cat(feats, dim=1)
        return self.project(x)

class DeepLabDecoder(nn.Module):
    def __init__(self, hidden_dim, num_classes, aspp_out=256, atrous_rates=(6, 12, 18), dropout=0.1):
        super().__init__()
        self.aspp = ASPP(int(hidden_dim), int(aspp_out), atrous_rates=atrous_rates, dropout=float(dropout))
        self.head = nn.Sequential(
            nn.Conv2d(int(aspp_out), int(aspp_out), kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=(32 if int(aspp_out) % 32 == 0 else (16 if int(aspp_out) % 16 == 0 else (8 if int(aspp_out) % 8 == 0 else 1))), num_channels=int(aspp_out)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Conv2d(int(aspp_out), int(num_classes), kernel_size=1),
        )

    def forward(self, x, original_size, **kwargs):
        x = self.aspp(x)
        x = self.head(x)
        return F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)

def build_decoder(decoder_cfg, hidden_dim, num_classes, skip_channels=None):
    cfg = decoder_cfg or {}
    decoder_type = str(cfg.get("type", "segmentation_head")).lower()
    dropout = float(cfg.get("dropout", 0.0))
    use_skips = bool(cfg.get("use_skips", False))
    if decoder_type in {"deeplab", "deeplabv3", "deeplabv3plus", "deeplab_v3", "deeplab_v3_plus"}:
        aspp_out = int(cfg.get("aspp_out", 256))
        atrous_rates = cfg.get("atrous_rates", (6, 12, 18))
        if isinstance(atrous_rates, (list, tuple)) and len(atrous_rates) > 0:
            atrous_rates = tuple(int(x) for x in atrous_rates)
        else:
            atrous_rates = (6, 12, 18)
        return DeepLabDecoder(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            aspp_out=aspp_out,
            atrous_rates=atrous_rates,
            dropout=dropout
        )
    if decoder_type in {"unet", "strong_unet", "strong-unet", "unet_strong"}:
        width = int(cfg.get("unet_width", cfg.get("width", 256)))
        depth = int(cfg.get("unet_depth", cfg.get("depth", 4)))
        if use_skips:
            return StrongUNetDecoderWithSkips(
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                width=width,
                depth=depth,
                dropout=dropout,
                skip_channels=skip_channels
            )
        return StrongUNetDecoder(hidden_dim=hidden_dim, num_classes=num_classes, width=width, depth=depth, dropout=dropout)
    return SegmentationDecoder(hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
