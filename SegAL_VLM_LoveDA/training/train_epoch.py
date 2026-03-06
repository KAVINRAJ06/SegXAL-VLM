import torch
import torch.nn.functional as F
from tqdm import tqdm
from training.metrics import Evaluator

def train_one_epoch(model, loader, optimizer, criterion, device, num_classes=7, grad_accum_steps=1, attn_supervision_weight=0.0, attn_logits_weight=0.0):
    model.train()
    total_loss = 0
    evaluator = Evaluator(num_classes, device)
    grad_accum_steps = max(1, int(grad_accum_steps))
    attn_supervision_weight = float(attn_supervision_weight) if attn_supervision_weight is not None else 0.0
    attn_logits_weight = float(attn_logits_weight) if attn_logits_weight is not None else 0.0
    prompt_focus_prob = 0.0
    focus_classes = None
    if hasattr(model, "config") and isinstance(getattr(model, "config", None), dict):
        prompt_cfg = model.config.get("prompt_training", {}) or {}
        prompt_focus_prob = float(prompt_cfg.get("focus_prob", 0.0) or 0.0)
        focus_classes = prompt_cfg.get("focus_classes", None)
        if isinstance(focus_classes, (list, tuple)) and len(focus_classes) > 0:
            focus_classes = [int(x) for x in focus_classes]
        else:
            focus_classes = None
    
    pbar = tqdm(loader, desc="Training")
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        prompts_list = [p[0] for p in batch['text_prompts']]
        
        if (masks != 255).sum() == 0:
            continue
            
        focus_class = None
        focus_mode = prompt_focus_prob > 0 and (torch.rand(()) < prompt_focus_prob)
        if focus_mode:
            if focus_classes is None:
                focus_class = int(torch.randint(low=1, high=int(num_classes), size=(1,)).item())
            else:
                idx = int(torch.randint(low=0, high=len(focus_classes), size=(1,)).item())
                focus_class = int(focus_classes[idx])
            if 0 <= focus_class < len(prompts_list):
                prompts_list = [prompts_list[0], prompts_list[focus_class]]

        outputs = model(images, prompts_list)
        logits = outputs['logits']
        
        train_masks = masks
        if focus_mode and focus_class is not None:
            keep = (masks == 0) | (masks == focus_class) | (masks == 255)
            train_masks = torch.where(keep, masks, torch.full_like(masks, 255))
        loss = criterion(logits, train_masks)
        if attn_logits_weight > 0:
            attn_logits = outputs.get("attn_logits", None)
            if attn_logits is not None and torch.is_tensor(attn_logits) and attn_logits.dim() == 4 and attn_logits.shape[1] == int(num_classes):
                loss = loss + (attn_logits_weight * criterion(attn_logits, train_masks))
        if attn_supervision_weight > 0:
            attn_weights = outputs.get("attn_weights", None)
            feature_hw = outputs.get("feature_hw", None)
            if attn_weights is not None and feature_hw is not None:
                h_f, w_f = int(feature_hw[0]), int(feature_hw[1])
                if h_f > 0 and w_f > 0:
                    masks_ds = F.interpolate(
                        train_masks.unsqueeze(1).float(),
                        size=(h_f, w_f),
                        mode="nearest"
                    ).squeeze(1).long()
                    attn = attn_weights.reshape(-1, attn_weights.shape[-1])
                    tgt = masks_ds.reshape(-1)
                    valid = tgt != 255
                    if valid.any():
                        tgt_v = tgt[valid].clamp(0, attn.shape[-1] - 1)
                        p = attn[valid].gather(1, tgt_v.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
                        attn_loss = (-torch.log(p)).mean()
                        loss = loss + (attn_supervision_weight * attn_loss)
        (loss / grad_accum_steps).backward()
        
        micro_step += 1
        if micro_step % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
        evaluator.update(logits.detach(), masks)
    
    if micro_step % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
    scores = evaluator.get_scores()
    avg_loss = total_loss / len(loader)
    
    return avg_loss, scores['miou'], scores['pixel_acc']
