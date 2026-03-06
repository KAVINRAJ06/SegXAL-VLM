import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_prediction(image, mask, pred, mem_map, save_path):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # Image (C, H, W) -> (H, W, C)
    if torch.is_tensor(image):
        img_t = image.detach().cpu()
        if img_t.ndim == 3 and img_t.shape[0] == 3:
            mn = float(img_t.min())
            mx = float(img_t.max())
            if mn < 0.0 or mx > 1.0:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_t = img_t * std + mean
            img_t = img_t.clamp(0.0, 1.0)
        img = img_t.permute(1, 2, 0).numpy()
    else:
        img = np.array(image)
        
    axs[0].imshow(img)
    axs[0].set_title("Input")
    
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    axs[1].imshow(mask, cmap='tab10', vmin=0, vmax=6)
    axs[1].set_title("Ground Truth")
    
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    axs[2].imshow(pred, cmap='tab10', vmin=0, vmax=6)
    axs[2].set_title("Prediction")
    
    if torch.is_tensor(mem_map):
        mem_map = mem_map.detach().cpu().numpy()
    axs[3].imshow(mem_map, cmap='jet')
    axs[3].set_title("MEM (Error Mask)")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
