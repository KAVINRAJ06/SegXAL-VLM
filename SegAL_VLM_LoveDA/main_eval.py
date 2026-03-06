import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.segal_vlm import SegAL_VLM
from dataset.loveda import LoveDADataset
from training.validate import validate
from training.losses import SegLoss
from training.metrics import Evaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (.pth). Defaults to best_miou.pth if present.')
    parser.add_argument('--config', type=str, default='configs/model.yaml')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--output_dir', type=str, default='experiments/test_predictions')
    args = parser.parse_args()

    if args.checkpoint is None:
        candidates = [
            os.path.join('experiments', 'checkpoints', 'best_miou.pth'),
            os.path.join('experiments', 'checkpoints', 'last_epoch.pth'),
        ]
        args.checkpoint = next((p for p in candidates if os.path.exists(p)), None)
        if args.checkpoint is None:
            ckpt_dir = os.path.join('experiments', 'checkpoints')
            if os.path.isdir(ckpt_dir):
                files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pth')])
                if files:
                    raise SystemExit(f"No default checkpoint found. Available: {', '.join(files)}")
            raise SystemExit("No checkpoint provided and none found in experiments/checkpoints.")
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = SegAL_VLM(config['model']).to(device)
    state = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state)
    
    # Setup Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LoveDADataset(
        root_dir='data/LoveDA', 
        split=args.split, 
        prompt_file='dataset/prompts.json',
        transform=transform,
        img_size=(512, 512)
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    criterion = SegLoss(num_classes=7, ignore_index=255).to(device)
    
    if args.split.lower() == 'test':
        import numpy as np
        from PIL import Image
        os.makedirs(args.output_dir, exist_ok=True)
        model.eval()
        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(device)
                prompts_list = [p[0] for p in batch['text_prompts']]
                outputs = model(images, prompts_list)
                logits = outputs['logits']
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                domains = batch['domain_name']
                names = batch['image_name']
                for i in range(preds.shape[0]):
                    dom_dir = os.path.join(args.output_dir, domains[i])
                    os.makedirs(dom_dir, exist_ok=True)
                    save_path = os.path.join(dom_dir, names[i])
                    Image.fromarray(preds[i].astype(np.uint8)).save(save_path)
        print(f"Saved predictions to {args.output_dir}")
        return
    
    avg_loss, miou, class_iou, pixel_acc, domain_stats = validate(model, loader, criterion, device, num_classes=7)

    print("\n" + "="*40)
    print("       SegAL-VLM Evaluation Report")
    print("="*40)
    print(f"Overall mIoU      : {miou:.4f}")
    print(f"Pixel Accuracy    : {pixel_acc:.4f}")
    print(f"Urban mIoU        : {domain_stats['urban']['miou']:.4f}")
    print(f"Urban Pixel Acc   : {domain_stats['urban']['pixel_acc']:.4f}")
    print(f"Rural mIoU        : {domain_stats['rural']['miou']:.4f}")
    print(f"Rural Pixel Acc   : {domain_stats['rural']['pixel_acc']:.4f}")
    print(f"Validation Loss   : {avg_loss:.4f}")
    print("-" * 40)
    print("Class-wise IoU:")
    
    if len(class_iou) == 7:
        classes = ['Background','Building','Road','Water','Barren','Forest','Agriculture']
    else:
        classes = [f'Class {i}' for i in range(len(class_iou))]
        
    df = pd.DataFrame({'Class': classes, 'IoU': class_iou})
    print(df.to_string(index=False))
    print("="*40)

if __name__ == "__main__":
    main()
