import torch
import matplotlib.pyplot as plt
import sys
import os
import yaml
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.segal_vlm import SegAL_VLM
from dataset.loveda import LoveDADataset
from xai.mem import MEMGenerator
from visualization.visualize_prediction import visualize_prediction

from torchvision import transforms

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='experiments/checkpoints/best_miou.pth', help='Path to checkpoint')
    parser.add_argument('--index', type=int, default=0, help='Index of image to visualize')
    args = parser.parse_args()

    # Load model
    with open('configs/model.yaml') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegAL_VLM(config['model']).to(device)
    
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print(f"Checkpoint not found at {args.checkpoint}, using random weights.")

    model.eval()
    
    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = LoveDADataset(
        root_dir='data/LoveDA', 
        split='val', 
        prompt_file='dataset/prompts.json', 
        transform=transform,
        img_size=(512, 512)
    )
    
    # Get a sample
    sample = dataset[args.index]
    image = sample['image'].unsqueeze(0).to(device)
    
    mask = sample['mask'].numpy() # Convert back for visualization
    prompts = sample['text_prompts'] # List of strings
    img_name = sample['image_name']

    print(f"Visualizing {img_name}...")

    # Inference
    with torch.no_grad():
        outputs = model(image, prompts)
        logits = outputs['logits']
        attn_weights = outputs['attn_weights']
        
        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
        
        # MEM
        mem_gen = MEMGenerator()
        mem_map = mem_gen.generate(logits, attn_weights, (512, 512))[0].cpu().numpy()
        
    # Denormalize image for visualization
    img_viz = denormalize(sample['image'].cpu())

    # Visualize
    save_path = f'visualization_result_{args.index}.png'
    visualize_prediction(img_viz, mask, pred, mem_map, save_path)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    main()
