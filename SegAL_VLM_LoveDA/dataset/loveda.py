import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageOps
import numpy as np
import json
import glob
import random
from torchvision import transforms as T

class LoveDADataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, prompt_file=None, img_size=None, require_mask=None, augment=None):
        """
        Args:
            root_dir (str): Path to the LoveDA dataset directory.
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample (image only).
            prompt_file (str, optional): Path to prompts.json.
            img_size (tuple, optional): (height, width) to resize both image and mask.
        """
        self.root_dir = root_dir
        self.split = split.lower()
        self.transform = transform
        self.img_size = img_size
        self.require_mask = require_mask if require_mask is not None else self.split in ('train', 'val')
        self.augment = augment if augment is not None else self.split in ('train', 'labeled')
        self.logged_success = False
        
        # Map split to folder name
        split_map = {
            'train': 'Train',
            'val': 'Val',
            'test': 'Test'
        }
        split_folder = split_map.get(self.split, self.split.capitalize())
        
        self.image_paths = []
        self.mask_paths = []
        
        # Sub-domains in LoveDA
        scenes = ['Urban', 'Rural']
        
        # Check if root dir exists
        if not os.path.exists(root_dir):
            print(f"Warning: {root_dir} not found. Using dummy data.")
            self.images = [f"{i}.png" for i in range(10)]
            self.dummy_mode = True
        else:
            # Collect files
            for scene in scenes:
                img_dir = os.path.join(root_dir, split_folder, scene, 'images_png')
                mask_dir = os.path.join(root_dir, split_folder, scene, 'masks_png')
                
                if os.path.exists(img_dir):
                    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
                    self.image_paths.extend(imgs)
                    
                    if self.split != 'test' and self.require_mask:
                        if os.path.exists(mask_dir):
                            masks = [os.path.join(mask_dir, os.path.basename(img)) for img in imgs]
                        else:
                            masks = [None for _ in imgs]
                        self.mask_paths.extend(masks)
            
            if len(self.image_paths) == 0:
                 print(f"Warning: No images found in {root_dir}/{split_folder}. Using dummy data.")
                 self.images = [f"{i}.png" for i in range(10)]
                 self.dummy_mode = True
        
        if prompt_file:
            with open(prompt_file, 'r') as f:
                self.prompts = json.load(f)
        else:
            self.prompts = {}
            
        self.num_classes = 7
        self.ignore_index = 255

    def rgb_to_mask(self, rgb_mask):
        """
        Convert RGB mask to class index mask.
        Handles both RGB color-coded masks and grayscale index masks (standard LoveDA).
        """
        # rgb_mask: numpy array (H, W, 3)
        # Returns: mask (H, W) with class indices
        
        h, w, c = rgb_mask.shape
        mask = np.full((h, w), self.ignore_index, dtype=np.int64)
        
        # Check if mask is grayscale (R=G=B) indicating index encoding
        if np.all(rgb_mask[:, :, 0] == rgb_mask[:, :, 1]) and np.all(rgb_mask[:, :, 0] == rgb_mask[:, :, 2]):
            indices = rgb_mask[:, :, 0].astype(np.int64, copy=False)
            if int(indices.max()) <= 7:
                valid_0_6 = (indices >= 0) & (indices <= 6)
                valid_1_7 = (indices >= 1) & (indices <= 7)
                mask[valid_0_6] = indices[valid_0_6]
                mask[(~valid_0_6) & valid_1_7] = indices[(~valid_0_6) & valid_1_7] - 1
                return mask

        colors = [
            ([255, 255, 255], 0),
            ([0, 0, 0], 0),
            ([255, 0, 0], 1),
            ([255, 255, 0], 2),
            ([0, 0, 255], 3),
            ([159, 129, 183], 4),
            ([0, 255, 0], 5),
            ([255, 195, 128], 6)
        ]
        
        for color, idx in colors:
            # Vectorized matching
            matches = np.all(rgb_mask == np.array(color), axis=-1)
            mask[matches] = idx
            
        return mask

    def __len__(self):
        if hasattr(self, 'dummy_mode') and self.dummy_mode:
            return len(self.images)
        return len(self.image_paths)

    def __getitem__(self, idx):
        if hasattr(self, 'dummy_mode') and self.dummy_mode:
            # Generate dummy data
            size = self.img_size if self.img_size else (512, 512)
            image = Image.new('RGB', size, color=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
            mask = Image.new('L', size, color=np.random.randint(0, 7))
            img_name = self.images[idx]
            domain = 0
            domain_name = "Urban"
            
            if self.transform:
                image = self.transform(image)
            mask = torch.from_numpy(np.array(mask)).long()
            
        else:
            img_path = self.image_paths[idx]
            img_name = os.path.basename(img_path)
            
            image = Image.open(img_path).convert('RGB')
            
            mask_path = None
            if self.split != 'test' and self.require_mask:
                mask_path = self.mask_paths[idx]
                mask_missing = (not mask_path) or (not os.path.exists(mask_path))
                if mask_missing:
                    if self.img_size:
                        image = image.resize(self.img_size[::-1], Image.BILINEAR)
                    h, w = self.img_size if self.img_size else (image.size[1], image.size[0])
                    mask = torch.full((h, w), self.ignore_index, dtype=torch.long)
                    mask_path = ""
                else:
                    mask_pil = Image.open(mask_path).convert('RGB')
                
                    if self.img_size:
                        image = image.resize(self.img_size[::-1], Image.BILINEAR)
                        mask_pil = mask_pil.resize(self.img_size[::-1], Image.NEAREST)
                    
                    if self.augment:
                        if random.random() < 0.5:
                            image = ImageOps.mirror(image)
                            mask_pil = ImageOps.mirror(mask_pil)
                        if random.random() < 0.5:
                            image = ImageOps.flip(image)
                            mask_pil = ImageOps.flip(mask_pil)
                        rot_k = random.randint(0, 3)
                        if rot_k:
                            angle = 90 * rot_k
                            image = image.rotate(angle, resample=Image.BILINEAR)
                            mask_pil = mask_pil.rotate(angle, resample=Image.NEAREST)
                        if random.random() < 0.8:
                            jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
                            image = jitter(image)
                        if random.random() < 0.1:
                            image = ImageOps.autocontrast(image)
                    
                    mask_np = np.array(mask_pil)
                    mask_indices = self.rgb_to_mask(mask_np)
                    mask = torch.from_numpy(mask_indices).long()
                    
                    if not self.logged_success:
                        print(f"Mask conversion success for image: {img_name}")
                        self.logged_success = True
                
            else:
                if self.img_size:
                    image = image.resize(self.img_size[::-1], Image.BILINEAR)
                h, w = self.img_size if self.img_size else (image.size[1], image.size[0])
                mask = torch.full((h, w), self.ignore_index, dtype=torch.long)
                mask_path = ""
            
            img_path_lower = img_path.lower()
            domain = 0 if "urban" in img_path_lower else 1
            domain_name = "Urban" if domain == 0 else "Rural"
        
            if self.transform:
                image = self.transform(image)
            
        text_prompts = None
        if isinstance(self.prompts, dict) and len(self.prompts) > 0:
            ordered = []
            for i in range(6):
                v = self.prompts.get(str(i))
                if isinstance(v, str) and v.strip():
                    ordered.append(v.strip())
            if len(ordered) == 6:
                text_prompts = ["satellite aerial view background and other land cover"] + ordered

        if text_prompts is None:
            text_prompts = [
                "satellite aerial view background and other land cover",
                "buildings",
                "roads",
                "water bodies",
                "barren land",
                "forest",
                "agricultural fields"
            ]

        return {
            "image": image,
            "mask": mask,
            "domain": domain,
            "domain_name": domain_name,
            "image_name": img_name,
            "image_path": img_path if not (hasattr(self, 'dummy_mode') and self.dummy_mode) else "",
            "mask_path": mask_path or "",
            "text_prompts": text_prompts
        }
