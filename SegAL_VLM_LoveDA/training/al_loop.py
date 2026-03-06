import os
import shutil
import math
import torch
import numpy as np
import random
import json
import glob
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
from PIL import Image

from dataset.loveda import LoveDADataset
from dataset.splits import create_initial_splits, get_subsets
from models.segal_vlm import SegAL_VLM
from training.train_epoch import train_one_epoch
from training.validate import validate
from training.losses import SegLoss
from xai.mem import MEMGenerator
from active_learning.oracle import Oracle
from active_learning.sampler import ALSampler
from visualization.plot_metrics import plot_metrics
from visualization.visualize_prediction import visualize_prediction

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ActiveLearningLoop:
    def __init__(self, config_dir='configs'):
        # Load configs
        with open(os.path.join(config_dir, 'model.yaml')) as f:
            self.model_config = yaml.safe_load(f)
        with open(os.path.join(config_dir, 'training.yaml')) as f:
            self.train_config = yaml.safe_load(f)
        with open(os.path.join(config_dir, 'active_learning.yaml')) as f:
            self.al_config = yaml.safe_load(f)
            
        requested_device = self.train_config['training']['device']
        if requested_device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available. Falling back to CPU.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(requested_device)

        training_cfg = self.train_config.get('training', {}) or {}
        self.augment = bool(training_cfg.get('augment', False))
        self.attn_supervision_weight = float(training_cfg.get('attn_supervision_weight', 0.0) or 0.0)
        self.attn_logits_weight = float(training_cfg.get('attn_logits_weight', 0.0) or 0.0)
        
        # Setup Transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Setup Dataset
        self.full_dataset = LoveDADataset(
            root_dir='data/LoveDA', 
            split='train', 
            prompt_file='dataset/prompts.json',
            transform=self.transform,
            img_size=(512, 512),
            augment=self.augment
        )
        self.val_dataset = LoveDADataset(
            root_dir='data/LoveDA', 
            split='val', 
            prompt_file='dataset/prompts.json',
            transform=self.transform,
            img_size=(512, 512)
        )

        data_cfg = (self.al_config.get('active_learning', {}) or {}).get('data', {}) or {}
        self.total_train_size = len(self.full_dataset)
        self.use_filesystem_pool = bool(data_cfg.get('use_filesystem_pool', False))
        self.pool_root = str(data_cfg.get('pool_root', os.path.join('data', 'LoveDA', 'Pool')))
        self.reset_pool = bool(data_cfg.get('reset_pool', False))
        
        # Splits
        self.labeled_indices, self.unlabeled_indices = create_initial_splits(
            self.full_dataset, 
            initial_ratio=self.al_config['active_learning']['data']['initial_labeled_ratio']
        )

        self.pool_labeled_dataset = None
        self.pool_unlabeled_infer_dataset = None
        self.pool_unlabeled_train_dataset = None
        if self.use_filesystem_pool:
            self._init_filesystem_pool()
            self._reload_pool_datasets()
        
        # Model
        self.model = SegAL_VLM(self.model_config['model']).to(self.device)
        
        # Loss
        label_smoothing = float(self.train_config.get('training', {}).get('label_smoothing', 0.0))
        focal_gamma = float(self.train_config.get('training', {}).get('focal_gamma', 0.0))
        dice_weight = float(self.train_config.get('training', {}).get('dice_weight', 1.0))
        class_weights = self.train_config.get('training', {}).get('class_weights', None)
        if isinstance(class_weights, (list, tuple)):
            class_weights = [float(x) for x in class_weights]
        else:
            class_weights = None
        self.criterion = SegLoss(
            num_classes=7,
            ignore_index=255,
            label_smoothing=label_smoothing,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
            dice_weight=dice_weight
        ).to(self.device)
        
        # Components
        strategy_cfg = (self.al_config.get('active_learning', {}) or {}).get('strategy', {}) or {}
        alpha = float(strategy_cfg.get('alpha', 0.5))
        beta = strategy_cfg.get('beta', None)
        adaptive = bool(strategy_cfg.get('adaptive', False))
        smooth_kernel = int(strategy_cfg.get('smooth_kernel', 0))
        smooth_sigma = float(strategy_cfg.get('smooth_sigma', 1.5))
        self.mem_gen = MEMGenerator(alpha=alpha, beta=beta, adaptive=adaptive, smooth_kernel=smooth_kernel, smooth_sigma=smooth_sigma)
        self.sampler = ALSampler(budget_per_round=self.al_config['active_learning']['budget_per_round'])

        oracle_cfg = (self.al_config.get('active_learning', {}) or {}).get('oracle', {}) or {}
        oracle_type = str(oracle_cfg.get('type', 'machine')).lower()
        oracle_acc = float(oracle_cfg.get('accuracy', 1.0))
        oracle_error = max(0.0, min(1.0, 1.0 - oracle_acc))
        self.oracle = Oracle(mode=oracle_type, error_rate=oracle_error, ignore_index=255)

        self._gt_mask_by_scene_name = {}
        if hasattr(self.full_dataset, "image_paths") and hasattr(self.full_dataset, "mask_paths"):
            for img_path, mask_path in zip(self.full_dataset.image_paths, self.full_dataset.mask_paths):
                if not mask_path:
                    continue
                if not os.path.exists(mask_path):
                    continue
                scene = self._pool_scene_from_path(img_path)
                self._gt_mask_by_scene_name[(scene, os.path.basename(img_path))] = mask_path
        
        # Directories
        self.exp_dir = 'experiments'
        self.ckpt_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.log_dir = os.path.join(self.exp_dir, 'logs')
        self.plot_dir = os.path.join(self.log_dir, 'plots')
        
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Metrics history (accumulated across epochs)
        self.history_path = os.path.join(self.log_dir, 'metrics_history.json')
        self.class_names = ['Background', 'Building', 'Road', 'Water', 'Barren', 'Forest', 'Agriculture']
        if os.path.exists(self.history_path):
            with open(self.history_path, 'r') as f:
                self.metrics_history_accum = json.load(f)
            if "val_miou_urban" not in self.metrics_history_accum:
                self.metrics_history_accum["val_miou_urban"] = []
            if "val_miou_rural" not in self.metrics_history_accum:
                self.metrics_history_accum["val_miou_rural"] = []
            if "val_acc_urban" not in self.metrics_history_accum:
                self.metrics_history_accum["val_acc_urban"] = []
            if "val_acc_rural" not in self.metrics_history_accum:
                self.metrics_history_accum["val_acc_rural"] = []
            if "classwise_iou" in self.metrics_history_accum:
                for name in self.class_names:
                    if name not in self.metrics_history_accum["classwise_iou"]:
                        self.metrics_history_accum["classwise_iou"][name] = []
            for k in ["labeled_count", "unlabeled_count", "labeled_fraction"]:
                if k not in self.metrics_history_accum:
                    self.metrics_history_accum[k] = []
            if "al_round" not in self.metrics_history_accum:
                self.metrics_history_accum["al_round"] = []
        else:
            self.metrics_history_accum = {
                "epochs": [],
                "train_loss": [],
                "val_loss": [],
                "train_accuracy": [],
                "val_accuracy": [],
                "train_miou": [],
                "val_miou": [],
                "val_miou_urban": [],
                "val_miou_rural": [],
                "val_acc_urban": [],
                "val_acc_rural": [],
                "classwise_iou": {name: [] for name in self.class_names},
                "labeled_count": [],
                "unlabeled_count": [],
                "labeled_fraction": [],
                "al_round": []
            }
        
        self.metrics_history = []
        self.best_miou = 0.0
        self.global_epoch = max(self.metrics_history_accum["epochs"]) if self.metrics_history_accum["epochs"] else 0

    def _reload_pool_datasets(self):
        self.pool_labeled_dataset = LoveDADataset(
            root_dir=self.pool_root,
            split='labeled',
            prompt_file='dataset/prompts.json',
            transform=self.transform,
            img_size=(512, 512),
            require_mask=True,
            augment=self.augment
        )
        self.pool_unlabeled_infer_dataset = LoveDADataset(
            root_dir=self.pool_root,
            split='unlabeled',
            prompt_file='dataset/prompts.json',
            transform=self.transform,
            img_size=(512, 512),
            require_mask=False,
            augment=False
        )
        self.pool_unlabeled_train_dataset = LoveDADataset(
            root_dir=self.pool_root,
            split='unlabeled',
            prompt_file='dataset/prompts.json',
            transform=self.transform,
            img_size=(512, 512),
            require_mask=True,
            augment=self.augment
        )
        self._pool_unlabeled_path_to_index = {p: i for i, p in enumerate(getattr(self.pool_unlabeled_infer_dataset, "image_paths", []))}

    def _lookup_gt_mask_path(self, pool_image_path):
        scene = self._pool_scene_from_path(pool_image_path)
        name = os.path.basename(pool_image_path)
        return self._gt_mask_by_scene_name.get((scene, name), None)

    def _batch_prompts(self, batch):
        prompts = batch.get("text_prompts", None)
        if prompts is None:
            return [
                "satellite aerial view background and other land cover",
                "buildings",
                "roads",
                "water bodies",
                "barren land",
                "forest",
                "agricultural fields"
            ]
        if isinstance(prompts, list) and len(prompts) > 0 and isinstance(prompts[0], list):
            return prompts[0]
        if isinstance(prompts, list):
            out = []
            for p in prompts:
                if isinstance(p, (list, tuple)) and len(p) > 0:
                    out.append(p[0])
                else:
                    out.append(p)
            return out
        return prompts

    def _pseudo_label_unlabeled_and_promote(self, top_k):
        if not self.use_filesystem_pool:
            return 0

        unlabeled_count = self._count_pool_images("Unlabeled")
        if unlabeled_count == 0 or top_k <= 0:
            return 0

        strategy_cfg = (self.al_config.get('active_learning', {}) or {}).get('strategy', {}) or {}
        strategy = str(strategy_cfg.get('name', 'mem')).lower()
        query_ratio = float(strategy_cfg.get('query_ratio', 0.2))
        query_ratio = max(0.0, min(1.0, query_ratio))
        training_cfg = self.train_config.get('training', {})
        batch_size = int(training_cfg.get('batch_size', 4))
        num_workers = int(training_cfg.get('num_workers', 0))
        pin_memory = self.device.type == 'cuda'
        persistent_workers = num_workers > 0
        max_pool = int((self.al_config.get('active_learning', {}) or {}).get('data', {}).get('unlabeled_pool_size', unlabeled_count))
        max_pool = max(0, min(max_pool, unlabeled_count))

        loader = DataLoader(
            self.pool_unlabeled_infer_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )

        scored = []
        self.model.eval()
        processed = 0
        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(self.device, non_blocking=True)
                prompts_list = self._batch_prompts(batch)
                outputs = self.model(images, prompts_list)
                logits = outputs['logits']
                attn_weights = outputs.get('attn_weights', None)
                feature_hw = outputs.get('feature_hw', None)
                original_size = images.shape[-2:]

                if strategy == "confidence":
                    probs = torch.softmax(logits, dim=1)
                    scores = (1.0 - probs.max(dim=1).values.mean(dim=(1, 2))).detach().cpu().tolist()
                else:
                    mem_map = self.mem_gen.generate(logits, attn_weights, original_size, feature_hw=feature_hw)
                    scores = [float(self.sampler.compute_score(mem_map[i])) for i in range(mem_map.shape[0])]

                image_paths = batch.get("image_path", [""] * logits.shape[0])
                image_names = batch.get("image_name", [""] * logits.shape[0])
                domain_names = batch.get("domain_name", [""] * logits.shape[0])

                for i in range(logits.shape[0]):
                    image_path = image_paths[i]
                    image_name = image_names[i]
                    domain_name = domain_names[i] if domain_names[i] else self._pool_scene_from_path(image_path)
                    scored.append((image_path, image_name, domain_name, float(scores[i])))
                    processed += 1
                    if processed >= max_pool:
                        break
                if processed >= max_pool:
                    break

        if not scored:
            return 0
        
        scored.sort(key=lambda x: x[3], reverse=True)
        k = max(0, min(int(top_k), len(scored)))
        balance_domains = bool(strategy_cfg.get('balance_domains', False))
        if balance_domains and k > 0:
            by_domain = {}
            for s in scored:
                by_domain.setdefault(s[2], []).append(s)
            domains = [d for d in by_domain.keys() if d]
            if len(domains) >= 2:
                per = k // len(domains)
                selected = []
                for d in domains:
                    selected.extend(by_domain[d][:per])
                rem = k - len(selected)
                if rem > 0:
                    leftovers = []
                    for d in domains:
                        leftovers.extend(by_domain[d][per:])
                    leftovers.sort(key=lambda x: x[3], reverse=True)
                    selected.extend(leftovers[:rem])
            else:
                selected = scored[:k]
        else:
            selected = scored[:k]

        moved = 0
        for image_path, image_name, domain_name, _ in selected:
            if not image_path:
                continue
            gt_mask_path = self._lookup_gt_mask_path(image_path)
            if not gt_mask_path:
                continue

            infer_idx = self._pool_unlabeled_path_to_index.get(image_path, None)
            if infer_idx is None:
                continue
            sample = self.pool_unlabeled_infer_dataset[infer_idx]
            img_tensor = sample["image"].unsqueeze(0).to(self.device)
            prompts = sample.get("text_prompts", None)
            prompts = prompts if prompts is not None else self._batch_prompts({})

            with torch.no_grad():
                out = self.model(img_tensor, prompts)
                logits = out["logits"]
                attn_weights = out.get("attn_weights", None)
                feature_hw = out.get("feature_hw", None)
                original_size = img_tensor.shape[-2:]
                mem_map = self.mem_gen.generate(logits, attn_weights, original_size, feature_hw=feature_hw)[0]

            mem_flat = mem_map.detach().cpu().numpy().reshape(-1)
            if mem_flat.size == 0:
                continue
            if query_ratio <= 0:
                query_mask = torch.zeros_like(mem_map, dtype=torch.bool)
            elif query_ratio >= 1.0:
                query_mask = torch.ones_like(mem_map, dtype=torch.bool)
            else:
                thr = float(np.quantile(mem_flat, 1.0 - query_ratio))
                query_mask = (mem_map >= thr)

            gt_pil = Image.open(gt_mask_path).convert("RGB")
            gt_pil = gt_pil.resize((original_size[1], original_size[0]), resample=Image.NEAREST)
            gt_np = np.array(gt_pil)
            gt_idx = self.full_dataset.rgb_to_mask(gt_np)
            gt_mask = torch.from_numpy(gt_idx).to(self.device)

            annotated = self.oracle.query(None, gt_mask, query_mask)
            annotated_cpu = annotated.detach().cpu().numpy().astype(np.int64)
            out_mask = np.full_like(annotated_cpu, fill_value=255, dtype=np.uint8)
            valid = (annotated_cpu >= 0) & (annotated_cpu < 7)
            out_mask[valid] = (annotated_cpu[valid] + 1).astype(np.uint8)

            src_img_dir, src_mask_dir = self._pool_dirs("Unlabeled", domain_name)
            dst_img_dir, dst_mask_dir = self._pool_dirs("Labeled", domain_name)
            os.makedirs(dst_img_dir, exist_ok=True)
            os.makedirs(dst_mask_dir, exist_ok=True)

            src_img_path = os.path.join(src_img_dir, image_name)
            src_mask_path = os.path.join(src_mask_dir, image_name)
            dst_img_path = os.path.join(dst_img_dir, image_name)
            dst_mask_path = os.path.join(dst_mask_dir, image_name)

            if os.path.exists(src_mask_path):
                try:
                    os.remove(src_mask_path)
                except OSError:
                    pass
            Image.fromarray(out_mask, mode="L").save(dst_mask_path)

            if os.path.exists(src_img_path) and not os.path.exists(dst_img_path):
                try:
                    shutil.move(src_img_path, dst_img_path)
                except shutil.Error:
                    if os.path.abspath(src_img_path) != os.path.abspath(dst_img_path):
                        shutil.copy2(src_img_path, dst_img_path)
                        try:
                            os.remove(src_img_path)
                        except OSError:
                            pass
                moved += 1

        self._reload_pool_datasets()
        return moved

    def _link_or_copy(self, src, dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            return
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    def _pool_scene_from_path(self, path):
        p = str(path).lower()
        if "urban" in p:
            return "Urban"
        return "Rural"

    def _pool_dirs(self, split_folder, scene):
        img_dir = os.path.join(self.pool_root, split_folder, scene, "images_png")
        mask_dir = os.path.join(self.pool_root, split_folder, scene, "masks_png")
        return img_dir, mask_dir

    def _count_pool_images(self, split_folder):
        total = 0
        for scene in ["Urban", "Rural"]:
            img_dir, _ = self._pool_dirs(split_folder, scene)
            total += len(glob.glob(os.path.join(img_dir, "*.png")))
        return total

    def _init_filesystem_pool(self):
        labeled_count = self._count_pool_images("Labeled") if os.path.exists(self.pool_root) else 0
        unlabeled_count = self._count_pool_images("Unlabeled") if os.path.exists(self.pool_root) else 0
        if (labeled_count > 0 or unlabeled_count > 0) and not self.reset_pool:
            return

        for split_folder in ["Labeled", "Unlabeled"]:
            for scene in ["Urban", "Rural"]:
                img_dir, mask_dir = self._pool_dirs(split_folder, scene)
                os.makedirs(img_dir, exist_ok=True)
                os.makedirs(mask_dir, exist_ok=True)
                for f in glob.glob(os.path.join(img_dir, "*.png")):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
                for f in glob.glob(os.path.join(mask_dir, "*.png")):
                    try:
                        os.remove(f)
                    except OSError:
                        pass

        for idx in self.labeled_indices:
            img_path = self.full_dataset.image_paths[idx]
            mask_path = self.full_dataset.mask_paths[idx] if hasattr(self.full_dataset, "mask_paths") else None
            scene = self._pool_scene_from_path(img_path)
            dst_img_dir, dst_mask_dir = self._pool_dirs("Labeled", scene)
            img_name = os.path.basename(img_path)
            self._link_or_copy(img_path, os.path.join(dst_img_dir, img_name))
            if mask_path and os.path.exists(mask_path):
                self._link_or_copy(mask_path, os.path.join(dst_mask_dir, img_name))

        for idx in self.unlabeled_indices:
            img_path = self.full_dataset.image_paths[idx]
            scene = self._pool_scene_from_path(img_path)
            dst_img_dir, _ = self._pool_dirs("Unlabeled", scene)
            img_name = os.path.basename(img_path)
            self._link_or_copy(img_path, os.path.join(dst_img_dir, img_name))

    def _save_stage_visualizations(self, stage_idx, labeled_fraction, max_samples=8):
        out_root = os.path.join(self.log_dir, 'visualizations', f"stage_{int(stage_idx):02d}_{int(round(labeled_fraction * 100.0)):02d}pct")
        os.makedirs(out_root, exist_ok=True)
        n = min(int(max_samples), len(self.val_dataset))
        if n <= 0:
            return

        self.model.eval()
        with torch.no_grad():
            for i in range(n):
                sample = self.val_dataset[i]
                image = sample['image'].unsqueeze(0).to(self.device)
                prompts = sample.get('text_prompts', [])
                mask = sample.get('mask', None)
                if torch.is_tensor(mask):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask) if mask is not None else None

                outputs = self.model(image, prompts)
                logits = outputs['logits']
                attn_weights = outputs.get('attn_weights', None)
                feature_hw = outputs.get('feature_hw', None)
                original_size = image.shape[-2:]
                pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
                mem_map = self.mem_gen.generate(logits, attn_weights, original_size, feature_hw=feature_hw)[0].cpu().numpy()

                domain = sample.get('domain_name', '')
                name = sample.get('image_name', f"sample_{i}.png")
                if name.lower().endswith('.png'):
                    base = name[:-4]
                else:
                    base = name
                tag = f"{domain}_" if domain else ""
                save_path = os.path.join(out_root, f"{i:04d}_{tag}{base}.png")
                if mask_np is None:
                    continue
                visualize_prediction(sample['image'].cpu(), mask_np, pred, mem_map, save_path)

    def _build_optimizer(self):
        training_cfg = self.train_config.get('training', {})
        optim_cfg = training_cfg.get('optimizer', {}) or {}
        optim_type = str(optim_cfg.get('type', 'adamw')).lower()
        lr = float(training_cfg.get('learning_rate', 1e-5))
        weight_decay = float(training_cfg.get('weight_decay', 0.0))
        backbone_lr_mult = float(training_cfg.get('backbone_lr_mult', 1.0))

        param_groups = None
        if backbone_lr_mult != 1.0 and hasattr(self.model, 'vision_encoder'):
            backbone_params = [p for p in self.model.vision_encoder.parameters() if p.requires_grad]
            head_params = [
                p for n, p in self.model.named_parameters()
                if p.requires_grad and not n.startswith('vision_encoder.')
            ]
            if len(backbone_params) > 0 and len(head_params) > 0:
                param_groups = [
                    {"params": backbone_params, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
                    {"params": head_params, "lr": lr, "weight_decay": weight_decay}
                ]

        params = param_groups if param_groups is not None else [p for p in self.model.parameters() if p.requires_grad]

        if optim_type == 'adamw':
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        if optim_type == 'adam':
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        if optim_type == 'sgd':
            momentum = float(optim_cfg.get('momentum', 0.9))
            nesterov = bool(optim_cfg.get('nesterov', False))
            return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        raise ValueError(f"Unsupported optimizer type: {optim_type}")

    def _build_scheduler(self, optimizer):
        training_cfg = self.train_config.get('training', {})
        sched_cfg = training_cfg.get('scheduler', None)
        if not sched_cfg:
            return None

        sched_type = str(sched_cfg.get('type', '')).lower()
        if sched_type == 'cosine_annealing':
            t_max = int(sched_cfg.get('T_max', training_cfg.get('num_epochs', 1)))
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, t_max))
        if sched_type in ('none', ''):
            return None
        raise ValueError(f"Unsupported scheduler type: {sched_type}")
        
    def run(self):
        set_seed(42)
        al_cfg = (self.al_config.get('active_learning', {}) or {})
        stage_fractions = al_cfg.get('stage_fractions', None)
        if stage_fractions:
            stages = sorted({float(x) for x in stage_fractions})
        else:
            total_rounds = int(al_cfg.get('total_rounds', 1))
            stages = [None for _ in range(max(1, total_rounds))]

        stage_summary_path = os.path.join(self.log_dir, 'al_stage_summary.json')
        stage_summary = []
        if os.path.exists(stage_summary_path):
            try:
                with open(stage_summary_path, 'r') as f:
                    stage_summary = json.load(f)
            except Exception:
                stage_summary = []

        for r in range(len(stages)):
            if self.use_filesystem_pool:
                labeled_count = self._count_pool_images("Labeled")
                unlabeled_count = self._count_pool_images("Unlabeled")
                frac = (labeled_count / self.total_train_size) if self.total_train_size else 0.0
                label = f"{frac * 100:.0f}%" if stages[r] is None else f"{stages[r] * 100:.0f}%"
                print(f"=== AL Stage {r+1}/{len(stages)} ({label}) ===")
                print(f"Labeled pool size: {labeled_count} ({frac * 100:.2f}%)")
                print(f"Unlabeled pool size: {unlabeled_count}")
            else:
                print(f"=== AL Stage {r+1}/{len(stages)} ===")
                print(f"Labeled pool size: {len(self.labeled_indices)}")

            phase_stats = self.train_phase(current_round=r + 1)
            stage_summary.append({
                "stage": int(r + 1),
                "target_fraction": None if stages[r] is None else float(stages[r]),
                "labeled_count": int(phase_stats.get("labeled_count", 0)),
                "labeled_fraction": float(phase_stats.get("labeled_fraction", 0.0)),
                "best_val_miou": float(phase_stats.get("best_val_miou", float("nan")))
            })
            with open(stage_summary_path, 'w') as f:
                json.dump(stage_summary, f, indent=4)

            viz_cfg = (al_cfg.get('visualization', {}) or {})
            viz_enabled = bool(viz_cfg.get('enabled', True))
            viz_max_samples = int(viz_cfg.get('max_samples', 8))
            if viz_enabled:
                self._save_stage_visualizations(stage_idx=r + 1, labeled_fraction=float(phase_stats.get("labeled_fraction", 0.0)), max_samples=viz_max_samples)

            if not self.use_filesystem_pool:
                break

            if r >= len(stages) - 1:
                break

            if stages[r + 1] is None:
                budget = int(al_cfg.get('budget_per_round', 0))
                top_k = max(0, budget)
            else:
                target_next = float(stages[r + 1])
                labeled_now = self._count_pool_images("Labeled")
                target_count = int(math.ceil(target_next * float(self.total_train_size)))
                top_k = max(0, target_count - int(labeled_now))

            if top_k <= 0:
                continue

            moved = self._pseudo_label_unlabeled_and_promote(top_k=top_k)
            if moved > 0:
                print(f"Oracle-annotated and promoted {moved} samples to Labeled.")
            else:
                print("No samples promoted this stage.")
                break
                
    def train_phase(self, current_round=1):
        if self.use_filesystem_pool:
            self._reload_pool_datasets()
        training_cfg = self.train_config.get('training', {})
        batch_size = int(training_cfg.get('batch_size', 4))
        grad_accum_steps = int(training_cfg.get('grad_accum_steps', 1))
        num_workers = int(training_cfg.get('num_workers', 0))
        pin_memory = self.device.type == 'cuda'
        persistent_workers = num_workers > 0
        freeze_backbone_epochs = int(training_cfg.get('freeze_backbone_epochs', 0))
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )

        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer)
        early_cfg = training_cfg.get('early_stopping', None)
        patience = int(early_cfg.get('patience', 0)) if early_cfg else 0
        min_delta = float(early_cfg.get('min_delta', 0.0)) if early_cfg else 0.0
        best_val_miou_phase = float("-inf")
        bad_epochs = 0
        backbone_frozen = None
        
        epochs = self.train_config['training']['num_epochs']
        # For fast check, let's keep it as configured (currently 1 in config from previous step)
        
        for e in range(epochs):
            if self.use_filesystem_pool:
                loader = DataLoader(
                    self.pool_labeled_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers
                )
            else:
                labeled_subset, _ = get_subsets(self.full_dataset, self.labeled_indices, self.unlabeled_indices)
                loader = DataLoader(
                    labeled_subset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers
                )

            should_freeze = e < freeze_backbone_epochs
            if backbone_frozen is None or should_freeze != backbone_frozen:
                if hasattr(self.model, 'vision_encoder'):
                    for param in self.model.vision_encoder.parameters():
                        param.requires_grad = not should_freeze
                optimizer = self._build_optimizer()
                scheduler = self._build_scheduler(optimizer)
                backbone_frozen = should_freeze

            self.global_epoch += 1
            print(f"--- Epoch {e+1}/{epochs} (Global: {self.global_epoch}) ---")
            
            # Train
            train_loss, train_miou, train_pixel_acc = train_one_epoch(
                self.model,
                loader,
                optimizer,
                self.criterion,
                self.device,
                grad_accum_steps=grad_accum_steps,
                attn_supervision_weight=self.attn_supervision_weight,
                attn_logits_weight=self.attn_logits_weight
            )
            print(f"Train Loss: {train_loss:.4f}, mIoU: {train_miou:.4f}, Acc: {train_pixel_acc:.4f}")
            
            # Validate
            val_loss, val_miou, val_class_iou, val_pixel_acc, domain_stats = validate(
                self.model,
                val_loader,
                self.criterion,
                self.device,
                focus_class=1
            )
            print(f"Val Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}, Acc: {val_pixel_acc:.4f}")
            if isinstance(domain_stats, dict) and isinstance(domain_stats.get("focus", None), dict):
                print(f"Val Focus (class 1) mIoU: {domain_stats['focus']['miou']:.4f}, Acc: {domain_stats['focus']['pixel_acc']:.4f}")

            if scheduler is not None:
                scheduler.step()

            if patience > 0:
                if val_miou > best_val_miou_phase + min_delta:
                    bad_epochs = 0
                else:
                    bad_epochs += 1
            if val_miou > best_val_miou_phase:
                best_val_miou_phase = val_miou

            if self.use_filesystem_pool:
                labeled_count = self._count_pool_images("Labeled")
                unlabeled_count = self._count_pool_images("Unlabeled")
            else:
                labeled_count = len(self.labeled_indices)
                unlabeled_count = len(self.unlabeled_indices)
            labeled_fraction = (labeled_count / self.total_train_size) if self.total_train_size else 0.0
            
            # Metrics Logging
            epoch_metrics = {
                'epoch': self.global_epoch,
                'al_round': int(current_round),
                'train_loss': train_loss,
                'train_miou': train_miou,
                'train_pixel_acc': train_pixel_acc,
                'val_loss': val_loss,
                'val_miou': val_miou,
                'val_pixel_acc': val_pixel_acc,
                'val_class_iou': val_class_iou,
                'labeled_count': labeled_count,
                'unlabeled_count': unlabeled_count,
                'labeled_fraction': labeled_fraction
            }
            self.metrics_history.append(epoch_metrics)
            
            # Save Metrics (list-of-dicts for quick inspection)
            with open(os.path.join(self.log_dir, 'metrics.json'), 'w') as f:
                json.dump(self.metrics_history, f, indent=4)
            
            # Strict append-only accumulation (NO reassignment, NO re-init, NO overwrite)
            if self.global_epoch in self.metrics_history_accum["epochs"]:
                raise RuntimeError("Epoch logged more than once")
            self.metrics_history_accum["epochs"].append(self.global_epoch)
            self.metrics_history_accum["train_loss"].append(float(train_loss))
            self.metrics_history_accum["val_loss"].append(float(val_loss))
            self.metrics_history_accum["train_accuracy"].append(float(train_pixel_acc))
            self.metrics_history_accum["val_accuracy"].append(float(val_pixel_acc))
            self.metrics_history_accum["train_miou"].append(float(train_miou))
            # Ensure domain-wise keys exist before appending (backward compatibility)
            for k in ["val_miou_urban", "val_miou_rural", "val_acc_urban", "val_acc_rural"]:
                if k not in self.metrics_history_accum:
                    self.metrics_history_accum[k] = []
            self.metrics_history_accum["val_miou"].append(float(val_miou))
            self.metrics_history_accum["val_miou_urban"].append(float(domain_stats["urban"]["miou"]))
            self.metrics_history_accum["val_miou_rural"].append(float(domain_stats["rural"]["miou"]))
            self.metrics_history_accum["val_acc_urban"].append(float(domain_stats["urban"]["pixel_acc"]))
            self.metrics_history_accum["val_acc_rural"].append(float(domain_stats["rural"]["pixel_acc"]))
            self.metrics_history_accum["al_round"].append(int(current_round))
            # Class-wise IoU mapping
            for i, name in enumerate(self.class_names):
                val = float(val_class_iou[i]) if i < len(val_class_iou) else float('nan')
                self.metrics_history_accum["classwise_iou"][name].append(val)
            self.metrics_history_accum["labeled_count"].append(int(labeled_count))
            self.metrics_history_accum["unlabeled_count"].append(int(unlabeled_count))
            self.metrics_history_accum["labeled_fraction"].append(float(labeled_fraction))
            # Persist accumulated history
            with open(self.history_path, 'w') as f:
                json.dump(self.metrics_history_accum, f, indent=4)
                
            # Save Class-wise IoU
            with open(os.path.join(self.log_dir, f'classwise_iou_epoch_{self.global_epoch}.json'), 'w') as f:
                json.dump({'class_iou': val_class_iou}, f, indent=4)
                
            # Checkpoint State
            state = {
                'epoch': self.global_epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'metrics': epoch_metrics
            }
            
            # Save Last Epoch
            last_ckpt_path = os.path.join(self.ckpt_dir, 'last_epoch.pth')
            torch.save(state, last_ckpt_path)
            print(f"Saved last_epoch.pth to {last_ckpt_path}")
            
            # Save Best mIoU
            if val_miou > self.best_miou:
                self.best_miou = val_miou
                torch.save(state, os.path.join(self.ckpt_dir, 'best_miou.pth'))
                print(f"New best mIoU: {self.best_miou:.4f} -> Saved best_miou.pth")
                
            # Generate Plots from metrics_history.json
            try:
                plot_metrics(self.history_path, self.plot_dir)
            except AssertionError as e:
                print(f"Plotting skipped: {e}")

            if patience > 0 and bad_epochs >= patience:
                print(f"Early stopping: no val mIoU improvement for {patience} epochs.")
                break

        best_ckpt_path = os.path.join(self.ckpt_dir, 'best_miou.pth')
        if os.path.exists(best_ckpt_path):
            best_state = torch.load(best_ckpt_path, map_location=self.device)
            if isinstance(best_state, dict) and 'state_dict' in best_state:
                self.model.load_state_dict(best_state['state_dict'])
            else:
                self.model.load_state_dict(best_state)
        if self.use_filesystem_pool:
            labeled_count = self._count_pool_images("Labeled")
            unlabeled_count = self._count_pool_images("Unlabeled")
        else:
            labeled_count = len(self.labeled_indices)
            unlabeled_count = len(self.unlabeled_indices)
        labeled_fraction = (labeled_count / self.total_train_size) if self.total_train_size else 0.0
        best_val = float(best_val_miou_phase) if best_val_miou_phase != float("-inf") else float("nan")
        return {"best_val_miou": best_val, "labeled_count": int(labeled_count), "unlabeled_count": int(unlabeled_count), "labeled_fraction": float(labeled_fraction)}
            
    def active_learning_phase(self):
        budget = int((self.al_config.get('active_learning', {}) or {}).get('budget_per_round', 0))
        return self._pseudo_label_unlabeled_and_promote(top_k=budget)

if __name__ == "__main__":
    loop = ActiveLearningLoop()
    loop.run()
