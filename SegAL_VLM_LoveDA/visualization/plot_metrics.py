import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import argparse
import numpy as np

def plot_metrics(log_file, output_dir):
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return

    with open(log_file, 'r') as f:
        content = json.load(f)
    
    # Determine format: list-of-dicts or history dict
    if isinstance(content, dict) and 'epochs' in content:
        hist = content
        epochs = hist.get('epochs', [])
        assert len(set(epochs)) == len(epochs), "Epochs contain duplicates"
        get_train_loss = lambda: hist.get('train_loss', [])
        get_val_loss = lambda: hist.get('val_loss', [])
        get_train_acc = lambda: hist.get('train_accuracy', [])
        get_val_acc = lambda: hist.get('val_accuracy', [])
        get_train_miou = lambda: hist.get('train_miou', [])
        get_val_miou = lambda: hist.get('val_miou', [])
        get_al_round = lambda: hist.get('al_round', [1 for _ in epochs])
        get_labeled_fraction = lambda: hist.get('labeled_fraction', [np.nan for _ in epochs])
        class_names = ['Background', 'Building', 'Road', 'Water', 'Barren', 'Forest', 'Agriculture']
        class_series = np.array([hist.get('classwise_iou', {}).get(name, []) for name in class_names]).T
    else:
        raise ValueError("Unsupported metrics format. Expected metrics_history.json structure.")
    
    # 1. Loss Curve
    plt.figure(figsize=(10, 6))
    train_loss = get_train_loss()
    val_loss = get_val_loss()
    for v in train_loss:
        assert not isinstance(v, list), "train_loss contains list values"
    for v in val_loss:
        assert not isinstance(v, list), "val_loss contains list values"
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    print("Saved loss_curve.png")
    plt.close()
    
    # 2. Accuracy Curve
    plt.figure(figsize=(10, 6))
    train_acc_series = get_train_acc()
    val_acc_series = get_val_acc()
    for v in train_acc_series:
        assert not isinstance(v, list), "train_accuracy contains list values"
    for v in val_acc_series:
        assert not isinstance(v, list), "val_accuracy contains list values"
    plt.plot(epochs, train_acc_series, label='Train Acc', marker='o')
    plt.plot(epochs, val_acc_series, label='Val Acc', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'), dpi=300, bbox_inches='tight')
    print("Saved accuracy_curve.png")
    plt.close()
    
    # 3. mIoU Curve
    plt.figure(figsize=(10, 6))
    train_miou_series = get_train_miou()
    val_miou_series = get_val_miou()
    for v in train_miou_series:
        assert not isinstance(v, list), "train_miou contains list values"
    for v in val_miou_series:
        assert not isinstance(v, list), "val_miou contains list values"
    plt.plot(epochs, train_miou_series, label='Train mIoU', marker='o')
    plt.plot(epochs, val_miou_series, label='Val mIoU', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title('Mean IoU over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'miou_curve.png'), dpi=300, bbox_inches='tight')
    print("Saved miou_curve.png")
    plt.close()
    
    # 4. Class-wise IoU Curve (Validation)
    plt.figure(figsize=(12, 8))
    classes = class_names
    for i in range(len(classes)):
        ys = class_series[:, i] if class_series.ndim == 2 else np.array([])
        for v in ys:
            assert not isinstance(v, list), f"classwise_iou[{classes[i]}] contains list values"
        if ys.size > 0:
            plt.plot(epochs, ys, label=classes[i], marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Validation Class-wise IoU Across Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'classwise_iou_curve.png'), dpi=300, bbox_inches='tight')
    print("Saved classwise_iou_curve.png")
    plt.close()
    
    print(f"Plots saved to {output_dir}")

    labeled_fraction = np.asarray(get_labeled_fraction(), dtype=np.float64)
    val_miou_series = np.asarray(get_val_miou(), dtype=np.float64)
    ok = (~np.isnan(labeled_fraction)) & (~np.isnan(val_miou_series))
    if ok.sum() > 0:
        plt.figure(figsize=(10, 6))
        xs = labeled_fraction[ok] * 100.0
        ys = val_miou_series[ok] * 100.0
        order = np.argsort(xs)
        plt.plot(xs[order], ys[order], marker='o')
        plt.xlabel('% Labeled Data')
        plt.ylabel('mIoU (%)')
        plt.title('mIoU vs % Labeled Data')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'miou_vs_labeled_fraction.png'), dpi=300, bbox_inches='tight')
        print("Saved miou_vs_labeled_fraction.png")
        plt.close()

    al_round = np.asarray(get_al_round(), dtype=np.int64)
    if al_round.size == val_miou_series.size and al_round.size > 0:
        plt.figure(figsize=(10, 6))
        unique_rounds = np.unique(al_round)
        round_miou = []
        for r in unique_rounds:
            idx = np.where(al_round == r)[0]
            if idx.size == 0:
                continue
            vals = val_miou_series[idx]
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                continue
            round_miou.append((int(r), float(np.max(vals) * 100.0)))
        if len(round_miou) > 0:
            xs = [t[0] for t in round_miou]
            ys = [t[1] for t in round_miou]
            plt.plot(xs, ys, marker='o')
            plt.xlabel('AL Cycle')
            plt.ylabel('Best mIoU (%)')
            plt.title('mIoU vs Active Learning Cycles')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'miou_vs_al_cycles.png'), dpi=300, bbox_inches='tight')
            print("Saved miou_vs_al_cycles.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, default='experiments/logs/metrics.json')
    parser.add_argument('--output_dir', type=str, default='experiments/logs/plots')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    plot_metrics(args.log_file, args.output_dir)
