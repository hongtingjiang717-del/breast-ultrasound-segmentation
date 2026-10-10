"""BUSI V2 training utilities, compatible with the existing GitHub project.

Only training/validation data are used. Test-set evaluation is intentionally
kept out of the tuning loop to prevent test-set leakage.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.augmentations import get_eval_augmentation, get_train_augmentation
from src.dataset import BUSIDataset
from src.loss import DiceBCELoss
from src.metrics import segmentation_metrics
from src.model_zoo import build_attention_resunet18, build_resunet18


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # No benchmark autotuning, to reduce run-to-run variability.
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    # DataLoader workers have independent seeds; Albumentations uses NumPy as well.
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_model(name, pretrained="imagenet"):
    """Build an SMP model. Output must be logits (not sigmoid probabilities)."""
    weights = "imagenet" if pretrained == "imagenet" else None
    if name == "resunet18":
        return build_resunet18(in_channels=1, classes=1, encoder_weights=weights)
    if name == "attention_resunet18":
        return build_attention_resunet18(in_channels=1, classes=1, encoder_weights=weights)
    raise ValueError("Only SMP ResNet-based models are supported: {}".format(name))


def make_loaders(args, include_val=True):
    """Build data loaders with paired augmentation only in the training split."""
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError("Dataset root not found: {}".format(data_root))

    transform = get_train_augmentation(256) if args.augmentation else get_eval_augmentation(256)
    train_ds = BUSIDataset(
        csv_file=args.train_csv,
        image_size=(256, 256),
        data_root=str(data_root),
        transform=transform,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if not include_val:
        return train_loader, None

    val_ds = BUSIDataset(
        csv_file=args.val_csv,
        image_size=(256, 256),
        data_root=str(data_root),
        transform=get_eval_augmentation(256),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )
    return train_loader, val_loader


def make_optimizer(model, lr_mode, base_lr, encoder_ratio, weight_decay):
    """AdamW: shared LR, or smaller encoder LR + base decoder/head LR.

    In differential mode, the scSE attention blocks belong to model.decoder.
    Assert that the two groups cover every trainable parameter exactly once.
    """
    if base_lr <= 0 or not (0 < encoder_ratio <= 1):
        raise ValueError("base_lr must be positive and encoder_ratio in (0, 1]")

    if lr_mode == "shared":
        return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    if lr_mode != "differential":
        raise ValueError("Unknown lr_mode: {}".format(lr_mode))

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
    decoder_params += [p for p in model.segmentation_head.parameters() if p.requires_grad]
    all_trainable = [p for p in model.parameters() if p.requires_grad]
    grouped_ids = [id(p) for p in encoder_params + decoder_params]
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != {id(p) for p in all_trainable}:
        raise RuntimeError("Parameter groups are duplicated or do not cover all trainable parameters")
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": base_lr * encoder_ratio, "name": "encoder"},
            {"params": decoder_params, "lr": base_lr, "name": "decoder_head"},
        ],
        weight_decay=weight_decay,
    )


def get_current_lrs(optimizer):
    if len(optimizer.param_groups) == 1:
        lr = optimizer.param_groups[0]["lr"]
        return {"encoder_lr": lr, "decoder_lr": lr}
    return {
        "encoder_lr": optimizer.param_groups[0]["lr"],
        "decoder_lr": optimizer.param_groups[1]["lr"],
    }


def evaluate_validation(model, loader, criterion, device):
    """Sample-weighted validation averages (handles the last short batch)."""
    model.eval()
    n = 0
    sums = {"val_loss": 0.0, "val_dice": 0.0, "val_iou": 0.0}
    with torch.inference_mode():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            dice, iou = segmentation_metrics(logits, y)
            bsz = x.shape[0]
            sums["val_loss"] += float(loss.item()) * bsz
            sums["val_dice"] += float(dice) * bsz
            sums["val_iou"] += float(iou) * bsz
            n += bsz
    if n == 0:
        raise ValueError("Validation loader is empty")
    return {key: value / n for key, value in sums.items()}


def make_criterion():
    return DiceBCELoss()
