"""BUSI V2: ImageNet transfer learning, shared/differential LR and plateau schedule.

This script reads TRAIN and VAL only, deliberately never TEST.
Select models/configuration by validation Dice, then freeze your experiment
protocol before any new held-out evaluation.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.training_v2_utils import (
    build_model, evaluate_validation, get_current_lrs,
    make_criterion, make_loaders, make_optimizer, set_seed,
)


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if args.pretrained == "none" and args.lr_mode == "differential":
        raise ValueError("Random encoder comparison should use --lr_mode shared")
    if args.early_stop_patience <= args.lr_patience:
        raise ValueError("Early-stop patience must exceed plateau patience")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    train_loader, val_loader = make_loaders(args, include_val=True)
    print("Train / Validation:", len(train_loader.dataset), len(val_loader.dataset))
    model = build_model(args.model, args.pretrained).to(device)
    criterion = make_criterion()
    optimizer = make_optimizer(
        model=model, lr_mode=args.lr_mode, base_lr=args.base_lr,
        encoder_ratio=args.encoder_ratio, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max", factor=args.lr_factor,
        patience=args.lr_patience, min_lr=args.min_lr,
    )
    best_val_dice = float("-inf")
    best_epoch = 0
    no_improve = 0
    history = []
    ckpt_path = save_dir / "best_model.pt"

    print("Model: {} | init={} | LR mode={}".format(args.model, args.pretrained, args.lr_mode))
    print("Initial LRs:", get_current_lrs(optimizer))
    print("Test data are intentionally NOT read by this training script.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_train = 0
        for batch in tqdm(train_loader, desc="Epoch {}/{}".format(epoch, args.epochs)):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite train loss at epoch {}".format(epoch))
            loss.backward()
            optimizer.step()
            bs = images.shape[0]
            train_loss_sum += float(loss.item()) * bs
            n_train += bs
        train_loss = train_loss_sum / n_train
        metrics = evaluate_validation(model, val_loader, criterion, device)
        # LR applied during THIS epoch (before scheduler changes it).
        lr_used = get_current_lrs(optimizer)
        val_dice = metrics["val_dice"]
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": args.model,
                    "pretrained": args.pretrained,
                    "lr_mode": args.lr_mode,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_iou": metrics["val_iou"],
                    "config": vars(args),
                },
                ckpt_path,
            )
            improved = True
        else:
            no_improve += 1
            improved = False

        scheduler.step(val_dice)
        next_lrs = get_current_lrs(optimizer)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
            "encoder_lr_used": lr_used["encoder_lr"],
            "decoder_lr_used": lr_used["decoder_lr"],
            "encoder_lr_next": next_lrs["encoder_lr"],
            "decoder_lr_next": next_lrs["decoder_lr"],
            "best_val_dice_so_far": best_val_dice,
        }
        history.append(row)  # IMPORTANT: exactly one row per epoch
        pd.DataFrame(history).to_csv(save_dir / "history.csv", index=False)
        print(
            "Epoch {}: train_loss={:.4f}, val_loss={:.4f}, val_dice={:.4f}, "
            "val_iou={:.4f}, encoder_lr={:.2e}, decoder_lr={:.2e}, best={}"
            .format(
                epoch, train_loss, metrics["val_loss"], val_dice,
                metrics["val_iou"], lr_used["encoder_lr"],
                lr_used["decoder_lr"], "YES" if improved else "no"
            )
        )
        if epoch >= args.min_epochs and no_improve >= args.early_stop_patience:
            print("Early stopping at epoch", epoch)
            break

    summary = {
        "model": args.model,
        "pretrained": args.pretrained,
        "lr_mode": args.lr_mode,
        "base_lr": args.base_lr,
        "encoder_ratio": args.encoder_ratio,
        "weight_decay": args.weight_decay,
        "augmentation": args.augmentation,
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "epochs_completed": len(history),
    }
    (save_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n==== TRAIN/VAL COMPLETED ====")
    print(json.dumps(summary, indent=2))
    print("Files:", save_dir)
    print("No test metrics have been computed in this run.")


def make_parser():
    p = argparse.ArgumentParser(description="BUSI advanced V2 training (train/val only)")
    p.add_argument("--data_root", required=True)
    p.add_argument("--train_csv", default="data/splits/train.csv")
    p.add_argument("--val_csv", default="data/splits/val.csv")
    p.add_argument("--model", choices=["resunet18", "attention_resunet18"], default="resunet18")
    p.add_argument("--pretrained", choices=["imagenet", "none"], default="imagenet")
    p.add_argument("--lr_mode", choices=["shared", "differential"], default="shared")
    p.add_argument("--base_lr", type=float, default=3e-4)
    p.add_argument("--encoder_ratio", type=float, default=0.3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr_factor", type=float, default=0.5)
    p.add_argument("--lr_patience", type=int, default=3)
    p.add_argument("--min_lr", type=float, default=1e-7)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--min_epochs", type=int, default=15)
    p.add_argument("--early_stop_patience", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augmentation", action="store_true", default=True)
    p.add_argument("--no_augmentation", dest="augmentation", action="store_false")
    p.add_argument("--save_dir", required=True)
    return p


if __name__ == "__main__":
    run(make_parser().parse_args())
