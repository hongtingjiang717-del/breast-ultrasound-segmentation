"""LR range test for the user's BUSI ResNet18 segmentation models.

Run on TRAIN only; model is discarded afterwards. Its suggestion is a heuristic,
not a guaranteed optimal learning rate. Differential mode sweeps the decoder
reference LR while maintaining encoder_lr = decoder_lr * encoder_ratio.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.training_v2_utils import (
    build_model, make_criterion, make_loaders, make_optimizer, set_seed,
)


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("LR Finder device:", device)
    print("Model: {}, init: {}, LR mode: {}".format(args.model, args.pretrained, args.lr_mode))

    if args.max_lr <= args.min_lr or args.min_lr <= 0:
        raise ValueError("Require 0 < min_lr < max_lr")
    if args.num_iters < 15:
        raise ValueError("Use at least 15 iterations")

    loader, _ = make_loaders(args, include_val=False)
    if len(loader) == 0:
        raise ValueError("Training loader is empty")

    model = build_model(args.model, args.pretrained).to(device)
    criterion = make_criterion()
    optimizer = make_optimizer(
        model, args.lr_mode, args.min_lr, args.encoder_ratio, args.weight_decay
    )

    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    beta = 0.95
    moving = 0.0
    best_smooth = float("inf")
    rows = []
    iterator = iter(loader)
    model.train()

    for step in range(args.num_iters):
        # Exponentially increasing decoder/reference LR.
        lr = args.min_lr * (args.max_lr / args.min_lr) ** (step / (args.num_iters - 1))
        if args.lr_mode == "shared":
            optimizer.param_groups[0]["lr"] = lr
            encoder_lr = lr
        else:
            optimizer.param_groups[0]["lr"] = lr * args.encoder_ratio
            optimizer.param_groups[1]["lr"] = lr
            encoder_lr = lr * args.encoder_ratio

        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        raw = float(loss.item())
        if not np.isfinite(raw):
            print("Non-finite loss at step", step + 1)
            break

        loss.backward()
        optimizer.step()
        moving = beta * moving + (1 - beta) * raw
        smooth = moving / (1 - beta ** (step + 1))
        rows.append({
            "iteration": step + 1,
            "encoder_lr": encoder_lr,
            "decoder_lr": lr,
            "raw_loss": raw,
            "smoothed_loss": smooth,
        })
        if smooth < best_smooth:
            best_smooth = smooth
        if step >= 15 and smooth > best_smooth * args.diverge_factor:
            print("Early stopping: loss diverged at LR {:.3e}".format(lr))
            break
        if (step + 1) % 10 == 0:
            print("Step {:3d}: base LR={:.3e}, smooth loss={:.4f}".format(step + 1, lr, smooth))

    if len(rows) < 10:
        raise RuntimeError("LR finder ended too early; check inputs and numerical stability")

    results = pd.DataFrame(rows)
    results.to_csv(out / "lr_finder.csv", index=False)

    # Locate the steepest downward slope in log-LR space (range-test heuristic).
    # Smoothing and excluding the ends reduce instability from small batches.
    smoothed = results["smoothed_loss"].rolling(
        window=7, center=True, min_periods=3
    ).median().to_numpy()
    xlog = np.log10(results["decoder_lr"].to_numpy())
    slope = np.gradient(smoothed, xlog)
    lo = max(5, int(len(results) * 0.15))
    hi = min(len(results) - 3, int(len(results) * 0.85))
    slope_idx = lo + int(np.argmin(slope[lo:hi]))
    fastest_drop_lr = float(results.iloc[slope_idx]["decoder_lr"])
    # Conservative reference: begin below the steepest descent point.
    suggested = max(args.min_lr, min(args.max_lr, fastest_drop_lr / 3.0))
    if slope[slope_idx] >= 0:
        print("WARNING: No clear decreasing-loss interval was found. Re-run with more steps or inspect manually.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results["decoder_lr"], results["smoothed_loss"], label="Smoothed train loss")
    ax.axvline(suggested, linestyle="--", label="Heuristic starting LR {:.2e}".format(suggested))
    ax.axvline(fastest_drop_lr, linestyle=":", label="Steepest downward-slope LR {:.2e}".format(fastest_drop_lr))
    ax.set_xscale("log")
    ax.set_xlabel("Decoder / reference LR (log scale)")
    ax.set_ylabel("Smoothed training loss")
    ax.set_title("LR range test: {} ({})".format(args.model, args.lr_mode))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "lr_finder.png", dpi=240)
    plt.close(fig)

    text = (
        "Model: {}\nPretrained: {}\nLR mode: {}\nEncoder ratio: {}\n"
        "LR at steepest smoothed-loss decrease: {:.8g}\n"
        "Suggested decoder/reference initial LR (heuristic): {:.8g}\n"
        "If differential: suggested encoder LR: {:.8g}\n"
        "IMPORTANT: inspect the curve; this is not a proven optimum.\n"
    ).format(args.model, args.pretrained, args.lr_mode, args.encoder_ratio,
             fastest_drop_lr, suggested,
             suggested * args.encoder_ratio if args.lr_mode == "differential" else suggested)
    (out / "suggestion.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print("Saved:", out / "lr_finder.png")
    print("The LR Finder changes weights internally; its model is NOT saved/reused for formal training.")


def make_parser():
    p = argparse.ArgumentParser(description="BUSI LR Range Test")
    p.add_argument("--data_root", required=True)
    p.add_argument("--train_csv", default="data/splits/train.csv")
    p.add_argument("--val_csv", default="data/splits/val.csv")
    p.add_argument("--model", choices=["resunet18", "attention_resunet18"], default="resunet18")
    p.add_argument("--pretrained", choices=["imagenet", "none"], default="imagenet")
    p.add_argument("--lr_mode", choices=["shared", "differential"], default="shared")
    p.add_argument("--encoder_ratio", type=float, default=0.3)
    p.add_argument("--min_lr", type=float, default=1e-7)
    p.add_argument("--max_lr", type=float, default=1e-2)
    p.add_argument("--num_iters", type=int, default=100)
    p.add_argument("--diverge_factor", type=float, default=5.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--augmentation", action="store_true", default=True,
                   help="Training augmentation is enabled by default")
    p.add_argument("--no_augmentation", dest="augmentation", action="store_false")
    p.add_argument("--save_dir", default="results/v2/lr_finder_resunet")
    return p


if __name__ == "__main__":
    run(make_parser().parse_args())
