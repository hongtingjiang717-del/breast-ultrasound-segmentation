# ==========================================================
# BUSI U-Net Baseline Training
# ==========================================================

from pathlib import Path

import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import BUSIDataset
from src.model import UNet
from src.loss import DiceBCELoss
from src.metrics import segmentation_metrics


# ==========================================================
# 1. 固定随机种子
# ==========================================================

def set_seed(seed=42):
    """
    尽可能保证不同实验之间具有可重复性。
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================================
# 2. 验证函数
# ==========================================================

def evaluate(
    model,
    data_loader,
    criterion,
    device
):
    """
    在 Validation / Test 数据集上评估模型。

    注意：
    验证阶段：
        不进行 backward
        不进行 optimizer.step
        不更新模型参数
    """

    # 切换到推理模式
    model.eval()


    total_loss = 0.0

    total_dice = 0.0

    total_iou = 0.0


    # 验证时不计算梯度
    with torch.no_grad():

        for batch in data_loader:

            # --------------------------------------
            # 数据送到 GPU
            # --------------------------------------

            images = batch["image"].to(
                device,
                non_blocking=True
            )

            masks = batch["mask"].to(
                device,
                non_blocking=True
            )


            # --------------------------------------
            # Forward
            # --------------------------------------

            logits = model(
                images
            )


            # --------------------------------------
            # Loss
            # --------------------------------------

            loss = criterion(
                logits,
                masks
            )


            # --------------------------------------
            # Dice / IoU
            # --------------------------------------

            dice, iou = segmentation_metrics(
                logits,
                masks
            )


            total_loss += loss.item()

            total_dice += dice

            total_iou += iou


    # Batch 平均
    n_batches = len(
        data_loader
    )


    return (
        total_loss / n_batches,
        total_dice / n_batches,
        total_iou / n_batches
    )


# ==========================================================
# 3. 主训练函数
# ==========================================================

def main(args):

    # ------------------------------------------------------
    # 固定随机性
    # ------------------------------------------------------

    set_seed(
        args.seed
    )


    # ------------------------------------------------------
    # Device
    # ------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        f"Using device: {device}"
    )


    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    # ======================================================
    # Dataset
    # ======================================================

    train_dataset = BUSIDataset(
        csv_file=args.train_csv,
        image_size=(256, 256),
        data_root=args.data_root
    )


    val_dataset = BUSIDataset(
        csv_file=args.val_csv,
        image_size=(256, 256),
        data_root=args.data_root
    )


    test_dataset = BUSIDataset(
        csv_file=args.test_csv,
        image_size=(256, 256),
        data_root=args.data_root
    )


    print(
        "Train samples:",
        len(train_dataset)
    )

    print(
        "Val samples:",
        len(val_dataset)
    )

    print(
        "Test samples:",
        len(test_dataset)
    )


    # ======================================================
    # DataLoader
    # ======================================================

    train_loader = DataLoader(
        train_dataset,

        batch_size=args.batch_size,

        shuffle=True,

        num_workers=args.num_workers,

        # GPU 训练时可加快 CPU → GPU 数据传输
        pin_memory=(
            device.type == "cuda"
        )
    )


    val_loader = DataLoader(
        val_dataset,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=args.num_workers,

        pin_memory=(
            device.type == "cuda"
        )
    )


    test_loader = DataLoader(
        test_dataset,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=args.num_workers,

        pin_memory=(
            device.type == "cuda"
        )
    )


    # ======================================================
    # Model
    # ======================================================

    model = UNet().to(
        device
    )


    # ======================================================
    # Loss
    # ======================================================

    criterion = DiceBCELoss()


    # ======================================================
    # Optimizer
    # ======================================================

    optimizer = optim.Adam(
        model.parameters(),

        lr=args.lr,

        weight_decay=1e-4
    )


    # ======================================================
    # 保存训练结果
    # ======================================================

    save_dir = Path(
        args.save_dir
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    best_model_path = (
        save_dir
        /
        "best_model.pt"
    )


    # 用于记录训练历史
    history = []


    # 当前最好 Validation Dice
    best_val_dice = -1.0


    # Early Stopping 计数器
    no_improve_epochs = 0


    # ======================================================
    # Training
    # ======================================================

    for epoch in range(
        args.epochs
    ):

        model.train()


        running_loss = 0.0


        # tqdm 用于显示进度条
        progress_bar = tqdm(
            train_loader,

            desc=(
                f"Epoch "
                f"{epoch + 1}/"
                f"{args.epochs}"
            )
        )


        # ==================================================
        # 遍历 Train Batch
        # ==================================================

        for batch in progress_bar:

            images = batch["image"].to(
                device,
                non_blocking=True
            )

            masks = batch["mask"].to(
                device,
                non_blocking=True
            )


            # ----------------------------------------------
            # 1. 清空旧梯度
            # ----------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )


            # ----------------------------------------------
            # 2. Forward
            # ----------------------------------------------

            logits = model(
                images
            )


            # ----------------------------------------------
            # 3. Loss
            # ----------------------------------------------

            loss = criterion(
                logits,
                masks
            )


            # ----------------------------------------------
            # 4. Backward
            # ----------------------------------------------

            loss.backward()


            # ----------------------------------------------
            # 5. 更新模型参数
            # ----------------------------------------------

            optimizer.step()


            # ----------------------------------------------
            # 6. 记录 Loss
            # ----------------------------------------------

            running_loss += (
                loss.item()
            )


            progress_bar.set_postfix(
                loss=(
                    f"{loss.item():.4f}"
                )
            )


        # ==================================================
        # 一个 Epoch 结束
        # ==================================================

        train_loss = (
            running_loss
            /
            len(train_loader)
        )


        # ==================================================
        # Validation
        # ==================================================

        (
            val_loss,
            val_dice,
            val_iou
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device
        )


        print(
            f"\n"
            f"Epoch {epoch + 1}\n"
            f"Train Loss: {train_loss:.4f}\n"
            f"Val Loss:   {val_loss:.4f}\n"
            f"Val Dice:   {val_dice:.4f}\n"
            f"Val IoU:    {val_iou:.4f}\n"
        )


        # ==================================================
        # 保存历史记录
        # ==================================================

        history.append(
            {
                "epoch": epoch + 1,

                "train_loss": train_loss,

                "val_loss": val_loss,

                "val_dice": val_dice,

                "val_iou": val_iou
            }
        )


        # 每个 Epoch 都保存一次 CSV
        pd.DataFrame(
            history
        ).to_csv(
            save_dir
            /
            "history.csv",

            index=False
        )


        # ==================================================
        # 保存最佳模型
        # ==================================================

        if val_dice > best_val_dice:

            best_val_dice = val_dice

            no_improve_epochs = 0


            torch.save(
                {
                    "epoch":
                        epoch + 1,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "val_dice":
                        val_dice,

                    "val_iou":
                        val_iou
                },

                best_model_path
            )


            print(
                "Saved new best model."
            )


        else:

            no_improve_epochs += 1


        # ==================================================
        # Early Stopping
        # ==================================================

        if (
            no_improve_epochs
            >=
            args.patience
        ):

            print(
                "Early stopping triggered."
            )

            break


    # ======================================================
    # 加载最佳模型
    # ======================================================

    checkpoint = torch.load(
        best_model_path,
        map_location=device
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    print(
        "\nLoaded best model:",
        best_model_path
    )


    # ======================================================
    # Test
    # ======================================================

    (
        test_loss,
        test_dice,
        test_iou
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device
    )


    print(
        "\n===== Test Results ====="
    )

    print(
        f"Test Loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Dice: "
        f"{test_dice:.4f}"
    )

    print(
        f"Test IoU: "
        f"{test_iou:.4f}"
    )


    # 保存最终测试指标
    with open(
        save_dir
        /
        "test_metrics.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"Test Loss: "
            f"{test_loss:.4f}\n"
        )

        f.write(
            f"Test Dice: "
            f"{test_dice:.4f}\n"
        )

        f.write(
            f"Test IoU: "
            f"{test_iou:.4f}\n"
        )


# ==========================================================
# 4. 命令行参数
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--data_root",
        type=str,
        required=True
    )


    parser.add_argument(
        "--train_csv",
        type=str,
        default=(
            "data/splits/train.csv"
        )
    )


    parser.add_argument(
        "--val_csv",
        type=str,
        default=(
            "data/splits/val.csv"
        )
    )


    parser.add_argument(
        "--test_csv",
        type=str,
        default=(
            "data/splits/test.csv"
        )
    )


    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )


    parser.add_argument(
        "--batch_size",
        type=int,
        default=8
    )


    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )


    parser.add_argument(
        "--patience",
        type=int,
        default=10
    )


    parser.add_argument(
        "--num_workers",
        type=int,
        default=2
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )


    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/baseline"
    )


    args = parser.parse_args()


    main(
        args
    )