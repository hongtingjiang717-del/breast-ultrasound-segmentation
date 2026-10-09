"""
Learning Rate Range Test for BUSI segmentation.

核心思想
--------
从非常小的学习率开始：

    1e-7

在每个 batch 后指数增加 LR：

    1e-7
    1.2e-7
    ...
    1e-4
    ...
    1e-2

同时记录 Training Loss。

如果使用 differential LR：

    Encoder LR = base LR
    Decoder LR = multiplier * base LR

默认：

    decoder = 3 × encoder

最终根据 smoothed loss 对 log(LR) 的下降速度，
给出一个建议初始学习率。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from src.dataset import BUSIDataset

from src.model_zoo import (
    build_resunet18,
    build_attention_resunet18
)

from src.augmentations import (
    get_train_augmentation
)

from src.loss import DiceBCELoss


# ============================================================
# 1. 模型
# ============================================================

def build_model(
    model_name
):

    if model_name == "resunet18":

        return build_resunet18(
            in_channels=1,
            classes=1,
            encoder_weights="imagenet"
        )


    elif model_name == "attention_resunet18":

        return build_attention_resunet18(
            in_channels=1,
            classes=1,
            encoder_weights="imagenet"
        )


    else:

        raise ValueError(
            "LR Finder currently supports "
            "resunet18 and attention_resunet18."
        )


# ============================================================
# 2. Optimizer
# ============================================================

def build_lr_finder_optimizer(
    model,
    start_lr,
    lr_mode,
    decoder_multiplier,
    weight_decay
):

    if lr_mode == "shared":

        return torch.optim.AdamW(

            model.parameters(),

            lr=start_lr,

            weight_decay=weight_decay
        )


    elif lr_mode == "differential":

        return torch.optim.AdamW(

            [

                {
                    "params":
                        model.encoder.parameters(),

                    "lr":
                        start_lr
                },

                {
                    "params":
                        (
                            list(
                                model.decoder.parameters()
                            )
                            +
                            list(
                                model.segmentation_head.parameters()
                            )
                        ),

                    "lr":
                        (
                            start_lr
                            *
                            decoder_multiplier
                        )
                }

            ],

            weight_decay=weight_decay
        )


    raise ValueError(
        f"Unknown lr_mode: {lr_mode}"
    )


# ============================================================
# 3. Main LR Range Test
# ============================================================

def run_lr_finder(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    start_lr,
    end_lr,
    num_iterations,
    lr_mode,
    decoder_multiplier,
    smooth_beta=0.98
):

    # --------------------------------------------------------
    # 每一步 LR 乘多少倍？
    #
    # 让 start_lr 在 num_iterations 后
    # 指数增长到 end_lr。
    # --------------------------------------------------------

    lr_multiplier = (

        end_lr
        /
        start_lr

    ) ** (
        1
        /
        max(
            num_iterations - 1,
            1
        )
    )


    current_base_lr = start_lr


    history = []


    avg_loss = 0.0

    best_loss = float(
        "inf"
    )


    iteration = 0


    model.train()


    # ========================================================
    # 可能需要循环多轮 DataLoader，
    # 直到达到 num_iterations。
    # ========================================================

    while iteration < num_iterations:


        for batch in train_loader:


            if iteration >= num_iterations:

                break


            image = batch[
                "image"
            ].to(
                device
            )


            mask = batch[
                "mask"
            ].to(
                device
            )


            # =================================================
            # 设置当前 LR
            # =================================================

            if lr_mode == "shared":

                optimizer.param_groups[
                    0
                ][
                    "lr"
                ] = current_base_lr


                encoder_lr = (
                    current_base_lr
                )

                decoder_lr = (
                    current_base_lr
                )


            else:

                encoder_lr = (
                    current_base_lr
                )

                decoder_lr = (

                    current_base_lr
                    *
                    decoder_multiplier

                )


                optimizer.param_groups[
                    0
                ][
                    "lr"
                ] = encoder_lr


                optimizer.param_groups[
                    1
                ][
                    "lr"
                ] = decoder_lr


            # =================================================
            # Forward / Backward
            # =================================================

            optimizer.zero_grad()


            logits = model(
                image
            )


            loss = criterion(
                logits,
                mask
            )


            loss.backward()


            optimizer.step()


            raw_loss = loss.item()


            # =================================================
            # Exponential moving average
            #
            # 原始 batch loss 会比较抖，
            # 用 EMA 平滑。
            # =================================================

            avg_loss = (

                smooth_beta
                *
                avg_loss

                +

                (
                    1
                    -
                    smooth_beta
                )
                *
                raw_loss

            )


            smoothed_loss = (

                avg_loss
                /
                (
                    1
                    -
                    smooth_beta
                    **
                    (
                        iteration
                        +
                        1
                    )
                )

            )


            # =================================================
            # 记录
            # =================================================

            history.append(
                {

                    "iteration":
                        iteration,

                    "encoder_lr":
                        encoder_lr,

                    "decoder_lr":
                        decoder_lr,

                    "raw_loss":
                        raw_loss,

                    "smoothed_loss":
                        smoothed_loss
                }
            )


            # =================================================
            # 如果 loss 已经明显爆炸，
            # 提前停止。
            # =================================================

            if smoothed_loss < best_loss:

                best_loss = (
                    smoothed_loss
                )


            if (

                iteration > 10

                and

                smoothed_loss
                >
                4.0
                *
                best_loss

            ):

                print(
                    "\nLoss diverged. "
                    "Stopping LR range test."
                )

                return pd.DataFrame(
                    history
                )


            # =================================================
            # LR 指数增长
            # =================================================

            current_base_lr *= (
                lr_multiplier
            )


            iteration += 1


            if (
                iteration
                %
                10
                ==
                0
            ):

                print(

                    f"Iteration "
                    f"{iteration}/"
                    f"{num_iterations} | "

                    f"Encoder LR "
                    f"{encoder_lr:.2e} | "

                    f"Decoder LR "
                    f"{decoder_lr:.2e} | "

                    f"Loss "
                    f"{smoothed_loss:.4f}"

                )


    return pd.DataFrame(
        history
    )


# ============================================================
# 4. 自动建议 LR
# ============================================================

def suggest_learning_rate(
    history_df
):
    """
    根据：

        smoothed loss

    对：

        log10(encoder LR)

    的梯度，

    找 loss 下降最快的位置。

    这个位置比直接拿 minimum loss 更保守，
    也更适合作为训练初始 LR。
    """

    df = history_df.copy()


    # 去掉最开始几个极不稳定点
    if len(df) > 20:

        df = df.iloc[
            10:
        ].copy()


    log_lr = np.log10(
        df[
            "encoder_lr"
        ].values
    )


    loss = df[
        "smoothed_loss"
    ].values


    # --------------------------------------------------------
    # 数值梯度
    # --------------------------------------------------------

    gradient = np.gradient(
        loss,
        log_lr
    )


    # --------------------------------------------------------
    # 梯度最负：
    #
    # loss 下降最快。
    # --------------------------------------------------------

    best_position = np.argmin(
        gradient
    )


    suggested_encoder_lr = df[
        "encoder_lr"
    ].iloc[
        best_position
    ]


    return float(
        suggested_encoder_lr
    )


# ============================================================
# 5. 绘图
# ============================================================

def plot_lr_finder(
    history_df,
    suggested_lr,
    save_path
):

    plt.figure(
        figsize=(9, 6)
    )


    plt.plot(

        history_df[
            "encoder_lr"
        ],

        history_df[
            "smoothed_loss"
        ],

        linewidth=2
    )


    plt.axvline(

        suggested_lr,

        linestyle="--",

        linewidth=1.5,

        label=(
            f"Suggested encoder LR = "
            f"{suggested_lr:.2e}"
        )
    )


    plt.xscale(
        "log"
    )


    plt.xlabel(
        "Encoder / Base Learning Rate"
    )


    plt.ylabel(
        "Smoothed Training Loss"
    )


    plt.title(
        "Learning Rate Range Test"
    )


    plt.legend()


    plt.grid(
        alpha=0.25
    )


    plt.tight_layout()


    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# 6. Main
# ============================================================

def main(
    args
):

    device = torch.device(

        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"

    )


    print(
        "Using device:",
        device
    )


    # ========================================================
    # Dataset
    # ========================================================

    train_dataset = BUSIDataset(

        csv_file=args.train_csv,

        image_size=(256, 256),

        data_root=args.data_root,

        transform=(
            get_train_augmentation(
                256
            )
        )
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=args.batch_size,

        shuffle=True,

        num_workers=args.num_workers,

        pin_memory=(
            device.type
            ==
            "cuda"
        )
    )


    # ========================================================
    # Model
    # ========================================================

    model = build_model(
        args.model
    ).to(
        device
    )


    # ========================================================
    # Loss
    # ========================================================

    criterion = DiceBCELoss()


    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = (
        build_lr_finder_optimizer(

            model=model,

            start_lr=args.start_lr,

            lr_mode=args.lr_mode,

            decoder_multiplier=(
                args.decoder_multiplier
            ),

            weight_decay=(
                args.weight_decay
            )
        )
    )


    # ========================================================
    # Run
    # ========================================================

    history_df = run_lr_finder(

        model=model,

        train_loader=train_loader,

        criterion=criterion,

        optimizer=optimizer,

        device=device,

        start_lr=args.start_lr,

        end_lr=args.end_lr,

        num_iterations=(
            args.num_iterations
        ),

        lr_mode=args.lr_mode,

        decoder_multiplier=(
            args.decoder_multiplier
        )
    )


    # ========================================================
    # Suggested LR
    # ========================================================

    suggested_encoder_lr = (
        suggest_learning_rate(
            history_df
        )
    )


    if args.lr_mode == "shared":

        suggested_decoder_lr = (
            suggested_encoder_lr
        )


    else:

        suggested_decoder_lr = (

            suggested_encoder_lr

            *
            args.decoder_multiplier

        )


    print(
        "\n"
        "========================================"
    )

    print(
        "LR Finder Result"
    )

    print(
        "========================================"
    )


    print(

        f"Suggested Encoder/Base LR: "
        f"{suggested_encoder_lr:.2e}"

    )


    print(

        f"Suggested Decoder LR: "
        f"{suggested_decoder_lr:.2e}"

    )


    # ========================================================
    # 保存
    # ========================================================

    output_dir = Path(
        args.output_dir
    )


    output_dir.mkdir(

        parents=True,

        exist_ok=True
    )


    history_df.to_csv(

        output_dir /
        "lr_finder_history.csv",

        index=False
    )


    plot_lr_finder(

        history_df,

        suggested_encoder_lr,

        output_dir /
        "lr_finder_curve.png"
    )


    with open(

        output_dir /
        "suggested_lr.txt",

        "w",

        encoding="utf-8"

    ) as f:

        f.write(

            f"model={args.model}\n"

            f"lr_mode={args.lr_mode}\n"

            f"encoder_lr="
            f"{suggested_encoder_lr:.10f}\n"

            f"decoder_lr="
            f"{suggested_decoder_lr:.10f}\n"

        )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(

        "--model",

        type=str,

        required=True,

        choices=[
            "resunet18",
            "attention_resunet18"
        ]
    )


    parser.add_argument(

        "--lr_mode",

        type=str,

        default="differential",

        choices=[
            "shared",
            "differential"
        ]
    )


    parser.add_argument(

        "--decoder_multiplier",

        type=float,

        default=3.0
    )


    parser.add_argument(

        "--start_lr",

        type=float,

        default=1e-7
    )


    parser.add_argument(

        "--end_lr",

        type=float,

        default=1e-2
    )


    parser.add_argument(

        "--num_iterations",

        type=int,

        default=120
    )


    parser.add_argument(

        "--weight_decay",

        type=float,

        default=1e-4
    )


    parser.add_argument(

        "--data_root",

        type=str,

        required=True
    )


    parser.add_argument(

        "--train_csv",

        type=str,

        default="data/splits/train.csv"
    )


    parser.add_argument(

        "--batch_size",

        type=int,

        default=16
    )


    parser.add_argument(

        "--num_workers",

        type=int,

        default=4
    )


    parser.add_argument(

        "--output_dir",

        type=str,

        required=True
    )


    args = parser.parse_args()


    main(
        args
    )