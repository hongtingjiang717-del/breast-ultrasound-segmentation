"""
三模型逐病例比较脚本

比较：
1. Scratch U-Net
2. ResNet18-U-Net
3. Attention ResNet18-U-Net

功能：
1. 对同一个 Test Set 进行推理
2. 对每张图计算 Dice / IoU
3. 保存 per-sample 结果
4. 自动寻找最值得展示的病例
5. 生成三模型定性比较图
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from src.dataset import BUSIDataset

from src.model import UNet

from src.model_zoo import (
    build_resunet18,
    build_attention_resunet18
)

from src.augmentations import (
    get_eval_augmentation
)


# ============================================================
# 1. 加载 checkpoint
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path,
    device
):
    """
    将保存的 best_model.pt 参数装载到模型中。

    参数
    ----------
    model:
        已经创建好的模型结构

    checkpoint_path:
        best_model.pt 的位置

    device:
        cuda 或 cpu
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # --------------------------------------------------------
    # 我们自己的 train.py 保存方式一般是：
    #
    # checkpoint = {
    #     "model_state_dict": ...
    # }
    #
    # 但这里同时兼容其他保存格式。
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]


    elif (
        isinstance(checkpoint, dict)
        and
        "state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]


    else:

        # 如果 pt 文件本身就是 state_dict
        state_dict = checkpoint


    # --------------------------------------------------------
    # 如果模型以前经过 DataParallel，
    # 参数名字可能出现：
    #
    # module.xxx
    #
    # 这里自动删除 module. 前缀
    # --------------------------------------------------------

    clean_state_dict = {}

    for key, value in state_dict.items():

        new_key = key.replace(
            "module.",
            ""
        )

        clean_state_dict[
            new_key
        ] = value


    model.load_state_dict(
        clean_state_dict
    )


    return model


# ============================================================
# 2. 单张图 Dice / IoU
# ============================================================

def calculate_metrics(
    pred,
    target,
    smooth=1e-6
):
    """
    pred 和 target：
        shape = [1, 1, H, W]

    并且都是二值 Mask。
    """

    intersection = (
        pred * target
    ).sum()


    pred_area = pred.sum()

    target_area = target.sum()


    # Dice
    dice = (
        2.0 * intersection
        +
        smooth
    ) / (
        pred_area
        +
        target_area
        +
        smooth
    )


    # IoU
    union = (
        pred_area
        +
        target_area
        -
        intersection
    )


    iou = (
        intersection
        +
        smooth
    ) / (
        union
        +
        smooth
    )


    return (
        dice.item(),
        iou.item()
    )


# ============================================================
# 3. 单个模型推理
# ============================================================

def predict_one_model(
    model,
    image,
    threshold=0.5
):
    """
    输入：
        image
        [1, 1, 256, 256]

    返回：
        probability map
        binary prediction
    """

    logits = model(
        image
    )


    # Raw Logits → Probability
    probability = torch.sigmoid(
        logits
    )


    # Probability → Binary Mask
    prediction = (
        probability >= threshold
    ).float()


    return (
        probability,
        prediction
    )


# ============================================================
# 4. 绘制 Overlay
# ============================================================

def draw_overlay(
    ax,
    image,
    gt,
    pred,
    title
):
    """
    在原始超声图上同时画：

    实线：
        Ground Truth

    虚线：
        Prediction
    """

    ax.imshow(
        image,
        cmap="gray"
    )


    # Ground Truth contour
    if gt.max() > 0:

        ax.contour(
            gt,
            levels=[0.5],
            linewidths=1.5,
            linestyles="-"
        )


    # Prediction contour
    if pred.max() > 0:

        ax.contour(
            pred,
            levels=[0.5],
            linewidths=1.5,
            linestyles="--"
        )


    ax.set_title(
        title
    )

    ax.axis(
        "off"
    )


# ============================================================
# 5. 保存一个病例的模型比较图
# ============================================================

def save_case_figure(
    sample,
    reason,
    save_path
):
    """
    每个病例做成 2 × 4 的最终展示图。
    """

    image = sample[
        "image"
    ]

    gt = sample[
        "gt"
    ]

    unet = sample[
        "unet_pred"
    ]

    resnet = sample[
        "resnet_pred"
    ]

    attention = sample[
        "attention_pred"
    ]


    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16, 8)
    )


    # ========================================================
    # 第一行：原图 + GT + 三模型 Prediction
    # ========================================================

    axes[0, 0].imshow(
        image,
        cmap="gray"
    )

    axes[0, 0].set_title(
        "Ultrasound"
    )

    axes[0, 0].axis(
        "off"
    )


    axes[0, 1].imshow(
        gt,
        cmap="gray"
    )

    axes[0, 1].set_title(
        "Ground Truth"
    )

    axes[0, 1].axis(
        "off"
    )


    axes[0, 2].imshow(
        unet,
        cmap="gray"
    )

    axes[0, 2].set_title(
        f"U-Net\n"
        f"Dice={sample['unet_dice']:.3f}"
    )

    axes[0, 2].axis(
        "off"
    )


    axes[0, 3].imshow(
        resnet,
        cmap="gray"
    )

    axes[0, 3].set_title(
        f"ResNet18-U-Net\n"
        f"Dice={sample['resnet_dice']:.3f}"
    )

    axes[0, 3].axis(
        "off"
    )


    # ========================================================
    # 第二行
    # ========================================================

    axes[1, 0].imshow(
        attention,
        cmap="gray"
    )

    axes[1, 0].set_title(
        f"Attention-ResUNet\n"
        f"Dice={sample['attention_dice']:.3f}"
    )

    axes[1, 0].axis(
        "off"
    )


    draw_overlay(
        axes[1, 1],
        image,
        gt,
        unet,
        "U-Net Overlay"
    )


    draw_overlay(
        axes[1, 2],
        image,
        gt,
        resnet,
        "ResNet18-U-Net Overlay"
    )


    draw_overlay(
        axes[1, 3],
        image,
        gt,
        attention,
        "Attention Overlay"
    )


    # ========================================================
    # 整张图标题
    # ========================================================

    fig.suptitle(

        f"{reason}\n"
        f"{sample['image_name']}",

        fontsize=14
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

def main(args):

    # ========================================================
    # Device
    # ========================================================

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
    # 保存目录
    # ========================================================

    save_dir = Path(
        args.save_dir
    )


    figure_dir = (
        save_dir
        /
        "case_comparisons"
    )


    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    figure_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # 读取 Test CSV
    # ========================================================

    test_df = pd.read_csv(
        args.test_csv
    )


    print(
        "Test samples:",
        len(test_df)
    )


    # ========================================================
    # 所有模型统一使用完全相同的 Test preprocessing
    #
    # 非常重要：
    #
    # Test 不允许任何随机数据增强。
    # ========================================================

    eval_transform = (
        get_eval_augmentation(
            image_size=256
        )
    )


    test_dataset = BUSIDataset(

        csv_file=args.test_csv,

        image_size=(256, 256),

        data_root=args.data_root,

        transform=eval_transform
    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=1,

        shuffle=False,

        num_workers=0
    )


    # ========================================================
    # 创建三个模型
    # ========================================================

    print(
        "\nBuilding models..."
    )


    # --------------------------------------------------------
    # Model 1
    # Scratch U-Net
    # --------------------------------------------------------

    unet = UNet()


    # --------------------------------------------------------
    # Model 2
    # ResNet18-U-Net
    #
    # encoder_weights=None：
    # 推理阶段不需要重新下载 ImageNet 参数。
    # --------------------------------------------------------

    resnet = build_resunet18(

        in_channels=1,

        classes=1,

        encoder_weights=None
    )


    # --------------------------------------------------------
    # Model 3
    # Attention ResNet18-U-Net
    # --------------------------------------------------------

    attention = (
        build_attention_resunet18(

            in_channels=1,

            classes=1,

            encoder_weights=None
        )
    )


    # ========================================================
    # 加载训练好的权重
    # ========================================================

    print(
        "Loading checkpoints..."
    )


    unet = load_checkpoint(

        unet,

        args.unet_checkpoint,

        device
    )


    resnet = load_checkpoint(

        resnet,

        args.resnet_checkpoint,

        device
    )


    attention = load_checkpoint(

        attention,

        args.attention_checkpoint,

        device
    )


    # ========================================================
    # GPU / CPU
    # ========================================================

    unet = unet.to(
        device
    )

    resnet = resnet.to(
        device
    )

    attention = attention.to(
        device
    )


    # ========================================================
    # Evaluation Mode
    # ========================================================

    unet.eval()

    resnet.eval()

    attention.eval()


    print(
        "All models loaded successfully."
    )


    # ========================================================
    # 保存每张图的数值结果
    # ========================================================

    rows = []


    # 同时暂存图像和 prediction
    #
    # Test Set 只有 98 张，
    # 内存完全足够。
    case_data = {}


    # ========================================================
    # 正式推理
    # ========================================================

    with torch.no_grad():

        for index, batch in enumerate(
            test_loader
        ):

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
            # 三模型预测
            # =================================================

            (
                unet_prob,
                unet_pred
            ) = predict_one_model(

                unet,
                image,
                args.threshold
            )


            (
                resnet_prob,
                resnet_pred
            ) = predict_one_model(

                resnet,
                image,
                args.threshold
            )


            (
                attention_prob,
                attention_pred
            ) = predict_one_model(

                attention,
                image,
                args.threshold
            )


            # =================================================
            # Dice / IoU
            # =================================================

            (
                unet_dice,
                unet_iou
            ) = calculate_metrics(

                unet_pred,
                mask
            )


            (
                resnet_dice,
                resnet_iou
            ) = calculate_metrics(

                resnet_pred,
                mask
            )


            (
                attention_dice,
                attention_iou
            ) = calculate_metrics(

                attention_pred,
                mask
            )


            # =================================================
            # Test CSV 信息
            # =================================================

            row = test_df.iloc[
                index
            ]


            image_name = row[
                "image_name"
            ]


            # =================================================
            # 模型差异
            # =================================================

            resnet_gain = (
                resnet_dice
                -
                unet_dice
            )


            attention_change = (
                attention_dice
                -
                resnet_dice
            )


            mean_dice = np.mean(
                [
                    unet_dice,
                    resnet_dice,
                    attention_dice
                ]
            )


            # =================================================
            # 保存 CSV 数据
            # =================================================

            result = {

                "index":
                    index,

                "image_name":
                    image_name,

                "class":
                    row.get(
                        "class",
                        ""
                    ),

                "size_group":
                    row.get(
                        "size_group",
                        ""
                    ),

                "lesion_area_ratio":
                    row.get(
                        "lesion_area_ratio",
                        np.nan
                    ),

                "unet_dice":
                    unet_dice,

                "unet_iou":
                    unet_iou,

                "resnet_dice":
                    resnet_dice,

                "resnet_iou":
                    resnet_iou,

                "attention_dice":
                    attention_dice,

                "attention_iou":
                    attention_iou,

                "resnet_gain_vs_unet":
                    resnet_gain,

                "attention_change_vs_resnet":
                    attention_change,

                "mean_dice":
                    mean_dice
            }


            rows.append(
                result
            )


            # =================================================
            # 保存图像，用于后续自动出图
            # =================================================

            case_data[
                index
            ] = {

                **result,

                "image":
                    image[
                        0,
                        0
                    ]
                    .cpu()
                    .numpy(),

                "gt":
                    mask[
                        0,
                        0
                    ]
                    .cpu()
                    .numpy(),

                "unet_pred":
                    unet_pred[
                        0,
                        0
                    ]
                    .cpu()
                    .numpy(),

                "resnet_pred":
                    resnet_pred[
                        0,
                        0
                    ]
                    .cpu()
                    .numpy(),

                "attention_pred":
                    attention_pred[
                        0,
                        0
                    ]
                    .cpu()
                    .numpy(),

                "unet_prob":
                    unet_prob[
                        0,
                        0
                    ]
                    .cpu()
                    .numpy(),

                "resnet_prob":
                    resnet_prob[
                        0,
                        0
                    ]
                    .cpu()
                    .numpy(),

                "attention_prob":
                    attention_prob[
                        0,
                        0
                    ]
                    .cpu()
                    .numpy()
            }


            if (
                (index + 1)
                % 10
                ==
                0
            ):

                print(
                    f"Processed "
                    f"{index + 1}/"
                    f"{len(test_dataset)}"
                )


    # ========================================================
    # 结果表
    # ========================================================

    result_df = pd.DataFrame(
        rows
    )


    csv_path = (
        save_dir
        /
        "per_sample_model_comparison.csv"
    )


    result_df.to_csv(
        csv_path,
        index=False
    )


    # ========================================================
    # 打印标准化后的总体指标
    # ========================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "Standardized Test Results"
    )

    print(
        "========================================"
    )


    print(
        "\nU-Net"
    )

    print(
        f"Dice: "
        f"{result_df['unet_dice'].mean():.4f}"
    )

    print(
        f"IoU : "
        f"{result_df['unet_iou'].mean():.4f}"
    )


    print(
        "\nResNet18-U-Net"
    )

    print(
        f"Dice: "
        f"{result_df['resnet_dice'].mean():.4f}"
    )

    print(
        f"IoU : "
        f"{result_df['resnet_iou'].mean():.4f}"
    )


    print(
        "\nAttention-ResNet18-U-Net"
    )

    print(
        f"Dice: "
        f"{result_df['attention_dice'].mean():.4f}"
    )

    print(
        f"IoU : "
        f"{result_df['attention_iou'].mean():.4f}"
    )


    # ========================================================
    # 自动寻找四类代表病例
    # ========================================================


    # --------------------------------------------------------
    # Case A：
    # ResNet 相比 U-Net 提升最大
    # --------------------------------------------------------

    best_improvement_idx = (
        result_df[
            "resnet_gain_vs_unet"
        ].idxmax()
    )


    # --------------------------------------------------------
    # Case B：
    # Attention 相比 ResNet 下降最大
    # --------------------------------------------------------

    attention_degradation_idx = (
        result_df[
            "attention_change_vs_resnet"
        ].idxmin()
    )


    # --------------------------------------------------------
    # Case C：
    # 三模型平均 Dice 最低
    #
    # 即大家都觉得困难的病例
    # --------------------------------------------------------

    failure_idx = (
        result_df[
            "mean_dice"
        ].idxmin()
    )


    # --------------------------------------------------------
    # Case D：
    # 典型病例
    #
    # 找平均 Dice 最接近中位数的样本
    # --------------------------------------------------------

    median_dice = (
        result_df[
            "mean_dice"
        ].median()
    )


    typical_idx = (

        (
            result_df[
                "mean_dice"
            ]
            -
            median_dice
        )
        .abs()
        .idxmin()

    )


    selected_cases = {

        "01_best_resnet_improvement":
            best_improvement_idx,

        "02_typical_case":
            typical_idx,

        "03_common_failure":
            failure_idx,

        "04_attention_degradation":
            attention_degradation_idx
    }


    # ========================================================
    # 保存 selected case 信息
    # ========================================================

    selected_rows = []


    for (
        reason,
        case_idx
    ) in selected_cases.items():

        sample = case_data[
            case_idx
        ]


        selected_rows.append(
            {
                "reason":
                    reason,

                "index":
                    case_idx,

                "image_name":
                    sample[
                        "image_name"
                    ],

                "unet_dice":
                    sample[
                        "unet_dice"
                    ],

                "resnet_dice":
                    sample[
                        "resnet_dice"
                    ],

                "attention_dice":
                    sample[
                        "attention_dice"
                    ],

                "resnet_gain_vs_unet":
                    sample[
                        "resnet_gain_vs_unet"
                    ],

                "attention_change_vs_resnet":
                    sample[
                        "attention_change_vs_resnet"
                    ]
            }
        )


        # ====================================================
        # 生成最终展示图
        # ====================================================

        save_path = (

            figure_dir
            /
            f"{reason}.png"

        )


        save_case_figure(

            sample=sample,

            reason=reason.replace(
                "_",
                " "
            ),

            save_path=save_path
        )


    selection_df = pd.DataFrame(
        selected_rows
    )


    selection_df.to_csv(

        save_dir
        /
        "selected_cases.csv",

        index=False
    )


    print(
        "\n"
        "========================================"
    )

    print(
        "Selected Cases"
    )

    print(
        "========================================"
    )


    print(
        selection_df.to_string(
            index=False
        )
    )


    print(
        "\nPer-sample results:"
    )

    print(
        csv_path
    )


    print(
        "\nCase figures:"
    )

    print(
        figure_dir
    )


# ============================================================
# 7. Command Line Arguments
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    parser.add_argument(

        "--data_root",

        type=str,

        required=True,

        help=(
            "BUSI Dataset_BUSI_with_GT "
            "root directory"
        )
    )


    parser.add_argument(

        "--test_csv",

        type=str,

        default=(
            "data/splits/test.csv"
        )
    )


    # --------------------------------------------------------
    # Three checkpoints
    # --------------------------------------------------------

    parser.add_argument(

        "--unet_checkpoint",

        type=str,

        default=(
            "results/"
            "unet_baseline_final/"
            "best_model.pt"
        )
    )


    parser.add_argument(

        "--resnet_checkpoint",

        type=str,

        default=(
            "results/"
            "resunet18_aug/"
            "best_model.pt"
        )
    )


    parser.add_argument(

        "--attention_checkpoint",

        type=str,

        default=(
            "results/"
            "attention_resunet18_aug/"
            "best_model.pt"
        )
    )


    # --------------------------------------------------------
    # Segmentation threshold
    # --------------------------------------------------------

    parser.add_argument(

        "--threshold",

        type=float,

        default=0.5
    )


    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(

        "--save_dir",

        type=str,

        default=(
            "results/final/"
            "prediction_analysis"
        )
    )


    args = parser.parse_args()


    main(
        args
    )