import argparse
from pathlib import Path

import pandas as pd
import torch

from torch.utils.data import DataLoader

from src.dataset import BUSIDataset
from src.model import UNet


# ==========================================================
# 计算单张图像的 Dice 和 IoU
# ==========================================================

def calculate_metrics(
    pred,
    target,
    smooth=1e-6
):
    """
    pred:
        二值预测 Mask
        shape = [B, 1, H, W]

    target:
        Ground Truth
        shape = [B, 1, H, W]
    """

    # ------------------------------------------
    # 对每张图片分别计算
    # ------------------------------------------

    dims = (
        1,
        2,
        3
    )


    # Prediction 与 Ground Truth 的交集
    intersection = (
        pred * target
    ).sum(
        dim=dims
    )


    # Prediction 面积
    pred_area = pred.sum(
        dim=dims
    )


    # Ground Truth 面积
    target_area = target.sum(
        dim=dims
    )


    # ------------------------------------------
    # Dice
    # ------------------------------------------

    dice = (
        2 * intersection + smooth
    ) / (
        pred_area
        +
        target_area
        +
        smooth
    )


    # ------------------------------------------
    # IoU
    # ------------------------------------------

    union = (
        pred_area
        +
        target_area
        -
        intersection
    )


    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )


    return dice, iou


# 遍历整个test set
def evaluate_test_set(
    model,
    loader,
    device,
    dataframe,
    threshold=0.5
):

    model.eval()

    results = []

    sample_index = 0


    with torch.no_grad():

        for batch in loader:

            # ======================================
            # 1. 取数据
            # ======================================

            images = batch["image"].to(
                device
            )

            masks = batch["mask"].to(
                device
            )


            # ======================================
            # 2. Forward
            # ======================================

            logits = model(
                images
            )


            # ======================================
            # 3. Logits → Probability
            # ======================================

            probs = torch.sigmoid(
                logits
            )


            # ======================================
            # 4. Probability → Binary Mask
            # ======================================

            preds = (
                probs >= threshold
            ).float()


            # ======================================
            # 5. 每张图片分别计算 Dice / IoU
            # ======================================

            dice, iou = calculate_metrics(
                preds,
                masks
            )


            batch_size = images.shape[0]


            # ======================================
            # 6. 保存每张图片的结果
            # ======================================

            for i in range(
                batch_size
            ):

                row = dataframe.iloc[
                    sample_index
                ]


                results.append(
                    {
                        "image_name":
                            row["image_name"],

                        "class":
                            row["class"],

                        "size_group":
                            row["size_group"],

                        "lesion_area_ratio":
                            row["lesion_area_ratio"],

                        "dice":
                            dice[i].item(),

                        "iou":
                            iou[i].item()
                    }
                )


                sample_index += 1


    return pd.DataFrame(
        results
    )


#主程序
def main(args):

    # ==========================================
    # Device
    # ==========================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "Device:",
        device
    )


    # ==========================================
    # Test CSV
    # ==========================================

    test_df = pd.read_csv(
        args.test_csv
    )


    # ==========================================
    # Dataset
    # ==========================================

    dataset = BUSIDataset(
        csv_file=args.test_csv,
        image_size=(256, 256),
        data_root=args.data_root
    )


    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )


    # ==========================================
    # Model
    # ==========================================

    model = UNet().to(
        device
    )


    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    print(
        "Loaded best model from epoch:",
        checkpoint["epoch"]
    )


    # ==========================================
    # Evaluation
    # ==========================================

    result_df = evaluate_test_set(
        model=model,
        loader=loader,
        device=device,
        dataframe=test_df,
        threshold=args.threshold
    )


    # ==========================================
    # 保存结果
    # ==========================================

    save_path = Path(
        args.output
    )


    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    result_df.to_csv(
        save_path,
        index=False
    )


    # ==========================================
    # Overall
    # ==========================================

    print()
    print(
        "===== Overall Test Results ====="
    )

    print(
        "Mean Dice:",
        result_df["dice"].mean()
    )

    print(
        "Mean IoU:",
        result_df["iou"].mean()
    )


    # ==========================================
    # 按类别
    # ==========================================

    print()
    print(
        "===== By Class ====="
    )

    print(
        result_df.groupby(
            "class"
        )[
            ["dice", "iou"]
        ].mean()
    )


    # ==========================================
    # 按病灶大小
    # ==========================================

    print()
    print(
        "===== By Lesion Size ====="
    )

    print(
        result_df.groupby(
            "size_group"
        )[
            ["dice", "iou"]
        ].mean()
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--data_root",
        required=True
    )


    parser.add_argument(
        "--test_csv",
        default="data/splits/test.csv"
    )


    parser.add_argument(
        "--checkpoint",
        default=(
            "results/baseline/"
            "best_model.pt"
        )
    )


    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )


    parser.add_argument(
        "--batch_size",
        type=int,
        default=8
    )


    parser.add_argument(
        "--output",
        default=(
            "results/baseline/"
            "test_per_sample.csv"
        )
    )


    args = parser.parse_args()


    main(
        args
    )