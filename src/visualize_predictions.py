# ==========================================================
# 可视化 U-Net Baseline 在 Test Set 上的预测结果
# ==========================================================

from pathlib import Path

import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from src.dataset import BUSIDataset
from src.model import UNet


# ==========================================
# 配置
# ==========================================

DATA_ROOT = (
    "D:/desktop/work-projects/breast-ultrasound-segmentation/data/Dataset_BUSI_with_GT"
)

TEST_CSV = (
    "data/splits/test.csv"
)

CHECKPOINT_PATH = (
    "results/baseline/best_model.pt"
)

SAVE_DIR = Path(
    "results/baseline/figures/predictions"
)

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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
# Dataset
# ==========================================

test_dataset = BUSIDataset(
    csv_file=TEST_CSV,
    image_size=(256, 256),
    data_root=DATA_ROOT
)


test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 加载模型
# ==========================================

model = UNet().to(
    device
)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


print(
    "Loaded checkpoint from epoch:",
    checkpoint["epoch"]
)


# ==========================================
# 可视化前 10 张 Test 图像
# ==========================================

with torch.no_grad():

    for i, batch in enumerate(
        test_loader
    ):

        if i >= 10:
            break


        image = batch["image"].to(
            device
        )

        mask = batch["mask"].to(
            device
        )


        # ----------------------------------
        # Forward
        # ----------------------------------

        logits = model(
            image
        )


        # ----------------------------------
        # Logits → Probability
        # ----------------------------------

        prob = torch.sigmoid(
            logits
        )


        # ----------------------------------
        # Probability → Binary Mask
        # ----------------------------------

        pred = (
            prob >= 0.5
        ).float()


        # ==================================
        # Tensor → NumPy
        # ==================================

        image_np = (
            image[0, 0]
            .cpu()
            .numpy()
        )

        mask_np = (
            mask[0, 0]
            .cpu()
            .numpy()
        )

        prob_np = (
            prob[0, 0]
            .cpu()
            .numpy()
        )

        pred_np = (
            pred[0, 0]
            .cpu()
            .numpy()
        )


        # ==================================
        # Visualization
        # ==================================

        plt.figure(
            figsize=(16, 4)
        )


        # 原始超声图
        plt.subplot(
            1,
            4,
            1
        )

        plt.imshow(
            image_np,
            cmap="gray"
        )

        plt.title(
            "Ultrasound"
        )

        plt.axis(
            "off"
        )


        # Ground Truth
        plt.subplot(
            1,
            4,
            2
        )

        plt.imshow(
            mask_np,
            cmap="gray"
        )

        plt.title(
            "Ground Truth"
        )

        plt.axis(
            "off"
        )


        # Probability Map
        plt.subplot(
            1,
            4,
            3
        )

        plt.imshow(
            prob_np
        )

        plt.title(
            "Probability"
        )

        plt.axis(
            "off"
        )


        # Prediction
        plt.subplot(
            1,
            4,
            4
        )

        plt.imshow(
            pred_np,
            cmap="gray"
        )

        plt.title(
            "Prediction"
        )

        plt.axis(
            "off"
        )


        plt.tight_layout()


        plt.savefig(
            SAVE_DIR
            /
            f"sample_{i:03d}.png",

            dpi=200
        )


        plt.close()


print(
    "Prediction figures saved to:",
    SAVE_DIR
)