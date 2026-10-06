# ==========================================================
# 绘制 U-Net Baseline 训练曲线
# ==========================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. 路径
# ==========================================

history_path = Path(
    "results/baseline/history.csv"
)

save_dir = Path(
    "results/baseline/figures"
)

save_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# 2. 读取训练记录
# ==========================================

df = pd.read_csv(
    history_path
)

print(df.head())
print()
print("Total epochs:", len(df))


# ==========================================
# 3. Loss 曲线
# ==========================================

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    df["epoch"],
    df["train_loss"],
    label="Train Loss"
)

plt.plot(
    df["epoch"],
    df["val_loss"],
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training and Validation Loss"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    save_dir / "loss_curve.png",
    dpi=300
)

plt.show()


# ==========================================
# 4. Validation Dice 曲线
# ==========================================

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    df["epoch"],
    df["val_dice"],
    label="Validation Dice"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Dice Score"
)

plt.title(
    "Validation Dice"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    save_dir / "val_dice_curve.png",
    dpi=300
)

plt.show()


# ==========================================
# 5. Validation IoU 曲线
# ==========================================

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    df["epoch"],
    df["val_iou"],
    label="Validation IoU"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "IoU"
)

plt.title(
    "Validation IoU"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    save_dir / "val_iou_curve.png",
    dpi=300
)

plt.show()