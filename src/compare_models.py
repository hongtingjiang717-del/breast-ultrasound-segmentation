"""
三种乳腺超声分割模型的最终结果比较。

比较模型：
1. Scratch U-Net
2. ResNet18 U-Net
3. Attention ResNet18 U-Net

输出：
1. model_comparison.csv
2. Dice 柱状图
3. IoU 柱状图
4. Test Loss 柱状图
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 创建最终结果保存目录
# ============================================================

SAVE_DIR = Path(
    "results/final"
)

FIGURE_DIR = (
    SAVE_DIR / "figures"
)

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 输入三个模型的真实测试结果
# ============================================================
#
# 注意：
# 这里全部来自我们真正训练得到的 Test Set 结果。
# 不是 GitHub 原作者结果。
#
# ============================================================

results = [

    {
        "Model": "U-Net",
        "Test Loss": 0.4544,
        "Test Dice": 0.6983,
        "Test IoU": 0.5795
    },

    {
        "Model": "ResNet18-U-Net",
        "Test Loss": 0.2935,
        "Test Dice": 0.8310,
        "Test IoU": 0.7377
    },

    {
        "Model": "Attention-ResNet18-U-Net",
        "Test Loss": 0.3462,
        "Test Dice": 0.7960,
        "Test IoU": 0.7117
    }

]


# ============================================================
# 3. 转成 DataFrame
# ============================================================

df = pd.DataFrame(
    results
)


print(
    "\n===== Final Model Comparison ====="
)

print(
    df.to_string(index=False)
)


# ============================================================
# 4. 保存 CSV
# ============================================================

csv_path = (
    SAVE_DIR /
    "model_comparison.csv"
)

df.to_csv(
    csv_path,
    index=False
)

print(
    "\nSaved:",
    csv_path
)


# ============================================================
# 5. Dice 柱状图
# ============================================================

plt.figure(
    figsize=(8, 5)
)

bars = plt.bar(
    df["Model"],
    df["Test Dice"]
)


plt.ylabel(
    "Dice Score"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Test Dice Comparison"
)

plt.ylim(
    0,
    1
)


# 在柱子上显示具体数字
for bar, value in zip(
    bars,
    df["Test Dice"]
):

    plt.text(

        bar.get_x()
        +
        bar.get_width() / 2,

        value + 0.015,

        f"{value:.4f}",

        ha="center"
    )


plt.tight_layout()


plt.savefig(
    FIGURE_DIR /
    "model_dice_comparison.png",

    dpi=300
)

plt.close()


# ============================================================
# 6. IoU 柱状图
# ============================================================

plt.figure(
    figsize=(8, 5)
)

bars = plt.bar(
    df["Model"],
    df["Test IoU"]
)


plt.ylabel(
    "IoU Score"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Test IoU Comparison"
)

plt.ylim(
    0,
    1
)


for bar, value in zip(
    bars,
    df["Test IoU"]
):

    plt.text(

        bar.get_x()
        +
        bar.get_width() / 2,

        value + 0.015,

        f"{value:.4f}",

        ha="center"
    )


plt.tight_layout()


plt.savefig(
    FIGURE_DIR /
    "model_iou_comparison.png",

    dpi=300
)

plt.close()


# ============================================================
# 7. Loss 柱状图
# ============================================================

plt.figure(
    figsize=(8, 5)
)

bars = plt.bar(
    df["Model"],
    df["Test Loss"]
)


plt.ylabel(
    "Test Loss"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Test Loss Comparison"
)


for bar, value in zip(
    bars,
    df["Test Loss"]
):

    plt.text(

        bar.get_x()
        +
        bar.get_width() / 2,

        value + 0.01,

        f"{value:.4f}",

        ha="center"
    )


plt.tight_layout()


plt.savefig(
    FIGURE_DIR /
    "model_loss_comparison.png",

    dpi=300
)

plt.close()


# ============================================================
# 8. 计算 ResUNet 相对 Baseline 的提升
# ============================================================

baseline = df[
    df["Model"] == "U-Net"
].iloc[0]


resunet = df[
    df["Model"] == "ResNet18-U-Net"
].iloc[0]


dice_gain = (
    resunet["Test Dice"]
    -
    baseline["Test Dice"]
)


iou_gain = (
    resunet["Test IoU"]
    -
    baseline["Test IoU"]
)


relative_dice_gain = (
    dice_gain
    /
    baseline["Test Dice"]
    *
    100
)


relative_iou_gain = (
    iou_gain
    /
    baseline["Test IoU"]
    *
    100
)


print(
    "\n===== ResNet18-U-Net vs U-Net ====="
)


print(
    f"Dice absolute gain: "
    f"{dice_gain:.4f}"
)


print(
    f"Dice relative gain: "
    f"{relative_dice_gain:.2f}%"
)


print(
    f"IoU absolute gain: "
    f"{iou_gain:.4f}"
)


print(
    f"IoU relative gain: "
    f"{relative_iou_gain:.2f}%"
)


print(
    "\nFigures saved to:",
    FIGURE_DIR
)