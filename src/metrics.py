"""
增加dice、iou指标
Dice：
\[
Dice=
\frac{2|P\cap G|}
{|P|+|G|}
\]
IoU：
\[
IoU=
\frac{|P\cap G|}
{|P\cup G|}
\]
通常：
Dice > IoU

这是正常现象。
它们都是：
越接近 1 越好
"""

import torch


def segmentation_metrics(
    logits,
    targets,
    threshold=0.5,
    smooth=1e-6
):
    """
    计算二分类图像分割的 Dice 和 IoU。

    参数
    ----------
    logits:
        模型原始输出。
        shape:
        [B, 1, H, W]

    targets:
        Ground Truth Mask。
        shape:
        [B, 1, H, W]

    threshold:
        二值化阈值。
        默认 0.5。

    smooth:
        防止分母为 0。

    返回
    ----------
    mean_dice:
        当前 Batch 的平均 Dice。

    mean_iou:
        当前 Batch 的平均 IoU。
    """

    # ==========================================
    # 1. Logits → Probability
    # ==========================================

    probs = torch.sigmoid(
        logits
    )


    # ==========================================
    # 2. Probability → Binary Mask
    # ==========================================

    preds = (
        probs >= threshold
    ).float()


    # ==========================================
    # 3. 对每张图片的 H、W、Channel 求和
    #
    # Tensor:
    # [B, C, H, W]
    #
    # 保留 Batch 维度，
    # 每张图片分别计算指标。
    # ==========================================

    dims = (
        1,
        2,
        3
    )


    # Prediction 和 Ground Truth 重叠区域
    intersection = (
        preds * targets
    ).sum(
        dim=dims
    )


    # Prediction 病灶面积
    pred_area = preds.sum(
        dim=dims
    )


    # Ground Truth 病灶面积
    target_area = targets.sum(
        dim=dims
    )


    # ==========================================
    # 4. Dice
    # ==========================================

    dice = (
        2.0 * intersection + smooth
    ) / (
        pred_area
        +
        target_area
        +
        smooth
    )


    # ==========================================
    # 5. IoU
    #
    # union =
    # prediction + target - intersection
    # ==========================================

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


    # 对 Batch 中所有样本求平均
    mean_dice = dice.mean().item()

    mean_iou = iou.mean().item()


    return (
        mean_dice,
        mean_iou
    )