import torch
import torch.nn as nn

class DiceBCELoss(nn.Module):
    """
    BCE Loss + Dice Loss的联合损失函数
    输入的是
        预测的概率值logits:
            U-Net的原始输出
            shape = [B,1,H,W]
        targets:
        Ground Truth Mask
        shape = [B,1,H,W]
        数值为0或1
    输出一个标量loss
    """
    def __init__(self):
        #初始化父类nn.module
        super().__init__()
        # Binary Cross Entropy二元交叉熵，会逐像素惩罚错误预测
        # BCEWithLogitsLoss 内部已经包含：
        #
        # Sigmoid + BCE
        #
        # 所以训练时不要提前对 logits 做 sigmoid。
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self,logits,targets):
        # ==========================================
        # 1. BCE Loss
        # ==========================================
        bce_loss = self.bce(
            logits,
            targets
        )
        # ==========================================
        # 2. 将 logits 转成概率
        # ==========================================
        #
        # Dice 的计算需要 0~1 的预测概率，
        # 所以这里需要 sigmoid。

        probs = torch.sigmoid(
            logits
        )


        # ==========================================
        # 3. 计算预测和真实 Mask 的交集
        # ==========================================

        intersection = (
            probs * targets
        ).sum(
            dim=(2, 3)
        )


        # ==========================================
        # 4. 计算两者区域大小之和---并集
        #dim(2,3)是高和宽的维度
        #dim 0 = Batch
        # dim 1 = Channel
        # dim 2 = Height
        # dim 3 = Width
        # ==========================================

        total = (
            probs.sum(dim=(2, 3))
            +
            targets.sum(dim=(2, 3))
        )


        # ==========================================
        # 5. Dice Score---两倍的交集比并集
        # ==========================================

        # 防止分母为 0
        smooth = 1e-6

        dice = (
            2.0 * intersection + smooth
        ) / (
            total + smooth
        )


        # ==========================================
        # 6. Dice Loss
        # ==========================================

        dice_loss = (
            1.0 - dice.mean()
        )


        # ==========================================
        # 7. BCE + Dice
        # ==========================================

        total_loss = (
            bce_loss
            +
            dice_loss
        )


        return total_loss