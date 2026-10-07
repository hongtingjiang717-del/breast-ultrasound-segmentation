"""
高级乳腺超声分割模型
1.ResNet18 U-Net
2.Attention ResNet18 U-Net
这些模型基于segmentation_models_pytorch实现
"""

import segmentation_models_pytorch as smp

def build_resunet18(
    in_channels=1,
    classes=1
):
    """
    构建 ResNet18-U-Net。

    核心思想：
    ------------------------------------------------
    Encoder:
        ResNet18
        使用 ImageNet 预训练参数，相当于已经学会了边缘、纹理、形状以及复杂视觉模式

    Decoder:
        U-Net Decoder

    输入：
        [B, 1, H, W]

    输出：
        [B, 1, H, W]

    注意：
    activation=None

    我们故意不在模型内部做 Sigmoid，
    因为当前 loss 使用 BCEWithLogitsLoss，
    它需要输入 raw logits。
    """

    model = smp.Unet(

        # ==========================================
        # Encoder
        # ==========================================

        encoder_name="resnet18",

        # 使用 ImageNet 预训练权重
        encoder_weights="imagenet",

        # BUSI 是灰度超声图
        in_channels=in_channels,

        # 二分类分割：
        # 只需要一个输出通道
        classes=classes,

        # ==========================================
        # 非常重要
        # ==========================================
        #
        # 不在模型内部 Sigmoid
        # 输出 raw logits
        #
        activation=None
    )

    return model


def build_attention_resunet18(
    in_channels=1,
    classes=1
):
    """
    构建 Attention ResNet18-U-Net。

    与普通 ResUNet 的主要区别：

        decoder_attention_type="scse"

    scSE:
        Spatial and Channel Squeeze & Excitation

    scSE:
    Feature
   │
   ├─ Channel Attention
   │
   └─ Spatial Attention
          ↓
        重新加权
            ↓
        Decoder

    它会从：
        Channel
        +
        Spatial

    两个维度重新给特征分配权重。
    Channel attention 决定“哪些 feature maps 更重要”；Spatial attention 决定“图像哪些位置更重要”。

    目的：
        增强病灶相关信息
        抑制无关背景信息
    """

    model = smp.Unet(

        encoder_name="resnet18",

        encoder_weights="imagenet",

        in_channels=in_channels,

        classes=classes,

        # ==========================================
        # Attention
        # ==========================================

        decoder_attention_type="scse",

        activation=None
    )

    return model
