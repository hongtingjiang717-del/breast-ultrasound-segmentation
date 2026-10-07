"""
version1:更保守更容易解释的医学图像增强办法
BUSI 乳腺超声数据增强。

关键原则：
Image 和 Mask 必须进行完全相同的几何变换。
"""

import cv2
import albumentations as A


def get_train_augmentation(
    image_size=256
):
    """
    训练集数据增强。

    只用于 Train。
    Validation / Test 不做随机增强。
    """

    transform = A.Compose(

        [

            # ==========================================
            # 统一尺寸
            # ==========================================

            A.Resize(
                height=image_size,
                width=image_size
            ),


            # ==========================================
            # 水平翻转
            # ==========================================
            #
            # p=0.5：
            # 每张图 50% 概率执行
            #
            A.HorizontalFlip(
                p=0.5
            ),


            # ==========================================
            # 小角度旋转
            # ==========================================

            A.Rotate(
                limit=15,

                border_mode=cv2.BORDER_CONSTANT,

                value=0,

                mask_value=0,

                p=0.5
            ),


            # ==========================================
            # 小范围平移 + 缩放
            # ==========================================

            A.ShiftScaleRotate(

                shift_limit=0.05,

                scale_limit=0.10,

                rotate_limit=0,

                border_mode=cv2.BORDER_CONSTANT,

                value=0,

                mask_value=0,

                p=0.4
            ),


            # ==========================================
            # 超声强度变化
            # ==========================================
            #
            # 这个操作只改变 Image，
            # 不会改变 Mask
            #
            A.RandomBrightnessContrast(

                brightness_limit=0.15,

                contrast_limit=0.15,

                p=0.3
            )
        ]
    )

    return transform


def get_eval_augmentation(
    image_size=256
):
    """
    Validation / Test：

    只 Resize，
    不做任何随机增强。
    """

    return A.Compose(
        [
            A.Resize(
                height=image_size,
                width=image_size
            )
        ]
    )