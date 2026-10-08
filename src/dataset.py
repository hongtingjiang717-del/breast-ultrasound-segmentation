"""
dataset.py

BUSI 乳腺超声分割数据集读取模块。

主要功能：
1. 从 CSV 中读取 image_path（找到原始超声图像.png）
2. 根据 image_path 自动寻找对应的一个或多个 mask
3. 如果存在多个 mask，将它们合并
4. image 和 mask resize 到统一尺寸
5. image 归一化到 [0, 1]
6. mask 二值化到 {0, 1}
7. 转换为 PyTorch Tensor
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset


# ============================================================
# 1. 根据原图路径寻找所有对应的 mask
# ============================================================

def find_mask_paths(image_path):
    """
    根据 BUSI 原始图像路径，自动寻找所有对应 mask。

    例如：

    原图：
        benign (1).png

    可能对应：
        benign (1)_mask.png
        benign (1)_mask_1.png
        benign (1)_mask_2.png

    返回：
        一个 mask 路径列表。
    """

    image_path = Path(image_path)

    # 原始文件名（不带扩展名）
    #
    # 例如：
    # benign (1).png
    #
    # stem = "benign (1)"
    image_stem = image_path.stem

    # 文件扩展名
    #
    # 例如：
    # ".png"
    image_ext = image_path.suffix

    # 原始图像所在文件夹
    image_dir = image_path.parent

    # glob：
    # 搜索所有：
    #
    # benign (1)_mask*.png
    #
    # 可以匹配：
    # benign (1)_mask.png
    # benign (1)_mask_1.png
    # benign (1)_mask_2.png
    mask_paths = sorted(
        image_dir.glob(
            f"{image_stem}_mask*{image_ext}"
        )
    )

    return mask_paths


# ============================================================
# 2. 合并多个 mask
# ============================================================

def load_merged_mask(mask_paths):
    """
    读取并合并一张图像对应的所有 mask。

    BUSI 中部分图像可能存在多个 mask。

    合并原则：

        某像素只要在任意一个 mask 中属于病灶，
        最终就认为该像素属于病灶。

    相当于逻辑 OR。
    """

    merged_mask = None

    for mask_path in mask_paths:

        # 读取为灰度图
        mask = Image.open(
            mask_path
        ).convert("L")

        # PIL → numpy
        mask = np.array(mask)

        # 二值化
        #
        # 背景：
        # 0
        #
        # 病灶：
        # 1
        mask = (
            mask > 0
        ).astype(np.uint8)

        # 第一张 mask
        if merged_mask is None:

            merged_mask = mask

        else:

            # np.maximum 相当于逐像素做 OR
            #
            # 例如：
            #
            # mask1 = 0 1
            #         0 0
            #
            # mask2 = 0 0
            #         1 0
            #
            # 合并：
            #
            #        0 1
            #        1 0
            merged_mask = np.maximum(
                merged_mask,
                mask
            )

    return merged_mask


# ============================================================
# 3. Dataset
# ============================================================


class BUSIDataset(Dataset):

    def __init__(
        self,
        csv_file,
        image_size=(256, 256),
        data_root=None,
        transform=None
    ):
        """
        Parameters
        ----------
        csv_file:
            train.csv / val.csv / test.csv

        image_size:
            最终输入网络的尺寸。
            当前 baseline 使用 256 × 256。
        data_root:
            BUSI 原始数据根目录。

            如果 data_root=None：
                使用 CSV 中原有的 image_path。

            如果指定 data_root：
                自动通过：
                    data_root / class / image_name
                重新构造图像路径。

            这样可以解决：
            Windows 本地路径无法直接用于 Linux 云服务器的问题。
        """

        # -----------------------------
        # 读取数据划分 CSV
        # -----------------------------

        self.df = pd.read_csv(csv_file)

        self.image_size = image_size
        #如果传入数据根目录，将他转换成path对象
        self.data_root=(
            Path(data_root)
            if data_root is not None
            else None
        )
        #保存albumentations数据增强
        self.transform = transform

        # -----------------------------
        # 检查必要字段
        # -----------------------------

        # 注意：
        #
        # 这里现在只需要 image_path。
        #
        # 因为 mask_path 并没有存进 CSV，
        # 而是在 __getitem__ 中动态寻找。
        required_columns = [
            "image_path"
        ]

        for column in required_columns:

            if column not in self.df.columns:

                raise ValueError(
                    f"CSV 中缺少必要列：{column}"
                )


        print(
            f"Loaded dataset: {len(self.df)} images"
        )


    def __len__(self):

        return len(self.df)
    
    def _resolve_image_path(
        self,
        row
    ):
        """
        根据当前运行环境得到真正的图像路径。
        """

        # ==========================================
        # 情况 1：
        # 云服务器等新环境指定了 data_root
        # ==========================================

        if self.data_root is not None:

            image_path = (
                self.data_root
                /
                row["class"]
                /
                row["image_name"]
            )

        # ==========================================
        # 情况 2：
        # 本地继续使用 CSV 原有 image_path
        # ==========================================

        else:

            image_path = Path(
                row["image_path"]
            )


        return image_path

    def __getitem__(
        self,
        idx
    ):

        # ======================================================
        # 1. 从 CSV 中取得第 idx 个样本
        # ======================================================

        row = self.df.iloc[idx]


        # ======================================================
        # 2. 得到当前环境下真正的图像路径
        # ======================================================

        image_path = self._resolve_image_path(
            row
        )


        # ======================================================
        # 3. 自动寻找当前超声图像对应的所有 Mask
        # ======================================================

        mask_paths = find_mask_paths(
            image_path
        )


        if len(mask_paths) == 0:

            raise FileNotFoundError(
                f"No mask found for image: {image_path}"
            )


        # ======================================================
        # 4. 读取灰度超声图像
        #
        # convert("L")
        # 表示转成单通道灰度图
        # ======================================================

        image = Image.open(
            image_path
        ).convert(
            "L"
        )


        # PIL Image → NumPy
        #
        # shape:
        # [H, W]
        image_array = np.array(
            image,
            dtype=np.uint8
        )


        # ======================================================
        # 5. 合并所有 Mask
        #
        # merged_mask:
        # [H, W]
        #
        # 值：
        # 0 = 背景
        # 1 = 病灶
        # ======================================================

        merged_mask = load_merged_mask(
            mask_paths
        )


        # ======================================================
        # 6. 数据增强
        # ======================================================

        if self.transform is not None:

            # Albumentations 最重要的特点：
            #
            # image 和 mask 一起传进去
            # 因此旋转、翻转、缩放等空间变化会保持同步。
            transformed = self.transform(

                image=image_array,

                mask=merged_mask
            )


            image_array = transformed[
                "image"
            ]

            merged_mask = transformed[
                "mask"
            ]


        # ======================================================
        # 7. 如果没有使用 Albumentations
        #
        # 继续保留原来的 Resize
        # ======================================================

        else:

            # -----------------------------
            # Image：
            # Bilinear interpolation
            # -----------------------------

            image_pil = Image.fromarray(
                image_array
            )

            image_pil = image_pil.resize(
                self.image_size,
                Image.Resampling.BILINEAR
            )


            # -----------------------------
            # Mask：
            # Nearest interpolation
            # -----------------------------

            mask_pil = Image.fromarray(
                (
                    merged_mask * 255
                ).astype(
                    np.uint8
                )
            )

            mask_pil = mask_pil.resize(
                self.image_size,
                Image.Resampling.NEAREST
            )


            image_array = np.array(
                image_pil,
                dtype=np.float32
            )


            merged_mask = (
                np.array(
                    mask_pil
                )
                > 127
            ).astype(
                np.float32
            )


        # ======================================================
        # 8. Image 转 float32
        # ======================================================

        image_array = image_array.astype(
            np.float32
        )


        # ======================================================
        # 9. 图像归一化到 [0, 1]
        # ======================================================

        image_array = (
            image_array
            /
            255.0
        )


        # ======================================================
        # 10. Mask 再次保证严格二值化
        #
        # 数据增强以后可能数据类型发生变化，
        # 所以这里重新保险处理一次。
        # ======================================================

        merged_mask = (
            merged_mask > 0.5
        ).astype(
            np.float32
        )


        # ======================================================
        # 11. NumPy → PyTorch Tensor
        #
        # 当前：
        #
        # image:
        # [256,256]
        #
        # mask:
        # [256,256]
        # ======================================================

        image_tensor = torch.from_numpy(
            image_array
        )

        mask_tensor = torch.from_numpy(
            merged_mask
        )


        # ======================================================
        # 12. 增加 Channel 维度
        #
        # [H,W]
        #
        # ↓
        #
        # [1,H,W]
        # ======================================================

        image_tensor = image_tensor.unsqueeze(
            0
        )

        mask_tensor = mask_tensor.unsqueeze(
            0
        )


        # ======================================================
        # 13. 返回
        # ======================================================

        return {

            "image":
                image_tensor,

            "mask":
                mask_tensor,

            "image_path":
                str(image_path),

            "num_masks":
                len(mask_paths)
        }