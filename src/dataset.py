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
        data_root=None
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

    def __getitem__(self, index):
        """
        获取第 index 个样本。

        最终返回：

        {
            image: [1, 256, 256],
            mask:  [1, 256, 256],
            ...
        }
        """

        # ==================================================
        # 1. 取出当前样本信息
        # ==================================================

        row = self.df.iloc[idx]

        image_path = self._resolve_image_path(
            row
        )

        # ==================================================
        # 2. 自动寻找 mask
        # ==================================================

        mask_paths = find_mask_paths(
            image_path
        )


        # 理论上昨天我们已经检查：
        #
        # Missing masks = 0
        #
        # 所以正常情况下这里一定 >= 1。
        if len(mask_paths) == 0:

            raise FileNotFoundError(
                f"没有找到 mask：{image_path}"
            )


        # ==================================================
        # 3. 读取超声图像
        # ==================================================

        image = Image.open(
            image_path
        ).convert("L")


        # ==================================================
        # 4. 合并多个 mask
        # ==================================================

        merged_mask = load_merged_mask(
            mask_paths
        )

        # 当前 merged_mask 是 numpy 数组
        #
        # 转回 PIL，
        # 后面方便进行 resize。
        #
        # 目前值为：
        # 0 / 1
        #
        # 乘 255 后变成：
        # 0 / 255
        mask = Image.fromarray(
            merged_mask * 255
        )


        # ==================================================
        # 5. Resize image
        # ==================================================

        # 原图是连续灰度图，
        # 使用双线性插值。
        image = image.resize(
            self.image_size,
            resample=Image.Resampling.BILINEAR
        )


        # ==================================================
        # 6. Resize mask
        # ==================================================

        # mask 是离散标签，
        # 使用最近邻插值。
        mask = mask.resize(
            self.image_size,
            resample=Image.Resampling.NEAREST
        )


        # ==================================================
        # 7. PIL → numpy
        # ==================================================

        image = np.array(
            image,
            dtype=np.float32
        )

        mask = np.array(
            mask,
            dtype=np.float32
        )


        # ==================================================
        # 8. image 归一化
        # ==================================================

        # 0~255
        #
        # ↓
        #
        # 0~1
        image = image / 255.0


        # ==================================================
        # 9. mask 二值化
        # ==================================================

        mask = (
            mask > 127
        ).astype(np.float32)


        # ==================================================
        # 10. numpy → Tensor
        # ==================================================

        image = torch.from_numpy(
            image
        )

        mask = torch.from_numpy(
            mask
        )


        # ==================================================
        # 11. 添加 channel 维度
        # ==================================================

        # [256, 256]
        #
        # ↓
        #
        # [1, 256, 256]

        image = image.unsqueeze(0)

        mask = mask.unsqueeze(0)


        # ==================================================
        # 12. 返回
        # ==================================================

        return {

            "image": image,

            "mask": mask,

            "image_path": str(image_path),

            # 这里返回当前图像到底有几个 mask，
            # 后面调试会比较方便。
            "num_masks": len(mask_paths)
        }