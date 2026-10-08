"""
Breast Ultrasound Lesion Segmentation Demo

功能：
1. 上传乳腺超声图像
2. 选择分割模型
3. 输出 Probability Map
4. 输出 Binary Mask
5. 输出 Prediction Overlay
6. 计算预测病灶面积占比
7. 统计单张图像推理时间

当前模型：
- Scratch U-Net
- ResNet18-U-Net
- Attention-ResNet18-U-Net
"""
import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

from PIL import Image

from src.model import UNet

from src.model_zoo import (
    build_resunet18,
    build_attention_resunet18
)


# ============================================================
# 1. 基本配置
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

IMAGE_SIZE = 256


print(
    f"Using device: {DEVICE}"
)


# ============================================================
# 2. 三个模型 checkpoint 的位置
# ============================================================

CHECKPOINT_PATHS = {

    "U-Net":
        Path(
            "results/"
            "unet_baseline_final/"
            "best_model.pt"
        ),

    "ResNet18-U-Net":
        Path(
            "results/"
            "resunet18_aug/"
            "best_model.pt"
        ),

    "Attention-ResNet18-U-Net":
        Path(
            "results/"
            "attention_resunet18_aug/"
            "best_model.pt"
        )
}


# ============================================================
# 3. 加载 checkpoint
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path
):
    """
    将训练保存的 best_model.pt 加载到模型。

    同时兼容：

    {
        "model_state_dict": ...
    }

    和直接保存的：

    state_dict
    """

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )


    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )


    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]


    elif (
        isinstance(checkpoint, dict)
        and
        "state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]


    else:

        state_dict = checkpoint


    # --------------------------------------------------------
    # 兼容 DataParallel：
    #
    # module.xxx
    #
    # ↓
    #
    # xxx
    # --------------------------------------------------------

    clean_state_dict = {}

    for key, value in state_dict.items():

        new_key = key.replace(
            "module.",
            ""
        )

        clean_state_dict[
            new_key
        ] = value


    model.load_state_dict(
        clean_state_dict
    )


    model = model.to(
        DEVICE
    )

    model.eval()


    return model


# ============================================================
# 4. 创建模型
# ============================================================

def create_model(
    model_name
):
    """
    根据用户选择创建对应模型结构。
    """

    if model_name == "U-Net":

        model = UNet()


    elif model_name == "ResNet18-U-Net":

        # 推理阶段已经有训练好的权重，
        # 因此不需要再次下载 ImageNet weights。
        model = build_resunet18(

            in_channels=1,

            classes=1,

            encoder_weights=None
        )


    elif model_name == "Attention-ResNet18-U-Net":

        model = (
            build_attention_resunet18(

                in_channels=1,

                classes=1,

                encoder_weights=None
            )
        )


    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )


    checkpoint_path = (
        CHECKPOINT_PATHS[
            model_name
        ]
    )


    model = load_checkpoint(

        model,

        checkpoint_path
    )


    return model


# ============================================================
# 5. 模型缓存
# ============================================================
#
# 如果用户每点一次 Predict，
# 我们都重新读取 best_model.pt，
# 会非常慢。
#
# 所以第一次用某个模型时加载，
# 后续直接从字典里取。
#
# ============================================================

MODEL_CACHE = {}


def get_model(
    model_name
):

    if model_name not in MODEL_CACHE:

        print(
            f"Loading model: "
            f"{model_name}"
        )

        MODEL_CACHE[
            model_name
        ] = create_model(
            model_name
        )


    return MODEL_CACHE[
        model_name
    ]


# ============================================================
# 6. 图像预处理
# ============================================================

def preprocess_image(
    input_image
):
    """
    Gradio 上传的图片
            ↓
    PIL Image
            ↓
    灰度图
            ↓
    resize 256 × 256
            ↓
    [0,1]
            ↓
    Tensor:
    [1,1,256,256]
    """

    if input_image is None:

        raise ValueError(
            "Please upload an ultrasound image."
        )


    # --------------------------------------------------------
    # Gradio 通常给我们 NumPy Array
    # --------------------------------------------------------

    if isinstance(
        input_image,
        np.ndarray
    ):

        input_image = Image.fromarray(
            input_image.astype(
                np.uint8
            )
        )


    # --------------------------------------------------------
    # 保存原始 RGB 图像用于最终展示
    # --------------------------------------------------------

    original_rgb = (
        input_image
        .convert(
            "RGB"
        )
    )


    # --------------------------------------------------------
    # 模型训练时是单通道灰度图
    # --------------------------------------------------------

    gray_image = (
        input_image
        .convert(
            "L"
        )
    )


    # --------------------------------------------------------
    # 与训练 / Test 相同：
    #
    # resize → 256 × 256
    # --------------------------------------------------------

    resized_gray = gray_image.resize(

        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),

        Image.Resampling.BILINEAR
    )


    image_array = np.array(

        resized_gray,

        dtype=np.float32
    )


    # --------------------------------------------------------
    # 归一化
    #
    # 0~255
    #
    # ↓
    #
    # 0~1
    # --------------------------------------------------------

    image_array = (
        image_array
        /
        255.0
    )


    # --------------------------------------------------------
    # [H,W]
    #
    # ↓
    #
    # [1,1,H,W]
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        image_array
    )


    tensor = (
        tensor
        .unsqueeze(0)
        .unsqueeze(0)
    )


    tensor = tensor.to(
        DEVICE
    )


    return (
        original_rgb,
        resized_gray,
        tensor
    )


# ============================================================
# 7. 创建 Overlay
# ============================================================

def create_overlay(
    gray_image,
    binary_mask
):
    """
    将预测病灶 Mask 轮廓叠加到原始超声图像上。

    红色线：
        模型预测病灶边界
    """

    # --------------------------------------------------------
    # 灰度图 → RGB
    # --------------------------------------------------------

    image_array = np.array(
        gray_image
    )


    rgb_image = cv2.cvtColor(

        image_array,

        cv2.COLOR_GRAY2RGB
    )


    # --------------------------------------------------------
    # Mask 转 uint8
    # --------------------------------------------------------

    mask_uint8 = (

        binary_mask.astype(
            np.uint8
        )

        * 255
    )


    # --------------------------------------------------------
    # 找病灶轮廓
    # --------------------------------------------------------

    contours, _ = cv2.findContours(

        mask_uint8,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE
    )


    # --------------------------------------------------------
    # 在图像上画红色预测轮廓
    # --------------------------------------------------------

    cv2.drawContours(

        rgb_image,

        contours,

        contourIdx=-1,

        color=(255, 0, 0),

        thickness=2
    )


    return rgb_image


# ============================================================
# 8. 核心预测函数
# ============================================================

def predict(
    input_image,
    model_name,
    threshold
):
    """
    Demo 最核心的函数。

    输入：
        image
        model
        threshold

    输出：
        Original
        Probability Map
        Binary Mask
        Overlay
        Information
    """

    if input_image is None:

        return (
            None,
            None,
            None,
            None,
            "Please upload an image first."
        )


    # ========================================================
    # 预处理
    # ========================================================

    (
        original_rgb,
        resized_gray,
        image_tensor
    ) = preprocess_image(
        input_image
    )


    # ========================================================
    # 读取模型
    # ========================================================

    model = get_model(
        model_name
    )


    # ========================================================
    # 推理计时
    # ========================================================

    start_time = time.perf_counter()


    with torch.no_grad():

        logits = model(
            image_tensor
        )


        # Logits → Probability
        probability = torch.sigmoid(
            logits
        )


    # GPU 是异步执行，
    # 如果使用 GPU，
    # 必须 synchronize 后再停止计时。
    if DEVICE.type == "cuda":

        torch.cuda.synchronize()


    elapsed_time = (

        time.perf_counter()
        -
        start_time

    )


    # ========================================================
    # Probability Map
    # ========================================================

    probability_map = (

        probability[
            0,
            0
        ]

        .cpu()

        .numpy()

    )


    # ========================================================
    # Binary Mask
    # ========================================================

    binary_mask = (

        probability_map
        >=
        threshold

    ).astype(
        np.uint8
    )


    # ========================================================
    # 病灶面积占比
    # ========================================================

    lesion_area_ratio = (

        binary_mask.sum()
        /
        binary_mask.size

    )


    # ========================================================
    # Probability Map 转成可显示图像
    #
    # 0~1
    #
    # ↓
    #
    # 0~255
    # ========================================================

    probability_display = (

        probability_map
        *
        255

    ).astype(
        np.uint8
    )


    # ========================================================
    # Binary Mask 显示
    # ========================================================

    mask_display = (

        binary_mask
        *
        255

    ).astype(
        np.uint8
    )


    # ========================================================
    # Overlay
    # ========================================================

    overlay = create_overlay(

        resized_gray,

        binary_mask
    )


    # ========================================================
    # 信息文本
    # ========================================================

    information = (

        f"Model: {model_name}\n\n"

        f"Threshold: "
        f"{threshold:.2f}\n\n"

        f"Predicted lesion area: "
        f"{lesion_area_ratio * 100:.2f}%\n\n"

        f"Inference time: "
        f"{elapsed_time * 1000:.2f} ms\n\n"

        f"Device: {DEVICE}"

    )


    # 为了保持和模型输入一致，
    # Original 也显示 resize 后的版本。
    original_display = np.array(
        resized_gray
    )


    return (

        original_display,

        probability_display,

        mask_display,

        overlay,

        information
    )


# ============================================================
# 9. Gradio Web UI
# ============================================================

with gr.Blocks(
    title=(
        "Breast Ultrasound "
        "Lesion Segmentation"
    )
) as demo:


    gr.Markdown(
        """
# Breast Ultrasound Lesion Segmentation

Upload a breast ultrasound image and perform automatic lesion segmentation.

**Best model:** ResNet18-U-Net  
**Test Dice:** 0.8310  
**Test IoU:** 0.7377

The model is intended for research and educational demonstration only,
not for clinical diagnosis.
"""
    )


    # ========================================================
    # 第一行：
    # 输入 + 设置
    # ========================================================

    with gr.Row():


        with gr.Column():

            input_image = gr.Image(

                label=(
                    "Upload "
                    "Ultrasound Image"
                ),

                type="numpy"
            )


        with gr.Column():

            model_selector = gr.Dropdown(

                choices=[

                    "ResNet18-U-Net",

                    "U-Net",

                    (
                        "Attention-"
                        "ResNet18-U-Net"
                    )

                ],

                value=(
                    "ResNet18-U-Net"
                ),

                label="Model"
            )


            threshold_slider = gr.Slider(

                minimum=0.1,

                maximum=0.9,

                value=0.5,

                step=0.05,

                label=(
                    "Segmentation "
                    "Threshold"
                )
            )


            predict_button = gr.Button(

                "Run Segmentation",

                variant="primary"
            )


    # ========================================================
    # 第二行：
    # Original + Probability
    # ========================================================

    with gr.Row():

        original_output = gr.Image(

            label="Preprocessed Image",

            image_mode="L"
        )


        probability_output = gr.Image(

            label="Probability Map",

            image_mode="L"
        )


    # ========================================================
    # 第三行：
    # Mask + Overlay
    # ========================================================

    with gr.Row():

        mask_output = gr.Image(

            label="Binary Mask",

            image_mode="L"
        )


        overlay_output = gr.Image(

            label="Prediction Overlay"
        )


    # ========================================================
    # 模型信息
    # ========================================================

    info_output = gr.Textbox(

        label="Prediction Information",

        lines=7,

        interactive=False
    )


    # ========================================================
    # Button → predict()
    # ========================================================

    predict_button.click(

        fn=predict,

        inputs=[

            input_image,

            model_selector,

            threshold_slider

        ],

        outputs=[

            original_output,

            probability_output,

            mask_output,

            overlay_output,

            info_output

        ]
    )


# ============================================================
# 10. 启动
# ============================================================

if __name__ == "__main__":

    demo.launch()