"""
Breast Ultrasound Lesion Segmentation Demo
==========================================

功能：
1. 上传乳腺超声图像
2. 支持三个模型切换
3. 输出 Probability Heatmap
4. 输出 Binary Mask
5. 输出 Prediction Overlay
6. 显示推理信息
7. 下载预测结果 ZIP
8. 提供 BUSI 示例病例

注意：
本 Demo 仅用于科研与算法展示，不用于临床诊断。
"""

import json
import shutil
import tempfile
import time
from datetime import datetime
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

from src.augmentations import (
    get_eval_augmentation
)


# ============================================================
# 1. 基本配置
# ============================================================

IMAGE_SIZE = 256

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {DEVICE}")


# ============================================================
# 2. 三个模型 checkpoint
# ============================================================
#
# 如果你的实际目录名字不同，
# 只需要修改这里。
#
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
# 3. 标准化 Test 结果
# ============================================================

MODEL_METRICS = {

    "U-Net": {
        "dice": 0.6983,
        "iou": 0.5795,
        "description":
            "Scratch U-Net baseline"
    },

    "ResNet18-U-Net": {
        "dice": 0.8310,
        "iou": 0.7377,
        "description":
            "ImageNet pretrained ResNet18 encoder + U-Net decoder"
    },

    "Attention-ResNet18-U-Net": {
        "dice": 0.7960,
        "iou": 0.7117,
        "description":
            "ResNet18-U-Net + scSE decoder attention"
    }
}


# ============================================================
# 4. Test 阶段预处理
# ============================================================
#
# 关键：
# Demo 必须尽量和 standardized test evaluation 保持一致。
#
# 不能 Demo 一套 preprocessing，
# Test 又是另一套。
#
# ============================================================

EVAL_TRANSFORM = get_eval_augmentation(
    image_size=IMAGE_SIZE
)


# ============================================================
# 5. 模型缓存
# ============================================================

MODEL_CACHE = {}


# ============================================================
# 6. 加载 checkpoint
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path
):

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )


    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )


    # -------------------------------------------
    # 兼容：
    #
    # {
    #   "model_state_dict": ...
    # }
    #
    # -------------------------------------------

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


    # -------------------------------------------
    # 如果以前 DataParallel 保存为：
    #
    # module.xxx
    #
    # 自动去掉 module.
    # -------------------------------------------

    clean_state_dict = {}

    for key, value in state_dict.items():

        clean_key = key.replace(
            "module.",
            ""
        )

        clean_state_dict[
            clean_key
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
# 7. 创建模型
# ============================================================

def create_model(
    model_name
):

    if model_name == "U-Net":

        model = UNet()


    elif model_name == "ResNet18-U-Net":

        # 已经有最终 checkpoint，
        # 推理时不再重新下载 ImageNet 权重。
        model = build_resunet18(

            in_channels=1,
            classes=1,
            encoder_weights=None
        )


    elif model_name == "Attention-ResNet18-U-Net":

        model = build_attention_resunet18(

            in_channels=1,
            classes=1,
            encoder_weights=None
        )


    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )


    return load_checkpoint(

        model,

        CHECKPOINT_PATHS[
            model_name
        ]
    )


# ============================================================
# 8. 获取模型
# ============================================================
#
# 第一次使用：
# 加载 best_model.pt
#
# 第二次以后：
# 直接从内存读取
#
# ============================================================

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
# 9. 模型性能信息
# ============================================================

def get_model_information(
    model_name
):

    metric = MODEL_METRICS[
        model_name
    ]

    return f"""
### Model Information

**{model_name}**

{metric["description"]}

| Metric | Standardized Test |
|---|---:|
| Dice | **{metric["dice"]:.4f}** |
| IoU | **{metric["iou"]:.4f}** |

Test set: **98 independent BUSI lesion images**
"""


# ============================================================
# 10. 图像预处理
# ============================================================

def preprocess_image(
    input_image
):

    if input_image is None:

        raise ValueError(
            "Please upload an ultrasound image."
        )


    # -------------------------------------------
    # Gradio Image(type="numpy")
    # 返回 NumPy array
    # -------------------------------------------

    if isinstance(
        input_image,
        np.ndarray
    ):

        input_image = Image.fromarray(
            input_image.astype(
                np.uint8
            )
        )


    # -------------------------------------------
    # BUSI 模型训练时使用灰度图
    # -------------------------------------------

    gray_image = input_image.convert(
        "L"
    )


    gray_array = np.array(
        gray_image,
        dtype=np.uint8
    )


    # -------------------------------------------
    # 使用与 Test Set 相同的 Resize pipeline
    # -------------------------------------------

    transformed = EVAL_TRANSFORM(
        image=gray_array
    )


    resized_array = transformed[
        "image"
    ]


    # -------------------------------------------
    # [0,255]
    #
    # ↓
    #
    # [0,1]
    # -------------------------------------------

    normalized = (

        resized_array.astype(
            np.float32
        )

        / 255.0
    )


    # -------------------------------------------
    # [H,W]
    #
    # ↓
    #
    # [1,1,H,W]
    # -------------------------------------------

    tensor = torch.from_numpy(
        normalized
    )

    tensor = (
        tensor
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )


    return (
        resized_array,
        tensor
    )


# ============================================================
# 11. Probability → 彩色 Heatmap
# ============================================================

def create_probability_heatmap(
    probability_map
):

    # 0~1
    #
    # ↓
    #
    # 0~255
    probability_uint8 = np.clip(

        probability_map * 255,

        0,
        255

    ).astype(
        np.uint8
    )


    # OpenCV 彩色 heatmap
    heatmap_bgr = cv2.applyColorMap(

        probability_uint8,

        cv2.COLORMAP_TURBO
    )


    # OpenCV 默认 BGR
    #
    # Gradio / PIL 使用 RGB
    heatmap_rgb = cv2.cvtColor(

        heatmap_bgr,

        cv2.COLOR_BGR2RGB
    )


    return heatmap_rgb


# ============================================================
# 12. Prediction Overlay
# ============================================================

def create_overlay(
    gray_image,
    binary_mask
):
    """
    预测区域：
        半透明红色

    病灶边界：
        黄色轮廓
    """

    # Gray → RGB
    rgb = cv2.cvtColor(

        gray_image,

        cv2.COLOR_GRAY2RGB
    )


    overlay = rgb.copy().astype(
        np.float32
    )


    mask_bool = (
        binary_mask > 0
    )


    # 半透明红色填充
    red = np.array(
        [255, 50, 50],
        dtype=np.float32
    )


    overlay[
        mask_bool
    ] = (

        0.65
        *
        overlay[
            mask_bool
        ]

        +

        0.35
        *
        red

    )


    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(
        np.uint8
    )


    # -------------------------------------------
    # 再加边界
    # -------------------------------------------

    mask_uint8 = (

        binary_mask.astype(
            np.uint8
        )

        * 255
    )


    contours, _ = cv2.findContours(

        mask_uint8,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE
    )


    cv2.drawContours(

        overlay,

        contours,

        contourIdx=-1,

        color=(255, 215, 0),

        thickness=2
    )


    return overlay


# ============================================================
# 13. 保存预测结果包
# ============================================================

def save_prediction_package(
    preprocessed_image,
    probability_heatmap,
    binary_mask,
    overlay,
    metadata
):
    """
    每次预测生成一个 ZIP：

    input.png
    probability_heatmap.png
    binary_mask.png
    overlay.png
    metadata.json
    """

    # 使用系统临时目录，
    # 不污染 Git 项目目录。
    output_dir = Path(

        tempfile.mkdtemp(
            prefix="busi_prediction_"
        )

    )


    Image.fromarray(
        preprocessed_image
    ).save(
        output_dir /
        "input_preprocessed.png"
    )


    Image.fromarray(
        probability_heatmap
    ).save(
        output_dir /
        "probability_heatmap.png"
    )


    Image.fromarray(

        (
            binary_mask
            *
            255
        ).astype(
            np.uint8
        )

    ).save(
        output_dir /
        "binary_mask.png"
    )


    Image.fromarray(
        overlay
    ).save(
        output_dir /
        "prediction_overlay.png"
    )


    with open(

        output_dir /
        "metadata.json",

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            metadata,

            f,

            ensure_ascii=False,

            indent=4
        )


    # -------------------------------------------
    # 文件夹
    #
    # ↓
    #
    # zip
    # -------------------------------------------

    zip_path = shutil.make_archive(

        str(output_dir),

        "zip",

        root_dir=output_dir
    )


    return zip_path


# ============================================================
# 14. 核心预测函数
# ============================================================

def predict(
    input_image,
    model_name,
    threshold
):

    if input_image is None:

        raise gr.Error(
            "Please upload an ultrasound image first."
        )


    # ========================================================
    # Step 1：Preprocess
    # ========================================================

    (
        preprocessed_image,
        image_tensor
    ) = preprocess_image(
        input_image
    )


    # ========================================================
    # Step 2：Model
    # ========================================================

    model = get_model(
        model_name
    )


    # 如果使用 CUDA，
    # 计时前同步一次。
    if DEVICE.type == "cuda":

        torch.cuda.synchronize()


    start_time = time.perf_counter()


    # ========================================================
    # Step 3：Inference
    # ========================================================

    with torch.no_grad():

        logits = model(
            image_tensor
        )


        probability = torch.sigmoid(
            logits
        )


    if DEVICE.type == "cuda":

        torch.cuda.synchronize()


    inference_time = (

        time.perf_counter()
        -
        start_time

    )


    # ========================================================
    # Step 4：Probability
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
    # Step 5：Threshold
    # ========================================================

    binary_mask = (

        probability_map
        >=
        threshold

    ).astype(
        np.uint8
    )


    # ========================================================
    # Step 6：Predicted lesion area ratio
    # ========================================================

    lesion_area_ratio = (

        binary_mask.sum()
        /
        binary_mask.size

    )


    # ========================================================
    # Step 7：Heatmap
    # ========================================================

    probability_heatmap = (
        create_probability_heatmap(
            probability_map
        )
    )


    # ========================================================
    # Step 8：Overlay
    # ========================================================

    overlay = create_overlay(

        preprocessed_image,

        binary_mask
    )


    mask_display = (

        binary_mask
        *
        255

    ).astype(
        np.uint8
    )


    # ========================================================
    # Step 9：Metadata
    # ========================================================

    metadata = {

        "model":
            model_name,

        "threshold":
            float(
                threshold
            ),

        "predicted_lesion_area_ratio":
            float(
                lesion_area_ratio
            ),

        "inference_time_ms":
            float(
                inference_time
                *
                1000
            ),

        "device":
            str(
                DEVICE
            ),

        "image_size":
            [
                IMAGE_SIZE,
                IMAGE_SIZE
            ],

        "generated_at":
            datetime.now().isoformat()
    }


    # ========================================================
    # Step 10：结果下载包
    # ========================================================

    zip_path = save_prediction_package(

        preprocessed_image,

        probability_heatmap,

        binary_mask,

        overlay,

        metadata
    )


    # ========================================================
    # Step 11：结果描述
    # ========================================================

    if binary_mask.sum() == 0:

        prediction_note = (
            "No region exceeded the current threshold. "
            "This does NOT represent a clinical negative finding."
        )

    else:

        prediction_note = (
            "A segmentation region was detected "
            "above the selected probability threshold."
        )


    information = f"""
### Prediction Summary

**Model:** {model_name}

**Threshold:** {threshold:.2f}

**Predicted lesion area ratio:** {lesion_area_ratio * 100:.2f}%

**Inference time:** {inference_time * 1000:.1f} ms

**Device:** `{DEVICE}`

{prediction_note}
"""


    return (

        preprocessed_image,

        probability_heatmap,

        mask_display,

        overlay,

        information,

        zip_path
    )


# ============================================================
# 15. Demo Examples
# ============================================================

EXAMPLE_DIR = Path(
    "demo_examples"
)


EXAMPLE_FILES = [

    EXAMPLE_DIR /
    "01_resnet_rescue_benign_342.png",

    EXAMPLE_DIR /
    "02_typical_benign_271.png",

    EXAMPLE_DIR /
    "03_failure_benign_407.png"

]


# 只加载实际存在的文件，
# 避免因为 Examples 文件缺失导致 Demo 启动失败。
AVAILABLE_EXAMPLES = [

    [str(path)]

    for path in EXAMPLE_FILES

    if path.exists()

]


# ============================================================
# 16. CSS
# ============================================================

CSS = """
.gradio-container {
    max-width: 1350px !important;
    margin: auto !important;
}

.hero {
    text-align: center;
    padding: 18px;
}

.result-card {
    border-radius: 12px;
}

.metric-box {
    padding: 10px 16px;
    border-radius: 10px;
}
"""


# ============================================================
# 17. Gradio UI
# ============================================================

with gr.Blocks(

    title=(
        "Breast Ultrasound "
        "Lesion Segmentation"
    ),

    css=CSS

) as demo:


    # ========================================================
    # Header
    # ========================================================

    gr.Markdown(
        """
# 🩻 Breast Ultrasound Lesion Segmentation

### Deep-learning-based lesion segmentation on the BUSI dataset

Compare **Scratch U-Net**, **ResNet18-U-Net**, and
**Attention-ResNet18-U-Net**.

🏆 **Best standardized test model: ResNet18-U-Net — Dice 0.8310 / IoU 0.7377**

> Research and educational demonstration only.  
> Not intended for clinical diagnosis.
""",
        elem_classes=[
            "hero"
        ]
    )


    # ========================================================
    # Input Area
    # ========================================================

    with gr.Row():

        with gr.Column(
            scale=5
        ):

            input_image = gr.Image(

                label=(
                    "Breast Ultrasound Image"
                ),

                type="numpy",

                height=400
            )


            # ------------------------------------------------
            # 示例病例
            # ------------------------------------------------

            if AVAILABLE_EXAMPLES:

                gr.Examples(

                    examples=AVAILABLE_EXAMPLES,

                    inputs=[
                        input_image
                    ],

                    label=(
                        "Representative "
                        "BUSI Cases"
                    ),

                    cache_examples=False
                )


        # ====================================================
        # Settings
        # ====================================================

        with gr.Column(
            scale=4
        ):

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

                label="Segmentation Model"
            )


            threshold_slider = gr.Slider(

                minimum=0.1,

                maximum=0.9,

                value=0.5,

                step=0.05,

                label=(
                    "Probability Threshold"
                )
            )


            model_information = gr.Markdown(

                value=get_model_information(
                    "ResNet18-U-Net"
                ),

                elem_classes=[
                    "metric-box"
                ]
            )


            predict_button = gr.Button(

                "Run Segmentation",

                variant="primary"
            )


    # ========================================================
    # Results
    # ========================================================

    gr.Markdown(
        "## Segmentation Results"
    )


    with gr.Row():

        original_output = gr.Image(

            label="Preprocessed Ultrasound",

            height=330,

            elem_classes=[
                "result-card"
            ]
        )


        heatmap_output = gr.Image(

            label="Probability Heatmap",

            height=330,

            elem_classes=[
                "result-card"
            ]
        )


    with gr.Row():

        mask_output = gr.Image(

            label="Binary Segmentation Mask",

            height=330,

            elem_classes=[
                "result-card"
            ]
        )


        overlay_output = gr.Image(

            label="Prediction Overlay",

            height=330,

            elem_classes=[
                "result-card"
            ]
        )


    # ========================================================
    # Prediction Information
    # ========================================================

    with gr.Row():

        with gr.Column(
            scale=3
        ):

            prediction_information = gr.Markdown(
                """
### Prediction Summary

Run segmentation to view results.
"""
            )


        with gr.Column(
            scale=1
        ):

            download_button = gr.DownloadButton(

                label=(
                    "Download Prediction Package"
                ),

                variant="secondary"
            )


    # ========================================================
    # Events
    # ========================================================

    model_selector.change(

        fn=get_model_information,

        inputs=[
            model_selector
        ],

        outputs=[
            model_information
        ]
    )


    predict_button.click(

        fn=predict,

        inputs=[

            input_image,

            model_selector,

            threshold_slider

        ],

        outputs=[

            original_output,

            heatmap_output,

            mask_output,

            overlay_output,

            prediction_information,

            download_button

        ]
    )


# ============================================================
# 18. Launch
# ============================================================

if __name__ == "__main__":

    demo.launch()