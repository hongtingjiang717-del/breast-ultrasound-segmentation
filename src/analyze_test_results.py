"""
Test-set statistical analysis for BUSI segmentation models.

输入：
    per_sample_model_comparison.xlsx

输出：
    1. overall_model_summary.csv
    2. improvement_summary.csv
    3. dice_by_size_group.csv
    4. dice_by_class.csv

    5. dice_boxplot.png
    6. iou_boxplot.png
    7. resnet_gain_hist.png
    8. best_model_counts.png
    9. dice_by_size_group.png
    10. dice_by_class.png

分析模型：
    - U-Net
    - ResNet18-U-Net
    - Attention-ResNet18-U-Net
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 模型相关列名
# ============================================================

MODEL_COLUMNS = {

    "U-Net": {
        "dice": "unet_dice",
        "iou": "unet_iou"
    },

    "ResNet18-U-Net": {
        "dice": "resnet_dice",
        "iou": "resnet_iou"
    },

    "Attention-ResNet18-U-Net": {
        "dice": "attention_dice",
        "iou": "attention_iou"
    }

}


# ============================================================
# 2. 读取输入表格
# ============================================================

def load_results(
    input_file
):
    """
    根据后缀自动读取 xlsx / csv。
    """

    input_path = Path(
        input_file
    )


    if not input_path.exists():

        raise FileNotFoundError(
            f"Input file not found: "
            f"{input_path}"
        )


    suffix = input_path.suffix.lower()


    if suffix in [
        ".xlsx",
        ".xls"
    ]:

        df = pd.read_excel(
            input_path
        )


    elif suffix == ".csv":

        df = pd.read_csv(
            input_path
        )


    else:

        raise ValueError(
            "Only .xlsx, .xls or .csv "
            "files are supported."
        )


    print(
        f"Loaded file: {input_path}"
    )

    print(
        f"Number of test cases: {len(df)}"
    )


    return df


# ============================================================
# 3. 检查必要列
# ============================================================

def validate_columns(
    df
):
    """
    检查 Excel 里是否包含分析所需列。
    """

    required_columns = [

        "class",
        "size_group",

        "unet_dice",
        "unet_iou",

        "resnet_dice",
        "resnet_iou",

        "attention_dice",
        "attention_iou",

        "resnet_gain_vs_unet",
        "attention_change_vs_resnet"
    ]


    missing_columns = [

        column

        for column
        in required_columns

        if column not in df.columns

    ]


    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            +
            ", ".join(
                missing_columns
            )
        )


# ============================================================
# 4. 整体模型统计
# ============================================================

def calculate_overall_summary(
    df
):
    """
    计算：
        mean Dice
        median Dice
        std Dice
        mean IoU
        median IoU
        std IoU
    """

    rows = []


    for (
        model_name,
        columns
    ) in MODEL_COLUMNS.items():


        dice_values = df[
            columns[
                "dice"
            ]
        ]


        iou_values = df[
            columns[
                "iou"
            ]
        ]


        rows.append(
            {
                "Model":
                    model_name,

                "Mean Dice":
                    dice_values.mean(),

                "Median Dice":
                    dice_values.median(),

                "Std Dice":
                    dice_values.std(),

                "Mean IoU":
                    iou_values.mean(),

                "Median IoU":
                    iou_values.median(),

                "Std IoU":
                    iou_values.std()
            }
        )


    summary_df = pd.DataFrame(
        rows
    )


    return summary_df


# ============================================================
# 5. 每病例最佳模型
# ============================================================

def calculate_best_model_counts(
    df
):
    """
    对每一个 Test Case：

    比较三个模型 Dice，

    哪个最高，
    哪个就是该病例的 winner。
    """

    winners = []


    for _, row in df.iterrows():


        case_scores = {

            "U-Net":
                row[
                    "unet_dice"
                ],

            "ResNet18-U-Net":
                row[
                    "resnet_dice"
                ],

            "Attention-ResNet18-U-Net":
                row[
                    "attention_dice"
                ]
        }


        winner = max(

            case_scores,

            key=case_scores.get

        )


        winners.append(
            winner
        )


    result_df = df.copy()


    result_df[
        "best_model_by_dice"
    ] = winners


    counts = (

        result_df[
            "best_model_by_dice"
        ]

        .value_counts()

        .reindex(
            [
                "U-Net",
                "ResNet18-U-Net",
                "Attention-ResNet18-U-Net"
            ],
            fill_value=0
        )

        .rename_axis(
            "Model"
        )

        .reset_index(
            name="Number of test cases"
        )

    )


    return (
        result_df,
        counts
    )


# ============================================================
# 6. 模型改善统计
# ============================================================

def calculate_improvement_summary(
    df
):
    """
    对 ResNet vs U-Net，
    Attention vs ResNet
    做病例级 improvement 统计。
    """

    comparisons = []


    # --------------------------------------------------------
    # ResNet - U-Net
    # --------------------------------------------------------

    resnet_gain = df[
        "resnet_gain_vs_unet"
    ]


    comparisons.append(
        {
            "Comparison":
                "ResNet18-U-Net - U-Net",

            "Mean Dice Change":
                resnet_gain.mean(),

            "Median Dice Change":
                resnet_gain.median(),

            "Positive Cases":
                int(
                    (
                        resnet_gain
                        >
                        0
                    ).sum()
                ),

            "Negative Cases":
                int(
                    (
                        resnet_gain
                        <
                        0
                    ).sum()
                ),

            "No Change Cases":
                int(
                    (
                        resnet_gain
                        ==
                        0
                    ).sum()
                )
        }
    )


    # --------------------------------------------------------
    # Attention - ResNet
    # --------------------------------------------------------

    attention_change = df[
        "attention_change_vs_resnet"
    ]


    comparisons.append(
        {
            "Comparison":
                (
                    "Attention-ResNet18-U-Net "
                    "- ResNet18-U-Net"
                ),

            "Mean Dice Change":
                attention_change.mean(),

            "Median Dice Change":
                attention_change.median(),

            "Positive Cases":
                int(
                    (
                        attention_change
                        >
                        0
                    ).sum()
                ),

            "Negative Cases":
                int(
                    (
                        attention_change
                        <
                        0
                    ).sum()
                ),

            "No Change Cases":
                int(
                    (
                        attention_change
                        ==
                        0
                    ).sum()
                )
        }
    )


    return pd.DataFrame(
        comparisons
    )


# ============================================================
# 7. 按 lesion size 分组
# ============================================================

def calculate_size_group_summary(
    df
):
    """
    按：
        small
        medium
        large

    比较三个模型 Mean Dice。
    """

    summary = (

        df.groupby(
            "size_group"
        )[
            [
                "unet_dice",
                "resnet_dice",
                "attention_dice"
            ]
        ]

        .agg(
            [
                "mean",
                "median",
                "std",
                "count"
            ]
        )

    )


    return summary


# ============================================================
# 8. 按 benign / malignant 分组
# ============================================================

def calculate_class_summary(
    df
):
    """
    按：
        benign
        malignant

    比较三个模型 Dice。
    """

    summary = (

        df.groupby(
            "class"
        )[
            [
                "unet_dice",
                "resnet_dice",
                "attention_dice"
            ]
        ]

        .agg(
            [
                "mean",
                "median",
                "std",
                "count"
            ]
        )

    )


    return summary


# ============================================================
# 9. Dice Boxplot
# ============================================================

def plot_dice_boxplot(
    df,
    save_path
):
    """
    Boxplot + all individual test cases.

    每个模型：
        98 个 Test Case
        →
        每个病例显示为一个散点

    箱线图用于展示：
        中位数、IQR、整体分布

    散点用于展示：
        每一个真实病例
    """

    data = [
        df["unet_dice"].values,
        df["resnet_dice"].values,
        df["attention_dice"].values
    ]

    labels = [
        "U-Net",
        "ResNet18-\nU-Net",
        "Attention-\nResNet18-U-Net"
    ]

    positions = [
        1,
        2,
        3
    ]

    plt.figure(
        figsize=(9, 6)
    )

    # ========================================================
    # 1. 箱线图
    # ========================================================

    plt.boxplot(
        data,
        tick_labels=labels,

        # 不再单独画箱线图的离群点，
        # 因为下面我们会把 98 个点全部显示出来
        showfliers=False,

        showmeans=True,

        widths=0.45
    )


    # ========================================================
    # 2. 把全部 98 个病例叠加到箱线图上
    # ========================================================

    # 固定随机种子：
    # 保证每次生成图片时 jitter 位置一致
    rng = np.random.default_rng(
        seed=42
    )


    for position, values in zip(
        positions,
        data
    ):

        # ----------------------------------------------------
        # 如果所有点都放在 x=1 / x=2 / x=3，
        # 98 个点会严重重叠。
        #
        # 所以给 x 轴增加一点很小的随机抖动 jitter。
        # ----------------------------------------------------

        jitter = rng.normal(
            loc=0,
            scale=0.055,
            size=len(values)
        )


        x = (
            position
            +
            jitter
        )


        plt.scatter(
            x,
            values,

            s=22,

            alpha=0.55,

            edgecolors="none"
        )


    # ========================================================
    # 3. 图形设置
    # ========================================================

    plt.ylabel(
        "Dice Score"
    )

    plt.xlabel(
        "Model"
    )

    plt.title(
        "Test-set Dice Distribution (n=98)"
    )

    plt.ylim(
        -0.03,
        1.03
    )

    plt.grid(
        axis="y",
        alpha=0.2
    )

    plt.tight_layout()


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# 10. IoU Boxplot
# ============================================================

def plot_iou_boxplot(
    df,
    save_path
):
    """
    IoU Boxplot + all 98 individual cases.
    """

    data = [
        df["unet_iou"].values,
        df["resnet_iou"].values,
        df["attention_iou"].values
    ]

    labels = [
        "U-Net",
        "ResNet18-\nU-Net",
        "Attention-\nResNet18-U-Net"
    ]

    positions = [
        1,
        2,
        3
    ]


    plt.figure(
        figsize=(9, 6)
    )


    # ========================================================
    # Boxplot
    # ========================================================

    plt.boxplot(
        data,

        tick_labels=labels,

        showfliers=False,

        showmeans=True,

        widths=0.45
    )


    # ========================================================
    # All test cases
    # ========================================================

    rng = np.random.default_rng(
        seed=42
    )


    for position, values in zip(
        positions,
        data
    ):

        jitter = rng.normal(
            loc=0,
            scale=0.055,
            size=len(values)
        )


        x = (
            position
            +
            jitter
        )


        plt.scatter(
            x,
            values,

            s=22,

            alpha=0.55,

            edgecolors="none"
        )


    # ========================================================
    # Figure settings
    # ========================================================

    plt.ylabel(
        "IoU Score"
    )

    plt.xlabel(
        "Model"
    )

    plt.title(
        "Test-set IoU Distribution (n=98)"
    )

    plt.ylim(
        -0.03,
        1.03
    )

    plt.grid(
        axis="y",
        alpha=0.2
    )

    plt.tight_layout()


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 11. ResNet Dice Gain Histogram
# ============================================================

def plot_pairwise_gain_hist(
    gains,
    model_new,
    model_reference,
    save_path
):
    """
    绘制两个模型之间的逐病例 Dice 差值分布。

    gains:
        Dice_new - Dice_reference

    > 0:
        新模型优于参考模型

    < 0:
        新模型劣于参考模型
    """

    gains = np.asarray(
        gains
    )


    # ========================================================
    # 基本统计
    # ========================================================

    mean_gain = np.mean(
        gains
    )

    median_gain = np.median(
        gains
    )

    positive_cases = np.sum(
        gains > 0
    )

    negative_cases = np.sum(
        gains < 0
    )

    equal_cases = np.sum(
        gains == 0
    )


    # ========================================================
    # 绘图
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )


    plt.hist(

        gains,

        bins=20,

        edgecolor="black",

        alpha=0.8
    )


    # --------------------------------------------------------
    # x = 0
    #
    # 代表两个模型表现完全一致
    # --------------------------------------------------------

    plt.axvline(

        x=0,

        linestyle="--",

        linewidth=1.5,

        label="No change"
    )


    # --------------------------------------------------------
    # Mean difference
    # --------------------------------------------------------

    plt.axvline(

        x=mean_gain,

        linestyle="-",

        linewidth=1.8,

        label=(
            f"Mean = "
            f"{mean_gain:+.3f}"
        )
    )


    # --------------------------------------------------------
    # Median difference
    # --------------------------------------------------------

    plt.axvline(

        x=median_gain,

        linestyle=":",

        linewidth=1.8,

        label=(
            f"Median = "
            f"{median_gain:+.3f}"
        )
    )


    # ========================================================
    # 坐标轴
    # ========================================================

    plt.xlabel(

        f"Dice Difference "
        f"({model_new} - {model_reference})"

    )


    plt.ylabel(
        "Number of Test Cases"
    )


    plt.title(

        f"Per-case Dice Difference\n"
        f"{model_new} vs {model_reference}"

    )


    # ========================================================
    # 在图中加入病例计数
    # ========================================================

    text = (

        f"Improved: {positive_cases}\n"
        f"Decreased: {negative_cases}\n"
        f"Equal: {equal_cases}"

    )


    plt.text(

        0.97,
        0.95,

        text,

        transform=plt.gca().transAxes,

        ha="right",

        va="top",

        bbox=dict(
            boxstyle="round",
            alpha=0.15
        )
    )


    plt.legend()


    plt.grid(
        axis="y",
        alpha=0.25
    )


    plt.tight_layout()


    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# 12. 每病例最佳模型柱状图
# ============================================================

def plot_best_model_counts(
    counts,
    save_path
):

    plt.figure(
        figsize=(8, 6)
    )


    bars = plt.bar(

        counts[
            "Model"
        ],

        counts[
            "Number of test cases"
        ]

    )


    plt.ylabel(
        "Number of Test Cases"
    )


    plt.xlabel(
        "Best Model by Dice"
    )


    plt.title(
        "Per-case Best Model Counts"
    )


    plt.xticks(
        rotation=10
    )


    # 在柱子上写具体数字
    for bar in bars:

        height = bar.get_height()


        plt.text(

            bar.get_x()
            +
            bar.get_width() / 2,

            height + 0.5,

            f"{int(height)}",

            ha="center",

            va="bottom"
        )


    plt.tight_layout()


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# 13. 按病灶大小画 Mean Dice
# ============================================================

def plot_dice_by_size(
    df,
    save_path
):

    size_order = [

        "small",
        "medium",
        "large"

    ]


    grouped = (

        df.groupby(
            "size_group"
        )[
            [
                "unet_dice",
                "resnet_dice",
                "attention_dice"
            ]
        ]

        .mean()

        .reindex(
            size_order
        )

    )


    x = np.arange(
        len(
            grouped.index
        )
    )


    width = 0.25


    plt.figure(
        figsize=(9, 6)
    )


    plt.bar(

        x - width,

        grouped[
            "unet_dice"
        ],

        width,

        label="U-Net"
    )


    plt.bar(

        x,

        grouped[
            "resnet_dice"
        ],

        width,

        label="ResNet18-U-Net"
    )


    plt.bar(

        x + width,

        grouped[
            "attention_dice"
        ],

        width,

        label=(
            "Attention-ResNet18-U-Net"
        )
    )


    plt.xticks(

        x,

        grouped.index
    )


    plt.ylabel(
        "Mean Dice"
    )


    plt.xlabel(
        "Lesion Size Group"
    )


    plt.title(
        "Dice Performance by Lesion Size"
    )


    plt.ylim(
        0,
        1
    )


    plt.legend()


    plt.grid(
        axis="y",
        alpha=0.25
    )


    plt.tight_layout()


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# 14. 按 benign / malignant 画 Mean Dice
# ============================================================

def plot_dice_by_class(
    df,
    save_path
):

    grouped = (

        df.groupby(
            "class"
        )[
            [
                "unet_dice",
                "resnet_dice",
                "attention_dice"
            ]
        ]

        .mean()

    )


    x = np.arange(
        len(
            grouped.index
        )
    )


    width = 0.25


    plt.figure(
        figsize=(8, 6)
    )


    plt.bar(

        x - width,

        grouped[
            "unet_dice"
        ],

        width,

        label="U-Net"
    )


    plt.bar(

        x,

        grouped[
            "resnet_dice"
        ],

        width,

        label="ResNet18-U-Net"
    )


    plt.bar(

        x + width,

        grouped[
            "attention_dice"
        ],

        width,

        label=(
            "Attention-ResNet18-U-Net"
        )
    )


    plt.xticks(

        x,

        grouped.index
    )


    plt.ylabel(
        "Mean Dice"
    )


    plt.xlabel(
        "BUSI Class"
    )


    plt.title(
        "Dice Performance by Class"
    )


    plt.ylim(
        0,
        1
    )


    plt.legend()


    plt.grid(
        axis="y",
        alpha=0.25
    )


    plt.tight_layout()


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()

def plot_pairwise_scatter(
    x_values,
    y_values,
    x_label,
    y_label,
    title,
    save_path
):

    plt.figure(
        figsize=(7, 7)
    )


    plt.scatter(

        x_values,

        y_values,

        s=35,

        alpha=0.65
    )


    # y = x
    #
    # 在线上的点：
    # 两模型性能相同
    plt.plot(

        [0, 1],

        [0, 1],

        linestyle="--",

        linewidth=1.5,

        label="Equal performance"
    )


    plt.xlim(
        -0.03,
        1.03
    )

    plt.ylim(
        -0.03,
        1.03
    )


    plt.xlabel(
        x_label
    )

    plt.ylabel(
        y_label
    )


    plt.title(
        title
    )


    plt.grid(
        alpha=0.2
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()
# ============================================================
# 15. Main
# ============================================================

def main(
    args
):

    # --------------------------------------------------------
    # 输出目录
    # --------------------------------------------------------

    output_dir = Path(
        args.output_dir
    )


    figure_dir = (

        output_dir
        /
        "figures"

    )


    output_dir.mkdir(

        parents=True,
        exist_ok=True

    )


    figure_dir.mkdir(

        parents=True,
        exist_ok=True

    )


    # --------------------------------------------------------
    # 读取 Excel
    # --------------------------------------------------------

    df = load_results(
        args.input
    )


    validate_columns(
        df
    )


    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    overall_summary = (
        calculate_overall_summary(
            df
        )
    )


    # --------------------------------------------------------
    # Best model counts
    # --------------------------------------------------------

    (
        df_with_winner,
        best_model_counts
    ) = calculate_best_model_counts(
        df
    )


    # --------------------------------------------------------
    # Improvement
    # --------------------------------------------------------

    improvement_summary = (
        calculate_improvement_summary(
            df
        )
    )


    # --------------------------------------------------------
    # Size
    # --------------------------------------------------------

    size_summary = (
        calculate_size_group_summary(
            df
        )
    )


    # --------------------------------------------------------
    # Class
    # --------------------------------------------------------

    class_summary = (
        calculate_class_summary(
            df
        )
    )


    # ========================================================
    # 保存 CSV
    # ========================================================

    overall_summary.to_csv(

        output_dir /
        "overall_model_summary.csv",

        index=False
    )


    improvement_summary.to_csv(

        output_dir /
        "improvement_summary.csv",

        index=False
    )


    best_model_counts.to_csv(

        output_dir /
        "best_model_counts.csv",

        index=False
    )


    size_summary.to_csv(

        output_dir /
        "dice_by_size_group.csv"

    )


    class_summary.to_csv(

        output_dir /
        "dice_by_class.csv"

    )


    df_with_winner.to_csv(

        output_dir /
        "per_sample_with_best_model.csv",

        index=False
    )


    # ========================================================
    # 画图
    # ========================================================

    plot_dice_boxplot(

        df,

        figure_dir /
        "dice_boxplot.png"
    )


    plot_iou_boxplot(

        df,

        figure_dir /
        "iou_boxplot.png"
    )


    # ============================================================
    # Pairwise comparison 1
    #
    # ResNet18-U-Net vs U-Net
    # ============================================================

    plot_pairwise_gain_hist(

        gains=df[
            "resnet_gain_vs_unet"
        ],

        model_new="ResNet18-U-Net",

        model_reference="U-Net",

        save_path=(
            figure_dir /
            "resnet_vs_unet_dice_difference.png"
        )
    )
    # ============================================================
    # Pairwise comparison 2
    #
    # Attention-ResNet18-U-Net vs ResNet18-U-Net
    # ============================================================

    plot_pairwise_gain_hist(

        gains=df[
            "attention_change_vs_resnet"
        ],

        model_new="Attention-ResNet18-U-Net",

        model_reference="ResNet18-U-Net",

        save_path=(
            figure_dir /
            "attention_vs_resnet_dice_difference.png"
        )
    )
    plot_pairwise_scatter(

        x_values=df[
            "unet_dice"
        ],

        y_values=df[
            "resnet_dice"
        ],

        x_label="U-Net Dice",

        y_label="ResNet18-U-Net Dice",

        title=(
            "Paired Test-case Comparison\n"
            "ResNet18-U-Net vs U-Net"
        ),

        save_path=(
            figure_dir /
            "resnet_vs_unet_scatter.png"
        )
    )
    plot_pairwise_scatter(
        x_values=df[
            "resnet_dice"
        ],

        y_values=df[
            "attention_dice"
        ],

        x_label="ResNet18-U-Net Dice",

        y_label="Attention-ResNet18-U-Net Dice",

        title=(
            "Paired Test-case Comparison\n"
            "Attention-ResNet18-U-Net vs ResNet18-U-Net"
        ),

        save_path=(
            figure_dir /
            "attention_vs_resnet_scatter.png"
        )
    )
    plot_best_model_counts(

        best_model_counts,

        figure_dir /
        "best_model_counts.png"
    )


    plot_dice_by_size(

        df,

        figure_dir /
        "dice_by_size_group.png"
    )


    plot_dice_by_class(

        df,

        figure_dir /
        "dice_by_class.png"
    )


    # ========================================================
    # 控制台输出
    # ========================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "Overall Model Summary"
    )

    print(
        "========================================"
    )


    print(

        overall_summary.to_string(
            index=False
        )

    )


    print(
        "\n"
        "========================================"
    )

    print(
        "Improvement Summary"
    )

    print(
        "========================================"
    )


    print(

        improvement_summary.to_string(
            index=False
        )

    )


    print(
        "\n"
        "========================================"
    )

    print(
        "Best Model Counts"
    )

    print(
        "========================================"
    )


    print(

        best_model_counts.to_string(
            index=False
        )

    )


    print(
        "\nResults saved to:"
    )

    print(
        output_dir
    )


    print(
        "\nFigures saved to:"
    )

    print(
        figure_dir
    )


# ============================================================
# 16. CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(

        "--input",

        type=str,

        default=(
            "results/final/"
            "prediction_analysis/"
            "per_sample_model_comparison.xlsx"
        ),

        help=(
            "Path to per-sample "
            "comparison table"
        )
    )


    parser.add_argument(

        "--output_dir",

        type=str,

        default=(
            "results/final/"
            "test_set_analysis"
        ),

        help=(
            "Directory for summary "
            "tables and figures"
        )
    )


    args = parser.parse_args()


    main(
        args
    )