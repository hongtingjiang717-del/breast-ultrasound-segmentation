# Day 01：BUSI 数据集探索与质量检查

## 项目目标

本项目基于 **BUSI 乳腺超声数据集**，构建乳腺病灶自动分割流程，实现从超声图像中自动识别并分割病灶区域。

---

## 今日完成

### 1. 数据集结构检查

完成 BUSI 数据集文件结构梳理，区分：

- benign：良性病灶
- malignant：恶性病灶
- normal：正常乳腺超声图像
- ultrasound image：原始超声图像
- mask：人工标注的病灶区域

本项目当前主要使用具有病灶标注的 **benign + malignant** 图像进行分割任务。

---

### 2. Image-Mask 配对检查

根据图像文件名自动寻找对应的 mask，并检查是否存在缺失标注。

最终共获得：

```text
病灶图像：647 张
Missing masks：0
```

说明所有纳入分割任务的图像均存在对应病灶标注。

---

### 3. 多 Mask 检测与合并

BUSI 中部分超声图像对应多个 mask。

通过文件名自动搜索：

```text
xxx_mask.png
xxx_mask_1.png
xxx_mask_2.png
...
```

对于同一张图像的多个 mask，采用逐像素合并的方式生成最终病灶 mask，保证所有标注病灶区域均被保留。

---

### 4. 图像尺寸统计

统计所有病灶图像的高度和宽度。

结果表明 BUSI 图像尺寸并不统一，因此后续进入神经网络前需要进行统一尺寸处理。

---

### 5. 类别分布统计

共包含 647 张病灶图像：

```text
Benign：437
Malignant：210
```

良性样本数量多于恶性样本。

需要注意：当前任务是**病灶分割任务**，模型主要预测每个像素是否属于病灶，而不是进行良恶性分类。

---

### 6. 病灶面积统计

根据 mask 计算：

```text
lesion_area_ratio
=
病灶像素数 / 图像总像素数
```

结果显示，大部分乳腺病灶在整张超声图像中占比较小。

因此后续模型评价不能只依赖 Pixel Accuracy，否则即使模型大量预测为背景，也可能获得较高 Accuracy。

后续主要使用：

- Dice
- IoU

作为病灶分割核心评价指标。

---

### 7. Ground Truth 可视化

完成以下可视化检查：

- 原始乳腺超声图像
- Ground Truth Mask
- Image + Mask Overlay

通过 Overlay 检查人工标注区域与实际病灶位置是否对应，为后续模型训练进行数据质量控制。

---

## 今日核心数据流程

```text
BUSI 原始数据
      ↓
识别原始超声图像
      ↓
Image-Mask 配对
      ↓
检测多个 Mask
      ↓
合并 Mask
      ↓
图像尺寸统计
      ↓
类别分布统计
      ↓
病灶面积占比统计
      ↓
Ground Truth / Overlay 可视化
```

---

## 今日掌握的关键知识

- 医学图像分割需要同时包含原始图像和对应 Mask
- BUSI 中一张图像可能对应多个 Mask，需要进行合并
- Mask 表示人工标注的真实病灶区域，即 Ground Truth
- 病灶通常只占整张图像的一小部分，存在明显的前景/背景像素不平衡
- 分割任务中 Accuracy 可能虚高，因此 Dice 和 IoU 更适合作为主要评价指标
- 模型训练前应先进行 Image-Mask 配对及 Overlay 人工质量检查

---

## 下一步

- 划分 Train / Validation / Test 数据集
- 保证不同数据集中的良恶性比例和病灶大小分布基本一致
- 构建 PyTorch Dataset 和 DataLoader
- 将原始 BUSI 数据转换为可以直接输入深度学习模型的数据格式

# Day 02：BUSI 数据划分与数据管线构建

## 今日完成

### 1. 数据集划分

将 647 张乳腺超声病灶图像划分为：

- Train：452 张
- Validation：97 张
- Test：98 张

划分时同时考虑：

- 良性 / 恶性类别比例
- small / medium / large 病灶面积分布

三组数据的类别比例和病灶大小分布基本一致，保证后续模型评估更加公平。

---

### 2. 构建 PyTorch Dataset

完成 `src/dataset.py`，实现：

- 根据 `image_path` 读取超声图像
- 自动寻找对应的一个或多个 mask
- 多 mask 自动合并
- 图像统一调整为 `256 × 256`
- 超声图使用双线性插值
- mask 使用最近邻插值
- 图像归一化到 `[0, 1]`
- mask 二值化为 `0 / 1`
- 转换为 PyTorch Tensor

单个样本格式：

```text
Image shape: [1, 256, 256]
Mask shape:  [1, 256, 256]
```

---

### 3. 构建 DataLoader

使用 DataLoader 将多个样本组合成 batch。

当前设置：

```text
Batch size = 8
```

模型训练时的数据格式：

```text
[Batch, Channel, Height, Width]
[8, 1, 256, 256]
```

---

### 4. 数据质量检查

完成：

- image 与 mask 尺寸检查
- image 数值范围检查
- mask 二值化检查
- 多 mask 合并检查
- Image / Mask / Overlay 可视化
- 人工检查图像与病灶标注是否正确对齐

目前数据管线可以正常用于后续模型训练。

---

## 今日理解的图像处理知识

### Resize 与插值

`Resize` 决定图像最终尺寸，例如：

```text
原图 → 256 × 256
```

插值算法决定 resize 后的新像素值如何计算。

- 超声图像：使用 Bilinear 双线性插值
- 分割 mask：使用 Nearest Neighbor 最近邻插值

mask 使用最近邻插值，是为了避免产生不存在的中间类别值。

### 长宽比问题

直接将不同尺寸图像 resize 到 `256 × 256` 可能造成一定几何形变。

后续可尝试：

```text
等比例 Resize + Padding
```

作为一种改进方案。

---

## 今日使用的核心 Python 库

| 库 | 作用 |
|---|---|
| `pandas` | 读取和处理 CSV、DataFrame |
| `numpy` | 数组和数值运算 |
| `PIL` | 读取图像、Resize |
| `pathlib` | 文件和路径处理 |
| `matplotlib` | 图像可视化 |
| `sklearn` | Train / Val / Test 数据划分 |
| `torch` | Tensor 和深度学习基础 |
| `torch.utils.data` | Dataset 和 DataLoader |

---

## 今日核心函数与语法

### `pd.read_csv()`

读取 CSV 文件：

```python
df = pd.read_csv("train.csv")
```

返回一个 DataFrame。

---

### `Path()`

用于处理文件路径：

```python
image_path = Path(image_path)
```

相比直接使用字符串，路径拼接和文件查找更加方便。

---

### `glob()`

用于根据规则搜索文件：

```python
image_dir.glob("*_mask*.png")
```

在本项目中用于寻找一张超声图对应的所有 mask。

---

### `np.maximum()`

逐像素比较两个数组并保留较大值。

在本项目中用于合并多个 mask，相当于：

```text
mask1 OR mask2
```

---

### `Image.open()`

读取图片：

```python
image = Image.open(image_path)
```

---

### `.convert("L")`

将图片转换成单通道灰度图。

---

### `.resize()`

改变图片尺寸：

```python
image.resize((256, 256))
```

---

### `torch.from_numpy()`

将 NumPy 数组转换成 PyTorch Tensor。

---

### `.unsqueeze(0)`

增加一个维度。

例如：

```text
[256, 256]

→

[1, 256, 256]
```

这里增加的是 Channel 维度。

---

### `torch.unique()`

查看 Tensor 中有哪些不同数值。

本项目中主要用于检查 mask 是否只有：

```text
0 = 背景
1 = 病灶
```

---

### `Dataset`

负责定义：

> 一张数据应该如何读取和处理。

核心函数包括：

```python
__init__()
__len__()
__getitem__()
```

---

### `DataLoader`

负责：

> 将 Dataset 中的数据按照 batch 一批一批送给模型。

例如：

```text
8 张图
↓
一个 batch
↓
[8, 1, 256, 256]
```

---

## 今日接触的 Python 基础

### `def`

用于定义函数：

```python
def find_mask_paths(image_path):
```

---

### `class`

用于定义一个类：

```python
class BUSIDataset(Dataset):
```

后续搭建 U-Net 时还会大量使用。

---

### `self`

表示当前创建出来的对象本身。

例如：

```python
self.image_size = image_size
```

表示把 `image_size` 保存到当前 Dataset 对象中。

---

### `return`

表示函数最终返回什么结果。

例如：

```python
return mask_paths
```

---

## 今日数据流程

```text
BUSI 原始数据
      ↓
数据统计
      ↓
Train / Val / Test 划分
      ↓
Dataset
      ↓
寻找并合并 Mask
      ↓
Resize
      ↓
归一化 / 二值化
      ↓
Tensor
      ↓
DataLoader
      ↓
Image / Mask / Overlay 检查
```

---

## 下一步

明天开始搭建 **U-Net Baseline 基本框架**。

学习重点：

### 深度学习部分

- U-Net 整体结构
- Encoder
- Decoder
- Skip Connection
- Feature Map
- Channel
- 卷积与下采样、上采样

### PyTorch 核心函数

- `nn.Module`
- `nn.Conv2d`
- `nn.ReLU`
- `nn.MaxPool2d`
- `nn.ConvTranspose2d`
- `nn.Sequential`
- `torch.cat()`
- `forward()`

### Python 基础

重点理解：

- `class`
- `self`
- `__init__`
- `super()`
- 类的继承
- 函数参数
- `return`
- 对象如何被创建和调用

目标：不仅能够运行 U-Net，还能够看懂并解释 U-Net 的核心代码。

# Day 03：U-Net 基础架构与前向预测流程

## 今日完成

### 1. 从零搭建 U-Net Baseline

完成 `src/model.py`，构建基础 U-Net，包括：

- `ConvBlock` 双卷积模块
- Encoder 编码器
- Bottleneck
- Decoder 解码器
- Skip Connection
- `1×1 Conv` 最终输出层

由于 BUSI 超声图像采用灰度输入，因此模型输入为：

```text
[B, 1, 256, 256]
```

最终输出：

```text
[B, 1, 256, 256]
```

与 Ground Truth Mask 尺寸保持一致。

---

### 2. Encoder

Encoder 通过卷积提取特征，并利用 MaxPooling 逐步降低空间尺寸：

```text
[1,256,256]
      ↓
[64,256,256]
      ↓
[128,128,128]
      ↓
[256,64,64]
      ↓
[512,32,32]
```

核心规律：

```text
空间尺寸逐渐减小
Channel 数逐渐增加
```

卷积主要负责提取特征并改变 Channel，池化主要负责降低特征图空间尺寸。

---

### 3. Bottleneck

Encoder 最深层继续经过：

```text
[512,32,32]
      ↓ MaxPool
[512,16,16]
      ↓ ConvBlock
[1024,16,16]
```

用于提取更加抽象、高层次的病灶特征。

---

### 4. Decoder

Decoder 通过转置卷积逐步恢复空间分辨率：

```text
16×16
 ↓
32×32
 ↓
64×64
 ↓
128×128
 ↓
256×256
```

同时 Channel 数逐渐减少。

---

### 5. Skip Connection

使用：

```python
torch.cat()
```

将 Decoder 特征与对应 Encoder 特征在 Channel 维进行拼接。

例如：

```text
Decoder：
[B,512,32,32]

Encoder：
[B,512,32,32]

        ↓ cat(dim=1)

[B,1024,32,32]
```

Skip Connection 可以将 Encoder 中保留的高分辨率细节重新提供给 Decoder，有助于恢复病灶的位置和边界信息。

---

### 6. 最终输出层

使用：

```python
nn.Conv2d(
    64,
    1,
    kernel_size=1
)
```

将：

```text
[B,64,256,256]
```

转换为：

```text
[B,1,256,256]
```

最终每个像素对应一个病灶预测分数。

---

### 7. 模型参数检查

使用：

```python
model.parameters()
p.numel()
p.requires_grad
```

统计模型总参数量与可训练参数量，进一步理解神经网络训练的本质：

```text
随机初始化参数
      ↓
计算 Loss
      ↓
反向传播
      ↓
更新卷积核等模型参数
```

---

### 8. 接入真实 BUSI 数据

从 `DataLoader` 中读取真实训练数据：

```text
Images：
[B,1,256,256]

Masks：
[B,1,256,256]
```

并成功将真实超声图像送入 U-Net：

```text
BUSI Image
      ↓
Dataset
      ↓
DataLoader
      ↓
U-Net
      ↓
Logits
```

完成数据管线与模型之间的连接。

---

## Logits、Probability 与 Prediction Mask

U-Net 最终直接输出的是：

```text
Logits
```

其数值可以是任意实数，并不是直接的 `0 / 1` Mask。

完整预测过程：

```text
U-Net
  ↓
Logits
  ↓
Sigmoid
  ↓
Probability（0~1）
  ↓
Threshold = 0.5
  ↓
Prediction Mask（0/1）
```

使用：

```python
probs = torch.sigmoid(logits)

pred_masks = (probs > 0.5).float()
```

目前模型尚未训练，因此预测 Mask 基本为随机结果，这是正常现象。

---

## 今日理解的 U-Net 核心逻辑

### Encoder

```text
卷积：
提取特征 + 改变 Channel

池化：
降低 H/W
```

整体：

```text
H/W ↓
Channel ↑
```

### Decoder

```text
上采样：
H/W ↑
Channel ↓

torch.cat：
拼接 Encoder 特征
Channel ↑
H/W 不变

ConvBlock：
融合特征
Channel ↓
H/W 不变
```

最终恢复为与原始 Mask 相同的空间尺寸。

---

## 今日学习的 Python / PyTorch 核心知识

| 函数 / 语法 | 作用 |
|---|---|
| `class` | 定义类 |
| `__init__()` | 初始化模型结构 |
| `self` | 当前对象 |
| `super().__init__()` | 初始化父类 `nn.Module` |
| `forward()` | 定义数据如何经过网络 |
| `nn.Module` | PyTorch 神经网络基础类 |
| `nn.Sequential` | 顺序组合多个网络层 |
| `nn.Conv2d` | 二维卷积 |
| `nn.BatchNorm2d` | 稳定特征分布 |
| `nn.ReLU` | 引入非线性 |
| `nn.MaxPool2d` | 下采样 |
| `nn.ConvTranspose2d` | 上采样 |
| `torch.cat()` | 在指定维度拼接 Tensor |
| `model.parameters()` | 获取模型参数 |
| `p.numel()` | 统计 Tensor 元素数量 |
| `model.eval()` | 切换到推理模式 |
| `torch.no_grad()` | 推理时关闭梯度计算 |
| `torch.sigmoid()` | 将 Logits 转换为 0~1 概率 |
| `iter()` / `next()` | 从 DataLoader 中读取一个 Batch |

---

## 今日完整流程

```text
BUSI Image
     ↓
Dataset
     ↓
DataLoader
     ↓
[B,1,256,256]
     ↓
Encoder
     ↓
Bottleneck
     ↓
Decoder + Skip Connection
     ↓
1×1 Conv
     ↓
Logits
[B,1,256,256]
     ↓
Sigmoid
     ↓
Probability
     ↓
Threshold
     ↓
Prediction Mask
```

---

## 下一步：Day 04

开始模型正式训练，重点学习：

- Binary Cross Entropy（BCE）
- Dice Score
- Dice Loss
- `BCEWithLogitsLoss`
- Dice + BCE 联合损失
- Optimizer
- `optimizer.zero_grad()`
- `loss.backward()`
- `optimizer.step()`
- Epoch 与 Batch
- 完整 PyTorch Training Loop

目标：真正理解模型如何根据 Ground Truth 计算误差，并通过反向传播自动更新卷积核参数。


### 补充：
torch
├── Tensor        ← 一种“数据类型 / 对象”（类）
├── nn            ← 搭神经网络的模块工具箱（对象）
├── optim         ← 优化器工具箱（对象）
└── autograd      ← 自动求导系统


# Day 04：Loss、反向传播与 U-Net 训练机制

## 今日目标

今天开始从“模型结构搭建”正式进入“模型训练”。

前 3 天已经完成：

- BUSI 数据探索与质量检查
- Dataset / DataLoader 构建
- U-Net Encoder、Bottleneck、Decoder、Skip Connection
- Logits → Sigmoid → Prediction Mask
- 真实 BUSI 数据成功送入 U-Net

Day 04 重点理解：

1. Loss 是什么
2. BCE Loss 和 Dice Loss 为什么适合分割任务
3. Optimizer 是什么
4. `loss.backward()` 到底做了什么
5. 梯度和链式法则
6. 卷积核参数如何通过反向传播更新
7. `Tensor`、`nn`、`autograd`、`optim` 的关系
8. 一个完整 Batch 的训练流程
9. Epoch 和 Batch 的关系

---

## 1. U-Net 训练的整体流程

一个 Batch 的基本训练过程：

BUSI Image
    ↓
U-Net
    ↓
Logits
    ↓
与 Ground Truth Mask 比较
    ↓
Loss
    ↓
loss.backward()
    ↓
计算每个模型参数的梯度
    ↓
optimizer.step()
    ↓
更新模型参数
    ↓
下一批数据继续训练

深度学习模型训练的核心可以概括为：
预测
↓
发现错误
↓
计算错误应该如何影响各个参数
↓
修改参数
↓
再次预测

## 2.BCEWihLogitsLoss
BCE Loss是逐像素判断模型预测是否接近真实标签；
BCEWihLogitsLoss内部包含sigmoid（0-1化）和BCE，正确的流程：
Logits
↓
Sigmoid
↓
Probability
↓
Threshold
↓
Prediction Mask

## 3.Dice score
用于判断模型预测病灶和真实病灶之间的重合程度这个值越大越好，但是loss是越小越好，所以做一个1-操作

$$
DiceLoss = 1 - Dice
$$

## 4.一次batch训练的核心代码
optimizer.zero_grad()  #清空上一轮batch留下的梯度，因为梯度会累加

logits = model(
    images
)

loss = criterion(
    logits,
    masks
)

loss.backward()  #torch.Tensor自带的方法

optimizer.step()

## 5.Tensor、nn、autograd、optim 的关系
torch
│
├── Tensor（pytorch中最核心的数据结构之一）
│   └── 存储数据、参数、中间结果、Loss
│
├── nn(pytorch专门用于构建神经网络的模块)
│   └── 搭建神经网络
│       ├── Conv2d
│       ├── ReLU
│       ├── MaxPool2d
│       ├── BatchNorm2d
│       └── BCEWithLogitsLoss
│
├── autograd
│   └── 自动求导
│
└── optim
    └── 优化器
        ├── Adam
        └── SGD

## 6.U-Net的Encoder的作用
提取越来越抽象的特征＋逐步降低空间尺寸
[1,256,256]
↓
[64,256,256]
↓
[128,128,128]
↓
[256,64,64]
↓
[512,32,32]

## 7.bottleneck
位于encoder和decoder之间，是整个架构中空间分辨率最低、特征最浓、语义最抽象的位置
e4
[512,32,32]

↓ MaxPool

[512,16,16]

↓ Bottleneck ConvBlock

[1024,16,16]

他和encoder其实都是在convblock，也就是说数学操作是差不多的，区别在于
Encoder
→ 是整个下采样过程

Bottleneck
→ 是下采样结束后最底部的特征融合层

## 8.反向传播
### （1）Skip Connection 的反向传播
U-Net 中：
torch.cat(    [decoder_feature, encoder_feature],    dim=1)


前向时：
Decoder Feature
       +
Encoder Feature
       ↓
Channel 拼接

例如：
64 channels
+
64 channels
=
128 channels

反向传播时：
128-channel Gradient
↓
按照原来的 Channel 切开
↓
分别传回 Decoder 和 Encoder

因此 Encoder 中的特征除了主路径，还可以通过 Skip Connection 接收到梯度。

### （2）ReLU 的反向传播
ReLU：
$$
ReLU(x)=
\max(0,x)
$$
导数：

$$
ReLU'(x)=
\begin{cases}
1,&x>0\\
0,&x<0
\end{cases}
$$

因此：
如果前向值 > 0
→ Gradient 可以继续传

如果前向值 < 0
→ 该位置 Gradient 变为 0

### （3）MaxPool 的反向传播
例如前向：
1  5
2  3

MaxPool 输出：
5

如果反向传来的梯度为：
0.8

则只会传回最大值所在位置：
0   0.8
0   0

因为：
前向输出 5
只来自原来的那个最大值位置

## 9.taining loop
num_epochs = 5


for epoch in range(
    num_epochs
):

    model.train()

    running_loss = 0.0


    for batch in train_loader:

        images = batch["image"]

        masks = batch["mask"]


        optimizer.zero_grad()


        logits = model(
            images
        )


        loss = criterion(
            logits,
            masks
        )


        loss.backward()


        optimizer.step()


        running_loss += (
            loss.item()
        )


    epoch_loss = (
        running_loss
        /
        len(train_loader)
    )


    print(
        f"Epoch "
        f"{epoch + 1}/{num_epochs} "
        f"- Train Loss: "
        f"{epoch_loss:.4f}"
    )

## 10.今日最重要的核心逻辑
模型刚开始：

卷积核参数随机初始化

        ↓

输入 BUSI 图像

        ↓

Forward

        ↓

模型预测 Mask

        ↓

Prediction 与 Ground Truth 比较

        ↓

BCE + Dice Loss

        ↓

loss.backward()

        ↓

通过链式法则计算：
Loss 对每一个模型参数的偏导

        ↓

所有 Gradient 存入 param.grad

        ↓

optimizer.step()

        ↓

根据 Gradient 修改参数

        ↓

下一批数据

        ↓

再次预测

        ↓

再次计算 Loss

        ↓

再次更新参数

        ↓

不断重复

        ↓

模型逐渐学习到
有利于乳腺病灶分割的特征

## 11.今日核心概念总结
| 概念 | 含义 |
|---|---|
| Loss | 衡量预测和 Ground Truth 的差距 |
| BCE Loss | 逐像素进行二分类误差计算 |
| Dice Score | 衡量预测区域与真实区域重合程度 |
| Dice Loss | `1 - Dice` |
| Gradient | Loss 对参数的局部变化率 |
| Chain Rule | 反向传播的数学基础 |
| Backpropagation | 从 Loss 开始逐层计算梯度 |
| `loss.backward()` | 启动 PyTorch 自动求导 |
| `param.grad` | 保存参数对应的梯度 |
| Optimizer | 根据梯度更新参数 |
| Adam | 当前使用的优化器 |
| Learning Rate | 控制参数每次更新的步长 |
| Batch | 一次送入模型的一小批样本 |
| Epoch | 完整遍历一次训练集 |
| Tensor | PyTorch 中存储数据和参数的核心对象 |
| `nn` | PyTorch 神经网络模块 |
| autograd | PyTorch 自动求导系统 |
| Feature Map | Kernel 扫描输入后产生的特征响应图 |
| Bottleneck | U-Net 最底部、特征最浓缩的位置 |

## 12.Day05
Day 05 开始正式完成 Baseline U-Net 的完整训练和评估。
主要内容：
Train
+
Validation
+
Dice
+
IoU
+
Early Stopping
+
Best Model Checkpoint
+
Loss Curve
+
Prediction Visualization

目标：
1. 完整训练 U-Net Baseline
2. 同时监控 Train Loss 和 Validation Loss
3. 计算 Validation Dice 和 IoU
4. 保存表现最好的模型
5. 避免过拟合
6. 在 Test Set 上进行最终评估
7. 可视化 Image / Ground Truth / Prediction
8. 得到第一版可以写入简历的实验结果


# Day05 : 正式训练
## 1.修改dataste.csv的路径---路径可移植性
## 2.上传git：
git init
git status
git add .  # 表示把这些修改放进“准备提交区”
git commit -m "Day04: implement UNet training fundamentals and Dice BCE loss"

git remote add origin https://github.com/你hongtingjiang-del/breast-ultrasound-segmentation.git

git remote -v  #检查是否连接成功
#提交
git branch -M main
git push -u origin main

## 3.上传数据集（scp或者使用filezillia）---scp更快
scp -P 24758 "D:\desktop\work-projects\breast-ultrasound-segmentation\data\Dataset_BUSI_with_GT.zip" root@connect.westb.seetacloud.com:/root/


#### 问题：
# 1. 软撤销：commit 取消，改动回到暂存区，文件都还在
git reset --soft HEAD~1

# 2. 把压缩包从暂存区拿出来（不删除本地文件）
git restore --staged data/Dataset_BUSI_with_GT.zip

# 3. 确认暂存区里没有它了
git status

# 4. 重新提交（此时只包含你要的文件）
git commit -m "提交说明"

# 5. 推送
git push


### 常用命令
pwd          # 显示当前所在完整路径
cd 目录名     # 进入某个目录
cd ..        # 返回上一级
cd ~         # 回到用户主目录

ls -l        # 详细列表：权限、大小、修改时间
ls -a        # 显示隐藏文件（以 . 开头的，如 .git）
ls -la       # 详细 + 隐藏文件，最常用组合
ls -lh       # 详细列表，大小用 K/M/G 显示，更易读



### 运行train
cd ~/projects/breast-ultrasound-segmentation
python -m src.train --data_root ~/datasets/BUSI/Dataset_BUSI_with_GT --epochs 1 --batch_size 4 --num_workers 2

### 正式train运行命令，注意不是train.py，保存对应的日志，并在屏幕上显示
python -m src.train \
  --data_root ~/datasets/BUSI/Dataset_BUSI_with_GT \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4 \
  --patience 10 \
  --num_workers 4 \
  --save_dir results/baseline 2>&1 | tee train.log

  ### 使用scp下载训练好的模型
首先压缩文件夹：tar -czf baseline_results.tar.gz results/baseline
然后使用scp命令下载，在powershell里面：
  scp -P 24758 root@connect.westb.seetacloud.com:~/projects/breast-ultrasound-segmentation/baseline_results.tar.gz "D:\desktop\work-projects\breast-ultrasound-segmentation\results\"



# Day06:训练之后检查


## 找到最佳epoch
(breast_seg) PS D:\desktop\work-projects\breast-ultrasound-segmentation> python -c "import pandas as pd; df = pd.read_csv('results/baseline/history.csv'); best = df.loc[df['val_dice'].idxmax()]; print('Best epoch:', int(best['epoch'])); print('Train Loss:', best['train_loss']); print('Val Loss:', best['val_loss']); print('Val Dice:', best['val_dice']); print('Val IoU:', best['val_iou'])"
Best epoch: 38
Train Loss: 0.2425456922828105
Val Loss: 0.5130691390771133
Val Dice: 0.7186055412659278
Val IoU: 0.6261377701392541

## 为什么只有48个epoch?因为设置了提前停止
(base) PS D:\desktop\work-projects\breast-ultrasound-segmentation> & D:/Anaconda_envs/envs/breast_seg/python.exe d:/desktop/work-projects/breast-ultrasound-segmentation/src/plot_history.py
   epoch  train_loss  val_loss  val_dice   val_iou
0      1    1.323266  1.267184  0.374003  0.265534
1      2    1.123721  1.186228  0.418291  0.307745
2      3    1.072471  1.092772  0.496884  0.408331
3      4    1.012991  1.055381  0.527776  0.431144
4      5    1.001086  1.020877  0.509304  0.407047

Total epochs: 48

### 报错解决（今天遇到很多）
Traceback (most recent call last):
  File "d:\desktop\work-projects\breast-ultrasound-segmentation\src\visualize_predictions.py", line 12, in <module>
    from src.dataset import BUSIDataset
ModuleNotFoundError: No module named 'src'

问题原因：这是 Python 导入路径问题——直接运行 src/visualize_predictions.py 时，Python 会把 src/ 目录加入 sys.path，而不是项目根目录，所以找不到 src 包。
解决办法：在命令行直接运行：python -m src.visualize_predictions

## 对 Test Set 的每一张图分别计算 Dice / IoU。

不是只得到：
Test Dice = 某一个平均值，因为batch可能不均匀吧

而是得到：
image	class	size_group	Dice	IoU
benign xxx	benign	small	0.42	0.27
benign xxx	benign	large	0.91	0.84
malignant xxx	malignant	medium	0.78	0.65
...	...	...	...	...


有了这个表以后，我们才能真正分析：
到底是不是小病灶最难？

良性和恶性哪个更难？

最差的病例到底是什么特点？

模型是不是经常漏分？

还是经常把背景误分成病灶？

## 还有一个问题：现在的 0.5 阈值不一定最优
你当前预测使用：
pred = (prob >= 0.5).float()


也就是：
Probability ≥ 0.5
→ Lesion

Probability < 0.5
→ Background

但 0.5 只是最常见的默认值，并不能保证它对 BUSI 是最佳阈值。
例如：
threshold = 0.3
0.4
0.5
0.6
0.7

可能得到：
0.3 → Dice 0.70
0.4 → Dice 0.73
0.5 → Dice 0.72
0.6 → Dice 0.68

如果：
0.4

最好，那么以后正式预测应该考虑 0.4。
但是这里有一个实验规范非常重要：
阈值只能在 Validation Set 上选择，不能拿 Test Set 调阈值。

否则相当于：
偷看考试答案

Test Set 就不再是真正独立测试集了。
所以正确流程：
Validation Set
↓
寻找最佳 threshold
↓
比如 threshold = 0.45
↓
固定 threshold
↓
只在最后用一次 Test Set

## 接下来的计划
Step 1
best_model.pt
↓
加载最佳模型 Epoch 38

Step 2
Validation Set
↓
测试不同 threshold

Step 3
确定最佳 threshold

Step 4
固定 threshold

Step 5
Test Set
↓
每张图计算 Dice / IoU

Step 6
生成 per_sample_metrics.csv

Step 7
分别统计：
Overall
Benign / Malignant
Small / Medium / Large

Step 8
找出：
Best cases
Median cases
Worst cases

Step 9
重新画预测结果

Step 10
分析失败原因