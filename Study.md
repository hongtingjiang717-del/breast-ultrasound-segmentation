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


## git提交大文件的时候push不上,最后在git提交吧，先存到本地，目前不知道怎么解决

## 本地到云服务器传文件---复制粘贴

## 模型架构
                        BUSI
                         │
                         ▼
             Strict Train / Val / Test
                         │
                ┌────────┴─────────┐
                │                  │
             Train             Val / Test
                │                  │
          Augmentation          Resize only
                │                  │
                └────────┬─────────┘
                         ▼

          ┌───────────────────────────────┐
          │                               │
          ▼                               ▼

     Scratch U-Net                ResNet18 U-Net
      Baseline                         │
                                      ▼
                           Attention ResNet18 U-Net

          │                 │                    │
          └─────────────────┼────────────────────┘
                            ▼

                   Dice / IoU / Recall
                            │
                            ▼

             Image / GT / Prediction / Overlay
                            │
                            ▼

                      Gradio Demo

## 测试结果
root@autodl-container-9e0b48b25a-2db78bb6:~/projects/breast-ultrasound-segmentation# p
ython - <<'PY'
> import torch
> 
> from src.model_zoo import (
>     build_resunet18,
>     build_attention_resunet18
> )
> 
> x = torch.randn(
>     2,
>     1,
>     256,
>     256
> )
> 
> model1 = build_resunet18()
> 
> y1 = model1(x)
> 
> print(
>     "ResUNet input:",
>     x.shape
> )
> 
> print(
>     "ResUNet output:",
>     y1.shape
> )
> 
> 
> model2 = build_attention_resunet18()
> 
> y2 = model2(x)
> 
> print(
>     "Attention ResUNet output:",
>     y2.shape
> )
> PY
Downloading: "https://download.pytorch.org/models/resnet18-5c106cde.pth" to /root/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth
100%|█████████████████████████████████████| 44.7M/44.7M [00:09<00:00, 5.04MB/s]
ResUNet input: torch.Size([2, 1, 256, 256])
ResUNet output: torch.Size([2, 1, 256, 256])
Attention ResUNet output: torch.Size([2, 1, 256, 256])

## 测试结果
### 1.
root@autodl-container-9e0b48b25a-2db78bb6:~/projects/breast-ultrasound-segmentation# p
ython - <<'PY'
> import torch
> 
> from src.model_zoo import (
>     build_resunet18,
>     build_attention_resunet18
> )
> 
> x = torch.randn(
>     2,
>     1,
>     256,
>     256
> )
> 
> model1 = build_resunet18()
> 
> y1 = model1(x)
> 
> print(
>     "ResUNet input:",
>     x.shape
> )
> 
> print(
>     "ResUNet output:",
>     y1.shape
> )
> 
> 
> model2 = build_attention_resunet18()
> 
> y2 = model2(x)
> 
> print(
>     "Attention ResUNet output:",
>     y2.shape
> )
> PY
Downloading: "https://download.pytorch.org/models/resnet18-5c106cde.pth" to /root/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth
100%|█████████████████████████████████████| 44.7M/44.7M [00:09<00:00, 5.04MB/s]
ResUNet input: torch.Size([2, 1, 256, 256])
ResUNet output: torch.Size([2, 1, 256, 256])
Attention ResUNet output: torch.Size([2, 1, 256, 256])

### 2.
root@autodl-container-9e0b48b25a-2db78bb6:~/projects/breast-ultrasound-segmentation# python - <<'PY'
import torch

from src.dataset import BUSIDataset
from src.augmentations import get_train_augmentation


dataset = BUSIDataset(

    csv_file="data/splits/train.csv",

    image_size=(256, 256),

    data_root="/root/datasets/BUSI/Dataset_BUSI_with_GT",

    transform=get_train_augmentation(256)
)


PY  sample["num_masks"]().item()
/root/miniconda3/lib/python3.8/site-packages/albumentations/__init__.py:13: UserWarning: A new version of Albumentations is available: 2.0.8 (you have 1.4.14). Upgrade using: pip install -U albumentations. To disable automatic update checks, set the environment variable NO_ALBUMENTATIONS_UPDATE to 1.
  check_for_updates()
Loaded dataset: 452 images
Image: torch.Size([1, 256, 256])
Mask: torch.Size([1, 256, 256])
Image min: 0.0
Image max: 0.8588235378265381
Mask unique: tensor([0., 1.])
Num masks: 1


### 3.冒烟测试---smoke test(只使用一个epoch)---可以看到加入attention一个epoch就有比较好的结果
#### resnet
root@autodl-container-9e0b48b25a-2db78bb6:~/projects/breroot@autodl-container-9   root@autodl-container-9e0b48root@autodl-container-9e0b48b25a-2db78bb6:~/projects/breast-uroot@autodl-container-9e0b48b25a-2db78bb6:~/projects/breast-ultrasound-segmroot@autodl-container-9e0b48b25a-2db78bb6:~/projects/breast-ultrasound-segmenta
tion# python -m src.train \
>   --model resunet18 \
>   --augmentation \
>   --data_root /root/datasets/BUSI/Dataset_BUSI_with_GT \
>   --epochs 1 \
>   --batch_size 8 \
>   --lr 0.0003 \
>   --num_workers 2 \
>   --save_dir results/resunet18_smoke
/root/miniconda3/lib/python3.8/site-packages/albumentations/__init__.py:13: UserWarning: A new version of Albumentations is available: 2.0.8 (you have 1.4.14). Upgrade using: pip install -U albumentations. To disable automatic update checks, set the environment variable NO_ALBUMENTATIONS_UPDATE to 1.
  check_for_updates()
Using device: cuda
GPU: NVIDIA GeForce RTX 4090 D
Training augmentation: ON
Loaded dataset: 452 images
Loaded dataset: 97 images
Loaded dataset: 98 images
Train samples: 452
Val samples: 97
Test samples: 98
Model: ResNet18 U-Net
Epoch 1/1: 100%|█████████████████████| 57/57 [00:03<00:00, 17.17it/s, loss=1.0224]

Epoch 1
Train Loss: 1.2839
Val Loss:   1.1426
Val Dice:   0.5323
Val IoU:    0.4031

Saved new best model.

Loaded best model: results/resunet18_smoke/best_model.pt

===== Test Results =====
Test Loss: 1.1564
Test Dice: 0.5217
Test IoU: 0.3907

#### attention
root@autodl-container-9e0b48b25a-2db78bb6:~/projects/breast-ultrasound-segmentation# python -m src.train \
>   --model attention_resunet18 \
>   --augmentation \
>   --data_root /root/datasets/BUSI/Dataset_BUSI_with_GT \
>   --epochs 1 \
>   --batch_size 8 \
>   --lr 0.0003 \
>   --num_workers 2 \
>   --save_dir results/attention_resunet18_smoke
/root/miniconda3/lib/python3.8/site-packages/albumentations/__init__.py:13: UserWarning: A new version of Albumentations is available: 2.0.8 (you have 1.4.14). Upgrade using: pip install -U albumentations. To disable automatic update checks, set the environment variable NO_ALBUMENTATIONS_UPDATE to 1.
  check_for_updates()
Using device: cuda
GPU: NVIDIA GeForce RTX 4090 D
Training augmentation: ON
Loaded dataset: 452 images
Loaded dataset: 97 images
Loaded dataset: 98 images
Train samples: 452
Val samples: 97
Test samples: 98
Model: Attention ResNet18 U-Net
Epoch 1/1: 100%|█| 57/57 [00:03<00:00, 15.90it/s, loss=1.07

Epoch 1
Train Loss: 1.1017
Val Loss:   0.8677
Val Dice:   0.6100
Val IoU:    0.4996

Saved new best model.

Loaded best model: results/attention_resunet18_smoke/best_model.pt

===== Test Results =====
Test Loss: 0.8294
Test Dice: 0.6658
Test IoU: 0.5566

### 4.正式训练
#### resnet
root@autodl-container-9e0b48b25a-2db78bb6:~/projects/breast-ultrasound-segmentation# python -m src.train \
>   --model resunet18 \
>   --augmentation \
>   --data_root /root/datasets/BUSI/Dataset_BUSI_with_GT \
>   --epochs 50 \
>   --batch_size 16 \
>   --lr 0.0003 \
>   --patience 10 \
>   --num_workers 4 \
>   --save_dir results/resunet18_aug
/root/miniconda3/lib/python3.8/site-packages/albumentations/__init__.py:13: UserWarning: A new version of Albumentations is available: 2.0.8 (you have 1.4.14). Upgrade using: pip install -U albumentations. To disable automatic update checks, set the environment variable NO_ALBUMENTATIONS_UPDATE to 1.
  check_for_updates()
Using device: cuda
GPU: NVIDIA GeForce RTX 4090 D
Training augmentation: ON
Loaded dataset: 452 images
Loaded dataset: 97 images
Loaded dataset: 98 images
Train samples: 452
Val samples: 97
Test samples: 98
Model: ResNet18 U-Net
Epoch 1/50: 100%|█| 29/29 [00:02<00:00, 14.20it/s, loss=1.1

Epoch 1
Train Loss: 1.3912
Val Loss:   1.7140
Val Dice:   0.3240
Val IoU:    0.2184

Saved new best model.
Epoch 2/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.62it/s, loss=0.9683]

Epoch 2
Train Loss: 1.0365
Val Loss:   1.0122
Val Dice:   0.5784
Val IoU:    0.4759

Saved new best model.
Epoch 3/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.48it/s, loss=0.9745]

Epoch 3
Train Loss: 0.8722
Val Loss:   0.8863
Val Dice:   0.6150
Val IoU:    0.5140

Saved new best model.
Epoch 4/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 18.32it/s, loss=0.7578]

Epoch 4
Train Loss: 0.7479
Val Loss:   0.7965
Val Dice:   0.6074
Val IoU:    0.5076

Epoch 5/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.54it/s, loss=0.7519]

Epoch 5
Train Loss: 0.6817
Val Loss:   0.7359
Val Dice:   0.6309
Val IoU:    0.5323

Saved new best model.
Epoch 6/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.69it/s, loss=0.5520]

Epoch 6
Train Loss: 0.6133
Val Loss:   0.7228
Val Dice:   0.6359
Val IoU:    0.5228

Saved new best model.
Epoch 7/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.75it/s, loss=0.4788]

Epoch 7
Train Loss: 0.5383
Val Loss:   0.7148
Val Dice:   0.6194
Val IoU:    0.5018

Epoch 8/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 18.08it/s, loss=0.7577]

Epoch 8
Train Loss: 0.4865
Val Loss:   0.6192
Val Dice:   0.6844
Val IoU:    0.5881

Saved new best model.
Epoch 9/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 18.05it/s, loss=0.5312]

Epoch 9
Train Loss: 0.4447
Val Loss:   0.6470
Val Dice:   0.6476
Val IoU:    0.5385

Epoch 10/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.91it/s, loss=0.3413]

Epoch 10
Train Loss: 0.4327
Val Loss:   0.5658
Val Dice:   0.6517
Val IoU:    0.5696

Epoch 11/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.47it/s, loss=0.4278]

Epoch 11
Train Loss: 0.3819
Val Loss:   0.5566
Val Dice:   0.6410
Val IoU:    0.5550

Epoch 12/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.96it/s, loss=0.9001]

Epoch 12
Train Loss: 0.3832
Val Loss:   0.5787
Val Dice:   0.6413
Val IoU:    0.5691

Epoch 13/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.93it/s, loss=0.4906]

Epoch 13
Train Loss: 0.3363
Val Loss:   0.6401
Val Dice:   0.6091
Val IoU:    0.5373

Epoch 14/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 15.31it/s, loss=0.3182]

Epoch 14
Train Loss: 0.3807
Val Loss:   0.5628
Val Dice:   0.6695
Val IoU:    0.5911

Epoch 15/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 16.18it/s, loss=0.7303]

Epoch 15
Train Loss: 0.3337
Val Loss:   0.5263
Val Dice:   0.6500
Val IoU:    0.5711

Epoch 16/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 15.22it/s, loss=0.2345]

Epoch 16
Train Loss: 0.3028
Val Loss:   0.4975
Val Dice:   0.6979
Val IoU:    0.6091

Saved new best model.
Epoch 17/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.57it/s, loss=0.4295]

Epoch 17
Train Loss: 0.3095
Val Loss:   0.6147
Val Dice:   0.6493
Val IoU:    0.5665

Epoch 18/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 18.16it/s, loss=0.2370]

Epoch 18
Train Loss: 0.2994
Val Loss:   0.5760
Val Dice:   0.6484
Val IoU:    0.5650

Epoch 19/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 18.00it/s, loss=0.2978]

Epoch 19
Train Loss: 0.2965
Val Loss:   0.6076
Val Dice:   0.6597
Val IoU:    0.5854

Epoch 20/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 18.00it/s, loss=0.1549]

Epoch 20
Train Loss: 0.2652
Val Loss:   0.5395
Val Dice:   0.6598
Val IoU:    0.5871

Epoch 21/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.68it/s, loss=0.3054]

Epoch 21
Train Loss: 0.2529
Val Loss:   0.5660
Val Dice:   0.6534
Val IoU:    0.5816

Epoch 22/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.78it/s, loss=0.5107]

Epoch 22
Train Loss: 0.2472
Val Loss:   0.5397
Val Dice:   0.6702
Val IoU:    0.5902

Epoch 23/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.81it/s, loss=0.3583]

Epoch 23
Train Loss: 0.2637
Val Loss:   0.5385
Val Dice:   0.6520
Val IoU:    0.5625

Epoch 24/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.65it/s, loss=0.1579]

Epoch 24
Train Loss: 0.2554
Val Loss:   0.5395
Val Dice:   0.6575
Val IoU:    0.5822

Epoch 25/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.32it/s, loss=0.3601]

Epoch 25
Train Loss: 0.2386
Val Loss:   0.5220
Val Dice:   0.6638
Val IoU:    0.5850

Epoch 26/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.78it/s, loss=0.2709]

Epoch 26
Train Loss: 0.2462
Val Loss:   0.5606
Val Dice:   0.6514
Val IoU:    0.5807

Early stopping triggered.

Loaded best model: results/resunet18_aug/best_model.pt

===== Test Results =====
Test Loss: 0.2935
Test Dice: 0.8377
Test IoU: 0.7449
#### attention
root@autodl-container-9e0b48b25a-2db78bb6:~/projects/breast-ultrasound-segmentation# python -m src.train \
>   --model attention_resunet18 \
>   --augmentation \
>   --data_root /root/datasets/BUSI/Dataset_BUSI_with_GT \
>   --epochs 50 \
>   --batch_size 16 \
>   --lr 0.0003 \
>   --patience 10 \
>   --num_workers 4 \
>   --save_dir results/attention_resunet18_aug
/root/miniconda3/lib/python3.8/site-packages/albumentations/__init__.py:13: UserWarning: A new version of Albumentations is available: 2.0.8 (you have 1.4.14). Upgrade using: pip install -U albumentations. To disable automatic update checks, set the environment variable NO_ALBUMENTATIONS_UPDATE to 1.
  check_for_updates()
Using device: cuda
GPU: NVIDIA GeForce RTX 4090 D
Training augmentation: ON
Loaded dataset: 452 images
Loaded dataset: 97 images
Loaded dataset: 98 images
Train samples: 452
Val samples: 97
Test samples: 98
Model: Attention ResNet18 U-Net
Epoch 1/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:02<00:00, 11.71it/s, loss=1.1112]

Epoch 1
Train Loss: 1.2132
Val Loss:   1.3502
Val Dice:   0.3442
Val IoU:    0.2327

Saved new best model.
Epoch 2/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 16.64it/s, loss=1.0302]

Epoch 2
Train Loss: 0.9062
Val Loss:   0.8600
Val Dice:   0.5622
Val IoU:    0.4308

Saved new best model.
Epoch 3/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.33it/s, loss=1.0744]

Epoch 3
Train Loss: 0.7658
Val Loss:   0.7788
Val Dice:   0.6135
Val IoU:    0.5149

Saved new best model.
Epoch 4/50:   3%|███▏                                                                                         | 1/29 [00:00<0Epoch 4/50:   3%|███▏                                                                                         | 1/29 [00:00<0Epoch 4/50:  14%|████████████▊                                                                                | 4/29 [00:00<0Epoch 4/50:  14%|████████████▊                                                                                | 4/29 [00:00<0Epoch 4/50:  14%|████████████▊                                                                                | 4/29 [00:00<0Epoch 4/50:  14%|████████████▊                                                                                | 4/29 [00:00<0Epoch 4/50:  24%|██████████████████████▍                                                                      | 7/29 [00:00<0Epoch 4/50:  24%|██████████████████████▍                                                                      | 7/29 [00:00<0Epoch 4/50:  24%|██████████████████████▍                                                                      | 7/29 [00:00<0Epoch 4/50:  24%|██████████████████████▍                                                                      | 7/29 [00:00<0Epoch 4/50:  34%|███████████████████████████████▋                                                            | 10/29 [00:00<0Epoch 4/50:  34%|███████████████████████████████▋                                                            | 10/29 [00:00<0Epoch 4/50:  34%|███████████████████████████████▋                                                            | 10/29 [00:00<0Epoch 4/50:  34%|███████████████████████████████▋                                                            | 10/29 [00:00<0Epoch 4/50:  45%|█████████████████████████████████████████▏                                                  | 13/29 [00:00<0Epoch 4/50:  45%|█████████████████████████████████████████▏                                                  | 13/29 [00:00<0Epoch 4/50:  45%|█████████████████████████████████████████▏                                                  | 13/29 [00:01<0Epoch 4/50:  45%|█████████████████████████████████████████▏                                                  | 13/29 [00:01<0Epoch 4/50:  55%|██████████████████████████████████████████████████▊                                         | 16/29 [00:01<0Epoch 4/50:  55%|██████████████████████████████████████████████████▊                                         | 16/29 [00:01<0Epoch 4/50:  55%|██████████████████████████████████████████████████▊                                         | 16/29 [00:01<0Epoch 4/50:  55%|██████████████████████████████████████████████████▊                                         | 16/29 [00:01<0Epoch 4/50:  66%|████████████████████████████████████████████████████████████▎                               | 19/29 [00:01<0Epoch 4/50:  66%|████████████████████████████████████████████████████████████▎                               | 19/29 [00:01<0Epoch 4/50:  66%|████████████████████████████████████████████████████████████▎                               | 19/29 [00:01<0Epoch 4/50:  66%|████████████████████████████████████████████████████████████▎                               | 19/29 [00:01<0Epoch 4/50:  76%|█████████████████████████████████████████████████████████████████████▊                      | 22/29 [00:01<0Epoch 4/50:  76%|█████████████████████████████████████████████████████████████████████▊                      | 22/29 [00:01<0Epoch 4/50:  76%|█████████████████████████████████████████████████████████████████████▊                      | 22/29 [00:01<0Epoch 4/50:  76%|█████████████████████████████████████████████████████████████████████▊                      | 22/29 [00:01<0Epoch 4/50:  86%|███████████████████████████████████████████████████████████████████████████████▎            | 25/29 [00:01<0Epoch 4/50:  86%|███████████████████████████████████████████████████████████████████████████████▎            | 25/29 [00:01<0Epoch 4/50:  86%|███████████████████████████████████████████████████████████████████████████████▎            | 25/29 [00:01<0Epoch 4/50:  86%|███████████████████████████████████████████████████████████████████████████████▎            | 25/29 [00:01<0Epoch 4/50:  97%|████████████████████████████████████████████████████████████████████████████████████████▊   | 28/29 [00:01<0Epoch 4/50:  97%|████████████████████████████████████████████████████████████████████████████████████████▊   | 28/29 [00:01<0Epoch 4/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 16.95it/s, loss=0.5285]

Epoch 4
Train Loss: 0.6172
Val Loss:   0.7471
Val Dice:   0.5936
Val IoU:    0.5033

Epoch 5/50: 100%|███████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.14it/s, loss=0.3839]

Epoch 5
Train Loss: 0.5303
Val Loss:   0.6340
Val Dice:   0.6169
Val IoU:    0.5338

Saved new best model.
Epoch 6/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.41it/s, loss=0.4256]

Epoch 6
Train Loss: 0.4655
Val Loss:   0.6103
Val Dice:   0.6195
Val IoU:    0.5381

Saved new best model.
Epoch 7/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.17it/s, loss=0.4530]

Epoch 7
Train Loss: 0.4106
Val Loss:   0.5328
Val Dice:   0.6369
Val IoU:    0.5561

Saved new best model.
Epoch 8/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.35it/s, loss=0.4392]

Epoch 8
Train Loss: 0.4319
Val Loss:   0.5868
Val Dice:   0.6853
Val IoU:    0.5904

Saved new best model.
Epoch 9/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 15.71it/s, loss=0.2065]

Epoch 9
Train Loss: 0.3697
Val Loss:   0.5252
Val Dice:   0.6408
Val IoU:    0.5555

Epoch 10/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 14.65it/s, loss=0.1948]

Epoch 10
Train Loss: 0.3584
Val Loss:   0.4815
Val Dice:   0.6666
Val IoU:    0.5806

Epoch 11/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 15.34it/s, loss=0.8719]

Epoch 11
Train Loss: 0.3155
Val Loss:   0.4791
Val Dice:   0.6677
Val IoU:    0.5847

Epoch 12/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 16.78it/s, loss=0.7022]

Epoch 12
Train Loss: 0.3206
Val Loss:   0.5271
Val Dice:   0.6494
Val IoU:    0.5614

Epoch 13/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.58it/s, loss=0.1965]

Epoch 13
Train Loss: 0.2989
Val Loss:   0.4923
Val Dice:   0.6556
Val IoU:    0.5787

Epoch 14/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.14it/s, loss=0.3794]

Epoch 14
Train Loss: 0.2923
Val Loss:   0.5047
Val Dice:   0.6972
Val IoU:    0.5961

Saved new best model.
Epoch 15/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 16.81it/s, loss=0.3802]

Epoch 15
Train Loss: 0.2875
Val Loss:   0.4751
Val Dice:   0.7215
Val IoU:    0.6242

Saved new best model.
Epoch 16/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.50it/s, loss=0.1306]

Epoch 16
Train Loss: 0.2611
Val Loss:   0.4907
Val Dice:   0.7004
Val IoU:    0.6044

Epoch 17/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.61it/s, loss=0.5653]

Epoch 17
Train Loss: 0.2678
Val Loss:   0.4623
Val Dice:   0.7248
Val IoU:    0.6297

Saved new best model.
Epoch 18/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.00it/s, loss=0.5100]

Epoch 18
Train Loss: 0.2877
Val Loss:   0.4376
Val Dice:   0.7428
Val IoU:    0.6397

Saved new best model.
Epoch 19/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.22it/s, loss=0.2767]

Epoch 19
Train Loss: 0.2520
Val Loss:   0.4378
Val Dice:   0.7490
Val IoU:    0.6536

Saved new best model.
Epoch 20/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.49it/s, loss=0.1639]

Epoch 20
Train Loss: 0.2353
Val Loss:   0.4197
Val Dice:   0.7585
Val IoU:    0.6604

Saved new best model.
Epoch 21/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.53it/s, loss=0.2459]

Epoch 21
Train Loss: 0.2262
Val Loss:   0.4175
Val Dice:   0.7646
Val IoU:    0.6774

Saved new best model.
Epoch 22/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.08it/s, loss=0.2788]

Epoch 22
Train Loss: 0.2377
Val Loss:   0.4418
Val Dice:   0.7492
Val IoU:    0.6512

Epoch 23/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.02it/s, loss=0.1791]

Epoch 23
Train Loss: 0.2317
Val Loss:   0.4456
Val Dice:   0.7384
Val IoU:    0.6395

Epoch 24/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 16.95it/s, loss=0.1611]

Epoch 24
Train Loss: 0.2265
Val Loss:   0.4500
Val Dice:   0.7369
Val IoU:    0.6357

Epoch 25/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 16.30it/s, loss=0.5371]

Epoch 25
Train Loss: 0.2504
Val Loss:   0.4427
Val Dice:   0.7472
Val IoU:    0.6513

Epoch 26/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 14.92it/s, loss=0.1137]

Epoch 26
Train Loss: 0.2368
Val Loss:   0.4340
Val Dice:   0.7393
Val IoU:    0.6411

Epoch 27/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 15.54it/s, loss=0.2510]

Epoch 27
Train Loss: 0.2456
Val Loss:   0.4831
Val Dice:   0.7223
Val IoU:    0.6227

Epoch 28/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:02<00:00, 14.03it/s, loss=0.1861]

Epoch 28
Train Loss: 0.2299
Val Loss:   0.4518
Val Dice:   0.7492
Val IoU:    0.6523

Epoch 29/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.44it/s, loss=0.5845]

Epoch 29
Train Loss: 0.2246
Val Loss:   0.4422
Val Dice:   0.7522
Val IoU:    0.6512

Epoch 30/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.29it/s, loss=0.1533]

Epoch 30
Train Loss: 0.2145
Val Loss:   0.4230
Val Dice:   0.7623
Val IoU:    0.6647

Epoch 31/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 17.45it/s, loss=0.1526]

Epoch 31
Train Loss: 0.2241
Val Loss:   0.5002
Val Dice:   0.7318
Val IoU:    0.6352

Early stopping triggered.

Loaded best model: results/attention_resunet18_aug/best_model.pt

attention-resnet
===== Test Results =====
Test Loss: 0.3462
Test Dice: 0.8008
Test IoU: 0.7123

#### baseline
root@autodl-container-9e0b48b25a-2db78bb6:~/projects/breast-ultrasound-segmentation# python -m src.train \
>   --model unet \
>   --data_root /root/datasets/BUSI/Dataset_BUSI_with_GT \
>   --epochs 50 \
>   --batch_size 16 \
>   --lr 0.0003 \
>   --patience 10 \
>   --num_workers 4 \
>   --seed 42 \
>   --save_dir results/unet_baseline_final
/root/miniconda3/lib/python3.8/site-packages/albumentations/__init__.py:13: UserWarning: A new version of Albumentations is available: 2.0.8 (you have 1.4.14). Upgrade using: pip install -U albumentations. To disable automatic update checks, set the environment variable NO_ALBUMENTATIONS_UPDATE to 1.
  check_for_updates()
Using device: cuda
GPU: NVIDIA GeForce RTX 4090 D
Training augmentation: OFF
Loaded dataset: 452 images
Loaded dataset: 97 images
Loaded dataset: 98 images
Train samples: 452
Val samples: 97
Test samples: 98
Model: Scratch U-Net
Epoch 1/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  5.96it/s, loss=1.1692]

Epoch 1
Train Loss: 1.3432
Val Loss:   7.9548
Val Dice:   0.1699
Val IoU:    0.1013

Saved new best model.
Epoch 2/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.66it/s, loss=1.1840]

Epoch 2
Train Loss: 1.1459
Val Loss:   1.2571
Val Dice:   0.2902
Val IoU:    0.1993

Saved new best model.
Epoch 3/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.71it/s, loss=1.0457]

Epoch 3
Train Loss: 1.0672
Val Loss:   1.1812
Val Dice:   0.3305
Val IoU:    0.2390

Saved new best model.
Epoch 4/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.70it/s, loss=0.8273]

Epoch 4
Train Loss: 1.0055
Val Loss:   1.0664
Val Dice:   0.4605
Val IoU:    0.3784

Saved new best model.
Epoch 5/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.74it/s, loss=1.2325]

Epoch 5
Train Loss: 0.9744
Val Loss:   1.0179
Val Dice:   0.4359
Val IoU:    0.3377

Epoch 6/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.71it/s, loss=1.0984]

Epoch 6
Train Loss: 0.9257
Val Loss:   1.0723
Val Dice:   0.3868
Val IoU:    0.2800

Epoch 7/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.67it/s, loss=0.8431]

Epoch 7
Train Loss: 0.8850
Val Loss:   0.9864
Val Dice:   0.4836
Val IoU:    0.3887

Saved new best model.
Epoch 8/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.65it/s, loss=0.9241]

Epoch 8
Train Loss: 0.8514
Val Loss:   1.1020
Val Dice:   0.4063
Val IoU:    0.2898

Epoch 9/50: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.62it/s, loss=0.6701]

Epoch 9
Train Loss: 0.8223
Val Loss:   0.9640
Val Dice:   0.4354
Val IoU:    0.3244

Epoch 10/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.70it/s, loss=0.8356]

Epoch 10
Train Loss: 0.8004
Val Loss:   0.9356
Val Dice:   0.4571
Val IoU:    0.3517

Epoch 11/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.67it/s, loss=0.8450]

Epoch 11
Train Loss: 0.7612
Val Loss:   0.8679
Val Dice:   0.4952
Val IoU:    0.3919

Saved new best model.
Epoch 12/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.70it/s, loss=0.5734]

Epoch 12
Train Loss: 0.7305
Val Loss:   0.8402
Val Dice:   0.5207
Val IoU:    0.4201

Saved new best model.
Epoch 13/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.65it/s, loss=0.5002]

Epoch 13
Train Loss: 0.6922
Val Loss:   0.8765
Val Dice:   0.4822
Val IoU:    0.3839

Epoch 14/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.68it/s, loss=0.6648]

Epoch 14
Train Loss: 0.6736
Val Loss:   0.7852
Val Dice:   0.5291
Val IoU:    0.4218

Saved new best model.
Epoch 15/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.66it/s, loss=0.6460]

Epoch 15
Train Loss: 0.6672
Val Loss:   0.8066
Val Dice:   0.5014
Val IoU:    0.3950

Epoch 16/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.68it/s, loss=1.1052]

Epoch 16
Train Loss: 0.6637
Val Loss:   0.7905
Val Dice:   0.5247
Val IoU:    0.4184

Epoch 17/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.69it/s, loss=0.8539]

Epoch 17
Train Loss: 0.6066
Val Loss:   0.7573
Val Dice:   0.5552
Val IoU:    0.4705

Saved new best model.
Epoch 18/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.68it/s, loss=0.6036]

Epoch 18
Train Loss: 0.5754
Val Loss:   0.7574
Val Dice:   0.5275
Val IoU:    0.4275

Epoch 19/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.69it/s, loss=0.4817]

Epoch 19
Train Loss: 0.5598
Val Loss:   1.0920
Val Dice:   0.3893
Val IoU:    0.2875

Epoch 20/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.73it/s, loss=0.5989]

Epoch 20
Train Loss: 0.5812
Val Loss:   0.7457
Val Dice:   0.5349
Val IoU:    0.4380

Epoch 21/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.68it/s, loss=0.4091]

Epoch 21
Train Loss: 0.5508
Val Loss:   0.9222
Val Dice:   0.4456
Val IoU:    0.3403

Epoch 22/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.69it/s, loss=0.6079]

Epoch 22
Train Loss: 0.5265
Val Loss:   0.8139
Val Dice:   0.5112
Val IoU:    0.4051

Epoch 23/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.64it/s, loss=0.4140]

Epoch 23
Train Loss: 0.5087
Val Loss:   0.7799
Val Dice:   0.5182
Val IoU:    0.4400

Epoch 24/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.72it/s, loss=0.3389]

Epoch 24
Train Loss: 0.4786
Val Loss:   0.7826
Val Dice:   0.5143
Val IoU:    0.4376

Epoch 25/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.68it/s, loss=0.5389]

Epoch 25
Train Loss: 0.4311
Val Loss:   0.7679
Val Dice:   0.5338
Val IoU:    0.4439

Epoch 26/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.67it/s, loss=0.6851]

Epoch 26
Train Loss: 0.4582
Val Loss:   0.7501
Val Dice:   0.5437
Val IoU:    0.4696

Epoch 27/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.72it/s, loss=0.6668]

Epoch 27
Train Loss: 0.4438
Val Loss:   0.6737
Val Dice:   0.5833
Val IoU:    0.4986

Saved new best model.
Epoch 28/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.71it/s, loss=0.3624]

Epoch 28
Train Loss: 0.4165
Val Loss:   0.6883
Val Dice:   0.5691
Val IoU:    0.4909

Epoch 29/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.67it/s, loss=0.5918]

Epoch 29
Train Loss: 0.3830
Val Loss:   0.7035
Val Dice:   0.5652
Val IoU:    0.4817

Epoch 30/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.65it/s, loss=0.4146]

Epoch 30
Train Loss: 0.3923
Val Loss:   0.7553
Val Dice:   0.5259
Val IoU:    0.4553

Epoch 31/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.68it/s, loss=0.9255]

Epoch 31
Train Loss: 0.3753
Val Loss:   0.7076
Val Dice:   0.5553
Val IoU:    0.4801

Epoch 32/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.73it/s, loss=0.3878]

Epoch 32
Train Loss: 0.3994
Val Loss:   0.6679
Val Dice:   0.5899
Val IoU:    0.5166

Saved new best model.
Epoch 33/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.66it/s, loss=0.3478]

Epoch 33
Train Loss: 0.3398
Val Loss:   0.6741
Val Dice:   0.5822
Val IoU:    0.5097

Epoch 34/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.69it/s, loss=0.3598]

Epoch 34
Train Loss: 0.3578
Val Loss:   0.9100
Val Dice:   0.4348
Val IoU:    0.3242

Epoch 35/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.67it/s, loss=0.2412]

Epoch 35
Train Loss: 0.3360
Val Loss:   0.6674
Val Dice:   0.5865
Val IoU:    0.4962

Epoch 36/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.68it/s, loss=0.5925]

Epoch 36
Train Loss: 0.3179
Val Loss:   0.6689
Val Dice:   0.5828
Val IoU:    0.5042

Epoch 37/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.68it/s, loss=0.5581]

Epoch 37
Train Loss: 0.3204
Val Loss:   0.6557
Val Dice:   0.5884
Val IoU:    0.5041

Epoch 38/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.62it/s, loss=0.3158]

Epoch 38
Train Loss: 0.2740
Val Loss:   0.7057
Val Dice:   0.5477
Val IoU:    0.4512

Epoch 39/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.69it/s, loss=0.3504]

Epoch 39
Train Loss: 0.3076
Val Loss:   0.6585
Val Dice:   0.5916
Val IoU:    0.5140

Saved new best model.
Epoch 40/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.71it/s, loss=0.4897]

Epoch 40
Train Loss: 0.2798
Val Loss:   0.7578
Val Dice:   0.5396
Val IoU:    0.4706

Epoch 41/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.68it/s, loss=0.5208]

Epoch 41
Train Loss: 0.2896
Val Loss:   0.7620
Val Dice:   0.5270
Val IoU:    0.4372

Epoch 42/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.71it/s, loss=0.4319]

Epoch 42
Train Loss: 0.3158
Val Loss:   0.6626
Val Dice:   0.5925
Val IoU:    0.5042

Saved new best model.
Epoch 43/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.65it/s, loss=0.1447]

Epoch 43
Train Loss: 0.2707
Val Loss:   0.6723
Val Dice:   0.5822
Val IoU:    0.5078

Epoch 44/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.71it/s, loss=0.3537]

Epoch 44
Train Loss: 0.2694
Val Loss:   0.6583
Val Dice:   0.5828
Val IoU:    0.5026

Epoch 45/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.63it/s, loss=0.4328]

Epoch 45
Train Loss: 0.2675
Val Loss:   0.6372
Val Dice:   0.6082
Val IoU:    0.5195

Saved new best model.
Epoch 46/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.69it/s, loss=0.2677]

Epoch 46
Train Loss: 0.2466
Val Loss:   0.8330
Val Dice:   0.4864
Val IoU:    0.4194

Epoch 47/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.72it/s, loss=0.2184]

Epoch 47
Train Loss: 0.2555
Val Loss:   0.7206
Val Dice:   0.5678
Val IoU:    0.4859

Epoch 48/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.68it/s, loss=0.3980]

Epoch 48
Train Loss: 0.2487
Val Loss:   0.6744
Val Dice:   0.5799
Val IoU:    0.5129

Epoch 49/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.72it/s, loss=0.4444]

Epoch 49
Train Loss: 0.2408
Val Loss:   0.7184
Val Dice:   0.5694
Val IoU:    0.4917

Epoch 50/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:04<00:00,  6.70it/s, loss=0.1791]

Epoch 50
Train Loss: 0.2605
Val Loss:   0.6408
Val Dice:   0.6051
Val IoU:    0.5271


Loaded best model: results/unet_baseline_final/best_model.pt

===== Test Results =====
Test Loss: 0.4544
Test Dice: 0.7382
Test IoU: 0.6245

#### 模型对比
resnet
===== Test Results =====
Test Loss: 0.2935
Test Dice: 0.8377
Test IoU: 0.7449

attention-resnet
===== Test Results =====
Test Loss: 0.3462
Test Dice: 0.8008
Test IoU: 0.7123


baseline
===== Test Results =====
Test Loss: 0.4544
Test Dice: 0.7382
Test IoU: 0.6245


## 压缩命令
tar -czf unet_baseline_final.tar.gz \
  results/unet_baseline_final

## 传输
scp -P 17738 root@connect.westc.seetacloud.com:/root/projects/breast-ultrasound-segmentation/unet_baseline_final.tar.gz "D:\desktop\work-projects\breast-ultrasound-segmentation\results\"


# 补充git命令(有待检查，大体如下)
上传某一个文件：
git add src.model
git commit -m "hhh"
git push origin main

下拉某一个文件：
git fetch
git checkout origin/main -- src/model.py

# github连接不上，先重置一下端口然后紧接着git push
git config --global --unset http.proxy
git config --global --unset https.proxy


# 一些基本的概念问题
## 1.什么是预训练？pretraining
先在一个大规模通用数据集（如 ImageNet）上训练模型，让网络学到通用的视觉特征（边缘、纹理、形状、语义），然后把这个权重作为起点，再在你的小规模目标任务（乳腺超声分割）上继续训练。

为什么有效：

底层特征通用：卷积网络的浅层学到的边缘、角点、纹理，在自然图像和医学图像里都适用。

缓解小数据问题：医学数据标注昂贵，预训练相当于"借"了 ImageNet 1400 万张图的知识。

收敛更快、效果更好：通常比随机初始化（from scratch）的 Dice 系数高几个百分点。

两种用法：

方式	做法	适用场景
特征提取（冻结）	冻结编码器权重，只训练解码器	目标数据极少
微调（fine-tune）	编码器也参与训练，用小学习率	目标数据中等规模，最常用
注意：ImageNet 是自然图像（猫狗汽车），医学超声是灰度图、纹理差异大。所以微调时学习率要调小（如 1e-4 或更低），避免把预训练学到的通用特征"冲掉"。

## 2.什么是ImageNet?
一个超大规模图像分类数据集，由李飞飞团队创建。包含 1400 多万张图片，2 万多个类别，其中常用的是 ILSVRC 子集（1000 类，约 128 万张训练图）。

为什么重要：

它是计算机视觉领域的"标准考卷"，ResNet、VGG、EfficientNet 等经典模型都是在这个数据集上比拼出来的。

它催生了预训练范式：先在 ImageNet 上训练一个通用视觉模型，再迁移到其他任务。

和你的任务的关系：乳腺超声分割数据集通常只有几百到几千张图，直接从头训练深层网络会严重过拟合。所以大家会用 ImageNet 预训练的 ResNet 作为编码器初始化权重。

## 3.什么是attention(注意力机制)？
让网络学会"看哪里更重要"的机制。不是对所有特征一视同仁，而是给重要区域更高的权重。

常见类型：

通道注意力（SE Block, Squeeze-and-Excitation）：判断"哪些特征通道更重要"。比如超声图像里，边缘纹理通道可能比颜色通道更关键。

空间注意力：判断"图像上哪些位置更重要"。比如乳腺超声里，肿瘤区域应该比背景脂肪组织获得更高权重。

自注意力 / Transformer：每个像素和所有其他像素计算相关性，建模长距离依赖。Vision Transformer (ViT) 和 Swin Transformer 就属于这一类。

在分割任务里的作用：

U-Net 的短板：它的 skip connection 是把编码器的特征无差别地直接拼接到解码器。但编码器里既有有用的边界信息，也有背景噪声，全部传过去会干扰解码。

Attention U-Net 的改进：在 skip connection 上加一个注意力门控（Attention Gate），让解码器自动筛选编码器传来的特征——只保留和当前分割目标相关的部分，抑制无关背景。

Transformer 类分割模型（如 Swin-UNet）：用自注意力替代卷积，能捕捉全局上下文，对边界模糊、对比度低的超声图像特别有帮助。


## 4.什么是残差网络（ResNet）?
一种卷积神经网络（CNN）架构，2015 年由何恺明等人提出，当年 ImageNet 图像分类冠军。

核心思想：残差连接（skip connection）。

传统网络是 y = F(x)，ResNet 把它改成 y = F(x) + x。这样梯度可以通过 +x 这条捷径直接回传，解决了深层网络的两个顽疾：

梯度消失：层数太深时，反向传播的梯度越来越小，浅层几乎学不到东西。

退化问题：理论上网络越深越强，但实际上 56 层比 20 层还差。ResNet 让 152 层甚至 1000 层都能正常训练。

关键结构：残差块（Residual Block），由两层 3×3 卷积 + BN + ReLU 组成，再加上那条恒等映射的捷径。

在分割任务里的作用：通常作为编码器（encoder）的主干网络，替代 U-Net 里手工堆叠的普通卷积层。因为 ResNet 已经在 ImageNet 上预训练过，特征提取能力远强于从零训练的小 U-Net 编码器。

## 5.对比unet基础框架，使用上面模块的改进总结
U-Net（2015）的结构是：编码器（下采样）+ 解码器（上采样）+ skip connection。它的编码器就是普通的卷积堆叠，没有残差、没有注意力、没有预训练。各改进方向如下：

改进	替换/增强了 U-Net 的哪部分	解决的问题
ResNet 编码器	把普通卷积编码器换成 ResNet（如 ResNet-34/50）	编码器太浅、梯度消失、特征提取能力弱

预训练	用 ImageNet 权重初始化编码器	医学数据太少、从零训练过拟合

Attention	在 skip connection 或编码器内部加注意力	skip connection 无差别传递噪声、缺乏全局上下文

Transformer	用自注意力替代部分/全部卷积	卷积感受野有限，难以建模长距离依赖
典型组合：ResNet-34（ImageNet 预训练）+ Attention Gate + U-Net 解码器，这在乳腺超声分割里是很常见的强 baseline。

## 6.什么是注意力门控？
注意力门控是 Attention U-Net（Oktay et al., 2018）提出的模块，专门加在 skip connection 上。

先回顾 U-Net 的 skip connection 问题
U-Net 编码器每一层都会把特征图直接拼到解码器对应层：

text
编码器第 l 层特征 ──────────────→ 直接 concat 到解码器第 l 层
问题是：编码器特征里既有有用的器官/病灶边界信息，也有背景噪声（脂肪、组织纹理、声影）。无差别拼接会把噪声也喂给解码器。

注意力门控怎么工作
它用解码器上一层的特征作为"查询（query）"，去筛选编码器传来的特征：

text
编码器特征 x  ──┐
                ├──→ 注意力系数 α ──→ x · α ──→ 传给解码器
解码器特征 g  ──┘
具体步骤：

两个输入：

x：编码器第 l 层的特征（要筛选的对象）

g：解码器上一层的特征（提供"当前在找什么"的上下文，也叫 gating signal）

各自做 1×1 卷积，把通道数降到同一维度，然后相加：

text
q = ReLU( W_x · x + W_g · g + b )
再经过 1×1 卷积 + Sigmoid，得到注意力系数图：

text
α = Sigmoid( W_ψ · q + b_ψ )
α 的每个值在 0~1 之间，表示对应位置该保留多少。

逐元素相乘，得到加权后的特征：

text
x̂ = x · α
然后把 x̂ 而不是原始的 x 拼接到解码器。

直觉理解
g 告诉网络："我现在正在重建肿瘤区域。"

注意力门控据此计算：编码器特征里哪些位置和肿瘤相关（α≈1），哪些是背景（α≈0）。

结果：相关区域被放大，无关背景被抑制。

为什么对医学图像特别有用
医学图像里目标往往很小、边界模糊、对比度低。注意力门控能：

抑制背景器官的干扰

突出小病灶区域

不需要额外标注，端到端训练

## 7.ResNet后边的数字表示什么意思？
ResNet 后面的数字（如 ResNet-18、ResNet-34、ResNet-50、ResNet-101、ResNet-152）表示网络中带有可学习权重的层数，也就是卷积层 + 全连接层的总数。

注意：它不包括池化层、BN 层（batch normalize）、ReLU 激活层，也不包括残差连接本身。
BasicBlock每块的卷积数是2；Bottleneck每块的卷积数是3

## ImageNet ResNet18输入不是 3 channel 吗？我们的 BUSI 是 1 channel？
smp支持in_channels = 1,它会对预训练的第一层权重进行适配
