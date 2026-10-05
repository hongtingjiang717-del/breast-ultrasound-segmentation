#encoder做的事情就是让图像尺寸越来越小，但是提取出来的特征越来越丰富。
import torch
import torch.nn as nn   #专门负责搭建神经网络

class ConvBlock(nn.Module):
    """
    U-Net 中最基础的双卷积模块。

    输入：
        [B, in_channels, H, W]

    输出：
        [B, out_channels, H, W]

    这里不会改变图像的高度 H 和宽度 W，
    主要改变 Channel 数并提取图像特征。
    """
    #定义一个卷积模块，可以同时保存数据和功能
    #继承inheritance pytorch给我们准备好的神经网络基础模块
    #ConvBlock 是一种 PyTorch 神经网络模块，所以请把 nn.Module 已经具备的能力继承过来
    def __init__(self, in_channels, out_channels):
        #初始化函数，初始化父类nn.Module
        super().__init__()
        #======================================================
        #建立一个连续执行的神经网络模块（双卷积模块）
        #u-net大量使用3*3不仅可以提取局部边缘和纹理；参数量比较小，连续两个3*3可以获得更大的感受野，同时加入更多的非线性
        #======================================================
        self.block = nn.Sequential(
            #nn.Sequential表示按顺序执行每个步骤
            #第一次3*3卷积
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size=3,
                padding = 1,
                bias = False 
            ),
            #对卷积结果做batch normalizatin
            #BatchNorm 会对中间 feature map 的数值分布进行标准化处理，让训练通常更加稳定
            #feature map相当于不同的卷积核，用来提取特征1→64
            nn.BatchNorm2d(out_channels),
            #激活函数，加入非线性变换
            nn.ReLU(inplace=True),

            #第二次3*3卷积
            nn.Conv2d(
                in_channels = out_channels,
                out_channels = out_channels,
                kernel_size=3,
                padding = 1,
                bias = False 
            ),
            #对卷积结果做batch normalizatin
            nn.BatchNorm2d(out_channels),
            #激活函数
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        '''
        定义数据经过这个模块的时候应该怎样前向传播
        '''
        x = self.block(x)

        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        #=====================================
        #encoder
        #=======================================
        
        #第一层：输入灰度图1channel,输出64channel
        self.enc1 = ConvBlock(
            in_channels=1,
            out_channels=64
        )
        #第二层：64→128
        self.enc2 = ConvBlock(
            in_channels=64,
            out_channels=128
        )

        # 第三层：
        #
        # 128 → 256
        self.enc3 = ConvBlock(
            in_channels=128,
            out_channels=256
        )


        # 第四层：
        #
        # 256 → 512
        self.enc4 = ConvBlock(
            in_channels=256,
            out_channels=512
        )

        # max pooling最大池化---在每个2*2区域里选择最大值，我的理解：不同于上边的卷积是升维，池化是降维的
        #随着encoder加深，空间越来越小，模型逐渐从像素→边缘→纹理→病灶局部结构→更高级的语义信息，不断抽象
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        #==========================================
        #bottleneck
        #可以把他理解为unet最底部、信息最浓缩的一层，encoder不断地让尺寸变小，通道数增加，到bottleneck这一层空间信息少但是抽象特征变多了
        #这里模型看到的不再只是局部像素，而是更高级的病灶结构信息
        #==========================================
        #encoder最深处的特征提取层
        self.bottleneck = ConvBlock(
            in_channels = 512,
            out_channels = 1024
        )

        #====================================
        #decoder第四层；把小尺寸的feature map一步一步的放大回来
        #====================================
        #上采样
        #[B,1024,16,16]
        #[B,512,32,32]

        #nn.ConvTranspose2d是转置卷积，相比于普通卷积只是负责提取特征，它主要负责扩大空间尺寸，里面有可以训练的参数
        self.up4 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2
        )
        # up4 输出 512 channel
        # e4       512 channel
        #
        # 拼接后：
        # 512 + 512 = 1024
        self.dec4 = ConvBlock(
            in_channels=1024,
            out_channels=512
        )

        # ==============================================
        # Decoder 第三层
        # ==============================================

        # [B,512,32,32]
        #        ↓
        # [B,256,64,64]
        self.up3 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2
        )

        # up3:
        # 256 channel
        #
        # e3:
        # 256 channel
        #
        # 拼接：
        # 512 channel
        self.dec3 = ConvBlock(
            in_channels=512,
            out_channels=256
        )


        # ==============================================
        # Decoder 第二层
        # ==============================================

        # [B,256,64,64]
        #        ↓
        # [B,128,128,128]
        self.up2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2
        )

        self.dec2 = ConvBlock(
            in_channels=256,
            out_channels=128
        )


        # ==============================================
        # Decoder 第一层
        # ==============================================

        # [B,128,128,128]
        #        ↓
        # [B,64,256,256]
        self.up1 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = ConvBlock(
            in_channels=128,
            out_channels=64
        )

        #===========================================
        #最终输出层
        #===========================================
        self.final = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1
        )

#========================================
# skip_connection---
# 将encoder早期保存的高分辨率特征重新送给decoder，
# 两边结合起来就是语义信息和精细空间信息的结合----有精细的边缘信息
#========================================

    def forward(self, x):

        # ==============================================
        # Encoder
        # ==============================================

        # [B,1,256,256]
        # ↓
        # [B,64,256,256]
        e1 = self.enc1(x)

        # pool：
        # [B,64,256,256]
        # ↓
        # [B,64,128,128]
        #
        # enc2：
        # ↓
        # [B,128,128,128]
        e2 = self.enc2(
            self.pool(e1)
        )

        # [B,128,128,128]
        # ↓
        # [B,256,64,64]
        e3 = self.enc3(
            self.pool(e2)
        )

        # [B,256,64,64]
        # ↓
        # [B,512,32,32]
        e4 = self.enc4(
            self.pool(e3)
        )


        # ==============================================
        # Bottleneck
        # ==============================================

        # e4：
        # [B,512,32,32]
        #
        # pool：
        # [B,512,16,16]
        #
        # bottleneck：
        # [B,1024,16,16]

        b = self.bottleneck(
            self.pool(e4)
        )


        # ==============================================
        # Decoder 4
        # ==============================================

        # 上采样：
        #
        # [B,1024,16,16]
        #
        # ↓
        #
        # [B,512,32,32]

        d4 = self.up4(b)


        # 与 Encoder 的 e4 拼接
        #
        # d4：
        # [B,512,32,32]
        #
        # e4：
        # [B,512,32,32]
        #
        # ↓ cat
        #
        # [B,1024,32,32]

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )


        # 双卷积：
        #
        # [B,1024,32,32]
        #
        # ↓
        #
        # [B,512,32,32]

        d4 = self.dec4(d4)


        # ==============================================
        # Decoder 3
        # ==============================================

        # [B,512,32,32]
        # ↓
        # [B,256,64,64]

        d3 = self.up3(d4)


        # 与 e3 拼接：
        #
        # 256 + 256 = 512 channel

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )


        # [B,512,64,64]
        # ↓
        # [B,256,64,64]

        d3 = self.dec3(d3)


        # ==============================================
        # Decoder 2
        # ==============================================

        # [B,256,64,64]
        # ↓
        # [B,128,128,128]

        d2 = self.up2(d3)


        # 128 + 128 = 256
        d2 = torch.cat(
            [d2, e2],
            dim=1
        )


        # [B,256,128,128]
        # ↓
        # [B,128,128,128]

        d2 = self.dec2(d2)


        # ==============================================
        # Decoder 1
        # ==============================================

        # [B,128,128,128]
        # ↓
        # [B,64,256,256]

        d1 = self.up1(d2)


        # 64 + 64 = 128
        d1 = torch.cat(
            [d1, e1],
            dim=1
        )


        # [B,128,256,256]
        # ↓
        # [B,64,256,256]

        d1 = self.dec1(d1)


        # ==============================================
        # 最终输出
        # ==============================================

        # [B,64,256,256]
        #
        # ↓ 1×1 Conv
        #
        # [B,1,256,256]

        output = self.final(d1)


        return output



