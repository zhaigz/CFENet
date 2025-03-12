from torch import nn


class FeatureFuse(nn.Module):
    def __init__(
        self,
        embed_dim,
        mid_dim,
        dropout,
    ):
        super(FeatureFuse, self).__init__()

        # 定义两个卷积层 (conv1 和 conv2),并保持输出尺寸不变
        self.conv1 = nn.Conv2d(embed_dim, mid_dim, kernel_size=3, stride=1, padding=1)
        # 在训练过程中随机丢弃一部分神经元输出，以防止过拟合
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(mid_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        # 定义两个 Layer Normalization 层 (norm1 和 norm2)，用于对输入特征进行归一化，提高模型的稳定性和训练效率。
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation =  nn.LeakyReLU()

    def forward(self,feature,feats):
        ##################################################################################
        # fuse feature
        ##################################################################################
        # 对 tgt 和融合后的 tgt2 进行融合操作，通过添加 (tgt + self.dropout1(tgt2)) 并使用 Dropout 进行正则化。
        tgt = feature + self.dropout1(feats)
        # 进行通道交换 (permute) 和 Layer Normalization (norm1)，并经过卷积 (conv1)、激活函数 (activation)、Dropout 正则化 (dropout)
        # 和另一个卷积 (conv2)，再次将结果与原始 tgt 进行融合 (tgt + self.dropout2(tgt2)).
        tgt = tgt.permute(0, 2, 3, 1).contiguous()
        tgt = self.norm1(tgt).permute(0, 3, 1, 2).contiguous()
        # Learnable Feature Fusion
        tgt2 = self.conv2(self.dropout(self.activation(self.conv1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        # 最后再进行一次通道交换和 Layer Normalization (norm2)，得到最终的融合特征
        tgt = tgt.permute(0, 2, 3, 1).contiguous()
        tgt = self.norm2(tgt).permute(0, 3, 1, 2).contiguous()
        return tgt