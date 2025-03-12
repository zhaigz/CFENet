from .mlp import MLP
from .positional_encoding import PositionalEncodingsFixed

import torch
from torch import nn

from torchvision.ops import roi_align


class OPEModule(nn.Module):

    def __init__(
        self,
        num_iterative_steps: int,
        emb_dim: int,
        kernel_dim: int,
        num_objects: int,
        num_heads: int,
        reduction: int,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
        zero_shot: bool,
    ):

        super(OPEModule, self).__init__()

        self.num_iterative_steps = num_iterative_steps
        self.zero_shot = zero_shot
        self.kernel_dim = kernel_dim
        self.num_objects = num_objects
        self.emb_dim = emb_dim
        self.reduction = reduction

        if num_iterative_steps > 0:
            self.iterative_adaptation = IterativeAdaptationModule(
                num_layers=num_iterative_steps, emb_dim=emb_dim, num_heads=num_heads,
                dropout=0, layer_norm_eps=layer_norm_eps,
                mlp_factor=mlp_factor, norm_first=norm_first,
                activation=activation, norm=norm,
                zero_shot=zero_shot
            )

        if not self.zero_shot:
            self.shape_or_objectness = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Linear(64, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, self.kernel_dim**2 * emb_dim)
            )
        else:
            self.shape_or_objectness = nn.Parameter(
                torch.empty((self.num_objects, self.kernel_dim**2, emb_dim))
            )
            nn.init.normal_(self.shape_or_objectness)

        self.pos_emb = PositionalEncodingsFixed(emb_dim)

    def forward(self, f_e, pos_emb, bboxes):
        bs, _, h, w = f_e.size()
        # extract the shape features or objectness
        # bboxes (1,3,4)
        if not self.zero_shot:
            # 初始化一个box_hw来存储box的宽度和高度信息
            box_hw = torch.zeros(bboxes.size(0), bboxes.size(1), 2).to(bboxes.device) # box_hw (1,3,2)
            box_hw[:, :, 0] = bboxes[:, :, 2] - bboxes[:, :, 0] # 宽度计算：右边界 - 左边界
            box_hw[:, :, 1] = bboxes[:, :, 3] - bboxes[:, :, 1] # 高度计算：下边界 - 上边界
            # (1,3,2)->(1,3,2034)->(1,3,9,256)->(1,27,256)->(27,1,256)
            shape_or_objectness = self.shape_or_objectness(box_hw).reshape(
                bs, -1, self.kernel_dim ** 2, self.emb_dim
            ).flatten(1, 2).transpose(0, 1)
        else:
            shape_or_objectness = self.shape_or_objectness.expand(
                bs, -1, -1, -1
            ).flatten(1, 2).transpose(0, 1)

        # if not zero shot add appearance
        if not self.zero_shot:
            # reshape bboxes into the format suitable for roi_align
            # 为了将每个bounding box的索引与其对应的坐标信息连接起来
            # 将bboxes(n=3)先展平为(3,4)然后与全为0的列向量在通道维度拼接->(3,5)
            bbox_index = torch.arange(
                    bs, requires_grad=False
                ).to(bboxes.device).repeat_interleave(self.num_objects).reshape(-1, 1)
            # bbox_index (3,1)tensor([[0],\n[0],\n[0]],device='cuda:0')
            bboxes = torch.cat([
                torch.arange(
                    bs, requires_grad=False
                ).to(bboxes.device).repeat_interleave(self.num_objects).reshape(-1, 1),
                bboxes.flatten(0, 1),
            ], dim=1) #(3,5) (0,x1,y1,x2,y2)
            # 特征图f_e进行感兴趣区域池化
            # 以bboxes作为输入，对每个bounding box在特征图f_e上提取固定大小（output_size=self.kernel_dim）的局部特征
            # appearance (27,1,256)
            appearance = roi_align(
                f_e,
                boxes=bboxes, output_size=self.kernel_dim,
                spatial_scale=1.0 / self.reduction, aligned=True
            ).permute(0, 2, 3, 1).reshape(
                bs, self.num_objects * self.kernel_dim ** 2, -1
            ).transpose(0, 1)
        else:
            appearance = None
        # 对于每一个对象，都生成了一组对应于特定位置编码的嵌入向量，并且这些向量与原始特征图的大小相关联。
        # 这些位置嵌入通常会在后续处理中与其它特征相结合，以便模型能够理解并学习不同位置之间的关系。
        # query_pos_emb (27,1,256)
        query_pos_emb = self.pos_emb(
            bs, self.kernel_dim, self.kernel_dim, f_e.device
        ).flatten(2).permute(2, 0, 1).repeat(self.num_objects, 1, 1)

        if self.num_iterative_steps > 0:
            # shape_or_objectness(27,1,256) appearance(27,1,256) memory (4096,1,256)
            # pos_emb(4096,1,256) query_pos_emb(27,1,256)
            memory = f_e.flatten(2).permute(2, 0, 1)
            all_prototypes = self.iterative_adaptation(
                shape_or_objectness, appearance, memory, pos_emb, query_pos_emb
            )
        else:
            if shape_or_objectness is not None and appearance is not None:
                all_prototypes = (shape_or_objectness + appearance).unsqueeze(0)
            else:
                all_prototypes = (
                    shape_or_objectness if shape_or_objectness is not None else appearance
                ).unsqueeze(0)

        return all_prototypes


class IterativeAdaptationModule(nn.Module):

    def __init__(
        self,
        num_layers: int,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
        zero_shot: bool
    ):

        super(IterativeAdaptationModule, self).__init__()

        self.layers = nn.ModuleList([
            IterativeAdaptationLayer(
                emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation, zero_shot
            ) for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(emb_dim, layer_norm_eps) if norm else nn.Identity()

    def forward(
        self, tgt, appearance, memory, pos_emb, query_pos_emb, tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None
    ):

        output = tgt
        outputs = list()
        for i, layer in enumerate(self.layers):
            output = layer(
                output, appearance, memory, pos_emb, query_pos_emb, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask
            )
            outputs.append(self.norm(output))
        # 分别将迭代1/2/3次后的原型堆叠
        return torch.stack(outputs)


class IterativeAdaptationLayer(nn.Module):

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        zero_shot: bool
    ):
        super(IterativeAdaptationLayer, self).__init__()
        #
        self.norm_first = norm_first
        self.zero_shot = zero_shot

        if not self.zero_shot:
            self.norm1 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm2 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm3 = nn.LayerNorm(emb_dim, layer_norm_eps)
        if not self.zero_shot:
            self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if not self.zero_shot:
            self.self_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout)
        self.enc_dec_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout)

        self.feed_forward = FeedForward(emb_dim,emb_dim)
        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)

    def with_emb(self, x, emb):
        return x if emb is None else x + emb

    def forward(
        self, tgt, appearance, memory, pos_emb, query_pos_emb, tgt_mask, memory_mask,
        tgt_key_padding_mask, memory_key_padding_mask
    ):
        tgt_norm=[]
        if self.norm_first:
            if not self.zero_shot:
                tgt = tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(tgt, query_pos_emb),
                    key=self.with_emb(appearance, query_pos_emb),
                    value=appearance,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0])
                tgt_norm = self.norm1(tgt)
            tgt_ffd = self.feed_forward(tgt_norm)
            tgt = tgt + tgt_ffd
            tgt_norm = self.norm1(tgt)

            tgt = tgt + self.dropout2(self.enc_dec_attn(
                query=self.with_emb(tgt_norm, query_pos_emb),
                key=memory+pos_emb,
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )[0])
            tgt_norm = self.norm2(tgt)
            tgt_ffd = self.feed_forward(tgt_norm)
            tgt = tgt + tgt_ffd
            tgt_norm = self.norm1(tgt)
            tgt = tgt + self.dropout3(self.mlp(tgt_norm))

        else:
            if not self.zero_shot:
                tgt = self.norm1(tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(tgt, query_pos_emb),
                    key=self.with_emb(appearance),
                    value=appearance,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0]))

            tgt = self.norm2(tgt + self.dropout2(self.enc_dec_attn(
                query=self.with_emb(tgt, query_pos_emb),
                key=memory+pos_emb,
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )[0]))

            tgt = self.norm3(tgt + self.dropout3(self.mlp(tgt)))

        return tgt

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int,
                 dropout: float = 0.1,
                 activation=nn.ReLU(),
                 bias1: bool = True,
                 bias2: bool = True,
                 ):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor):
        g = self.activation(self.layer1(x))
        x = self.dropout(g)
        return self.layer2(x)
