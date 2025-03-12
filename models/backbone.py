import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.ops.misc import FrozenBatchNorm2d


class Backbone(nn.Module):

    def __init__(
        self,
        name: str,
        pretrained: bool,
        dilation: bool,
        reduction: int,
        swav: bool,
        requires_grad: bool
    ):

        super(Backbone, self).__init__()

        resnet = getattr(models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained, norm_layer=FrozenBatchNorm2d
        )

        self.backbone = resnet
        self.reduction = reduction

        if name == 'resnet50' and swav:
            checkpoint = torch.hub.load_state_dict_from_url(
                'https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar',
                map_location="cpu"
            )
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            self.backbone.load_state_dict(state_dict, strict=False)

        # concatenation of layers 2, 3 and 4
        self.num_channels = 896 if name in ['resnet18', 'resnet34'] else 3584

        for n, param in self.backbone.named_parameters():
            if 'layer2' not in n and 'layer3' not in n and 'layer4' not in n:
                param.requires_grad_(False)
            else:
                param.requires_grad_(requires_grad)

        self.de_pred4 = nn.Sequential(
            Conv2d(2048, 2048, 3, same_padding=True, NL='relu'),
        )
        self.de_pred3 = nn.Sequential(
            Conv2d(2048 + 1024, 1024, 3, same_padding=True, NL='relu'),
        )

        self.de_pred2 = nn.Sequential(
            Conv2d(1024 + 512, 512, 3, same_padding=True, NL='relu'),
        )

    def forward(self, x):
        # x(1,3,512,512)
        size = x.size(-2) // self.reduction, x.size(-1) // self.reduction #  size(64,64)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = layer2 = self.backbone.layer2(x) # layer2(1,512,64,64)
        x = layer3 = self.backbone.layer3(x) # layer3(1,1024,32,32)
        x = layer4 = self.backbone.layer4(x) # layer4(1,2048,16,16)

        # begining of decoding
        x = self.de_pred4(layer4)
        x4_out = x # (1,2048,16,16)
        x = F.interpolate(x, size=layer3.size()[2:])
        x = torch.cat([layer3, x], 1)
        x = self.de_pred3(x)
        x3_out = x # (1,1024,32,32)
        x = F.interpolate(x, size=layer2.size()[2:])
        x = torch.cat([layer2, x], 1)
        x = self.de_pred2(x)
        x2_out = x # (1,512,64,64)
        x = torch.cat([
            F.interpolate(f, size=size, mode='bilinear', align_corners=True)
            for f in [x2_out, x3_out, x4_out]
        ], dim=1)

        return x
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, \
                stride=1, NL='relu', same_padding=False, bn=False, dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = []
        if dilation == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else nn.Identity()
        if NL == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x