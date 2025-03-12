import torch
from torch import nn


def crop_roi_feat(feat, boxes, out_stride):
    """
    feat: 1 x c x h x w
    boxes: m x 4, 4: [y_tl, x_tl, y_br, x_br]
    """
    _, _, h, w = feat.shape
    boxes = boxes.squeeze(0)
    # 边界框放缩因子，将其从原始图像空间映射到特征图空间
    boxes_scaled = boxes / out_stride
    # 对每个边界框的坐标值进行取整操作，左上角坐标向下取整
    # [:, :2]保留所有行，取前两列的坐标，即每个边界框的左上角坐标
    boxes_scaled[:, :2] = torch.floor(boxes_scaled[:, :2])  # y_tl, x_tl: floor
    # 右上角坐标向上取整
    boxes_scaled[:, 2:] = torch.ceil(boxes_scaled[:, 2:])  # y_br, x_br: ceil
    # 确保边界框在特征图上的像素位置不超出特征图的边界范围
    # 确保左上坐标不小于0
    boxes_scaled[:, :2] = torch.clamp_min(boxes_scaled[:, :2], 0)
    boxes_scaled[:, 2] = torch.clamp_max(boxes_scaled[:, 2], h)
    boxes_scaled[:, 3] = torch.clamp_max(boxes_scaled[:, 3], w)
    feat_boxes = []
    # 遍历所有边界框，根据其在特征图上的坐标，在输入特征图中裁剪对应的感兴趣区域
    for idx_box in range(0, boxes.shape[0]):
        y_tl, x_tl, y_br, x_br = boxes_scaled[idx_box]
        y_tl, x_tl, y_br, x_br = int(y_tl), int(x_tl), int(y_br), int(x_br)
        # 其中 : 表示选择所有的样本（第一个维度）、所有的通道（第二个维度），
        # y_tl : (y_br + 1) 和 x_tl : (x_br + 1) 则表示在垂直和水平方向上选择边界框所包含的像素区域。
        feat_box = feat[:, :, y_tl : (y_br + 1), x_tl : (x_br + 1)]
        feat_boxes.append(feat_box)
    return feat_boxes

def get_activation(activation):
    if activation == "relu":
        return nn.ReLU
    if activation == "leaky_relu":
        return nn.LeakyReLU
    raise NotImplementedError