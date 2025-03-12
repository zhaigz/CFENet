import torch
from torch import nn
class DynamicSimilarityMatcher(nn.Module):
    def __init__(self, hidden_dim, proj_dim, dynamic_proj_dim, activation='tanh', num_objects=3, use_bias=False):
        super().__init__()
        self.num_objects = num_objects
        self.query_conv = nn.Linear(in_features=hidden_dim, out_features=proj_dim, bias=use_bias)
        self.key_conv = nn.Linear(in_features=hidden_dim, out_features=proj_dim, bias=use_bias)
        self.dynamic_pattern_conv = nn.Sequential(nn.Linear(in_features=proj_dim, out_features=dynamic_proj_dim),
                                                  nn.ReLU(),
                                                  nn.Linear(in_features=dynamic_proj_dim, out_features=proj_dim))

        self.softmax = nn.Softmax(dim=-1)
        self._weight_init_()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            raise NotImplementedError

    def forward(self, features, patches):
        bs, c, h, w = features.shape
        features = features.flatten(2).permute(2, 0, 1)  # hw * bs * dim

        proj_feat = self.query_conv(features) + features
        patches_feat = self.key_conv(patches) + patches
        patches_ca = self.activation(self.dynamic_pattern_conv(patches_feat))

        proj_feat = proj_feat.permute(1, 0, 2)
        patches_feat = (patches_feat * (patches_ca + 1)).reshape(self.num_objects,bs,c,-1)  # bs * c * exemplar_number
        # 添加注意力机制
        # patches_feat = self.attention(proj_feat, patches_feat.permute(1, 2, 0), proj_feat)[0].permute(1, 0, 2)

        energy = []
        for feat in patches_feat:
            similarity_weights  = torch.matmul(proj_feat, feat)  # bs * hw * exemplar_number
            # similarity_map = torch.sum(similarity_weights, dim=-1, keepdim=True)
            similarity_map = torch.mean(similarity_weights, dim=-1, keepdim=True)
            energy.append(similarity_map)
        similar_maps = torch.cat(energy, dim=-1)
        return similar_maps

    def _weight_init_(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



