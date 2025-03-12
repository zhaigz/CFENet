import torch
import torch.nn as nn
import torch.nn.functional as F

class SimilarityCalculator(nn.Module):
    def __init__(self, kernel_dim,num_objects):
        super(SimilarityCalculator, self).__init__()
        self.kernel_dim = kernel_dim
        self.num_objects = num_objects

    def forward(self, f_e, all_prototypes):
        bs, _, h, w = f_e.size()
        num_objects = self.num_objects
        similar_maps_list = []

        for i in range(all_prototypes.size(0)):
            prototypes = all_prototypes[i, ...].permute(1, 0, 2).reshape(
                bs, num_objects, self.kernel_dim, self.kernel_dim, -1
            ).permute(0, 1, 4, 2, 3).flatten(0, 2)[:, None, ...]
            # ([768, 1, 3, 3])

            f_e_expanded = torch.cat([f_e for _ in range(num_objects)], dim=1).flatten(0, 1).unsqueeze(0)
            # (1,768,64,64)

            similar_map = F.conv2d(
                f_e_expanded,
                prototypes,
                bias=None,
                padding=self.kernel_dim // 2,
                groups=prototypes.size(0)
            )

            similar_maps_list.append(similar_map)

        # similar_maps = torch.cat(similar_maps_list, dim=2)

        return similar_maps_list
