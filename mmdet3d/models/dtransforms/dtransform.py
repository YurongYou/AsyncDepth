from mmdet.models.builder import BACKBONES, NECKS
import torch
from torch import nn
from mmdet3d.models import DTRANSFORM
from mmcv.runner import BaseModule

@DTRANSFORM.register_module()
class Dtransform(BaseModule):
    def __init__(self,
                 backbone,
                 neck,
                 reduction='mean'
                 ):
        super().__init__()
        self.backbone = BACKBONES.build(backbone)
        self.out_channels = [neck.out_channels] * neck.num_outs
        self.neck = NECKS.build(neck)
        self.reduction = reduction
        # self.backbone.init_weights()

    def forward(self, hindsight_points_depth_map):
        # len: B
        n_traversals = [len(d) for d in hindsight_points_depth_map]

        # sum(B), 1, H, W
        combined_depth_map = torch.stack(sum(hindsight_points_depth_map, [])).unsqueeze(1)

        total_traversals, _, H, W = combined_depth_map.shape

        combined_depth_map = combined_depth_map.view(total_traversals, 1, H, W)

        outs = self.neck(self.backbone(combined_depth_map))
        outs = [x.view(total_traversals, *x.shape[-3:]) for x in outs]
        if self.reduction == 'mean':
            outs = [torch.cat(
                [d.mean(dim=0, keepdim=True) for d in x.split(n_traversals, dim=0)],
                dim=0)
                for x in outs]
        elif self.reduction == 'max':
            outs = [torch.cat(
                [d.max(dim=0, keepdim=True).values for d in x.split(n_traversals, dim=0)],
                dim=0)
                for x in outs]
        else:
            raise NotImplementedError()
        return outs
