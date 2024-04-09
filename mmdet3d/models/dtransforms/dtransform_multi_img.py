from mmdet.models.builder import BACKBONES, NECKS
import torch
from torch import nn
from mmdet3d.models import DTRANSFORM
from mmcv.runner import BaseModule
from torch.nn import functional as F

@DTRANSFORM.register_module()
class DtransformMultiImgs(BaseModule):
    def __init__(self,
                 backbone,
                 neck,
                 reduction='mean',
                 **kwargs
                 ):
        super().__init__()
        self.backbone = BACKBONES.build(backbone)
        self.out_channels = [neck.out_channels] * neck.num_outs
        self.neck = NECKS.build(neck)
        self.reduction = reduction

    def forward(self, hindsight_points_depth_map):
        # len: B
        n_traversals = [len(d) for d in hindsight_points_depth_map]

        # sum(B), N, 1, H, W
        combined_depth_map = torch.stack(sum(hindsight_points_depth_map, [])).unsqueeze(2)

        total_traversals, N, _, H, W = combined_depth_map.shape

        combined_depth_map = combined_depth_map.view(total_traversals * N, 1, H, W)

        outs = self.neck(self.backbone(combined_depth_map))
        outs = [x.view(total_traversals, N, *x.shape[-3:]) for x in outs]
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
        elif self.reduction == "none":
            outs = [torch.cat(x.split(n_traversals, dim=0), dim=0) for x in outs]
        else:
            raise NotImplementedError()
        return outs

@DTRANSFORM.register_module()
class IdentityMultiImgs(nn.Module):
    def __init__(self,
                 downsampling_factors=[8, 4, 2],
                 interpolation_mode='bilinear',
                 reduction='mean',
                 **kwargs
                 ):
        super().__init__()
        self.downsampling_factors = downsampling_factors
        self.interpolation_mode = interpolation_mode
        self.reduction = reduction

    def forward(self, hindsight_points_depth_map):
        # len: B
        n_traversals = [len(d) for d in hindsight_points_depth_map]

        # sum(B), N, 1, H, W
        combined_depth_map = torch.stack(sum(hindsight_points_depth_map, [])).unsqueeze(2)

        total_traversals, N, _, H, W = combined_depth_map.shape

        combined_depth_map = combined_depth_map.view(total_traversals * N, 1, H, W)

        outs = [F.interpolate(combined_depth_map, scale_factor=1 / f, mode=self.interpolation_mode) for f in self.downsampling_factors]
        outs = [x.view(total_traversals, N, *x.shape[-3:]) for x in outs]
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

