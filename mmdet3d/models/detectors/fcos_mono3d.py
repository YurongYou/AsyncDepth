# Copyright (c) OpenMMLab. All rights reserved.
# from mmdet.models.builder import DETECTORS
from mmdet3d.models import FUSIONMODELS
from .single_stage_mono3d import SingleStageMono3DDetector


@FUSIONMODELS.register_module()
class FCOSMono3D(SingleStageMono3DDetector):
    r"""`FCOS3D <https://arxiv.org/abs/2104.10956>`_ for monocular 3D object detection.

    Currently please refer to our entry on the
    `leaderboard <https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Camera>`_.
    """  # noqa: E501

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 depth_condition=None):
        super(FCOSMono3D, self).__init__(backbone, neck, bbox_head, train_cfg,
                                         test_cfg, pretrained, depth_condition=depth_condition)
