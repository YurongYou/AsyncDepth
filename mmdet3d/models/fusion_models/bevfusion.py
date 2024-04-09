from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

from mmdet3d.models import FUSIONMODELS
# from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.models.builder import (build_backbone, build_fuser, build_head,
                                    build_neck, build_vtransform, build_dtransform)
from mmdet3d.ops import DynamicScatter, Voxelization

from .base import Base3DFusionModel

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )

            if 'Dtransform' in encoders['camera']:
                self.encoders['camera']['Dtransform'] = build_dtransform(encoders['camera']['Dtransform'])
                self.Dtransform_use_gt_depth = encoders['camera']['Dtransform']['use_gt_depth']

        # if encoders.get("hindsight_camera") is not None:
        #     self.hindsight_gradient_checkpoint = encoders["hindsight_camera"].get("gradient_checkpoint", False)
        #     self.hindsight_image_share_backbone = encoders["hindsight_camera"].get("share_backbone", False)
        #     self.hindsight_image_share_neck = encoders["hindsight_camera"].get("share_neck", False)
        #     self.hindsight_image_share_vtransform = encoders["hindsight_camera"].get("share_vtransform", False)

        #     modules = {"fuser": build_fuser(encoders["hindsight_camera"]["fuser"])}

        #     if not self.hindsight_image_share_backbone:
        #         modules["backbone"] = build_backbone(encoders["hindsight_camera"]["backbone"])
        #     if not self.hindsight_image_share_neck:
        #         modules["neck"] = build_neck(encoders["hindsight_camera"]["neck"])
        #     if not self.hindsight_image_share_vtransform:
        #         modules["vtransform"] = build_vtransform(encoders["hindsight_camera"]["vtransform"])

        #     self.encoders["hindsight_camera"] = nn.ModuleDict(modules)
        #     self.hindsight_maxpool_sample_loc = encoders["hindsight_camera"].get(
        #         "maxpool_sample_loc", False)
        #     self.hindsight_num_cam = encoders["hindsight_camera"].get(
        #         "hindsight_num_cam", 6)
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        # if encoders.get("hindsight_lidar") is not None:
        #     if encoders["hindsight_lidar"]["voxelize"].get("max_num_points", -1) > 0:
        #         voxelize_module = Voxelization(**encoders["hindsight_lidar"]["voxelize"])
        #     else:
        #         voxelize_module = DynamicScatter(**encoders["hindsight_lidar"]["voxelize"])
        #     self.encoders["hindsight_lidar"] = nn.ModuleDict(
        #         {
        #             "voxelize": voxelize_module,
        #             "backbone": build_backbone(encoders["hindsight_lidar"]["backbone"]),
        #             "fuser": build_fuser(encoders["hindsight_lidar"]["fuser"]),
        #         }
        #     )
        #     self.voxelize_reduce_hindsight = encoders["hindsight_lidar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        self.depth_supervision = (encoders.get("camera", None) is not None and
                                  encoders["camera"]["vtransform"].get('depth_loss', None) is not None)

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0
            if self.depth_supervision:
                self.loss_scale['depth'] = 1.0

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            print("initing encoders[\"camera\"][\"backbone\"]")
            self.encoders["camera"]["backbone"].init_weights()

            if "Dtransform" in self.encoders["camera"]:
                print("initing encoders[\"camera\"][\"Dtransform\"]")
                self.encoders["camera"]["Dtransform"].init_weights()

        # if "hindsight_camera" in self.encoders and not self.hindsight_image_share_backbone:
        #     print("initing encoders[\"hindsight_camera\"][\"backbone\"]")
        #     self.encoders["hindsight_camera"]["backbone"].init_weights()

        if "lidar" in self.encoders:
            print("initing encoders[\"lidar\"][\"backbone\"]")
            self.encoders["lidar"]["backbone"].init_weights()

        # if "hindsight_lidar" in self.encoders:
        #     print("initing encoders[\"hindsight_lidar\"][\"backbone\"]")
        #     self.encoders["hindsight_lidar"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        hindsight_points,
        hindsight_points_depth_map,
        points_depth_map,
        return_depth=False,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # start.record()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)
        # end.record()

        # # Waits for everything to finish running
        # torch.cuda.synchronize()
        # print()
        # print(start.elapsed_time(end))

        if "Dtransform" in self.encoders["camera"]:
            if self.Dtransform_use_gt_depth:
                # increase by one dimension to match the shape
                gt_map_input = [[i] for i in points_depth_map]
                # a list of fms at different resolution
                # Each tensor is of shape (B, N, C, H, W)
                depth_fm = self.encoders["camera"]["Dtransform"](gt_map_input)
            else:
                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)

                # start.record()
                # a list of fms at different resolution
                # Each tensor is of shape (B, N, C, H, W)
                depth_fm = self.encoders["camera"]["Dtransform"](hindsight_points_depth_map)
                # end.record()

                # # Waits for everything to finish running
                # torch.cuda.synchronize()
                # print()
                # print(start.elapsed_time(end))
            x = torch.cat([x, depth_fm[0]], dim=2)

        return self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            hindsight_points=hindsight_points,
            hindsight_points_depth_map=hindsight_points_depth_map,
            return_depth=return_depth
        )

    # def extract_camera_hindsight_features(
    #     self,
    #     hindsight_imgs,
    #     hindsight_camera_intrinsics,
    #     hindsight_camera2lidar,
    #     hindsight_img_aug_matrix,
    #     lidar_aug_matrix,
    # ) -> torch.Tensor:

    #     B, Nh, Nc, C, H, W = hindsight_imgs.size()
    #     x = hindsight_imgs.view(B * Nh * Nc, C, H, W)

    #     if self.hindsight_image_share_backbone:
    #         backbone = self.encoders["camera"]["backbone"]
    #     else:
    #         backbone = self.encoders["hindsight_camera"]["backbone"]
    #     if self.hindsight_gradient_checkpoint:
    #         x.requires_grad = True
    #         x = checkpoint(backbone, x)
    #     else:
    #         x = backbone(x)
    #     if self.hindsight_image_share_neck:
    #         x = self.encoders["camera"]["neck"](x)
    #     else:
    #         x = self.encoders["hindsight_camera"]["neck"](x)

    #     if not isinstance(x, torch.Tensor):
    #         x = x[0]

    #     _, C, H, W = x.size()
    #     x = x.view(B * Nh, Nc, C, H, W)
    #     if not self.hindsight_maxpool_sample_loc:
    #         camera_intrinsics = hindsight_camera_intrinsics.view(
    #             B * Nh, Nc, *hindsight_camera_intrinsics.size()[-2:])
    #         camera2lidar = hindsight_camera2lidar.view(
    #             B * Nh, Nc, *hindsight_camera2lidar.size()[-2:])
    #         img_aug_matrix = hindsight_img_aug_matrix.view(
    #             B * Nh, Nc, *hindsight_img_aug_matrix.size()[-2:])
    #         lidar_aug_matrix = (
    #             lidar_aug_matrix.view(B, 1, *lidar_aug_matrix.size()[-2:])
    #             .repeat(1, Nh, 1, 1)
    #             .view(B * Nh, *lidar_aug_matrix.size()[-2:])
    #         )
    #     else:
    #         Ns = Nc // self.hindsight_num_cam
    #         assert Nc % self.hindsight_num_cam == 0

    #         x = x.view(B * Nh * Ns, self.hindsight_num_cam, C, H, W)
    #         camera_intrinsics = hindsight_camera_intrinsics.view(
    #             B * Nh * Ns, self.hindsight_num_cam, *hindsight_camera_intrinsics.size()[-2:])
    #         camera2lidar = hindsight_camera2lidar.view(
    #             B * Nh * Ns, self.hindsight_num_cam,
    #             *hindsight_camera2lidar.size()[-2:])
    #         img_aug_matrix = hindsight_img_aug_matrix.view(
    #             B * Nh * Ns, self.hindsight_num_cam,
    #             *hindsight_img_aug_matrix.size()[-2:])
    #         lidar_aug_matrix = (
    #             lidar_aug_matrix.view(B, 1, *lidar_aug_matrix.size()[-2:])
    #             .repeat(1, Nh * Ns, 1, 1)
    #             .view(B * Nh * Ns, *lidar_aug_matrix.size()[-2:])
    #         )

    #     if self.hindsight_image_share_vtransform:
    #         vtransform = self.encoders["camera"]["vtransform"]
    #     else:
    #         vtransform = self.encoders["hindsight_camera"]["vtransform"]
    #     x = vtransform(
    #         x,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         camera_intrinsics,
    #         camera2lidar,
    #         img_aug_matrix,
    #         lidar_aug_matrix,
    #     )
    #     if self.hindsight_maxpool_sample_loc:
    #         x = x.view(B, Nh, Ns, *x.size()[1:])
    #         x = torch.max(x, dim=2)[0]
    #         bev_hindsight_features = x.view(B, Nh, *x.size()[2:])
    #     else:
    #         bev_hindsight_features = x.view(B, Nh, *x.size()[1:])

    #     return self.encoders["hindsight_camera"]["fuser"](bev_hindsight_features)

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    def extract_lidar_hindsight_features(self, traversals) -> torch.Tensor:
        # traversals is a list of list of traversals
        # traversals[0] contains all the traversals for the first sample
        # traversals_flatten = sum(traversals, [])
        # print(traversals[0][0].shape)
        # for i in range(len(traversals_flatten)):
        #     print(i, traversals_flatten[i].shape)
        # print(len(traversals_flatten))
        # # print(traversals_flatten[0].shape)

        # # traversals_batch_idx = sum([[ind] * len(i) for ind, i in enumerate(traversals)], [])
        # # print(traversals_flatten.__len__())
        # # print(traversals_batch_idx)
        # feats, coords, sizes = self.voxelize_hindsight(traversals_flatten)
        # batch_size = coords[-1, 0] + 1
        # print(feats.size())
        # print(batch_size)
        # print(coords.shape)
        #  x = self.encoders["hindsight_lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        # print(x.shape)
        # fms = x
        fms = []
        for traversal in traversals:
            feats, coords, sizes = self.voxelize_hindsight(traversal)
            batch_size = coords[-1, 0] + 1
            x = self.encoders["hindsight_lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
            fms.append(x)

        fms = self.encoders["hindsight_lidar"]["fuser"](fms)
        return fms

    @torch.no_grad()
    @force_fp32()
    def voxelize_hindsight(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["hindsight_lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce_hindsight:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        # hindsight_img=None,
        # hindsight_camera_intrinsics=None,
        # hindsight_camera2lidar=None,
        # hindsight_img_aug_matrix=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)

            # start.record()
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                # hindsight_img,
                # hindsight_camera_intrinsics,
                # hindsight_camera2lidar,
                # hindsight_img_aug_matrix,
                **kwargs,
            )
            # end.record()

            # # Waits for everything to finish running
            # torch.cuda.synchronize()
            # print()
            # print(start.elapsed_time(end))
            return outputs

    @auto_fp16(apply_to=("img", "points", "hindsight_img"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        # hindsight_img=None,
        # hindsight_camera_intrinsics=None,
        # hindsight_camera2lidar=None,
        # hindsight_img_aug_matrix=None,
        hindsight_points=None,
        hindsight_points_depth_map=None,
        points_depth_map=None,
        single_sweep_points=None,
        **kwargs,
    ):
        features = []
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                    hindsight_points=hindsight_points,
                    hindsight_points_depth_map=hindsight_points_depth_map,
                    points_depth_map=points_depth_map,
                    return_depth=self.depth_supervision
                )
                if self.depth_supervision:
                    feature, pred_depth = feature
            # elif sensor == "hindsight_camera":
            #     feature = self.extract_camera_hindsight_features(
            #         hindsight_img,
            #         hindsight_camera_intrinsics,
            #         hindsight_camera2lidar,
            #         hindsight_img_aug_matrix,
            #         lidar_aug_matrix,
            #     )
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
            elif sensor == "hindsight_lidar":
                feature = self.extract_lidar_hindsight_features(hindsight_points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)


        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            if self.depth_supervision:
                assert single_sweep_points is not None
                outputs["loss/depth"] = self.encoders['camera']['vtransform'].compute_depth_loss(
                    pred_depth,
                    single_sweep_points,
                    lidar2image,
                    img_aug_matrix,
                    lidar_aug_matrix,
                ) * self.loss_scale["depth"]
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs
