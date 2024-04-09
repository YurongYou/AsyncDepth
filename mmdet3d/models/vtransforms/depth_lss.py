from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn
from torch.utils.checkpoint import checkpoint

from mmdet3d.models.builder import VTRANSFORMS

from .base import BaseDepthTransform, BaseTransform

__all__ = ["DepthLSSTransform", "HindsightDepthLSSTransform", "HindsightDepthLSSTransform_v2", "HindsightFeatureLSSTransform"]


@VTRANSFORMS.register_module()
class DepthLSSTransform(BaseDepthTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
        depth_loss: dict = None,
        raw_image_size: Tuple[int, int] = (),
        with_grad_ckpt: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
            depth_loss=depth_loss,
            raw_image_size=raw_image_size,
            with_grad_ckpt=with_grad_ckpt
        )
        feature_down_sample = image_size[0] // feature_size[0]
        if feature_down_sample == 8:
            self.dtransform = nn.Sequential(
                nn.Conv2d(1, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
        elif feature_down_sample == 4:
            self.dtransform = nn.Sequential(
                nn.Conv2d(1, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 5, padding='same'),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
        elif feature_down_sample == 2:
            self.dtransform = nn.Sequential(
                nn.Conv2d(1, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=2, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 5, padding='same'),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    @force_fp32()
    def get_cam_feats(self, x, d):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x, depth.view(B, N, self.D, fH, fW)

    def forward(self, *args, **kwargs):
        if kwargs.get("return_depth", False):
            x, depth = super().forward(*args, **kwargs)
        else:
            x = super().forward(*args, **kwargs)
        x = self.downsample(x)
        if kwargs.get("return_depth", False):
            return x, depth
        else:
            return x


@VTRANSFORMS.register_module()
class HindsightDepthLSSTransform(BaseTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_traversals: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
        depth_loss: dict = None,
        raw_image_size: Tuple[int, int] = (),
        with_grad_ckpt: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
            depth_loss=depth_loss,
            raw_image_size=raw_image_size,
            with_grad_ckpt=with_grad_ckpt
        )
        self.n_traversals = n_traversals
        feature_down_sample = image_size[0] // feature_size[0]
        if feature_down_sample == 8:
            self.dtransform = nn.Sequential(
                nn.Conv2d(n_traversals, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
        elif feature_down_sample == 4:
            self.dtransform = nn.Sequential(
                nn.Conv2d(n_traversals, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 5, padding='same'),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
        elif feature_down_sample == 2:
            self.dtransform = nn.Sequential(
                nn.Conv2d(n_traversals, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=2, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 5, padding='same'),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    @force_fp32()
    def get_cam_feats(self, x, d):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D: (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x, depth.view(B, N, self.D, fH, fW)

    @force_fp32()
    def forward(self,
                img,
                points,
                sensor2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                cam_intrinsic,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                hindsight_points,
                return_depth=False,
                **kwargs,):
        # assert points is None
        assert hindsight_points is not None
        rots = sensor2ego[..., :3, :3]
        trans = sensor2ego[..., :3, 3]
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        # print(img.shape, self.image_size, self.feature_size)
        # print(len(hindsight_points))
        # print(len(hindsight_points))
        # import pdb; pdb.set_trace()
        batch_size = len(hindsight_points)
        depth = torch.zeros(batch_size, img.shape[1], self.n_traversals, *self.image_size).to(
            points[0].device
        )

        for b in range(batch_size):
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]
            for traversal_id in range(min(self.n_traversals, len(hindsight_points[b]))):
                cur_coords = hindsight_points[b][traversal_id][:, :3]

                # inverse aug
                cur_coords -= cur_lidar_aug_matrix[:3, 3]
                cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                    cur_coords.transpose(1, 0)
                )
                # lidar2image
                cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
                cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
                # get 2d coords
                dist = cur_coords[:, 2, :]
                cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
                cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

                # imgaug
                cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
                cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
                cur_coords = cur_coords[:, :2, :].transpose(1, 2)

                # normalize coords for grid sample
                cur_coords = cur_coords[..., [1, 0]]

                on_img = (
                    (cur_coords[..., 0] < self.image_size[0])
                    & (cur_coords[..., 0] >= 0)
                    & (cur_coords[..., 1] < self.image_size[1])
                    & (cur_coords[..., 1] >= 0)
                )
                for c in range(on_img.shape[0]):
                    masked_coords = cur_coords[c, on_img[c]].long()
                    masked_dist = dist[c, on_img[c]]
                    depth[b, c, traversal_id, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        # x, depth = self.get_cam_feats(img, depth)
        if self.with_grad_ckpt:
            x, depth = checkpoint(self.get_cam_feats, img, depth)
        else:
            x, depth = self.get_cam_feats(img, depth)
        x = self.bev_pool(geom, x)

        x = self.downsample(x)
        if return_depth:
            return x, depth
        else:
            return x


@VTRANSFORMS.register_module()
class HindsightDepthLSSTransform_v2(BaseTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
        depth_loss: dict = None,
        raw_image_size: Tuple[int, int] = (),
        with_grad_ckpt: bool = False,
        depth_feature_reduction: str = "mean",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
            depth_loss=depth_loss,
            raw_image_size=raw_image_size,
            with_grad_ckpt=with_grad_ckpt
        )
        feature_down_sample = image_size[0] // feature_size[0]
        if feature_down_sample == 8:
            self.dtransform = nn.Sequential(
                nn.Conv2d(1, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
        elif feature_down_sample == 4:
            self.dtransform = nn.Sequential(
                nn.Conv2d(1, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 5, padding='same'),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
        elif feature_down_sample == 2:
            self.dtransform = nn.Sequential(
                nn.Conv2d(1, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=2, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 5, padding='same'),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()
        self.depth_feature_reduction = depth_feature_reduction

    @force_fp32()
    def get_cam_feats(self, x, hindsight_points_depth_map):
        B, N, C, fH, fW = x.shape

        # len: B
        n_traversals = [len(d) for d in hindsight_points_depth_map]

        # sum(B), N, 1, H, W
        combined_depth_map = torch.stack(sum(hindsight_points_depth_map, [])).unsqueeze(2)

        total_traversals, N, _, H, W = combined_depth_map.shape

        combined_depth_map = combined_depth_map.view(total_traversals * N, 1, H, W)

        combined_depth_map = self.dtransform(combined_depth_map)
        combined_depth_map = combined_depth_map.view(total_traversals, N, *combined_depth_map.shape[-3:])

        if self.depth_feature_reduction == 'mean':
            # average across all the past traversals associated to each scan
            aggregated_depth_map = torch.cat([d.mean(dim=0, keepdim=True) for d in combined_depth_map.split(n_traversals, dim=0)], dim=0)
        elif self.depth_feature_reduction == 'max':
            aggregated_depth_map = torch.cat(
                [d.max(dim=0, keepdim=True).values for d in combined_depth_map.split(n_traversals, dim=0)], dim=0)
        else:
            raise NotImplementedError()

        d = aggregated_depth_map.view(B * N, -1, fH, fW)
        x = x.view(B * N, C, fH, fW)

        # d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D: (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x, depth.view(B, N, self.D, fH, fW)

    @force_fp32()
    def forward(self,
                img,
                points,
                sensor2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                cam_intrinsic,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                hindsight_points,
                hindsight_points_depth_map,
                return_depth=False,
                **kwargs,):
        # assert points is None
        assert hindsight_points is not None
        assert hindsight_points_depth_map is not None

        rots = sensor2ego[..., :3, :3]
        trans = sensor2ego[..., :3, 3]
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]


        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        # x, depth = self.get_cam_feats(img, depth)
        if self.with_grad_ckpt:
            x, depth = checkpoint(self.get_cam_feats, img, hindsight_points_depth_map)
        else:
            x, depth = self.get_cam_feats(img, hindsight_points_depth_map)
        x = self.bev_pool(geom, x)

        x = self.downsample(x)
        if return_depth:
            return x, depth
        else:
            return x


@VTRANSFORMS.register_module()
class HindsightFeatureLSSTransform(BaseTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
        depth_loss: dict = None,
        raw_image_size: Tuple[int, int] = (),
        with_grad_ckpt: bool = False,
        use_hindsight_depth: bool = False,
        hindsight_config: dict = None,
        depth_feature_reduction: str = "mean",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
            depth_loss=depth_loss,
            raw_image_size=raw_image_size,
            with_grad_ckpt=with_grad_ckpt
        )

        self.use_hindsight_depth = use_hindsight_depth
        self.hindsight_config = hindsight_config
        self.depth_rep_dim = 64

        self.hindsight_features_dim = self.hindsight_config["out_channels"]

        # Need to figure out downsampling factor
        feature_down_sample = image_size[0] // feature_size[0]
        if feature_down_sample == 8:
            self.dtransform = nn.Sequential(
                nn.Conv2d(1, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, self.depth_rep_dim, 5, stride=2, padding=2),
                nn.BatchNorm2d(self.depth_rep_dim),
                nn.ReLU(True),
            ) if self.use_hindsight_depth else None

            self.hindsight_downsampler = nn.Sequential(
                nn.Conv2d(self.hindsight_features_dim, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, self.depth_rep_dim, 5, stride=2, padding=2),
                nn.BatchNorm2d(self.depth_rep_dim),
                nn.ReLU(True),
            )  
        elif feature_down_sample == 4:
            self.dtransform = nn.Sequential(
                nn.Conv2d(1, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, self.depth_rep_dim, 5, padding='same'),
                nn.BatchNorm2d(self.depth_rep_dim),
                nn.ReLU(True),
            ) if self.use_hindsight_depth else None

            self.hindsight_downsampler = nn.Sequential(
                nn.Conv2d(self.hindsight_features_dim, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, self.depth_rep_dim, 5, padding='same'),
                nn.BatchNorm2d(self.depth_rep_dim),
                nn.ReLU(True),
            )          
        elif feature_down_sample == 2:
            self.dtransform = nn.Sequential(
                nn.Conv2d(1, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=2, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 5, padding='same'),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            ) if self.use_hindsight_depth else None

            self.hindsight_downsampler = nn.Sequential(
                nn.Conv2d(self.hindsight_features_dim, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=2, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 5, padding='same'),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )

        self.depth_net_input_features_dim = self.depth_rep_dim

        if self.use_hindsight_depth:
            self.depth_net_input_features_dim += self.depth_rep_dim

        self.depthnet = nn.Sequential(
            nn.Conv2d(self.depth_net_input_features_dim + in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

        self.depth_feature_reduction = depth_feature_reduction

    @force_fp32()
    def get_cam_feats(self, x, hindsight_feature_map):
        B, N, C, fH, fW = x.shape
        B, N, C_hindsight, fH, fW = hindsight_feature_map.shape

        x = x.view(B * N, C, fH, fW)
        hindsight_feature_map = hindsight_feature_map.view(B * N, C_hindsight, fH, fW)

        # d = self.dtransform(d)
        x = torch.cat([hindsight_feature_map, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D: (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x, depth.view(B, N, self.D, fH, fW)

    @force_fp32()
    def process_hindsight_depth(self, hindsight_points_depth_map):
        # len: B
        n_traversals = [len(d) for d in hindsight_points_depth_map]

        # sum(B), N, 1, H, W
        combined_depth_map = torch.stack(sum(hindsight_points_depth_map, [])).unsqueeze(2)
        
        total_traversals, N, _, H, W = combined_depth_map.shape

        combined_depth_map = combined_depth_map.view(total_traversals * N, 1, H, W)

        combined_depth_map = self.dtransform(combined_depth_map)
        combined_depth_map = combined_depth_map.view(total_traversals, N, *combined_depth_map.shape[-3:])

        if self.depth_feature_reduction == 'mean':
            # average across all the past traversals associated to each scan
            aggregated_depth_map = torch.cat(
                [d.mean(dim=0, keepdim=True) for d in combined_depth_map.split(n_traversals, dim=0)], dim=0)
        elif self.depth_feature_reduction == 'max':
            aggregated_depth_map = torch.cat(
                [d.max(dim=0, keepdim=True).values for d in combined_depth_map.split(n_traversals, dim=0)], dim=0)
        else:
            raise NotImplementedError()
        return aggregated_depth_map

    @force_fp32()
    def process_hindsight_features(self, hindsight_features):
        # Need to dowsample the hindsight features 
        B, N, C, fH, fW = hindsight_features.shape

        fm = self.hindsight_downsampler(hindsight_features.view(B * N, C, fH, fW))
        fm = fm.view(B, N, *fm.shape[-3:])

        # should return B, N, C_hindsight, H, W
        return fm

    @force_fp32()
    def forward(self,
                img,
                points,
                sensor2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                cam_intrinsic,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                hindsight_points,
                hindsight_points_depth_map, 
                hindsight_features,
                return_depth=False,
                **kwargs,):
        # assert points is None
        assert hindsight_points is not None
        assert hindsight_points_depth_map is not None

        rots = sensor2ego[..., :3, :3]
        trans = sensor2ego[..., :3, 3]
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]


        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        # downsample the 2d hindsight features maps (assume this is properly projected to 2D)
        hindsight_feature_map = self.process_hindsight_features(hindsight_features)

        if self.use_hindsight_depth:
            hindsight_depth_map = self.process_hindsight_depth(hindsight_points_depth_map)
            hindsight_feature_map = torch.cat([hindsight_feature_map, hindsight_depth_map], dim=2)

        # x, depth = self.get_cam_feats(img, depth)
        if self.with_grad_ckpt:
            x, depth = checkpoint(self.get_cam_feats, img, hindsight_feature_map)
        else:
            x, depth = self.get_cam_feats(img, hindsight_feature_map)
        x = self.bev_pool(geom, x)

        x = self.downsample(x)
        if return_depth:
            return x, depth
        else:
            return x

@VTRANSFORMS.register_module()
class HindsightFeatureLSSTransform_v2(BaseTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
        depth_loss: dict = None,
        raw_image_size: Tuple[int, int] = (),
        with_grad_ckpt: bool = False,
        use_hindsight_depth: bool = False,
        hindsight_config: dict = None,
        depth_feature_reduction: str = "mean",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
            depth_loss=depth_loss,
            raw_image_size=raw_image_size,
            with_grad_ckpt=with_grad_ckpt
        )

        self.use_hindsight_depth = use_hindsight_depth
        self.hindsight_config = hindsight_config
        self.depth_rep_dim = 64

        self.hindsight_features_dim = self.hindsight_config["out_channels"]

        # Need to figure out downsampling factor
        feature_down_sample = image_size[0] // feature_size[0]
        if feature_down_sample == 8:
            self.dtransform = nn.Sequential(
                nn.Conv2d(1, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, self.depth_rep_dim, 5, stride=2, padding=2),
                nn.BatchNorm2d(self.depth_rep_dim),
                nn.ReLU(True),
            ) if self.use_hindsight_depth else None

            self.hindsight_downsampler = nn.Sequential(
                nn.Conv2d(self.hindsight_features_dim, self.hindsight_features_dim, 1),
                nn.BatchNorm2d(self.hindsight_features_dim),
                nn.ReLU(True),
                nn.Conv2d(self.hindsight_features_dim, self.hindsight_features_dim*2, 5, stride=4, padding=2),
                nn.BatchNorm2d(self.hindsight_features_dim*2),
                nn.ReLU(True),
                nn.Conv2d(self.hindsight_features_dim*2, self.hindsight_features_dim*4, 5, stride=2, padding=2),
                nn.BatchNorm2d(self.hindsight_features_dim*4),
                nn.ReLU(True),
            )
        elif feature_down_sample == 4:
            self.dtransform = nn.Sequential(
                nn.Conv2d(1, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, self.depth_rep_dim, 5, padding='same'),
                nn.BatchNorm2d(self.depth_rep_dim),
                nn.ReLU(True),
            ) if self.use_hindsight_depth else None

            self.hindsight_downsampler = nn.Sequential(
                nn.Conv2d(self.hindsight_features_dim, self.hindsight_features_dim, 1),
                nn.BatchNorm2d(self.hindsight_features_dim),
                nn.ReLU(True),
                nn.Conv2d(self.hindsight_features_dim, self.hindsight_features_dim*2, 5, stride=4, padding=2),
                nn.BatchNorm2d(self.hindsight_features_dim*2),
                nn.ReLU(True),
                nn.Conv2d(self.hindsight_features_dim*2, self.hindsight_features_dim*4, 5, padding='same'),
                nn.BatchNorm2d(self.hindsight_features_dim*4),
                nn.ReLU(True),
            )          
        elif feature_down_sample == 2:
            self.dtransform = nn.Sequential(
                nn.Conv2d(1, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=2, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 5, padding='same'),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            ) if self.use_hindsight_depth else None

            self.hindsight_downsampler = nn.Sequential(
                nn.Conv2d(self.hindsight_features_dim, self.hindsight_features_dim, 1),
                nn.BatchNorm2d(self.hindsight_features_dim),
                nn.ReLU(True),
                nn.Conv2d(self.hindsight_features_dim, self.hindsight_features_dim*2, 5, stride=2, padding=2),
                nn.BatchNorm2d(self.hindsight_features_dim*2),
                nn.ReLU(True),
                nn.Conv2d(self.hindsight_features_dim*2, self.hindsight_features_dim*4, 5, padding='same'),
                nn.BatchNorm2d(self.hindsight_features_dim*4),
                nn.ReLU(True),
            )

        self.depth_net_input_features_dim = self.hindsight_features_dim*4 

        if self.use_hindsight_depth:
            self.depth_net_input_features_dim += self.depth_rep_dim

        self.depthnet = nn.Sequential(
            nn.Conv2d(self.depth_net_input_features_dim + in_channels, self.depth_net_input_features_dim + in_channels, 3, padding=1),
            nn.BatchNorm2d(self.depth_net_input_features_dim + in_channels),
            nn.ReLU(True),
            nn.Conv2d(self.depth_net_input_features_dim + in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

        self.depth_feature_reduction = depth_feature_reduction

    @force_fp32()
    def get_cam_feats(self, x, hindsight_feature_map):
        B, N, C, fH, fW = x.shape
        B, N, C_hindsight, fH, fW = hindsight_feature_map.shape

        x = x.view(B * N, C, fH, fW)
        hindsight_feature_map = hindsight_feature_map.view(B * N, C_hindsight, fH, fW)

        # d = self.dtransform(d)
        x = torch.cat([hindsight_feature_map, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D: (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x, depth.view(B, N, self.D, fH, fW)

    @force_fp32()
    def process_hindsight_depth(self, hindsight_points_depth_map):
        # len: B
        n_traversals = [len(d) for d in hindsight_points_depth_map]

        # sum(B), N, 1, H, W
        combined_depth_map = torch.stack(sum(hindsight_points_depth_map, [])).unsqueeze(2)
        
        total_traversals, N, _, H, W = combined_depth_map.shape

        combined_depth_map = combined_depth_map.view(total_traversals * N, 1, H, W)

        combined_depth_map = self.dtransform(combined_depth_map)
        combined_depth_map = combined_depth_map.view(total_traversals, N, *combined_depth_map.shape[-3:])

        if self.depth_feature_reduction == 'mean':
            # average across all the past traversals associated to each scan
            aggregated_depth_map = torch.cat(
                [d.mean(dim=0, keepdim=True) for d in combined_depth_map.split(n_traversals, dim=0)], dim=0)
        elif self.depth_feature_reduction == 'max':
            aggregated_depth_map = torch.cat(
                [d.max(dim=0, keepdim=True).values for d in combined_depth_map.split(n_traversals, dim=0)], dim=0)
        else:
            raise NotImplementedError()
        return aggregated_depth_map

    @force_fp32()
    def process_hindsight_features(self, hindsight_features):
        # Need to dowsample the hindsight features 
        B, N, C, fH, fW = hindsight_features.shape

        fm = self.hindsight_downsampler(hindsight_features.view(B * N, C, fH, fW))
        fm = fm.view(B, N, *fm.shape[-3:])

        # should return B, N, C_hindsight, H, W
        return fm

    @force_fp32()
    def forward(self,
                img,
                points,
                sensor2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                cam_intrinsic,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                hindsight_points,
                hindsight_points_depth_map, 
                hindsight_features,
                return_depth=False,
                **kwargs,):
        # assert points is None
        assert hindsight_points is not None
        assert hindsight_points_depth_map is not None

        rots = sensor2ego[..., :3, :3]
        trans = sensor2ego[..., :3, 3]
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]


        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        # downsample the 2d hindsight features maps (assume this is properly projected to 2D)
        hindsight_feature_map = self.process_hindsight_features(hindsight_features)

        if self.use_hindsight_depth:
            hindsight_depth_map = self.process_hindsight_depth(hindsight_points_depth_map)
            hindsight_feature_map = torch.cat([hindsight_feature_map, hindsight_depth_map], dim=2)

        # x, depth = self.get_cam_feats(img, depth)
        if self.with_grad_ckpt:
            x, depth = checkpoint(self.get_cam_feats, img, hindsight_feature_map)
        else:
            x, depth = self.get_cam_feats(img, hindsight_feature_map)
        x = self.bev_pool(geom, x)

        x = self.downsample(x)
        if return_depth:
            return x, depth
        else:
            return x