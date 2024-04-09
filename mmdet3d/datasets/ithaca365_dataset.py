import copy
import tempfile
from os import path as osp
from typing import Any, Dict, List

import mmcv
import numpy as np
import pyquaternion
import torch
from ithaca365.utils.data_classes import Box as NuScenesBox
# from lyft_dataset_sdk.lyftdataset import LyftDataset as Lyft
from ithaca365.ithaca365 import Ithaca365
from ithaca365.eval.detection.evaluate import NuScenesEval, DetectionEval
from ithaca365.eval.common.utils import center_distance


from pyquaternion import Quaternion

from mmdet.datasets import DATASETS

from ..core import show_result
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset
from .pipelines import Compose

# this is copy from ithaca365-devkit/ithaca365/eval/detection/data_classes.py
class ithaca365DetectionConfig:
    """ Data class that specifies the detection evaluation settings. """

    def __init__(self,
                 class_range: Dict[str, List[List[int]]],
                 dist_fcn: str,
                 dist_ths: List[float],
                 dist_th_tp: float,
                 min_recall: float,
                 min_precision: float,
                 max_boxes_per_sample: int,
                 mean_ap_weight: int,
                 only_accurate_localization: bool = False):
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight
        self.only_accurate_localization = only_accurate_localization

        self.class_names = self.class_range.keys()

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'class_range': self.class_range,
            'dist_fcn': self.dist_fcn,
            'dist_ths': self.dist_ths,
            'dist_th_tp': self.dist_th_tp,
            'min_recall': self.min_recall,
            'min_precision': self.min_precision,
            'max_boxes_per_sample': self.max_boxes_per_sample,
            'mean_ap_weight': self.mean_ap_weight,
            'only_accurate_localization': self.only_accurate_localization
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized dictionary. """
        return cls(content['class_range'],
                   content['dist_fcn'],
                   content['dist_ths'],
                   content['dist_th_tp'],
                   content['min_recall'],
                   content['min_precision'],
                   content['max_boxes_per_sample'],
                   content['mean_ap_weight'],
                   content['only_accurate_localization'])

    @property
    def dist_fcn_callable(self):
        """ Return the distance function corresponding to the dist_fcn string. """
        if self.dist_fcn == 'center_distance':
            return center_distance
        else:
            raise Exception('Error: Unknown distance function %s!' % self.dist_fcn)



@DATASETS.register_module()
class Ithaca365Dataset(Custom3DDataset):
    r"""Ithaca365 Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        dataset_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool): Whether to use `use_valid_flag` key in the info
            file as mask to filter gt_boxes and gt_names. Defaults to False.
    """
    NameMapping = {
        'bicyclist': 'bicyclist',
        'bus': 'bus',
        'car': 'car',
        'motorcyclist': 'motorcyclist',
        'pedestrian': 'pedestrian',
        'truck': 'truck',
    }
    DefaultAttribute = {
        "car": "is_stationary",
        "pedestrian": "is_stationary",
        "truck": "is_stationary",
        "bus": "is_stationary",
        "motorcyclist": "is_stationary",
        "bicyclist": "is_stationary",
    }

    CLASSES = ("car", "truck", "bus", "bicyclist", "motorcyclist","pedestrian")

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 dataset_root=None,
                 object_classes=None,
                 load_interval=1,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 use_valid_flag=False,
                 select_traversal_frames=None,
                 load_relative_path=False,
                 **kwargs):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        self.select_traversal_frames = select_traversal_frames
        super().__init__(
            dataset_root=dataset_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=object_classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        **kwargs)
        #TODO: add argument to dataset and config to add differant eval versions
        from ithaca365.eval.detection.config import config_factory
        eval_version = 'detection_by_range'
        # self.eval_detection_configs = config_factory(eval_version)

        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
                use_hindsight_camera=False,
            )
        self.load_relative_path = load_relative_path

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        # loading data from a file-like object needs file format
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): sample index
                - lidar_path (str): filename of point clouds
                - sweeps (list[dict]): infos of sweeps
                - timestamp (float): sample timestamp
                - img_filename (str, optional): image filename
                - lidar2img (list[np.ndarray], optional): transformations
                    from lidar to different cameras
                - ann_info (dict): annotation info
        """
        info = self.data_infos[index]

        data = dict(
            token=info["token"],
            sample_idx=info['token'],
            lidar_path=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'],
        )
        if self.load_relative_path:
            data['lidar_path'] = osp.join(self.dataset_root, data['lidar_path'])
            data['sweeps'] = copy.deepcopy(data['sweeps'])
            for sweep in data['sweeps']:
                sweep['data_path'] = osp.join(self.dataset_root, sweep['data_path'])

        # ego to global transform
        ego2global = np.eye(4).astype(np.float32)
        ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = info["ego2global_translation"]
        data["ego2global"] = ego2global

        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        data["lidar2ego"] = lidar2ego

        if self.modality["use_camera"]:
            data["image_paths"] = []
            data["lidar2camera"] = []
            data["lidar2image"] = []
            data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []

            for _, camera_info in info["cams"].items():
                if self.load_relative_path:
                    data["image_paths"].append(
                        osp.join(self.dataset_root, camera_info["data_path"])
                    )
                else:
                    data["image_paths"].append(camera_info["data_path"])

                # lidar to camera transform
                lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
                lidar2camera_t = (
                    camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
                )
                lidar2camera_rt = np.eye(4).astype(np.float32)
                lidar2camera_rt[:3, :3] = lidar2camera_r.T
                lidar2camera_rt[3, :3] = -lidar2camera_t
                data["lidar2camera"].append(lidar2camera_rt.T)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = camera_info["cam_intrinsic"]
                data["camera_intrinsics"].append(camera_intrinsics)

                # lidar to image transform
                lidar2image = camera_intrinsics @ lidar2camera_rt.T
                data["lidar2image"].append(lidar2image)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                camera2ego[:3, :3] = Quaternion(
                    camera_info["sensor2ego_rotation"]
                ).rotation_matrix
                camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
                data["camera2ego"].append(camera2ego)

                # camera to lidar transform
                camera2lidar = np.eye(4).astype(np.float32)
                camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
                camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
                data["camera2lidar"].append(camera2lidar)

        if self.modality.get("use_hindsight_camera", False):
            data["hindsight_image_paths"] = []
            data["hindsight_camera_intrinsics"] = []
            data["hindsight_camera2lidar"] = []

            sensors = list(info["cams"].keys())
            for traversal in info["history_traversals"]:
                data["hindsight_image_paths"].append([])
                data["hindsight_camera_intrinsics"].append([])
                data["hindsight_camera2lidar"].append([])

                traversal_sweeps = copy.deepcopy(traversal)
                traversal_sweeps = {k: traversal_sweeps[k] for k in sensors}
                if self.select_traversal_frames:
                    traversal_sweeps = {
                        k: [traversal_sweeps[k][_i] for _i in self.select_traversal_frames]
                        for k in sensors}
                for camera_infos in zip(*traversal_sweeps.values()):
                    for camera_info in camera_infos:
                        if self.load_relative_path:
                            data["hindsight_image_paths"][-1].append(
                                osp.join(self.dataset_root, camera_info["data_path"])
                            )
                        else:
                            data["hindsight_image_paths"][-1].append(camera_info["data_path"])

                        # camera intrinsics
                        camera_intrinsics = np.eye(4).astype(np.float32)
                        camera_intrinsics[:3, :3] = camera_info["cam_intrinsic"]
                        data["hindsight_camera_intrinsics"][-1].append(camera_intrinsics)

                        # camera to lidar transform
                        camera2lidar = np.eye(4).astype(np.float32)
                        camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
                        camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
                        data["hindsight_camera2lidar"][-1].append(camera2lidar)

        if self.modality.get("use_hindsight_lidar", False):
            data["hindsight_lidar_paths"] = []
            data["hindsight_lidar2lidar"] = []

            for traversal in info["history_traversals"]:
                data["hindsight_lidar_paths"].append([])
                data["hindsight_lidar2lidar"].append([])

                traversal_sweeps = traversal['LIDAR_TOP']
                if self.select_traversal_frames:
                    traversal_sweeps = [traversal_sweeps[_i]
                                            for _i in self.select_traversal_frames]

                for lidar_info in traversal_sweeps:
                    data["hindsight_lidar_paths"][-1].append(lidar_info["data_path"])
                    sensor2lidar = np.eye(4).astype(np.float32)
                    sensor2lidar[:3, :3] = lidar_info["sensor2lidar_rotation"]
                    sensor2lidar[:3, 3] = lidar_info["sensor2lidar_translation"]
                    data["hindsight_lidar2lidar"][-1].append(sensor2lidar)

        if self.modality.get("use_hindsight_lidar", False):
            data["hindsight_lidar_paths"] = []
            data["hindsight_lidar2lidar"] = []

            for traversal in info["history_traversals"]:
                data["hindsight_lidar_paths"].append([])
                data["hindsight_lidar2lidar"].append([])

                traversal_sweeps = traversal['LIDAR_TOP']
                if self.select_traversal_frames:
                    traversal_sweeps = [traversal_sweeps[_i]
                                            for _i in self.select_traversal_frames]

                for lidar_info in traversal_sweeps:
                    data_path = lidar_info["data_path"]
                    if self.load_relative_path:
                        data_path = osp.join(self.dataset_root, data_path)
                    data["hindsight_lidar_paths"][-1].append(data_path)
                    sensor2lidar = np.eye(4).astype(np.float32)
                    sensor2lidar[:3, :3] = lidar_info["sensor2lidar_rotation"]
                    sensor2lidar[:3, 3] = lidar_info["sensor2lidar_translation"]
                    data["hindsight_lidar2lidar"][-1].append(sensor2lidar)

        # if not self.test_mode:
        annos = self.get_ann_info(index)
        data['ann_info'] = annos

        return data

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        ##TO DO: Does valid/empty check need to be removed?
        # if self.use_valid_flag:
        #     mask = info["valid_flag"]
        # else:
        #     mask = info["num_lidar_pts"] > 0
        # gt_bboxes_3d = info["gt_boxes"][mask]
        # gt_names_3d = info["gt_names"][mask]
        gt_bboxes_3d = info['gt_boxes']
        gt_names_3d = info['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if 'gt_shape' in info:
            gt_shape = info['gt_shape']
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_shape], axis=-1)

        # TODO(cad297): why does lyft have a 'gt_shape' check
        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
             box_dim=gt_bboxes_3d.shape[-1],
             origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_ithaca365_box(det)
            sample_token = self.data_infos[sample_id]["token"]
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]

                if name in [
                    "car",
                    "truck",
                    "bus",
                ]:
                    attr = "vehicle.moving"
                elif name in ["bicyclist", "motorcyclist"]:
                    attr = "cycle.with_rider"
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    velocity=box.velocity.tolist()[:2],
                    attribute_name=attr
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_ithaca365.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(
        self,
        result_path,
        logger=None,
        metric="bbox",
        result_name="pts_bbox",
        close_only=False,
    ):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        import json

        #TODO: chose where to fix 'version' issue (when generating pickle?)
        if self.version == 'v1.1-trainval':
            _version = 'v1.1'
        output_dir = osp.join(*osp.split(result_path)[:-1])
        ith365 = Ithaca365(version=_version,
                    dataroot=self.dataset_root,
                    verbose=True)
        eval_set_map = {
            "v1.1-trainval": "val"}
        with open(result_path, "r") as outfile:
            r = json.load(outfile)

        eval_criteria = {
            "dist_fcn": "center_distance",
            "dist_ths": [0.5, 1.0, 2.0, 4.0],
            "dist_th_tp": 2.0,
            "min_recall": 0.1,
            "min_precision": 0.1,
            "max_boxes_per_sample": 500,
            "mean_ap_weight": 3,
            "only_accurate_localization": True
        }

        if close_only:
            eval_criteria["class_range"] = {
                cl: [[0, 30], [30, 50], [0, 50]] for cl in self.CLASSES
            }
        else:
            eval_criteria["class_range"] = {
                cl: [[0, 30], [30, 50], [50, 80], [0, 80]] for cl in self.CLASSES
            }

        # from ithaca365.eval.detection.config import config_factory
        # eval_version = 'detection_by_range'
        # # self.eval_detection_configs = config_factory(eval_version)
        # eval_detection_configs = config_factory(eval_version)
        eval_detection_configs = ithaca365DetectionConfig(**eval_criteria)

        ith365_eval = DetectionEval(
            ith365,
            config=eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True
        )

        ith365_eval.main(plot_examples = 0, render_curves=False)
        metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
        metric_prefix = f'{result_name}_Ithaca365'

        # with open(output_path / 'metrics_summary.json', 'r') as f:
        #     metrics = json.load(f)

        result_str, result_dict = format_ithaca365_results(metrics,
                eval_detection_configs.class_names, metric_prefix)
        return result_dict
        # # record metrics
        # # TODO(cad297): change to map? (see lyft_dataset.py)
        # metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
        # detail = dict()
        # for name in self.CLASSES:
        #     for k, v in metrics["label_aps"][name].items():
        #         val = float("{:.4f}".format(v))
        #         detail["object/{}_ap_dist_{}".format(name, k)] = val
        #     for k, v in metrics["label_tp_errors"][name].items():
        #         val = float("{:.4f}".format(v))
        #         detail["object/{}_{}".format(name, k)] = val
        #     for k, v in metrics["tp_errors"].items():
        #         val = float("{:.4f}".format(v))
        #         detail["object/{}".format(self.ErrNameMapping[k])] = val

        # detail["object/nds"] = metrics["nd_score"]
        # detail["object/map"] = metrics["mean_ap"]
        # return detail

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(
            self
        ), "The length of results is not equal to the dataset len: {} != {}".format(
            len(results), len(self)
        )

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                # not evaluate 2D predictions on nuScenes
                if '2d' in name:
                    continue
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def evaluate_map(self, results):
        thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])

        num_classes = len(self.map_classes)
        num_thresholds = len(thresholds)

        tp = torch.zeros(num_classes, num_thresholds)
        fp = torch.zeros(num_classes, num_thresholds)
        fn = torch.zeros(num_classes, num_thresholds)

        for result in results:
            pred = result["masks_bev"]
            label = result["gt_masks_bev"]

            pred = pred.detach().reshape(num_classes, -1)
            label = label.detach().bool().reshape(num_classes, -1)

            pred = pred[:, :, None] >= thresholds
            label = label[:, :, None]

            tp += (pred & label).sum(dim=1)
            fp += (pred & ~label).sum(dim=1)
            fn += (~pred & label).sum(dim=1)

        ious = tp / (tp + fp + fn + 1e-7)

        metrics = {}
        for index, name in enumerate(self.map_classes):
            metrics[f"map/{name}/iou@max"] = ious[index].max().item()
            for threshold, iou in zip(thresholds, ious[index]):
                metrics[f"map/{name}/iou@{threshold.item():.2f}"] = iou.item()
        metrics["map/mean/iou@max"] = ious.max(dim=1).values.mean().item()
        return metrics

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 close_only=False):
        """Evaluation in Lyft protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: 'bbox'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str, optional): The prefix of json files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            csv_savepath (str, optional): The path for saving csv files.
                It includes the file path and the csv filename,
                e.g., "a/b/filename.csv". If not specified,
                the result will not be converted to csv file.
            result_names (list[str], optional): Result names in the
                metric prefix. Default: ['pts_bbox'].
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Evaluation results.
        """
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print(f'Evaluating bboxes of {name}')
                ret_dict = self._evaluate_single(result_files[name], close_only=close_only)
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files, close_only=close_only)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return results_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=3,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=False, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)

def output_to_ithaca365_box(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2
    # velocity = np.empty((box3d.shape[0],3))  # no velocity for ith365
    # velocity[:] = np.NaN
    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        # velocity = (*box3d.tensor[i, 7:9], 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
        )
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(
    info, boxes):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs : Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list

def format_ithaca365_results(metrics, class_names, metric_prefix='pts_bbox_ithaca365'):
    result = '----------------Ithca365 %s results-----------------\n' % metric_prefix
    for name in class_names:
        result += f'***{name}\n'
        for det_range in metrics['label_aps'][name].keys():
            threshs = ', '.join([str(t) for t in metrics['label_aps'][name][det_range].keys()])
            ap_list = list(metrics['label_aps'][name][det_range].values())

            err_name =', '.join([x.split('_')[0] for x in list(metrics['label_tp_errors'][name][det_range].keys())])
            error_list = list(metrics['label_tp_errors'][name][det_range].values())

            result += f'range {det_range} error@{err_name} | AP@{threshs}\n'
            result += ', '.join(['%.2f' % x for x in error_list]) + ' | '
            result += ', '.join(['%.2f' % (x * 100) for x in ap_list])
            result += f" | mean AP: {metrics['mean_dist_aps'][name][det_range]}"
            result += '\n'

    result += '--------------average performance-------------\n'
    details = {}
    for key, val in metrics['tp_errors'].items():
        result += '%s:\t %.4f\n' % (key, val)
        details[f'{metric_prefix}/{key}'] = val

    result += 'mAP:\t %.4f\n' % metrics['mean_ap']
    result += 'NDS:\t %.4f\n' % metrics['nd_score']

    match_thresh = {
        "car": 1.0,
        "truck": 1.0,
        "bus": 1.0,
        "motorcyclist": 1.0,
        "bicyclist": 1.0,
        "pedestrian": 1.0,
    }


    tp_errors_keys = ['trans_err', 'scale_err', 'orient_err']

    result += '--------------table log summary-------------\n'
    for name in class_names:
        result += f'***{name}\n'
        result += f"Range:\t" + ', '.join(list(metrics['label_aps'][name].keys())) + '\n'
        metrics_by_ranges = []
        for det_range in metrics['label_aps'][name].keys():
            # print(metrics['label_aps'][name][det_range].keys)
            ap_val = metrics['label_aps'][name][det_range][str(match_thresh[name])]
            metrics_by_ranges.append(ap_val)
            # result += f"{det_range} match {match_thresh[name]}:\t"
            # result += ", ".join([f"{x:3.1f} / {y:3.1f}" for x,
            #                     y in zip(bev05, threeD05)]) + "\n\n"
        result += f"AP@{match_thresh[name]:.2f}m:\t" + ", ".join(['%.2f' % (x * 100) for x in metrics_by_ranges]) + "\n"

        tp_errors_dict = { key: [] for key in tp_errors_keys }
        # for tp errors
        for det_range in metrics['label_tp_errors'][name].keys():
            tp_errors = metrics['label_tp_errors'][name][det_range]
            for key in tp_errors_keys:
                tp_errors_dict[key].append(tp_errors[key])

        for key in tp_errors_keys:
            result += f"{key}@{metrics['cfg']['dist_th_tp']:.2f}m:\t" + ", ".join(['%.2f' % x for x in tp_errors_dict[key]]) + "\n"

    details.update({
        f'{metric_prefix}/mAP': metrics['mean_ap'],
        f'{metric_prefix}/NDS': metrics['nd_score'],
    })

    return result, details

