# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import tempfile
from os import path as osp

import mmcv
import numpy as np
import pandas as pd
from lyft_dataset_sdk.lyftdataset import LyftDataset as Lyft
from lyft_dataset_sdk.utils.data_classes import Box as LyftBox
from pyquaternion import Quaternion

from mmdet3d.core.evaluation.lyft_eval import lyft_eval, lyft_eval_by_distance

from ..core import show_result
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose


@DATASETS.register_module()
class LyftDataset(Custom3DDataset):
    r"""Lyft Dataset.

    This class serves as the API for experiments on the Lyft Dataset.

    Please refer to
    `<https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/data>`_
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
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """  # noqa: E501
    NameMapping = {
        'bicycle': 'bicycle',
        'bus': 'bus',
        'car': 'car',
        'emergency_vehicle': 'emergency_vehicle',
        'motorcycle': 'motorcycle',
        'other_vehicle': 'other_vehicle',
        'pedestrian': 'pedestrian',
        'truck': 'truck',
        'animal': 'animal'
    }
    DefaultAttribute = {
        'car': 'is_stationary',
        'truck': 'is_stationary',
        'bus': 'is_stationary',
        'emergency_vehicle': 'is_stationary',
        'other_vehicle': 'is_stationary',
        'motorcycle': 'is_stationary',
        'bicycle': 'is_stationary',
        'pedestrian': 'is_stationary',
        'animal': 'is_stationary'
    }
    CLASSES = ('car', 'truck', 'bus', 'emergency_vehicle', 'other_vehicle',
               'motorcycle', 'bicycle', 'pedestrian', 'animal')

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
                 load_relative_path=True,
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

        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
                use_hindsight_camera=False,
                use_hindsight_lidar=False,
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
            data['lidar_path'] = os.path.join(self.dataset_root, data['lidar_path'])
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
                data_path = camera_info["data_path"]
                if self.load_relative_path:
                    data_path = os.path.join(self.dataset_root, data_path)
                data["image_paths"].append(data_path)

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
                # traversal_sweeps is {sensor: [image_at_loc1, image_at_loc2, ...]}
                traversal_sweeps = {k: traversal_sweeps[k] for k in sensors}
                if self.select_traversal_frames:
                    traversal_sweeps = {
                        k: [traversal_sweeps[k][_i] for _i in self.select_traversal_frames]
                        for k in sensors}
                for camera_infos in zip(*traversal_sweeps.values()):
                    for camera_info in camera_infos:
                        data_path = camera_info["data_path"]
                        if self.load_relative_path:
                            data_path = os.path.join(self.dataset_root, data_path)
                        data["hindsight_image_paths"][-1].append(data_path)

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
                    data_path = lidar_info["data_path"]
                    if self.load_relative_path:
                        data_path = os.path.join(self.dataset_root, data_path)
                    data["hindsight_lidar_paths"][-1].append(data_path)
                    sensor2lidar = np.eye(4).astype(np.float32)
                    sensor2lidar[:3, :3] = lidar_info["sensor2lidar_rotation"]
                    sensor2lidar[:3, 3] = lidar_info["sensor2lidar_translation"]
                    data["hindsight_lidar2lidar"][-1].append(sensor2lidar)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            data['ann_info'] = annos

        return data

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # if self.use_valid_flag:
        #     mask = info["valid_flag"]
        # else:
        #     mask = info["num_lidar_pts"] > 0

        # gt_bboxes_3d = info['gt_boxes'][mask]
        # gt_names_3d = info['gt_names'][mask]
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

        # the lyft box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # TODO(yurongyou): check if this is correct
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
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
        lyft_annos = {}
        # boxes_local = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_lyft_box(det)
            sample_token = self.data_infos[sample_id]['token']
            boxes = lidar_lyft_box_to_global(self.data_infos[sample_id], boxes)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                lyft_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    name=name,
                    score=box.score)
                annos.append(lyft_anno)
            lyft_annos[sample_token] = annos
        lyft_submissions = {
            'meta': self.modality,
            'results': lyft_annos,
            # 'boxes_local': boxes_local
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_lyft.json')
        print('Results writes to', res_path)
        mmcv.dump(lyft_submissions, res_path)
        return res_path

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox',
                         close_only=False,
                         eval_by_distance=False,
                         **kwargs):
        """Evaluation for a single model in Lyft protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str, optional): Metric name used for evaluation.
                Default: 'bbox'.
            result_name (str, optional): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """

        output_dir = osp.join(*osp.split(result_path)[:-1])
        lyft = Lyft(
            data_path=osp.join(self.dataset_root, self.version),
            json_path=osp.join(self.dataset_root, self.version, self.version),
            verbose=True)
        # eval_set_map = {
        #     'v1.01-train': 'test',
        #     'v1.01-train_debug': 'train_debug',
        # }
        valid_sample_tokens = [x['token'] for x in self.data_infos]
        metric_prefix = f'{result_name}_Lyft'

        if eval_by_distance:
            metrics, detail = lyft_eval_by_distance(lyft, valid_sample_tokens, result_path, output_dir, logger, metric_prefix, close_only=close_only, **kwargs)
        else:
            metrics = lyft_eval(lyft, valid_sample_tokens, result_path, output_dir, logger, **kwargs)

            # record metrics
            detail = dict()

            for i, name in enumerate(metrics['class_names']):
                AP = float(metrics['mAPs_cate'][i])
                detail[f'{metric_prefix}/{name}_AP'] = AP

            detail[f'{metric_prefix}/mAP'] = metrics['Final mAP']
        return detail

    def format_results(self, results, jsonfile_prefix=None, csv_savepath=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            csv_savepath (str): The path for saving csv files.
                It includes the file path and the csv filename,
                e.g., "a/b/filename.csv". If not specified,
                the result will not be converted to csv file.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on Lyft
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        if csv_savepath is not None:
            self.json2csv(result_files['pts_bbox'], csv_savepath)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 csv_savepath=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 eval_by_distance=False,
                 close_only=False,
                 **kwargs):
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
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix,
                                                    csv_savepath)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print(f'Evaluating bboxes of {name}')
                ret_dict = self._evaluate_single(result_files[name], eval_by_distance=eval_by_distance, close_only=close_only, **kwargs)
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files, eval_by_distance=eval_by_distance, close_only=close_only, **kwargs)

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
                load_dim=5,
                use_dim=5,
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

    def json2csv(self, json_path, csv_savepath):
        """Convert the json file to csv format for submission.

        Args:
            json_path (str): Path of the result json file.
            csv_savepath (str): Path to save the csv file.
        """
        results = mmcv.load(json_path)['results']
        sample_list_path = osp.join(self.dataset_root, 'sample_submission.csv')
        data = pd.read_csv(sample_list_path)
        Id_list = list(data['Id'])
        pred_list = list(data['PredictionString'])
        cnt = 0
        print('Converting the json to csv...')
        for token in results.keys():
            cnt += 1
            predictions = results[token]
            prediction_str = ''
            for i in range(len(predictions)):
                prediction_str += \
                    str(predictions[i]['score']) + ' ' + \
                    str(predictions[i]['translation'][0]) + ' ' + \
                    str(predictions[i]['translation'][1]) + ' ' + \
                    str(predictions[i]['translation'][2]) + ' ' + \
                    str(predictions[i]['size'][0]) + ' ' + \
                    str(predictions[i]['size'][1]) + ' ' + \
                    str(predictions[i]['size'][2]) + ' ' + \
                    str(Quaternion(list(predictions[i]['rotation']))
                        .yaw_pitch_roll[0]) + ' ' + \
                    predictions[i]['name'] + ' '
            prediction_str = prediction_str[:-1]
            idx = Id_list.index(token)
            pred_list[idx] = prediction_str
        df = pd.DataFrame({'Id': Id_list, 'PredictionString': pred_list})
        mmcv.mkdir_or_exist(os.path.dirname(csv_savepath))
        df.to_csv(csv_savepath, index=False)


def output_to_lyft_box(detection):
    """Convert the output to the box class in the Lyft.

    Args:
        detection (dict): Detection results.

    Returns:
        list[:obj:`LyftBox`]: List of standard LyftBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    # # our LiDAR coordinate system -> Lyft box coordinate system
    # lyft_box_dims = box_dims[:, [1, 0, 2]]
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        box = LyftBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i])
        box_list.append(box)
    return box_list


def lidar_lyft_box_to_global(info, boxes):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`LyftBox`]): List of predicted LyftBoxes.

    Returns:
        list: List of standard LyftBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate_around_origin(Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # Move box to global coord system
        box.rotate_around_origin(Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list
