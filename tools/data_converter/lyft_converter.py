# Copyright (c) OpenMMLab. All rights reserved.
import os
import pickle
from functools import partial
from logging import warning
from os import path as osp

import mmcv
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from lyft_dataset_sdk.lyftdataset import LyftDataset as Lyft
from pyquaternion import Quaternion

from mmdet3d.datasets import LyftDataset

from .nuscenes_converter import (get_2d_boxes,
                                 obtain_sensor2top)

lyft_categories = ('car', 'truck', 'bus', 'emergency_vehicle', 'other_vehicle',
                   'motorcycle', 'bicycle', 'pedestrian', 'animal')


def create_lyft_infos(root_path,
                      info_prefix,
                      version='v1.01-train',
                      sample_info_prefix='',
                      max_sweeps=10):
    """Create info file of lyft dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.01-train'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """
    lyft = Lyft(
        data_path=osp.join(root_path, version),
        json_path=osp.join(root_path, version, version),
        verbose=True)
    assert version == 'v1.01-train'

    valid_samples_train = pickle.load(
        open(f"{root_path}/{sample_info_prefix}train_valid_samples.pkl", "rb"))
    valid_samples_test = pickle.load(
        open(f"{root_path}/{sample_info_prefix}test_valid_samples.pkl", "rb"))
    train_lyft_infos, val_lyft_infos = _fill_trainval_infos(
        lyft, valid_samples_train, valid_samples_test, False, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    # if test:
    #     print(f'test sample: {len(train_lyft_infos)}')
    #     data = dict(infos=train_lyft_infos, metadata=metadata)
    #     info_name = f'{info_prefix}_infos_test'
    #     info_path = osp.join(root_path, f'{info_name}.pkl')
    #     mmcv.dump(data, info_path)
    # else:
    print(f'train sample: {len(train_lyft_infos)}, \
            val sample: {len(val_lyft_infos)}')
    data = dict(infos=train_lyft_infos, metadata=metadata)
    train_info_name = f'{info_prefix}_infos_train'
    info_path = osp.join(root_path, f'{train_info_name}.pkl')
    mmcv.dump(data, info_path)
    data['infos'] = val_lyft_infos
    val_info_name = f'{info_prefix}_infos_val'
    info_val_path = osp.join(root_path, f'{val_info_name}.pkl')
    mmcv.dump(data, info_val_path)

def _process_sample(sample_tkn, lyft, traversal_infos, max_sweeps, test):
    sample = lyft.get('sample', sample_tkn)
    lidar_token = sample['data']['LIDAR_TOP']
    sd_rec = lyft.get('sample_data', sample['data']['LIDAR_TOP'])
    cs_record = lyft.get('calibrated_sensor',
                            sd_rec['calibrated_sensor_token'])
    pose_record = lyft.get('ego_pose', sd_rec['ego_pose_token'])
    abs_lidar_path, boxes, _ = lyft.get_sample_data(lidar_token)
    # nuScenes devkit returns more convenient relative paths while
    # lyft devkit returns absolute paths
    abs_lidar_path = str(abs_lidar_path)  # absolute path
    lidar_path = abs_lidar_path.split(f'{os.getcwd()}/')[-1]
    # relative path

    mmcv.check_file_exist(lidar_path)

    info = {
        'lidar_path': lidar_path,
        'token': sample['token'],
        'sweeps': [],
        'cams': dict(),
        'lidar2ego_translation': cs_record['translation'],
        'lidar2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sample['timestamp'],
    }

    l2e_r = info['lidar2ego_rotation']
    l2e_t = info['lidar2ego_translation']
    e2g_r = info['ego2global_rotation']
    e2g_t = info['ego2global_translation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    # obtain 6 image's information per frame
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    for cam in camera_types:
        cam_token = sample['data'][cam]
        cam_path, _, cam_intrinsic = lyft.get_sample_data(cam_token)
        cam_info = obtain_sensor2top(lyft, cam_token, l2e_t, l2e_r_mat,
                                        e2g_t, e2g_r_mat, cam)
        cam_info.update(cam_intrinsic=cam_intrinsic)
        info['cams'].update({cam: cam_info})

    # obtain sweeps for a single key-frame
    sd_rec = lyft.get('sample_data', sample['data']['LIDAR_TOP'])
    sweeps = []
    while len(sweeps) < max_sweeps:
        if not sd_rec['prev'] == '':
            sweep = obtain_sensor2top(lyft, sd_rec['prev'], l2e_t,
                                        l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
            sweeps.append(sweep)
            sd_rec = lyft.get('sample_data', sd_rec['prev'])
        else:
            break
    info['sweeps'] = sweeps

    # obtain history traversals
    history_traversals = []
    for traversal in traversal_infos[sample_tkn][1]:
        traversal_info = {}
        traversal_info['scene_tkn'] = traversal['scene_token']
        traversal_sample_tkn_list = traversal['sample_token_list']
        for sensor in camera_types + ['LIDAR_TOP']:
            traversal_sweeps = []
            for traversal_sample_tkn in traversal_sample_tkn_list:
                traversal_sample = lyft.get('sample', traversal_sample_tkn)
                traversal_sample_data_tkn = traversal_sample['data'][sensor]
                traversal_sample_info = obtain_sensor2top(
                    lyft, traversal_sample_data_tkn, l2e_t, l2e_r_mat,
                    e2g_t, e2g_r_mat, sensor)
                if sensor.startswith('CAM'):
                    _, _, sensor_intrinsic = lyft.get_sample_data(traversal_sample_data_tkn)
                    traversal_sample_info.update(cam_intrinsic=sensor_intrinsic)
                traversal_sweeps.append(traversal_sample_info)
            traversal_info.update({sensor: traversal_sweeps})
        history_traversals.append(traversal_info)
    info['history_traversals'] = history_traversals

    # obtain annotation
    if not test:
        annotations = [
            lyft.get('sample_annotation', token)
            for token in sample['anns']
        ]
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        # TODO(yurongyou): check the correctness of the rotation
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                            for b in boxes]).reshape(-1, 1)

        names = [b.name for b in boxes]
        for i in range(len(names)):
            if names[i] in LyftDataset.NameMapping:
                names[i] = LyftDataset.NameMapping[names[i]]
        names = np.array(names)

        # # we need to convert box size to
        # # the format of our lidar coordinate system
        # # which is x_size, y_size, z_size (corresponding to l, w, h)
        # gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
        # we need to convert rot to SECOND format.
        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
        assert len(gt_boxes) == len(
            annotations), f'{len(gt_boxes)}, {len(annotations)}'
        info['gt_boxes'] = gt_boxes
        info['gt_names'] = names
        info['num_lidar_pts'] = np.array(
            [a['num_lidar_pts'] for a in annotations])
        info['num_radar_pts'] = np.array(
            [a['num_radar_pts'] for a in annotations])
        info["valid_flag"] = np.array(
            [
                (anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0
                for anno in annotations
            ],
            dtype=bool,
        ).reshape(-1)
    return info

def _fill_trainval_infos(lyft,
                         train_sample_info,
                         val_sample_info,
                         test=False,
                         max_sweeps=10,
                         parallel_n_jobs=16):
    """Generate the train/val infos from the raw data.

    Args:
        lyft (:obj:`LyftDataset`): Dataset class in the Lyft dataset.
        train_sample_info (dict): Basic information of training samples.
        val_sample_info (dict): Basic information of validation samples.
        test (bool, optional): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and
            validation set that will be saved to the info file.
    """
    train_lyft_infos = []
    val_lyft_infos = []

    train_samples = list(train_sample_info.keys())
    val_samples = list(val_sample_info.keys())
    # full_sample_infos = {**train_sample_info, **val_sample_info}


    train_lyft_infos = mmcv.track_parallel_progress(
        partial(_process_sample, lyft=lyft, traversal_infos=train_sample_info,
                max_sweeps=max_sweeps, test=test), train_samples, nproc=parallel_n_jobs,
        chunksize=100)
    val_lyft_infos = mmcv.track_parallel_progress(
        partial(_process_sample, lyft=lyft, traversal_infos=val_sample_info,
                max_sweeps=max_sweeps, test=test), val_samples, nproc=parallel_n_jobs,
        chunksize=100)
    # with parallel_backend("loky", n_jobs=parallel_n_jobs):
    #     train_lyft_infos = Parallel()(delayed(process_sample)(sample_token)
    #         for sample_token in mmcv.track_iter_progress(train_samples))
    # with parallel_backend("loky", n_jobs=parallel_n_jobs):
    #     val_lyft_infos = Parallel()(delayed(process_sample)(sample_token)
    #         for sample_token in mmcv.track_iter_progress(val_samples))

    return train_lyft_infos, val_lyft_infos


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
    """
    # warning.warn('DeprecationWarning: 2D annotations are not used on the '
    #              'Lyft dataset. The function export_2d_annotation will be '
    #              'deprecated.')
    # get bbox annotations for camera
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    lyft_infos = mmcv.load(info_path)['infos']
    lyft = Lyft(
        data_path=osp.join(root_path, version),
        json_path=osp.join(root_path, version, version),
        verbose=True)
    # info_2d_list = []
    cat2Ids = [
        dict(id=lyft_categories.index(cat_name), name=cat_name)
        for cat_name in lyft_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmcv.track_iter_progress(lyft_infos):
        for cam in camera_types:
            cam_info = info['cams'][cam]
            coco_infos = get_2d_boxes(
                lyft,
                cam_info['sample_data_token'],
                visibilities=['', '1', '2', '3', '4'],
                mono3d=mono3d,
                dataset_type='lyft')
            (height, width, _) = mmcv.imread(
                osp.join(root_path, cam_info['data_path'])).shape
            coco_2d_dict['images'].append(
                dict(
                    file_name=cam_info['data_path'],
                    id=cam_info['sample_data_token'],
                    token=info['token'],
                    cam2ego_rotation=cam_info["sensor2ego_rotation"],
                    cam2ego_translation=cam_info["sensor2ego_translation"],
                    ego2global_rotation=info["ego2global_rotation"],
                    ego2global_translation=info["ego2global_translation"],
                    camera_intrinsics=cam_info["cam_intrinsic"],
                    sensor2lidar_rotation=cam_info["sensor2lidar_rotation"],
                    sensor2lidar_translation=cam_info["sensor2lidar_translation"],
                    width=width,
                    height=height))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f"{info_path[:-4]}_mono3d"
    else:
        json_prefix = f"{info_path[:-4]}"
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')
