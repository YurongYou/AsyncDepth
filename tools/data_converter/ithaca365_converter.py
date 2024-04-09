import os
from logging import warning
from os import path as osp
import pickle
from joblib import Parallel, delayed, parallel_backend
from functools import partial

import mmcv
import numpy as np
from ithaca365.ithaca365 import Ithaca365
from ithaca365.utils import splits

from pyquaternion import Quaternion
from .nuscenes_converter import (get_2d_boxes, get_available_scenes,
                                 obtain_sensor2top)

ithaca365_categories = ('car', 'truck', 'bus', 'pedestrian', 'motorcyclist',
                   'bicyclist')

NameMapping = {
    'bicyclist': 'bicyclist',
    'bus': 'bus',
    'car': 'car',
    'motorcyclist': 'motorcyclist',
    'pedestrian': 'pedestrian',
    'truck': 'truck',
}

def create_ithaca365_infos(root_path, info_prefix, version="v1.1", max_sweeps=10):
    """Create info file of ithaca365 dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.1-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """
    ith365 = Ithaca365(
        version=version,
        dataroot=root_path,
        verbose=True,
    )
    train_scenes = splits.train
    val_scenes = splits.val

    # filter existing scenes.
    available_scenes = get_available_scenes(ith365)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set(
        [
            available_scenes[available_scene_names.index(s)]["token"]
            for s in train_scenes
        ]
    )
    val_scenes = set(
        [available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes]
    )
    def get_sample_tokens(scene_tokens, ith365, only_accurate_localization=False):
        sample_token_dict={}
        scene_to_sample_token_list = {}
        for sc_tkn in scene_tokens:
            scene = ith365.get('scene', sc_tkn)
            token = scene['first_sample_token']
            scene_to_sample_token_list[scene['token']] = []
            while token != "":
                sample = ith365.get("sample", token)
                if (
                    ith365.get("sample_data", sample["data"]["LIDAR_TOP"])["bestpos"][
                        "field.pos_type.type"
                    ]
                    >= 56.0
                ):
                    sample_token_dict[token]={'scene_token': scene['token']}
                token = ith365.get('sample', token)['next']

        return sample_token_dict

    valid_samples_train  = get_sample_tokens(train_scenes, ith365, only_accurate_localization=True)
    valid_samples_test = get_sample_tokens(val_scenes, ith365, only_accurate_localization=True)

    test = "test" in version
    if test:
        print("test scene: {}".format(len(train_scenes)))
    else:
        print(
            "train scene: {}, val scene: {}".format(len(train_scenes), len(val_scenes))
        )

    metadata = dict(version=version)
    train_ith365_infos, val_ith365_infos = _fill_trainval_infos(
        ith365, valid_samples_train, valid_samples_test, False,
        train_scenes=train_scenes, max_sweeps=max_sweeps)
    data = dict(infos=train_ith365_infos, metadata=metadata)
    train_info_name = f'{info_prefix}_infos_train'
    info_path = osp.join(root_path, f'{train_info_name}_temp.pkl')
    mmcv.dump(data, info_path)
    data['infos'] = val_ith365_infos
    val_info_name = f'{info_prefix}_infos_val'
    info_val_path = osp.join(root_path, f'{val_info_name}_temp.pkl')
    mmcv.dump(data, info_val_path)


def _process_sample(sample_tkn, ith365, test, max_sweeps,
                    history_allow_train_only=False, train_scenes=set()):
    sample = ith365.get('sample', sample_tkn)
    lidar_token = sample['data']['LIDAR_TOP']
    sd_rec = ith365.get('sample_data', sample['data']['LIDAR_TOP'])
    cs_record = ith365.get('calibrated_sensor',
                            sd_rec['calibrated_sensor_token'])
    pose_record = ith365.get('ego_pose', sd_rec['ego_pose_token'])
    abs_lidar_path, boxes, _ = ith365.get_sample_data(lidar_token)
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

    # obtain 1 image's information per frame
    camera_types = ["cam0", "cam2"]
    for cam in camera_types:
        cam_token = sample['data'][cam]
        cam_path, _, cam_intrinsic = ith365.get_sample_data(cam_token)
        cam_info = obtain_sensor2top(ith365, cam_token, l2e_t, l2e_r_mat,
                                        e2g_t, e2g_r_mat, cam)
        cam_info.update(cam_intrinsic=cam_intrinsic)
        info['cams'].update({cam: cam_info})

    # obtain sweeps for a single key-frame
    sd_rec = ith365.get('sample_data', sample['data']['LIDAR_TOP'])
    sweeps = []
    while len(sweeps) < max_sweeps:
        if not sd_rec['prev'] == '':
            sweep = obtain_sensor2top(ith365, sd_rec['prev'], l2e_t,
                                        l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
            sweeps.append(sweep)
            sd_rec = ith365.get('sample_data', sd_rec['prev'])
        else:
            break
    info['sweeps'] = sweeps

    past_traversals = ith365.get_other_traversals_tokens(
        sample['data']['LIDAR_TOP'],
        num_history=20,
        ranges=(0, 70),
        every_x_meter=5.,
        sorted_by='hgt_stdev',
        increasing_order=True,
        accurate_history_only=True)

    if history_allow_train_only:
        # remove traversals that are not in train set
        tmp_past_traversals = {}
        for center_sd_tkn in past_traversals:
            center_sd = ith365.get('sample_data', center_sd_tkn)
            center_s = ith365.get('sample', center_sd['sample_token'])
            center_scene_tkn = center_s['scene_token']
            if center_scene_tkn in train_scenes:
                tmp_past_traversals[center_sd_tkn] = past_traversals[center_sd_tkn]
        past_traversals = tmp_past_traversals

    history_traversals = []
    for init_tkn, traversal_lidar_sample_data_tkn_list in past_traversals.items():
        traversal_info = {}
        traversal_sweeps = []
        for traversal_lidar_sample_data_tkn in traversal_lidar_sample_data_tkn_list:
            traversal_sample_info = obtain_sensor2top(
                ith365, traversal_lidar_sample_data_tkn, l2e_t, l2e_r_mat,
                e2g_t, e2g_r_mat, 'LIDAR_TOP')

            traversal_sweeps.append(traversal_sample_info)
        traversal_info.update({
            'LIDAR_TOP': traversal_sweeps
        })
        history_traversals.append(traversal_info)
    info['history_traversals'] = history_traversals

    # obtain annotation
    if not test:
        # print('annot')
        annotations = [
            ith365.get('sample_annotation', token)
            for token in sample['anns']
        ]

        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                            for b in boxes]).reshape(-1, 1)

        names = [b.name for b in boxes]
        for i in range(len(names)):
            if names[i] in NameMapping:
                names[i] = NameMapping[names[i]]
        names = np.array(names)

        # we need to convert box size to
        # the format of our lidar coordinate system
        # which is x_size, y_size, z_size (corresponding to l, w, h)
        # gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
        assert len(gt_boxes) == len(
            annotations), f'{len(gt_boxes)}, {len(annotations)}'
        info['gt_boxes'] = gt_boxes
        info['gt_names'] = names
        info['num_lidar_pts'] = np.array(
            [a['num_lidar_pts'] for a in annotations])
        info['num_radar_pts'] = np.array(
            [a['num_radar_pts'] for a in annotations])
    return info

def _fill_trainval_infos(ith365,
                         train_sample_info,
                         val_sample_info,
                         test=False,
                         max_sweeps=10,
                         train_scenes=set(),
                         parallel_n_jobs=16):
    """Generate the train/val infos from the raw data.

    Args:
        ith365 (:obj:`Ithaca365`): Dataset class in the Lyft dataset.
        train_sample_info (dict): Basic information of training samples.
        val_sample_info (dict): Basic information of validation samples.
        test (bool, optional): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and
            validation set that will be saved to the info file.
    """
    train_ith365_infos = []
    val_ith365_infos = []

    train_samples = list(train_sample_info.keys())
    val_samples = list(val_sample_info.keys())

    train_ith365_infos = mmcv.track_parallel_progress(
        partial(_process_sample, ith365=ith365,
                max_sweeps=max_sweeps, test=test, history_allow_train_only=True, train_scenes=train_scenes), train_samples, nproc=parallel_n_jobs,
        chunksize=500)
    val_ith365_infos = mmcv.track_parallel_progress(
        partial(_process_sample, ith365=ith365,
                max_sweeps=max_sweeps, test=test), val_samples, nproc=parallel_n_jobs,
        chunksize=500)

    return train_ith365_infos, val_ith365_infos

def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
    """
    # get bbox annotations for camera
    camera_types = [
        'cam0', 'cam2']
    ith365_infos = mmcv.load(info_path)['infos']
    ith365 = Ithaca365(
        version="v1.1",
        dataroot="/share/campbell/Skynet/nuScene_format/v1.1",
        verbose=True,
    )
    cat2Ids = [
        dict(id=ithaca365_categories.index(cat_name), name=cat_name)
        for cat_name in ithaca365_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmcv.track_iter_progress(ith365_infos):
        for cam in camera_types:
            cam_info = info['cams'][cam]
            coco_infos = get_2d_boxes(
                ith365,
                cam_info['sample_data_token'],
                visibilities=['', '1', '2', '3', '4'],
                mono3d=mono3d,
                dataset_type='ithaca365')
            # print(osp.join(root_path, cam_info['data_path']))
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
