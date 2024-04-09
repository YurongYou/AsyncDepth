"""Generate the lyft dataset splits.

Reference:
https://github.com/YurongYou/Hindsight/blob/master/data_preprocessing/lyft/split_traintest.py
"""
import argparse
import os.path as osp
import pickle

import numpy as np
from lyft_dataset_sdk.lyftdataset import LyftDataset as Lyft
from pyquaternion import Quaternion
from tqdm.auto import tqdm

def form_trans_mat(translation, rotation):
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = Quaternion(rotation).rotation_matrix
    mat[:3, 3] = translation
    return mat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root-folder',
        type=str,
        default='./data/lyft',
        help='specify the root path of Lyft dataset')
    parser.add_argument(
        '--version',
        type=str,
        default='v1.01-train',
        help='specify Lyft dataset version')
    parser.add_argument(
        '--prefix',
        type=str,
        default='',
        help='prefix of the output file')
    parser.add_argument(
        '--every_x_meters',
        type=float,
        default=5.0,
        help='gather history traversal points every x meters'
    )
    parser.add_argument(
        '--max_distance',
        type=float,
        default=40.0,
        help='gather history traversal points up to x meters'
    )
    parser.add_argument(
        '--exclude_beta_plus_plus',
        action='store_true',
        help='whether to exclude the beta++ data. See https://github.com/lyft/nuscenes-devkit/issues/60'
    )
    parser.add_argument(
        '--cutoff',
        type=float,
        default=1700.0,
        help='cutoff for train and val'
    )
    parser.add_argument(
        '--upper_part_train',
        action='store_true',
    )
    return parser.parse_args()


def main(args):
    lyft = Lyft(
        data_path=osp.join(args.root_folder, args.version),
        json_path=osp.join(args.root_folder, args.version, args.version),
        verbose=True)

    # obtaining the sequences
    scene_to_sample_token_list = {}
    for scene in lyft.scene:
        token = scene['first_sample_token']
        if args.exclude_beta_plus_plus:
            image_info = lyft.get('sample_data',
                     lyft.get('sample', token)['data']['CAM_FRONT'])
            if image_info['width'] == 1920:
                continue
        scene_to_sample_token_list[scene['token']] = []
        while token != "":
            scene_to_sample_token_list[scene['token']].append(token)
            token = lyft.get('sample', token)['next']

    # obtaining the geo location
    sample_token_to_geo = {}
    for scene_token, sample_tokens in scene_to_sample_token_list.items():
        for sample_token in sample_tokens:
            sample = lyft.get("sample", sample_token)
            lidar_token = sample["data"]['LIDAR_TOP']
            sd_record_lid = lyft.get("sample_data", lidar_token)
            ego_record_lid = lyft.get(
                "ego_pose", sd_record_lid["ego_pose_token"])
            sample_token_to_geo[sample_token] = form_trans_mat(
                ego_record_lid['translation'], ego_record_lid["rotation"])

    # cutoff = 1700  # a simple cut off by the geo location
    train_scenes = []
    test_scenes = []
    for scene_token, sample_tokens in scene_to_sample_token_list.items():
        seq_pose = []
        for sample_token in sample_tokens:
            seq_pose.append(sample_token_to_geo[sample_token][:3, 3])
        seq_pose = np.array(seq_pose)
        if args.upper_part_train:
            if (seq_pose[:, 1] < args.cutoff).sum() == len(seq_pose):
                test_scenes.append(scene_token)
            if (seq_pose[:, 1] >= args.cutoff).sum() == len(seq_pose):
                train_scenes.append(scene_token)
        else:
            if (seq_pose[:, 1] >= args.cutoff).sum() == len(seq_pose):
                test_scenes.append(scene_token)
            if (seq_pose[:, 1] < args.cutoff).sum() == len(seq_pose):
                train_scenes.append(scene_token)
    print(f"initial #train_scenes: {len(train_scenes)}")
    print(f"initial #test_scenes: {len(test_scenes)}")
    dis_choice = np.arange(args.every_x_meters, args.max_distance + 0.1, args.every_x_meters)
    max_allow_dist = 3.

    for subset_name, subset_scenes in zip(['train', 'test'], [train_scenes, test_scenes]):
        print(f"processing {subset_name} set")
        loc_cache = {}
        for scene_token in subset_scenes:
            loc_cache[scene_token] = np.array(
                [sample_token_to_geo[sample_token][:2, 3] for sample_token in scene_to_sample_token_list[scene_token]])

        valid_samples = {}
        for scene_token in tqdm(subset_scenes):
            sample_tokens = scene_to_sample_token_list[scene_token]
            for frame_idx, sample_token in enumerate(sample_tokens):
                origin_pose = sample_token_to_geo[sample_token]
                valid_scenes = []
                for _scene_token in subset_scenes:
                    _sample_tokens = scene_to_sample_token_list[_scene_token]
                    if _scene_token == scene_token:
                        continue
                    distance = np.linalg.norm(loc_cache[_scene_token] - origin_pose[:2, 3], axis=1)
                    min_dist_indices = np.argmin(distance)
                    min_dist = distance[min_dist_indices]
                    if min_dist > max_allow_dist:
                        continue
                    # pick samples, bw and fw
                    indices = [min_dist_indices]
                    for dis in dis_choice:
                        temp = np.where(distance > dis)[0]
                        if len(temp[temp < min_dist_indices]) == 0:
                            break
                        indices.append(temp[temp < min_dist_indices].max())
                        if len(temp[temp > min_dist_indices]) == 0:
                            break
                        indices.append(temp[temp > min_dist_indices].min())
                    if len(indices) < 2 * len(dis_choice) + 1:
                        continue
                    valid_scenes.append({'scene_token': _scene_token, 'sample_token_list': [
                                        _sample_tokens[idx] for idx in indices]})
                if len(valid_scenes) > 1:
                    valid_samples[sample_token] = (
                        scene_token, valid_scenes)
        print(f"valid {subset_name} samples: {len(valid_samples)}")
        print(f"saving to {args.root_folder}/{args.prefix}{subset_name}_valid_samples.pkl")
        pickle.dump(valid_samples,
                    open(f'{args.root_folder}/{args.prefix}{subset_name}_valid_samples.pkl', 'wb'))


if __name__ == "__main__":
    main(parse_args())
