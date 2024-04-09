import os
import os.path as osp
import pickle
import sys
import yaml
import numpy as np
import argparse
from tqdm.auto import tqdm

from scipy.spatial import cKDTree


def count_neighbors(ptc, trees, max_neighbor_dist=0.3):
    neighbor_count = {}
    for seq in trees.keys():
        neighbor_count[seq] = trees[seq].query_ball_point(
            ptc[:, :3], r=max_neighbor_dist,
            return_length=True)
    return np.stack(list(neighbor_count.values())).T


def compute_ephe_score(count):
    N = count.shape[1]
    P = count / (np.expand_dims(count.sum(axis=1), -1) + 1e-8)
    H = (-P * np.log(P + 1e-8)).sum(axis=1) / np.log(N)
    return H

def cart2hom(pts_3d):
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1), dtype=np.float32)))
    return pts_3d_hom

def shuffle(points, nums=200000):
    return points[np.random.choice(points.shape[0], nums)]

def transform_points(pts_3d_ref, Tr):
    pts_3d_ref = cart2hom(pts_3d_ref)  # nx4
    return np.dot(pts_3d_ref, np.transpose(Tr)).reshape(-1, 4)[:, 0:3]

def load_lidar(path,dim=5):
    return np.fromfile(path, np.float32).reshape(-1, dim)[:, :3]

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--info_path', type=str,
        default='../data/lyft_nfs/beta_v0_dist_20_cutoff_1000__relative-path-infos_train.pkl')
    parser.add_argument('--abs_data_path', type=str, default='../data/lyft_nfs')
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--max_neighbor_dist', type=float, default=0.3)
    parser.add_argument('--lidar_dims', type=int, default=5)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def display_args(args):
    eprint("========== ephemerality info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(yaml.dump(vars(args), default_flow_style=False))
    eprint("=======================================")

def main(args):
    display_args(args)
    os.makedirs(args.save_path, exist_ok=True)
    infos = pickle.load(open(args.info_path, "rb"))['infos']
    idx_list = np.array(list(range(len(infos))))
    if args.world_size > 1:
        idx_list = np.array_split(
            idx_list, args.world_size)[args.local_rank]

    for idx in tqdm(idx_list):
        info = infos[idx]
        curr_lidar = load_lidar(osp.join(args.abs_data_path, info['lidar_path']),
                                dim=args.lidar_dims)

        combined_lidar = {}
        historical_traversal = info['history_traversals']
        for seq_id, seq in enumerate(historical_traversal):
            hist_lidar = []
            for _, item in enumerate(seq['LIDAR_TOP']):
                _l = load_lidar(osp.join(args.abs_data_path, item['data_path']),
                                dim=args.lidar_dims)
                sensor2lidar_rotation = item['sensor2lidar_rotation']
                sensor2lidar_translation = item['sensor2lidar_translation']
                trans = np.eye(4, dtype=np.float32)
                trans[:3, :3] = sensor2lidar_rotation
                trans[:3, 3] = sensor2lidar_translation
                _l = transform_points(_l, trans)
                hist_lidar.append(_l)
            combined_lidar[seq_id] = np.concatenate(hist_lidar,axis=0)

        trees = {}
        for seq, ptc in combined_lidar.items():
            trees[seq] = cKDTree(ptc)
        count = count_neighbors(curr_lidar, trees)
        H = compute_ephe_score(count)
        np.save(osp.join(args.save_path, f"{idx:06d}_{info['token']}"),
                H.astype(np.float32))

if __name__ == "__main__":
    main(parse_args())