import argparse
import pickle
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--info_pattern", type=str, default="data/ithaca365/ithaca365-new-past_infos_*")
    parser.add_argument("--pattern_to_replace", type=str, default="/share/campbell/Skynet/nuScene_format/")
    return parser.parse_args()

def inplace_clean_up_infos(infos, pattern_to_replace):
    # clean_up_info
    for info in infos['infos']:
        info['lidar_path'] = info['lidar_path'].replace(pattern_to_replace, "")
        for cam, cam_info in info['cams'].items():
            cam_info['data_path'] = cam_info['data_path'].replace(pattern_to_replace, "")
        for sweep in info['sweeps']:
            sweep['data_path'] = sweep['data_path'].replace(pattern_to_replace, "")
        for traversal in info['history_traversals']:
            for sensor, sensor_data in traversal.items():
                if sensor != 'scene_tkn':
                    for data_info in sensor_data:
                        data_info['data_path'] = data_info['data_path'].replace(pattern_to_replace, "")
    return infos

def main(args):
    info_files = glob.glob(args.info_pattern)
    for info_file in info_files:
        print(">> Found", info_file)
    for info_file in info_files:
        print(">> Loading", info_file)
        infos = pickle.load(open(info_file, "rb"))
        print(">> Finish loading", info_file)
        infos = inplace_clean_up_infos(infos, args.pattern_to_replace)
        print(">> Finish cleaning up", info_file)
        new_file_name = info_file.replace("infos", "relative-path-infos")
        print(">> Saving to", new_file_name)
        pickle.dump(infos, open(new_file_name, "wb"))

if __name__ == "__main__":
    main(parse_args())