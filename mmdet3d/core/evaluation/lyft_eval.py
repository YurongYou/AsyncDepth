# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp
from typing import Tuple, Dict, List

import mmcv
import numpy as np
import copy
from lyft_dataset_sdk.eval.detection.mAP_evaluation import (Box3D, get_ap,
                                                            get_class_names,
                                                            get_ious,
                                                            group_by_key,
                                                            wrap_in_box)
from mmcv.utils import print_log
from terminaltables import AsciiTable

from ithaca365.eval.common.data_classes import EvalBoxes
from ithaca365.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox

import pandas as pd


class LyftDetectionBox(DetectionBox):
    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 ego_translation: Tuple[float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
                 num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
                 detection_name: str = 'car',  # The class name used in the detection challenge.
                 detection_score: float = -1.0,  # GT samples do not have a score.
                 attribute_name: str = ''):  # Box attribute. Each box can have at most 1 attribute.

        super().__init__(sample_token, translation, size, rotation, velocity, ego_translation, num_pts)

        assert type(detection_score) == float, 'Error: detection_score must be a float!'
        assert not np.any(np.isnan(detection_score)), 'Error: detection_score may not be NaN!'

        # Assign.
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                   else tuple(content['ego_translation']),
                   num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                   detection_name=content['name'] if not 'detection_name' in content else content['detection_name'],
                   detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']))


class LyftDetectionConfig(DetectionConfig):
    """ Data class that specifies the detection evaluation settings. """

    def __init__(self,
                 class_range: Dict[str, List[List[int]]],
                 dist_fcn: str,
                 dist_ths: List[float],
                 dist_th_tp: float,
                 min_recall: float,
                 min_precision: float,
                 max_boxes_per_sample: int,
                 mean_ap_weight: int,):
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight
        self.only_accurate_localization = False

        self.class_names = self.class_range.keys()


def load_lyft_gts(lyft, valid_sample_tokens, logger=None, load_as_eval_boxes=True, name_mapping=None):
    """Loads ground truth boxes from database.

    Args:
        lyft (:obj:`LyftDataset`): Lyft class in the sdk.
        valid_sample_tokens (List[str]): list of valid sample tokens.
        logger (logging.Logger | str, optional): Logger used for printing
        related information during evaluation. Default: None.

    Returns:
        list[dict]: List of annotation dictionaries.
    """
    # split_scenes = mmcv.list_from_file(
    #     osp.join(data_root, f'{eval_split}.txt'))
    # valid_sample_info = pickle.load(open(valid_sample_file, "rb"))
    # sample_tokens = list(valid_sample_info.keys())

    # # Read out all sample_tokens in DB.
    # sample_tokens_all = [s['token'] for s in lyft.sample]
    # assert len(sample_tokens_all) > 0, 'Error: Database has no samples!'

    # if eval_split == 'test':
    #     # Check that you aren't trying to cheat :)
    #     assert len(lyft.sample_annotation) > 0, \
    #         'Error: You are trying to evaluate on the test set \
    #          but you do not have the annotations!'

    all_annotations = EvalBoxes() if load_as_eval_boxes else []

    print_log('Loading ground truth annotations...', logger=logger)
    # Load annotations and filter predictions and annotations.
    for sample_token in mmcv.track_iter_progress(valid_sample_tokens):
        sample = lyft.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        sample_boxes = []
        for sample_annotation_token in sample_annotation_tokens:
            # Get label name in detection task and filter unused labels.
            sample_annotation = \
                lyft.get('sample_annotation', sample_annotation_token)
            detection_name = sample_annotation['category_name']
            if detection_name is None:
                continue
            if name_mapping is not None and detection_name in name_mapping:
                detection_name = name_mapping[detection_name]
            if load_as_eval_boxes:
                sample_boxes.append(
                    LyftDetectionBox(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=lyft.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        detection_name=detection_name,
                        detection_score=-1.0,  # GT samples do not have a score.
                    )
                )
            else:
                annotation = {
                    'sample_token': sample_token,
                    'translation': sample_annotation['translation'],
                    'size': sample_annotation['size'],
                    'rotation': sample_annotation['rotation'],
                    'name': detection_name,
                }
                all_annotations.append(annotation)
        if load_as_eval_boxes:
            all_annotations.add_boxes(sample_token, sample_boxes)

    return all_annotations


def load_lyft_predictions(res_path, load_as_eval_boxes=True, name_mapping=None):
    """Load Lyft predictions from json file.

    Args:
        res_path (str): Path of result json file recording detections.

    Returns:
        list[dict]: List of prediction dictionaries.
    """
    predictions = mmcv.load(res_path)
    predictions = predictions['results']
    all_preds = EvalBoxes() if load_as_eval_boxes else []
    for sample_token in mmcv.track_iter_progress(predictions.keys()):
        if load_as_eval_boxes:
            sample_boxes = []
            for pred in predictions[sample_token]:
                detection_name = pred['name']
                if name_mapping is not None and detection_name in name_mapping:
                    detection_name = name_mapping[detection_name]
                sample_boxes.append(
                    LyftDetectionBox(
                        sample_token=sample_token,
                        translation=pred['translation'],
                        size=pred['size'],
                        rotation=pred['rotation'],
                        detection_name=detection_name,
                        detection_score=pred['score'],  # GT samples do not have a score.
                    )
                )
            all_preds.add_boxes(sample_token, sample_boxes)
        else:
            all_preds.extend(predictions[sample_token])
    return all_preds


def lyft_eval(lyft, valid_sample_tokens, res_path, output_dir, logger=None):
    """Evaluation API for Lyft dataset.

    Args:
        lyft (:obj:`LyftDataset`): Lyft class in the sdk.
        valid_sample_tokens (List[str]): list of valid sample tokens.
        res_path (str): Path of result json file recording detections.
        output_dir (str): Output directory for output json files.
        logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.

    Returns:
        dict[str, float]: The evaluation results.
    """
    # evaluate by lyft metrics
    gts = load_lyft_gts(lyft, valid_sample_tokens, logger, load_as_eval_boxes=False)
    predictions = load_lyft_predictions(res_path, load_as_eval_boxes=False)

    class_names = get_class_names(gts)
    print('Calculating mAP@0.5:0.95...')

    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    metrics = {}
    average_precisions = \
        get_classwise_aps(gts, predictions, class_names, iou_thresholds)
    APs_data = [['IOU', 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]]

    mAPs = np.mean(average_precisions, axis=0)
    mAPs_cate = np.mean(average_precisions, axis=1)
    final_mAP = np.mean(mAPs)

    metrics['average_precisions'] = average_precisions.tolist()
    metrics['mAPs'] = mAPs.tolist()
    metrics['Final mAP'] = float(final_mAP)
    metrics['class_names'] = class_names
    metrics['mAPs_cate'] = mAPs_cate.tolist()

    APs_data = [['class', 'mAP@0.5:0.95']]
    for i in range(len(class_names)):
        row = [class_names[i], round(mAPs_cate[i], 3)]
        APs_data.append(row)
    APs_data.append(['Overall', round(final_mAP, 3)])
    APs_table = AsciiTable(APs_data, title='mAPs@0.5:0.95')
    APs_table.inner_footing_row_border = True
    print_log(APs_table.table, logger=logger)
    print_log(f'AP\IOU:', logger=logger)
    print_log(pd.DataFrame(metrics['average_precisions'], index=metrics['class_names'], columns=iou_thresholds), logger=logger)

    res_path = osp.join(output_dir, 'lyft_metrics.json')
    mmcv.dump(metrics, res_path)
    return metrics


def filter_eval_boxes(
    nusc,
    eval_boxes: EvalBoxes,
    max_dist: Dict[str, List[List[int]]],
    verbose: bool = False,
) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param max_dist: Maps the detection name to the eval distance threshold for that class.
    :param verbose: Whether to print to stdout.
    """

    # ASSUME ALL DETECTION RANGES ARE THE SAME ACROSS CLASS
    # TODO: change to be more elegant
    det_ranges = list(max_dist.values())[0]
    filtered_eval_box_by_range = {
        tuple(det_range): copy.deepcopy(eval_boxes) for det_range in det_ranges
    }

    for det_range, eval_boxes in filtered_eval_box_by_range.items():
        # Accumulators for number of filtered boxes.
        total, dist_filter, point_filter, fov_filter = 0, 0, 0, 0
        for ind, sample_token in enumerate(eval_boxes.sample_tokens):
            # Filter on distance first.
            total += len(eval_boxes[sample_token])
            eval_boxes.boxes[sample_token] = [
                box
                for box in eval_boxes[sample_token]
                if det_range[0] <= box.ego_dist < det_range[1]
            ]
            dist_filter += len(eval_boxes[sample_token])

            # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
            eval_boxes.boxes[sample_token] = [
                box for box in eval_boxes[sample_token] if not box.num_pts == 0
            ]
            point_filter += len(eval_boxes[sample_token])

        if verbose:
            print("Detection range:", det_range)
            print("=> Original number of boxes: %d" % total)
            print("=> After distance based filtering: %d" % dist_filter)
            print("=> After LIDAR and RADAR points based filtering: %d" % point_filter)
            print("=> After fov based filtering: %d" % fov_filter)

    return filtered_eval_box_by_range


def lyft_eval_by_distance(lyft, valid_sample_tokens, res_path, output_dir,
                          logger=None, metric_prefix='lyft', close_only=False, merge_class=False):
    from ithaca365.eval.detection.algo import accumulate, calc_ap, calc_tp
    from ithaca365.eval.detection.data_classes import DetectionMetrics, DetectionRangeMetricDataList
    from ithaca365.eval.common.loaders import add_center_dist
    from ithaca365.eval.detection.constants import TP_METRICS

    name_mapping = None
    if close_only:
        if merge_class:
            eval_config = LyftDetectionConfig(
                class_range = {
                    "vehicle": [[0, 30], [30, 50], [0, 50]],
                    "human": [[0, 30], [30, 50], [0, 50]],
                },
                dist_fcn="center_distance",
                dist_ths=[0.5, 1.0, 2.0, 4.0],
                dist_th_tp=2.0,
                min_recall=0.1,
                min_precision=0.1,
                max_boxes_per_sample=500,
                mean_ap_weight=3,
            )
            name_mapping = {
                'car': 'vehicle',
                'truck': 'vehicle',
                'bus': 'vehicle',
                'bicycle': 'human',
                'pedestrian': 'human',
            }
        else:
            eval_config = LyftDetectionConfig(
                class_range = {
                    "car": [[0, 30], [30, 50], [0, 50]],
                    "truck": [[0, 30], [30, 50], [0, 50]],
                    "bus": [[0, 30], [30, 50], [0, 50]],
                    "bicycle": [[0, 30], [30, 50], [0, 50]],
                    "pedestrian": [[0, 30], [30, 50], [0, 50]],
                },
                dist_fcn="center_distance",
                dist_ths=[0.5, 1.0, 2.0, 4.0],
                dist_th_tp=2.0,
                min_recall=0.1,
                min_precision=0.1,
                max_boxes_per_sample=500,
                mean_ap_weight=3,
            )
    else:
        eval_config = LyftDetectionConfig(
            class_range = {
                "car": [[0, 30], [30, 50], [50, 80], [0, 80]],
                "truck": [[0, 30], [30, 50], [50, 80], [0, 80]],
                "bus": [[0, 30], [30, 50], [50, 80], [0, 80]],
                # "emergency_vehicle": [[0, 30], [30, 50], [50, 80], [0, 80]],
                # "other_vehicle": [[0, 30], [30, 50], [50, 80], [0, 80]],
                # "motorcycle": [[0, 30], [30, 50], [50, 80], [0, 80]],
                "bicycle": [[0, 30], [30, 50], [50, 80], [0, 80]],
                "pedestrian": [[0, 30], [30, 50], [50, 80], [0, 80]],
                # "animal": [[0, 30], [30, 50], [50, 80], [0, 80]]
            },
            dist_fcn="center_distance",
            dist_ths=[0.5, 1.0, 2.0, 4.0],
            dist_th_tp=2.0,
            min_recall=0.1,
            min_precision=0.1,
            max_boxes_per_sample=500,
            mean_ap_weight=3,
        )

    gt_boxes = load_lyft_gts(lyft, valid_sample_tokens, logger, load_as_eval_boxes=True, name_mapping=name_mapping)
    pred_boxes = load_lyft_predictions(res_path, load_as_eval_boxes=True, name_mapping=name_mapping)
    assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

    pred_boxes = add_center_dist(lyft, pred_boxes)
    gt_boxes = add_center_dist(lyft, gt_boxes)
    pred_boxes = filter_eval_boxes(lyft, pred_boxes, eval_config.class_range, verbose=True)
    gt_boxes = filter_eval_boxes(lyft, gt_boxes, eval_config.class_range, verbose=True)

    metric_data_list = DetectionRangeMetricDataList()
    for class_name in eval_config.class_names:
        for det_range in eval_config.class_range[class_name]:
            det_range = tuple(det_range)
            for dist_th in eval_config.dist_ths:
                md = accumulate(gt_boxes[det_range], pred_boxes[det_range], class_name, eval_config.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, det_range, dist_th, md)

    metrics = DetectionMetrics(eval_config)
    for class_name in eval_config.class_names:
        for det_range in eval_config.class_range[class_name]:
            det_range = tuple(det_range)
            # Compute APs.
            for dist_th in eval_config.dist_ths:
                metric_data = metric_data_list[(class_name, det_range, dist_th)]
                ap = calc_ap(metric_data, eval_config.min_recall, eval_config.min_precision)
                metrics.add_label_ap(class_name, det_range, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, det_range, eval_config.dist_th_tp)]
                tp = calc_tp(metric_data, eval_config.min_recall, metric_name)
                metrics.add_label_tp(class_name, det_range, metric_name, tp)

    metrics_summary = metrics.serialize()

    results_msg, results_details = format_lyft_results(metrics_summary, eval_config.class_names, metric_prefix)
    print_log(results_msg, logger=logger)

    res_path = osp.join(output_dir, 'lyft_metrics_by_range.json')
    mmcv.dump(metrics_summary, res_path)

    return metrics_summary, results_details


def get_classwise_aps(gt, predictions, class_names, iou_thresholds):
    """Returns an array with an average precision per class.

    Note: Ground truth and predictions should have the following format.

    .. code-block::

    gt = [{
        'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207
                         fbb039a550991a5149214f98cec136ac',
        'translation': [974.2811881299899, 1714.6815014457964,
                        -23.689857123368846],
        'size': [1.796, 4.488, 1.664],
        'rotation': [0.14882026466054782, 0, 0, 0.9888642620837121],
        'name': 'car'
    }]

    predictions = [{
        'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207
                         fbb039a550991a5149214f98cec136ac',
        'translation': [971.8343488872263, 1713.6816097857359,
                        -25.82534357061308],
        'size': [2.519726579986132, 7.810161372666739, 3.483438286096803],
        'rotation': [0.10913582721095375, 0.04099572636992043,
                     0.01927712319721745, 1.029328402625659],
        'name': 'car',
        'score': 0.3077029437237213
    }]

    Args:
        gt (list[dict]): list of dictionaries in the format described below.
        predictions (list[dict]): list of dictionaries in the format
            described below.
        class_names (list[str]): list of the class names.
        iou_thresholds (list[float]): IOU thresholds used to calculate
            TP / FN

    Returns:
        np.ndarray: an array with an average precision per class.
    """
    assert all([0 <= iou_th <= 1 for iou_th in iou_thresholds])

    gt_by_class_name = group_by_key(gt, 'name')
    pred_by_class_name = group_by_key(predictions, 'name')

    average_precisions = np.zeros((len(class_names), len(iou_thresholds)))

    for class_id, class_name in enumerate(class_names):
        if class_name in pred_by_class_name:
            recalls, precisions, average_precision = get_single_class_aps(
                gt_by_class_name[class_name], pred_by_class_name[class_name],
                iou_thresholds)
            average_precisions[class_id, :] = average_precision

    return average_precisions


def get_single_class_aps(gt, predictions, iou_thresholds):
    """Compute recall and precision for all iou thresholds. Adapted from
    LyftDatasetDevkit.

    Args:
        gt (list[dict]): list of dictionaries in the format described above.
        predictions (list[dict]): list of dictionaries in the format
            described below.
        iou_thresholds (list[float]): IOU thresholds used to calculate
            TP / FN

    Returns:
        tuple[np.ndarray]: Returns (recalls, precisions, average precisions)
            for each class.
    """
    num_gts = len(gt)
    image_gts = group_by_key(gt, 'sample_token')
    image_gts = wrap_in_box(image_gts)

    sample_gt_checked = {
        sample_token: np.zeros((len(boxes), len(iou_thresholds)))
        for sample_token, boxes in image_gts.items()
    }

    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

    # go down dets and mark TPs and FPs
    num_predictions = len(predictions)
    tps = np.zeros((num_predictions, len(iou_thresholds)))
    fps = np.zeros((num_predictions, len(iou_thresholds)))

    for prediction_index, prediction in enumerate(predictions):
        predicted_box = Box3D(**prediction)

        sample_token = prediction['sample_token']

        max_overlap = -np.inf
        jmax = -1

        if sample_token in image_gts:
            gt_boxes = image_gts[sample_token]
            # gt_boxes per sample
            gt_checked = sample_gt_checked[sample_token]
            # gt flags per sample
        else:
            gt_boxes = []
            gt_checked = None

        if len(gt_boxes) > 0:
            overlaps = get_ious(gt_boxes, predicted_box)

            max_overlap = np.max(overlaps)

            jmax = np.argmax(overlaps)

        for i, iou_threshold in enumerate(iou_thresholds):
            if max_overlap > iou_threshold:
                if gt_checked[jmax, i] == 0:
                    tps[prediction_index, i] = 1.0
                    gt_checked[jmax, i] = 1
                else:
                    fps[prediction_index, i] = 1.0
            else:
                fps[prediction_index, i] = 1.0

    # compute precision recall
    fps = np.cumsum(fps, axis=0)
    tps = np.cumsum(tps, axis=0)

    recalls = tps / float(num_gts)
    # avoid divide by zero in case the first detection
    # matches a difficult ground truth
    precisions = tps / np.maximum(tps + fps, np.finfo(np.float64).eps)

    aps = []
    for i in range(len(iou_thresholds)):
        recall = recalls[:, i]
        precision = precisions[:, i]
        assert np.all(0 <= recall) & np.all(recall <= 1)
        assert np.all(0 <= precision) & np.all(precision <= 1)
        ap = get_ap(recall, precision)
        aps.append(ap)

    aps = np.array(aps)

    return recalls, precisions, aps


def format_lyft_results(metrics, class_names, metric_prefix='pts_bbox_Lyft'):
    result = '----------------Lyft %s results-----------------\n' % metric_prefix
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
        "emergency_vehicle": 1.0,
        "other_vehicle": 1.0,
        "motorcycle": 1.0,
        "bicycle": 1.0,
        "pedestrian": 1.0,
        "animal": 1.0,
        "human": 1.0,
        "vehicle": 1.0,
    }

    tp_errors_keys = ['trans_err', 'scale_err', 'orient_err']

    result += '--------------table log summary-------------\n'
    for name in class_names:
        result += f'***{name}\n'
        result += f"Range:\t" + ', '.join(list(metrics['label_aps'][name].keys())) + '\n'
        metrics_by_ranges = []
        for det_range in metrics['label_aps'][name].keys():
            ap_val = metrics['label_aps'][name][det_range][match_thresh[name]]
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

    result += '--------------table log summary v2-------------\n'
    result += "Classes: " + ', '.join(class_names) + '\n'
    det_ranges = list(metrics['label_aps'][name].keys())
    result += "Range: " + ', '.join([', '.join(det_ranges) ] * len(class_names)) + '\n'
    result += "AP@1.00m: " + ', , '.join([', '.join(['%.2f' % (metrics['label_aps'][name][det_range][1.0] * 100)  for det_range in det_ranges]) for name in class_names]) + '\n'
    result += f"trans_err@{metrics['cfg']['dist_th_tp']}m: " + ', , '.join([', '.join(['%.2f' % (metrics['label_tp_errors'][name][det_range]['trans_err'])  for det_range in det_ranges]) for name in class_names]) + '\n'
    result += f"scale_err@{metrics['cfg']['dist_th_tp']}m: " + ', , '.join([', '.join(['%.2f' % (metrics['label_tp_errors'][name][det_range]['scale_err'])  for det_range in det_ranges]) for name in class_names]) + '\n'
    result += f"orient_err@{metrics['cfg']['dist_th_tp']}m: " + ', , '.join([', '.join(['%.2f' % (metrics['label_tp_errors'][name][det_range]['orient_err'])  for det_range in det_ranges]) for name in class_names]) + '\n'

    details.update({
        f'{metric_prefix}/mAP': metrics['mean_ap'],
        f'{metric_prefix}/NDS': metrics['nd_score'],
    })

     # log additional details for tensorboard
    for name in class_names:
        # look for the widest range
        ranges = metrics['cfg']['class_range'][name]
        target_range = str(tuple(ranges[np.argmax([i[1] - i[0] for i in ranges]).item()]))
        details[f'{metric_prefix}/{name}_AP@1.00m_{target_range}'] = metrics['label_aps'][name][target_range][1.0]

    result += '--------------table log summary v3-------------\n'

    for name in class_names:
        result += f'***{name}\n'
        result += f"Range:\t" + ', '.join(list(metrics['label_aps'][name].keys())) + '\n'
        metrics_by_ranges = []
        for det_range in metrics['mean_dist_aps'][name].keys():
            ap_val = metrics['mean_dist_aps'][name][det_range]
            metrics_by_ranges.append(ap_val)
        result += f"mean_dist_aps:\t\t" + ", ".join([f"{x*100:.2f}" for x in metrics_by_ranges]) + "\n"

        tp_errors_dict = { key: [] for key in tp_errors_keys }
        # for tp errors
        for det_range in metrics['label_tp_errors'][name].keys():
            tp_errors = metrics['label_tp_errors'][name][det_range]
            for key in tp_errors_keys:
                tp_errors_dict[key].append(tp_errors[key])

        for key in tp_errors_keys:
            result += f"{key}@{metrics['cfg']['dist_th_tp']:.2f}m:\t" + ", ".join([f"{x*100:.2f}" for x in tp_errors_dict[key]]) + "\n"

    result += 'for copying: \n'
    result += '\t\t'.join(['\t'.join(['%.2f' % (metrics['mean_dist_aps'][name][det_range] * 100)  for det_range in det_ranges]) for name in class_names]) + '\n'
    result += '\t\t'.join(['\t'.join(['%.2f' % (metrics['label_tp_errors'][name][det_range]['trans_err'])  for det_range in det_ranges]) for name in class_names]) + '\n'
    result += '\t\t'.join(['\t'.join(['%.2f' % (metrics['label_tp_errors'][name][det_range]['scale_err'])  for det_range in det_ranges]) for name in class_names]) + '\n'
    result += '\t\t'.join(['\t'.join(['%.2f' % (metrics['label_tp_errors'][name][det_range]['orient_err'])  for det_range in det_ranges]) for name in class_names]) + '\n'
    # log additional details for tensorboard
    for name in class_names:
        for det_range in det_ranges:
            details[f'{metric_prefix}/{name}_mdistAP_{target_range}'] = metrics['mean_dist_aps'][name][det_range]

    return result, details
