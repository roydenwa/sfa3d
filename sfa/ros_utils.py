import argparse
import sys
import os
import time
import warnings
import zipfile

warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import torch
import numpy as np

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.demo_dataset import Demo_KittiDataset
from models.model_utils import create_model
from utils.evaluation_utils import draw_predictions, convert_det_to_real_values
import config.kitti_config as cnf
from data_process.transformation import lidar_to_camera_box
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from data_process.kitti_data_utils import Calibration
from utils.demo_utils import parse_demo_configs, do_detect, download_and_unzip, write_credit

from data_process.kitti_data_utils import get_filtered_lidar
from data_process.kitti_bev_utils import makeBEVMap
import config.kitti_config as cnf

from utils.misc import time_synchronized
from utils.evaluation_utils import decode
from utils.torch_utils import _sigmoid


def read_lidar_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def load_bevmap(pcd, boundary: dict = None, n_lasers: int = 128, center_y: bool = True, is_back: bool = False, is_left: bool = False, is_right: bool = False):
    if not boundary:
        lidar = get_filtered_lidar(pcd, cnf.boundary)
        bevmap = makeBEVMap(lidar, cnf.boundary, n_lasers, center_y)
    else:
        lidar = get_filtered_lidar(pcd, boundary)
        lidar[:, 0] = lidar[:, 0] - boundary["minX"]
        bevmap = makeBEVMap(lidar, boundary, n_lasers, center_y)

    bevmap = torch.from_numpy(bevmap)

    if is_back:
        bevmap = torch.flip(bevmap, [1, 2])
    elif is_left:
        bevmap = torch.rot90(bevmap, 3, [1, 2])
    elif is_right:
        bevmap = torch.rot90(bevmap, 1, [1, 2])

    return bevmap


def detect(configs, model, bevmaps, peak_thresh: float = None, considered_classes: tuple = (0, 1, 2)):
    input_bev_maps = bevmaps.to(configs.device, non_blocking=True).float()
    
    t1 = time_synchronized()
    
    outputs = model(input_bev_maps)
    outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
    outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
    
    # detections size (batch_size, K, 10)
    detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                        outputs['dim'], K=configs.K)
    detections = detections.cpu().numpy().astype(np.float32)
    
    if peak_thresh:
        detections = post_processing(detections, configs.num_classes, configs.down_ratio, peak_thresh, considered_classes)
    else:
        detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh, considered_classes)
    
    t2 = time_synchronized()

    # Inference speed
    fps = 1 / (t2 - t1)

    return detections, None, fps


def min_max_scaling(x: np.ndarray) -> np.ndarray:
    x_min, x_max = x.min(), x.max()
    x -= x_min
    x /= (x_max - x_min) + np.finfo(np.float32).eps

    return x


def get_yaw(direction):
    return np.arctan2(direction[:, 0:1], direction[:, 1:2])


def post_processing(detections, num_classes=3, down_ratio=4, peak_thresh=0.2, considered_classes: tuple = (0, 1, 2)):
    """
    :param detections: [batch_size, K, 10] -> K = topk results (default 50)
    # (scores x 1, xs x 1, ys x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
    # (scores-0:1, xs-1:2, ys-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
    :return:
    """
    ret = []

    for i in range(detections.shape[0]):
        top_preds = {}
        classes = detections[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            # x, y, z, h, w, l, yaw
            top_preds[j] = np.concatenate([
                detections[i, inds, 0:1],
                detections[i, inds, 1:2] * down_ratio,
                detections[i, inds, 2:3] * down_ratio,
                detections[i, inds, 3:4],
                detections[i, inds, 4:5],
                detections[i, inds, 5:6] / cnf.bound_size_y * cnf.BEV_WIDTH,
                detections[i, inds, 6:7] / cnf.bound_size_x * cnf.BEV_HEIGHT,
                get_yaw(detections[i, inds, 7:9]).astype(np.float32)], axis=1)

            # Filter by peak_thresh
            if len(top_preds[j]) > 0:
                keep_inds = (top_preds[j][:, 0] > peak_thresh)
                top_preds[j] = top_preds[j][keep_inds]

            # Filter considered classes
            if j not in considered_classes:
                top_preds[j] = []
        
        ret.append(top_preds)

    return ret