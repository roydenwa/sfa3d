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


def min_max_scaling(x: np.ndarray) -> np.ndarray:
    x_min, x_max = x.min(), x.max()
    x -= x_min
    x /= (x_max - x_min) + np.finfo(np.float32).eps

    return x