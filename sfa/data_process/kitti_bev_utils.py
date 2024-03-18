"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
"""

import math
import os
import sys

import cv2
import numpy as np

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import config.kitti_config as cnf

from timeit import default_timer as timer


def rasterize_bevmap(point_cloud, z_max=1.27, z_min=-2.73, n_lasers=128, center_y: bool = True):
    """Optimized version of kitti_bev_utils.makeBEVMap using int16 and float32 dtypes"""
    height = cnf.BEV_HEIGHT + 1
    width = cnf.BEV_WIDTH + 1

    # Discretize x and y coordinates
    _point_cloud = np.copy(point_cloud)
    _point_cloud[:, 0] = (np.floor(_point_cloud[:, 0] / cnf.DISCRETIZATION))
    
    if center_y:
        _point_cloud[:, 1] = (np.floor(_point_cloud[:, 1] / cnf.DISCRETIZATION) + width / 2)
    else:
        _point_cloud[:, 1] = (np.floor(_point_cloud[:, 1] / cnf.DISCRETIZATION))
    
    _point_cloud = _point_cloud.astype(np.int16)

    # Sort 3times
    sorted_indices = np.lexsort((-_point_cloud[:, 2], _point_cloud[:, 1], _point_cloud[:, 0]))
    _point_cloud = _point_cloud[sorted_indices]

    _, unique_indices, unique_counts = np.unique(_point_cloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    _point_cloud_top = _point_cloud[unique_indices]

    # Intensity, height and density maps
    intensity_map = np.zeros((height, width), dtype=np.float32)
    height_map = np.zeros((height, width), dtype=np.float32)
    density_map = np.zeros((height, width), dtype=np.float32)

    # Image coordinates are (y, x), not (x, y)
    max_height = np.float32(np.abs(z_max - z_min))
    height_map[_point_cloud_top[:, 0], _point_cloud_top[:, 1]] = _point_cloud_top[:, 2] / max_height

    normalized_counts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(n_lasers))
    intensity_map[_point_cloud_top[:, 0], _point_cloud_top[:, 1]] = _point_cloud_top[:, 3]
    density_map[_point_cloud_top[:, 0], _point_cloud_top[:, 1]] = normalized_counts

    ihd_map = np.zeros((3, cnf.BEV_HEIGHT, cnf.BEV_WIDTH), dtype=np.float32)
    ihd_map[0, :, :] = intensity_map[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]
    ihd_map[1, :, :] = height_map[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]
    ihd_map[2, :, :] = density_map[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]

    return ihd_map


def makeBEVMap(PointCloud_, boundary, n_lasers=128, center_y: bool = True): # KITTI LiDAR has 64 lasers
    Height = cnf.BEV_HEIGHT + 1
    Width = cnf.BEV_WIDTH + 1

    discretize_start = timer()

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / cnf.DISCRETIZATION))
    if center_y:
        PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / cnf.DISCRETIZATION) + Width / 2)
    else:
        PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / cnf.DISCRETIZATION))

    discretize_end = timer()

    # sort-3times, not required but speeds up subsequent np.unique
    # sorted_indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    # PointCloud = PointCloud[sorted_indices]

    sorting_end = timer()
    _, unique_indices, unique_counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[unique_indices]
    unique_counts_end = timer()

    # Height Map, Intensity Map & Density Map
    heightMap = np.zeros((Height, Width))
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    # some important problem is image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(boundary['maxZ'] - boundary['minZ']))
    heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 2] / max_height

    normalizedCounts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(n_lasers))
    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    RGB_Map = np.zeros((3, Height - 1, Width - 1))
    RGB_Map[2, :, :] = densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # r_map
    RGB_Map[1, :, :] = heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # g_map
    RGB_Map[0, :, :] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # b_map

    bevmap_set_end = timer()
    
    print(f"Bevmap discretization latency: {discretize_end - discretize_start} s")
    print(f"Bevmap sorting latency: {sorting_end - discretize_end} s")
    print(f"Bevmap unique elems latency: {unique_counts_end - sorting_end} s")
    print(f"Bevmap set elems latency: {bevmap_set_end - unique_counts_end} s")

    return RGB_Map


# bev image coordinates format
def get_corners(x, y, w, l, yaw):
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    # front left
    bev_corners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw

    # rear left
    bev_corners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw

    # rear right
    bev_corners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw

    # front right
    bev_corners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw

    return bev_corners


def drawRotatedBox(img, x, y, w, l, yaw, color):
    bev_corners = get_corners(x, y, w, l, yaw)
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
    cv2.polylines(img, [corners_int], True, color, 2)
    corners_int = bev_corners.reshape(-1, 2).astype(int)
    cv2.line(img, (corners_int[0, 0], corners_int[0, 1]), (corners_int[3, 0], corners_int[3, 1]), (255, 255, 0), 2)
