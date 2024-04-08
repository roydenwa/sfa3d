import os
import pcl
import torch
import signal
import rosnode
import numpy as np


from sensor_msgs.msg import Image
from tf.transformations import quaternion_from_euler
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray


def unique_torch(x, dim: int = 0):
    # Src: https://github.com/pytorch/pytorch/issues/36748
    unique, inverse, counts = torch.unique(
        x, dim=dim, sorted=True, return_inverse=True, return_counts=True
    )
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]

    return unique, index, inverse, counts


def filter_point_cloud(point_cloud, x_min, x_max, y_min, y_max, z_min, z_max):
    mask = np.where(
        (point_cloud[:, 0] >= x_min)
        & (point_cloud[:, 0] <= x_max)
        & (point_cloud[:, 1] >= y_min)
        & (point_cloud[:, 1] <= y_max)
        & (point_cloud[:, 2] >= z_min)
        & (point_cloud[:, 2] <= z_max)
    )
    point_cloud = point_cloud[mask]

    return point_cloud


def rasterize_bev_pillars(
    point_cloud,
    discretization_coefficient: float = 50 / 608,
    bev_height: int = 608,
    bev_width: int = 608,
    z_max: float = 1.27,
    z_min: float = -2.73,
    n_lasers: int = 128,
):
    """Optimized PyTorch version of kitti_bev_utils.makeBEVMap"""
    height = bev_height + 1
    width = bev_width + 1

    _point_cloud = torch.clone(point_cloud)  # Local copy required?
    _point_cloud = _point_cloud.to("cuda", dtype=torch.float32)

    # Discretize x and y coordinates
    _point_cloud[:, 0] = torch.floor(_point_cloud[:, 0] / discretization_coefficient)
    _point_cloud[:, 1] = torch.floor(_point_cloud[:, 1] / discretization_coefficient)

    # Get unique indices to rasterize and unique counts to compute the point density
    _, unique_indices, _, unique_counts = unique_torch(_point_cloud[:, 0:2], dim=0)
    _point_cloud_top = _point_cloud[unique_indices]

    # Intensity, height and density maps
    intensity_map = torch.zeros((height, width), dtype=torch.float32, device="cuda")
    height_map = torch.zeros((height, width), dtype=torch.float32, device="cuda")
    density_map = torch.zeros((height, width), dtype=torch.float32, device="cuda")

    x_indices = _point_cloud_top[:, 0].int()
    y_indices = _point_cloud_top[:, 1].int()

    intensity_map[x_indices, y_indices] = _point_cloud_top[:, 3]

    max_height = np.float32(np.abs(z_max - z_min))
    height_map[x_indices, y_indices] = _point_cloud_top[:, 2] / max_height

    normalized_counts = torch.log(unique_counts + 1) / np.log(n_lasers)
    normalized_counts = torch.clamp(normalized_counts, min=0.0, max=1.0)
    density_map[x_indices, y_indices] = normalized_counts

    ihd_map = torch.zeros(
        (3, bev_height, bev_width), dtype=torch.float32, device="cuda"
    )
    ihd_map[0, :, :] = intensity_map[:bev_height, :bev_width]
    ihd_map[1, :, :] = height_map[:bev_height, :bev_width]
    ihd_map[2, :, :] = density_map[:bev_height, :bev_width]

    return ihd_map


def convert_det_to_real_values(
    detections,
    discretization_coefficient: float = 50 / 608,
    x_min: float = 0.0,
    y_min: float = -25.0,
    z_min: float = -2.73,
    num_classes=3,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    z_offset: float = 0.0,
    backwards: bool = False,
    rot_90: bool = False,
):
    kitti_dets = []
    for cls_id in range(num_classes):
        if len(detections[cls_id]) > 0:
            for det in detections[cls_id]:
                # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                _score, _x, _y, _z, _h, _w, _l, _yaw = det
                _yaw = -_yaw
                x = _y * discretization_coefficient + x_min
                y = _x * discretization_coefficient + y_min
                z = _z + z_min
                w = _w * discretization_coefficient
                l = _l * discretization_coefficient
                x += x_offset
                y += y_offset
                z += z_offset

                if backwards:
                    kitti_dets.append(
                        [
                            _score,
                            cls_id,
                            x * -1,
                            y * -1,
                            z,
                            _h,
                            w,
                            l,
                            _yaw + np.deg2rad(180),
                        ]
                    )
                elif rot_90:
                    kitti_dets.append(
                        [_score, cls_id, -y, x, z, _h, w, l, _yaw - np.deg2rad(90)]
                    )
                else:
                    kitti_dets.append([_score, cls_id, x, y, z, _h, w, l, _yaw])

    return np.array(kitti_dets)


def preprocess_point_cloud(pc: pcl.PointCloud) -> np.ndarray:
    pcd_np = pc.to_ndarray()
    x = pcd_np["x"]
    x = np.nan_to_num(x, nan=0.0)
    y = pcd_np["y"]
    y = np.nan_to_num(y, nan=0.0)
    z = pcd_np["z"]
    z = np.nan_to_num(z, nan=0.0)
    i = pcd_np["intensity"]
    i = np.nan_to_num(i, nan=0.0)
    i /= 255.0
    points_32 = np.transpose(np.vstack((x, y, z, i)))

    return points_32


def cv2_to_imgmsg(cv_image: np.ndarray) -> Image:
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]

    if cv_image.shape[-1] == 3:
        img_msg.encoding = "bgr8"
    else:
        img_msg.encoding = "mono8"

    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height

    return img_msg


def bboxes_to_rosmsg(bboxes: list, timestamp) -> BoundingBoxArray:
    # TODO: JIT (numba)
    rosboxes = BoundingBoxArray()

    for bbox in bboxes:
        confidence, cls_id, x, y, z, h, w, l, yaw = bbox

        rosbox = BoundingBox()
        rosbox.header.stamp = timestamp
        rosbox.header.frame_id = "sensor/lidar/box_top/center/vls128_ap"

        # roll, pitch and yaw
        q = quaternion_from_euler(0, 0, yaw)

        rosbox.pose.orientation.x = q[0]
        rosbox.pose.orientation.y = q[1]
        rosbox.pose.orientation.z = q[2]
        rosbox.pose.orientation.w = q[3]
        rosbox.pose.position.x = x
        rosbox.pose.position.y = y
        rosbox.pose.position.z = z
        rosbox.dimensions.x = l
        rosbox.dimensions.y = w
        rosbox.dimensions.z = h

        rosbox.label = np.uint(cls_id)  # TODO: convert from KITTI to Joy classes
        rosbox.value = confidence

        rosboxes.boxes.append(rosbox)

    rosboxes.header.frame_id = "sensor/lidar/box_top/center/vls128_ap"
    rosboxes.header.stamp = timestamp

    return rosboxes


def bev_center_nms(
    bboxes_in: np.ndarray,
    thresh_x: float = 1.0,
    thresh_y: float = 1.0,
    class_id: float = 1.0,
) -> list:
    bboxes_out = []  # [confidence, cls_id, x, y, z, h, w, l, yaw]

    for idx_a, box_a in enumerate(bboxes_in):
        keep = True
        for idx_b, box_b in enumerate(bboxes_in):
            if (
                idx_a != idx_b
                and box_a[1] == class_id
                and box_b[1] == class_id
                and box_a[0] < box_b[0]
                and np.abs(box_a[2] - box_b[2]) < thresh_x
                and np.abs(box_a[3] - box_b[3]) < thresh_y
            ):
                keep = False
        if keep:
            bboxes_out.append(box_a)

    return bboxes_out


def shutdown_callback(event):
    if not "rviz" in "".join(rosnode.get_node_names()):
        os.kill(os.getpid(), signal.SIGTERM)


def ego_nms(
    bboxes_in: np.ndarray, x_thresh: float = 1.5, y_thresh: float = 1.5
) -> list:
    bboxes_out = []

    for bbox in bboxes_in:
        if np.abs(bbox[2]) > x_thresh or np.abs(bbox[3]) > y_thresh:
            bboxes_out.append(bbox)

    return bboxes_out
