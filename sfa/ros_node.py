import cv2
import pcl
import torch
import rospy
import typer
import numpy as np
import message_filters

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from tf.transformations import quaternion_from_euler

import config.kitti_config as cnf
from ros_utils import load_bevmap, min_max_scaling

from utils.evaluation_utils import draw_predictions, convert_det_to_real_values
from models.model_utils import create_model
from utils.demo_utils import parse_demo_configs, do_detect
from utils.visualization_utils import merge_rgb_to_bev


def main(log_level: int = rospy.ERROR) -> None:
    def perception_callback(*data):
        point_cloud = pcl.PointCloud(data[0])

        point_cloud = preprocess_point_cloud(point_cloud)
       
        # z_offset -> TODO: dbl check should be negative?
        # point_cloud[:, 2] += 0.55 # prob. times 1.1/ 1.2 if < 0.0 to stretch area between sensor and ground level

        front_bevmap_0 = load_bevmap(point_cloud)
        front_bevmap_1 = load_bevmap(
            point_cloud, 
            n_lasers=64, 
            boundary={
                "minX": 40,
                "maxX": 90,
                "minY": -25,
                "maxY": 25,
                "minZ": -2.73,
                "maxZ": 1.27,
            }
        )
        # 9040 config
        back_bevmap = load_bevmap(
            point_cloud,
            is_back=True,
            boundary={
                "minX": -40,
                "maxX": 10,
                "minY": -25,
                "maxY": 25,
                "minZ": -2.73,
                "maxZ": 1.27
            },
        )
        # point_cloud[:, 1] += 10 # workaround to get overlap in bag
        # T-config left
        # back_bevmap = load_bevmap(
        #     point_cloud,
        #     boundary={
        #         "minX": -25,
        #         "maxX": 25,
        #         "minY": 0,
        #         "maxY": 50,
        #         "minZ": -2.73,
        #         "maxZ": 1.27
        #     },
        #     center_y=False,
        #     is_left=True,
        # )

        with torch.no_grad():
            detections_0, bev_map_0, fps_0 = do_detect(
                configs, model, front_bevmap_0,
            )
            detections_1, bev_map_1, fps_1 = do_detect(
                configs, model, front_bevmap_1, peak_thresh=0.4, class_idx=1, # Only vehicles
            )
            detections_2, bev_map, fps_2 = do_detect(
                # 9040 config
                configs, model, back_bevmap,
                # T-config left
                # configs, model, back_bevmap, peak_thresh=0.2, is_left=True, class_idx=1,
            )

        print(f"fps: {(fps_0 + fps_1 + fps_2) / 6}")

        bev_map = (bev_map.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
        debug_img = bev_map
        debug_img_ros = cv2_to_imgmsg(debug_img)
        debug_img_pub.publish(debug_img_ros)

        # [confidence, cls_id, x, y, z, h, w, l, yaw]
        bboxes_0 = convert_det_to_real_values(detections=detections_0, z_offset=0.55)
        bboxes_1 = convert_det_to_real_values(detections=detections_1, x_offset=40, z_offset=0.55)
        # 9040 config
        bboxes_2 = convert_det_to_real_values(detections=detections_2, x_offset=-10, z_offset=0.55, backwards=True)
        # T-config left
        # bboxes_2 = convert_det_to_real_values(detections=detections_2, rot_90=True) #x_offset=-25, y_offset=-15)

        bboxes = np.array([], dtype=np.float32).reshape(0, 9)
        if bboxes_0.shape[0]:
            bboxes = np.concatenate((bboxes, bboxes_0), axis=0)
        if bboxes_1.shape[0]:
            bboxes = np.concatenate((bboxes, bboxes_1), axis=0)
        if bboxes_2.shape[0]:
            bboxes = np.concatenate((bboxes, bboxes_2), axis=0)

        # print("before nms")
        # print(bboxes)
        bboxes = bev_center_nms(bboxes, thresh_x=1.0, thresh_y=1.0)
        # print("after nms")
        # print(bboxes)

        rosboxes = bboxes_to_rosmsg(bboxes, data[0].header.stamp)

        bbox_pub.publish(rosboxes)

    configs = parse_demo_configs()
    configs.device = torch.device(
        "cpu" if configs.no_cuda else "cuda:{}".format(configs.gpu_idx)
    )

    model = create_model(configs)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location="cpu"))
    model = model.to(configs.device)

    rospy.init_node("sfa3d_detector", log_level=log_level)

    point_cloud_sub = message_filters.Subscriber(
        "/sensor/lidar/box_top/center/vls128_ap/points", PointCloud2
    )

    debug_img_pub = rospy.Publisher(
        name="/perception/sfa3d/debug_image",
        data_class=Image,
        queue_size=10,
    )

    bbox_pub = rospy.Publisher(
        name="/perception/sfa3d/bboxes",
        data_class=BoundingBoxArray,
        queue_size=10,
    )

    ts = message_filters.ApproximateTimeSynchronizer(
        fs=[
            point_cloud_sub,
        ],
        queue_size=10,
        slop=0.1,  # in secs
    )
    ts.registerCallback(perception_callback)

    rospy.spin()


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


def bev_center_nms(bboxes_in: np.ndarray, thresh_x: float = 1.0, thresh_y: float = 1.0) -> np.ndarray:
    # TODO: JIT (numba)
    bboxes_out = [] # [confidence, cls_id, x, y, z, h, w, l, yaw]  
    bboxes_in = bboxes_in[np.lexsort((bboxes_in[:, 0], bboxes_in[:, 2], bboxes_in[:, 3]))[::-1]] # Sort in descending order
    prev_bbox = np.zeros(9, dtype=np.float32)

    for bbox in bboxes_in:
        if not (np.abs(bbox[2] - prev_bbox[2]) < thresh_x and np.abs(bbox[3] - prev_bbox[3]) < thresh_y):
            bboxes_out.append(bbox)
        prev_bbox = bbox

    return np.array(bboxes_out)
    

if __name__ == "__main__":
    try:
        typer.run(main)
    except rospy.ROSInterruptException:
        rospy.logerr_once("Exiting due to ROSInterruptException")
