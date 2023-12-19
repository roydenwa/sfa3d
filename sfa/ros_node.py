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

import config.kitti_config as cnf
from ros_utils import load_bevmap, preprocess_point_cloud, cv2_to_imgmsg, bev_center_nms, bboxes_to_rosmsg

from utils.evaluation_utils import draw_predictions, convert_det_to_real_values
from models.model_utils import create_model
from utils.demo_utils import parse_demo_configs, do_detect
from utils.visualization_utils import merge_rgb_to_bev


def main(log_level: int = rospy.ERROR) -> None:
    def perception_callback(*data):
        point_cloud = pcl.PointCloud(data[0])
        point_cloud = preprocess_point_cloud(point_cloud)
       
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
            )
           
        print(f"fps: {(fps_0 + fps_1 + fps_2) / 6}")
        bev_map = back_bevmap

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

        bboxes = np.array([], dtype=np.float32).reshape(0, 9)
        if bboxes_0.shape[0]:
            bboxes = np.concatenate((bboxes, bboxes_0), axis=0)
        if bboxes_1.shape[0]:
            bboxes = np.concatenate((bboxes, bboxes_1), axis=0)
        if bboxes_2.shape[0]:
            bboxes = np.concatenate((bboxes, bboxes_2), axis=0)

        bboxes = bev_center_nms(bboxes, thresh_x=1.0, thresh_y=1.0)
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


if __name__ == "__main__":
    try:
        typer.run(main)
    except rospy.ROSInterruptException:
        rospy.logerr_once("Exiting due to ROSInterruptException")
