import cv2
import pcl
import torch
import rospy
import typer
import numpy as np

import concurrent.futures as cf
from timeit import default_timer as timer

from sensor_msgs.msg import PointCloud2
from jsk_recognition_msgs.msg import BoundingBoxArray

import config.kitti_config as cnf
from ros_utils import (
    load_bevmap,
    preprocess_point_cloud,
    bev_center_nms,
    bboxes_to_rosmsg,
    shutdown_callback,
    ego_nms,
)

from utils.evaluation_utils import convert_det_to_real_values
from models.model_utils import create_model
from utils.demo_utils import parse_demo_configs, do_detect as detect


def main(log_level: int = rospy.ERROR) -> None:
    def perception_callback(*data):
        start_time = timer()
        point_cloud = pcl.PointCloud(data[0])
        point_cloud = preprocess_point_cloud(point_cloud)

        with cf.ThreadPoolExecutor(3) as pool:
            future_0 = pool.submit(load_bevmap, point_cloud)
            future_1 = pool.submit(
                load_bevmap,
                point_cloud,
                n_lasers=64,
                boundary={
                    "minX": 40,
                    "maxX": 90,
                    "minY": -25,
                    "maxY": 25,
                    "minZ": -2.73,
                    "maxZ": 1.27,
                },
            )
            future_2 = pool.submit(
                load_bevmap,
                point_cloud,
                is_back=True,
                boundary={
                    "minX": -40,
                    "maxX": 10,
                    "minY": -25,
                    "maxY": 25,
                    "minZ": -2.73,
                    "maxZ": 1.27,
                },
            )

            front_bevmap_0 = future_0.result()
            front_bevmap_1 = future_1.result()
            back_bevmap = future_2.result()

        preprocessing_end = timer()

        with torch.no_grad():
            detections_0, *_ = detect(
                configs,
                model,
                front_bevmap_0,
                peak_thresh=0.2,
            )
            detections_1, *_ = detect(
                configs,
                model,
                front_bevmap_1,
                peak_thresh=0.4,
                class_idx=1, # Only vehicles
            )
            detections_2, *_ = detect(
                configs,
                model,
                back_bevmap,
                peak_thresh=0.2,
                class_idx=1,
            )

        inference_end = timer()

        # [confidence, cls_id, x, y, z, h, w, l, yaw]
        # bboxes_0 = convert_det_to_real_values(detections=detections_0, z_offset=0.55)
        bboxes_0 = convert_det_to_real_values(detections=detections_0)
        # bboxes_1 = convert_det_to_real_values(detections=detections_1, x_offset=40, z_offset=0.55)
        bboxes_1 = convert_det_to_real_values(detections=detections_1, x_offset=40)
        # 9040 config
        # bboxes_2 = convert_det_to_real_values(detections=detections_2, x_offset=-10, z_offset=0.55, backwards=True)
        bboxes_2 = convert_det_to_real_values(
            detections=detections_2, x_offset=-10, backwards=True
        )

        bboxes = np.array([], dtype=np.float32).reshape(0, 9)
        if bboxes_0.shape[0]:
            bboxes = np.concatenate((bboxes, bboxes_0), axis=0)
        if bboxes_1.shape[0]:
            bboxes = np.concatenate((bboxes, bboxes_1), axis=0)
        if bboxes_2.shape[0]:
            bboxes = np.concatenate((bboxes, bboxes_2), axis=0)

        bboxes = bev_center_nms(bboxes, thresh_x=2.0, thresh_y=1.5)
        # bboxes = ego_nms(np.array(bboxes))
        bboxes = ego_nms(bboxes)
        rosboxes = bboxes_to_rosmsg(bboxes, data[0].header.stamp)

        end_time = timer()
        bbox_pub.publish(rosboxes)
        end_publish = timer()

        if log_level == rospy.DEBUG:
            print(f"Pre-processing latency: {preprocessing_end - start_time}")
            print(f"Inference latency: {inference_end - preprocessing_end}")
            print(f"Post-processing latency: {end_time - inference_end}")
            print(f"Total latency: {end_time - start_time}")
            print(f"Message publishing latency: {end_publish - end_time}")

    configs = parse_demo_configs()
    configs.device = torch.device(
        "cpu" if configs.no_cuda else "cuda:{}".format(configs.gpu_idx)
    )

    model = create_model(configs)
    # model.load_state_dict(torch.load(configs.pretrained_path, map_location="cpu"))
    model.load_state_dict(
        torch.load(
            "../checkpoints/fpn_resnet_18/Model_fpn_resnet_18_epoch_8.pth",
            map_location="cpu",
        )
    )
    model = model.to(configs.device)

    rospy.init_node("sfa3d_detector", log_level=log_level)

    point_cloud_sub = rospy.Subscriber(
        name="/sensor/lidar/box_top/center/vls128_ap/points",
        data_class=PointCloud2,
        callback=perception_callback,
    )

    bbox_pub = rospy.Publisher(
        name="/perception/sfa3d/bboxes",
        data_class=BoundingBoxArray,
        queue_size=1,
    )

    rospy.Timer(rospy.Duration(secs=10), callback=shutdown_callback)
    rospy.spin()


if __name__ == "__main__":
    try:
        typer.run(main)
    except rospy.ROSInterruptException:
        rospy.logerr_once("Exiting due to ROSInterruptException")
