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


# def main(log_level: int = rospy.ERROR) -> None:
def main(log_level: int = rospy.DEBUG) -> None:
    def perception_callback(*data):
        pcd_msg_delay = rospy.rostime.Time.now() - data[0].header.stamp

        if not log_level == rospy.DEBUG:
            if pcd_msg_delay.to_sec() > 0.15:
                rospy.loginfo(
                    "Dropping point cloud message since it is delayed by more than 0.15 s."
                )
                return
        else:
            rospy.logdebug(f"Point cloud message delay: {pcd_msg_delay.to_sec()} s")

        start_time = timer()
        point_cloud = pcl.PointCloud(data[0])
        deserialization_end = timer()

        point_cloud = preprocess_point_cloud(point_cloud)
        preprocessing_end = timer()

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

        to_bevmap_end = timer()

        with torch.inference_mode():
            front_detections_0, *_ = detect(
                configs,
                model,
                front_bevmap_0,
                peak_thresh=0.2,
            )
            front_detections_1, *_ = detect(
                configs,
                model,
                front_bevmap_1,
                peak_thresh=0.4,
                class_idx=1,  # Only vehicles
            )
            back_detections, *_ = detect(
                configs,
                model,
                back_bevmap,
                peak_thresh=0.2,
                class_idx=1,
            )

        inference_end = timer()

        # Post-processing for 9040 config
        # [confidence, cls_id, x, y, z, h, w, l, yaw]
        bboxes_0 = convert_det_to_real_values(detections=front_detections_0)
        bboxes_1 = convert_det_to_real_values(detections=front_detections_1, x_offset=40)
        bboxes_2 = convert_det_to_real_values(
            detections=back_detections, x_offset=-10, backwards=True
        )

        bboxes = np.array([], dtype=np.float32).reshape(0, 9)
        if bboxes_0.shape[0]:
            bboxes = np.concatenate((bboxes, bboxes_0), axis=0)
        if bboxes_1.shape[0]:
            bboxes = np.concatenate((bboxes, bboxes_1), axis=0)
        if bboxes_2.shape[0]:
            bboxes = np.concatenate((bboxes, bboxes_2), axis=0)

        bboxes = bev_center_nms(bboxes, thresh_x=2.0, thresh_y=1.5)
        bboxes = ego_nms(bboxes)
        rosboxes = bboxes_to_rosmsg(bboxes, data[0].header.stamp)

        postprocessing_end = timer()
        bbox_pub.publish(rosboxes)
        publish_end = timer()

        if log_level == rospy.DEBUG:
            rospy.logdebug(
                f"Deserialization latency: {deserialization_end - start_time} s"
            )
            rospy.logdebug(
                f"Pre-processing latency: {preprocessing_end - deserialization_end} s"
            )
            rospy.logdebug(f"To bevmap latency: {to_bevmap_end - preprocessing_end} s")
            rospy.logdebug(f"Inference latency: {inference_end - to_bevmap_end} s")
            rospy.logdebug(
                f"Post-processing latency: {postprocessing_end - inference_end} s"
            )
            rospy.logdebug(
                f"Message publishing latency: {publish_end - postprocessing_end} s"
            )
            rospy.logdebug(f"Total latency: {publish_end - start_time} s")

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

    # Init model and jit post processing
    bevmap = torch.zeros((1, 3, 608, 608), dtype=torch.float32, device="cuda")
    bboxes = np.array([], dtype=np.float32).reshape(0, 9)

    model(bevmap)
    bev_center_nms(bboxes)
    print("Init model.")

    rospy.init_node("sfa3d_detector", log_level=log_level)

    point_cloud_sub = rospy.Subscriber(
        name="/sensor/lidar/box_top/center/vls128_ap/points",
        data_class=PointCloud2,
        callback=perception_callback,
        queue_size=1,
        buff_size=int(10e9),
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
