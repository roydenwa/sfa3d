import cv2
import pcl
import torch
import rospy
import typer
import numpy as np

from timeit import default_timer as timer
from sensor_msgs.msg import PointCloud2
from jsk_recognition_msgs.msg import BoundingBoxArray

from ros_utils import (
    preprocess_point_cloud,
    bev_center_nms,
    bboxes_to_rosmsg,
    shutdown_callback,
    ego_nms,
    rasterize_bev_pillars,
    filter_point_cloud,
    convert_det_to_real_values,
)
from models.model_utils import create_model
from utils.demo_utils import parse_demo_configs, do_detect as detect


def main(log_level: int = rospy.INFO) -> None:
# def main(log_level: int = rospy.DEBUG) -> None:
    def perception_callback(*data):
        pcd_msg_delay = rospy.rostime.Time.now() - data[0].header.stamp

        if not log_level == rospy.DEBUG:
            if pcd_msg_delay.to_sec() > 0.15:
                rospy.loginfo(
                    "Dropping point cloud message since the delay to ROS time now > 0.15 s."
                )
                return
        else:
            rospy.logdebug(
                f"Point cloud message delay to ROS time now: {pcd_msg_delay.to_sec()} s"
            )

        start_time = timer()
        point_cloud = pcl.PointCloud(data[0])
        deserialization_end = timer()

        point_cloud = preprocess_point_cloud(point_cloud)
        point_cloud = filter_point_cloud(
            point_cloud,
            x_min=-25,
            x_max=75,
            y_min=-25,
            y_max=25,
            z_min=-2.73,
            z_max=1.27,
        )

        # Set min to (0, 0, 0)
        point_cloud[:, 0] -= -25
        point_cloud[:, 1] -= -25
        point_cloud[:, 2] -= -2.73

        point_cloud = torch.from_numpy(point_cloud)
        preprocessing_end = timer()

        bev_pillars = rasterize_bev_pillars(point_cloud, bev_height=608 * 2)
        to_bev_pillars_end = timer()

        with torch.inference_mode():
            detections, *_ = detect(
                configs,
                model,
                bev_pillars,
                peak_thresh=0.2,
            )
        inference_end = timer()

        # Post-processing
        # [confidence, cls_id, x, y, z, h, w, l, yaw]
        bboxes = convert_det_to_real_values(detections=detections, x_offset=-25)
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
            rospy.logdebug(
                f"To BEV pillars latency: {to_bev_pillars_end - preprocessing_end} s"
            )
            rospy.logdebug(f"Inference latency: {inference_end - to_bev_pillars_end} s")
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
    model.load_state_dict(
        torch.load(
            "../checkpoints/fpn_resnet_18/Model_fpn_resnet_18_epoch_8.pth",
            map_location="cpu",
        )
    )
    model = model.to(configs.device)
    bev_pillars = torch.zeros((1, 3, 1216, 608), dtype=torch.float32, device="cuda")
    model(bev_pillars)
    print("Model initialized.")

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
