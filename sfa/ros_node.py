#!/usr/bin/env python3
import pcl
import torch
import rospy

from omegaconf import OmegaConf
from timeit import default_timer as timer
from sensor_msgs.msg import PointCloud2
from jsk_recognition_msgs.msg import BoundingBoxArray

from ros_utils import (
    bboxes_to_rosmsg,
)
from point_cloud_utils import (
    preprocess_point_cloud,
    filter_point_cloud,
    rasterize_bev_pillars,
)
from bounding_box_utils import ego_nms, bev_center_nms
from centernet_utils import detect, convert_det_to_real_values
from centernet_model import get_center_net


previous_total_latency = 0.0


def main(log_level: int = rospy.INFO) -> None:
    def perception_callback(*data):
        global previous_total_latency
        pcd_msg_delay = rospy.rostime.Time.now() - data[0].header.stamp

        if previous_total_latency > 0.2:
            rospy.loginfo(
                f"Dropping point cloud message since the previous total latency was > 0.2s ({previous_total_latency:.2}s)."
            )
            previous_total_latency = 0.0
            return

        start_time = timer()
        point_cloud = pcl.PointCloud(data[0])
        deserialization_end = timer()

        point_cloud = preprocess_point_cloud(point_cloud)
        point_cloud = filter_point_cloud(
            point_cloud,
            **config["filter_point_cloud"]
        )

        # Set min to (0, 0, 0)
        point_cloud[:, 0] -= config["filter_point_cloud"]["x_min"]
        point_cloud[:, 1] -= config["filter_point_cloud"]["y_min"]
        point_cloud[:, 2] -= config["filter_point_cloud"]["z_min"]

        point_cloud = torch.from_numpy(point_cloud)
        preprocessing_end = timer()

        bev_pillars = rasterize_bev_pillars(point_cloud, bev_height=config["bev_height"], device=config["device"])
        to_bev_pillars_end = timer()

        with torch.inference_mode():
            detections = detect(
                model,
                bev_pillars,
                peak_thresh=0.2,
            )
        inference_end = timer()

        # Post-processing
        # bbox = [confidence, cls_id, x, y, z, h, w, l, yaw]
        bboxes = convert_det_to_real_values(detections=detections, x_offset=config["filter_point_cloud"]["x_min"])
        bboxes = bev_center_nms(bboxes, thresh_x=2.0, thresh_y=1.5)
        bboxes = ego_nms(bboxes)
        rosboxes = bboxes_to_rosmsg(bboxes, data[0].header.stamp)

        postprocessing_end = timer()
        bbox_pub.publish(rosboxes)
        publish_end = timer()

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
        rospy.logdebug(
            f"Point cloud message delay to ROS time now: {pcd_msg_delay.to_sec()} s"
        )
        previous_total_latency = publish_end - start_time

    config = OmegaConf.load("/workspace/catkin_ws/src/config/inference.yaml")

    model = get_center_net(
        **config["model"]
    )

    model.load_state_dict(
        torch.load(
            config["checkpoint"],
            map_location="cpu",
        )
    )
    model = model.to(config["device"])
    bev_pillars = torch.zeros((1, 3, config["bev_height"], config["bev_width"]), dtype=torch.float32, device=config["device"])
    model(bev_pillars)
    print("Model initialized.")

    rospy.init_node("sfa3d_detector", log_level=log_level)

    point_cloud_sub = rospy.Subscriber(
        name=config["point_cloud_topic"],
        data_class=PointCloud2,
        callback=perception_callback,
        queue_size=1,
        buff_size=int(10e9),
    )

    bbox_pub = rospy.Publisher(
        name=config["bbox_topic"],
        data_class=BoundingBoxArray,
        queue_size=1,
    )

    rospy.spin()


if __name__ == "__main__":
    main()
