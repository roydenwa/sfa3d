#!/usr/bin/env python3
import pcl
import torch
import rospy

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
            x_min=-25,
            x_max=75,
            y_min=-25,
            y_max=25,
            z_min=-2.73,
            z_max=1.27,
        )

        # Set min to (0, 0, 0)
        point_cloud[:, 0] += 25
        point_cloud[:, 1] += 25
        point_cloud[:, 2] += 2.73

        point_cloud = torch.from_numpy(point_cloud)
        preprocessing_end = timer()

        bev_pillars = rasterize_bev_pillars(point_cloud, bev_height=608 * 2)
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
        bboxes = convert_det_to_real_values(detections=detections, x_offset=-25)
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

    model = get_center_net(
        num_layers=18,
        heads={
            "hm_cen": 3,
            "cen_offset": 2,
            "direction": 2,
            "z_coor": 1,
            "dim": 3,
        },
        head_conv=64,
        imagenet_pretrained=False,
    )

    model.load_state_dict(
        torch.load(
            "/workspace/catkin_ws/src/checkpoints/fpn_resnet_18_epoch_8.pth",
            map_location="cpu",
        )
    )
    model = model.to("cuda")
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

    rospy.spin()


if __name__ == "__main__":
    main()
