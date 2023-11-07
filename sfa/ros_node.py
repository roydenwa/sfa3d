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
from ros_utils import load_bevmap_front, min_max_scaling

from utils.evaluation_utils import draw_predictions, convert_det_to_real_values
from models.model_utils import create_model
from utils.demo_utils import parse_demo_configs, do_detect
from utils.visualization_utils import merge_rgb_to_bev


def main(log_level: int = rospy.ERROR):
    def perception_callback(*data):
        point_cloud = pcl.PointCloud(data[0])
        front_img = opencv_bridge.imgmsg_to_cv2(data[1])

        point_cloud = preprocess_point_cloud(point_cloud)
        front_bevmap = load_bevmap_front(point_cloud)

        with torch.no_grad():
            detections, bev_map, fps = do_detect(
                configs, model, front_bevmap, is_front=True
            )

        bev_map = (bev_map.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
        bev_map = draw_predictions(bev_map, detections, configs.num_classes)
        bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
        debug_img = merge_rgb_to_bev(
            front_img, bev_map, output_width=configs.output_width
        )
        debug_img_ros = cv2_to_imgmsg(debug_img)
        debug_img_pub.publish(debug_img_ros)

        # [cls_id, x, y, z, _h, w, l, _yaw]
        bboxes = convert_det_to_real_values(detections=detections)
        print(f"bboxes: {bboxes}")
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
    opencv_bridge = CvBridge()

    point_cloud_sub = message_filters.Subscriber(
        "/sensor/lidar/box_top/center/vls128_ap/points", PointCloud2
    )

    front_img_sub = message_filters.Subscriber(
        "/sensor/camera/box_ring/front/atl071s_cc/raw/image", Image
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
            front_img_sub,
        ],
        queue_size=10,
        slop=0.1,  # in secs
    )
    ts.registerCallback(perception_callback)

    rospy.spin()


def preprocess_point_cloud(pc):
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


def cv2_to_imgmsg(cv_image):
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


def bboxes_to_rosmsg(bboxes, timestamp):
    # TODO: JIT (numba)
    # [cls_id, x, y, z, _h, w, l, _yaw]
    rosboxes = BoundingBoxArray()

    for bbox in bboxes:
        # http://otamachan.github.io/sphinxros/indigo/packages/jsk_recognition_msgs.html#message-jsk_recognition_msgs/BoundingBox
        # Header header
        # geometry_msgs/Pose pose
        # geometry_msgs/Vector3 dimensions  # size of bounding box (x, y, z)
        # # You can use this field to hold value such as likelihood
        # float32 value
        # uint32 label
        cls_id, x, y, z, h, w, l, yaw = bbox

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

        rosboxes.boxes.append(rosbox)

    rosboxes.header.frame_id = "sensor/lidar/box_top/center/vls128_ap"
    rosboxes.header.stamp = timestamp

    return rosboxes


if __name__ == "__main__":
    try:
        typer.run(main)
    except rospy.ROSInterruptException:
        rospy.logerr_once("Exiting due to ROSInterruptException")
