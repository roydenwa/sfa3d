import os
import signal
import rosnode
import numpy as np

from sensor_msgs.msg import Image
from tf.transformations import quaternion_from_euler
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray


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


def bboxes_to_rosmsg(
    bboxes: list,
    timestamp,
    frame_id: str = "sensor/lidar/box_top/center/vls128_ap",
) -> BoundingBoxArray:
    rosboxes = BoundingBoxArray()

    for bbox in bboxes:
        confidence, cls_id, x, y, z, h, w, l, yaw = bbox

        rosbox = BoundingBox()
        rosbox.header.stamp = timestamp
        rosbox.header.frame_id = frame_id

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

    rosboxes.header.frame_id = frame_id
    rosboxes.header.stamp = timestamp

    return rosboxes


def shutdown_callback(event):
    if not "rviz" in "".join(rosnode.get_node_names()):
        os.kill(os.getpid(), signal.SIGTERM)
