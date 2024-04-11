import numpy as np


def ego_nms(
    bboxes_in: np.ndarray, x_thresh: float = 1.5, y_thresh: float = 1.5
) -> list:
    bboxes_out = []

    for bbox in bboxes_in:
        if np.abs(bbox[2]) > x_thresh or np.abs(bbox[3]) > y_thresh:
            bboxes_out.append(bbox)

    return bboxes_out


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
