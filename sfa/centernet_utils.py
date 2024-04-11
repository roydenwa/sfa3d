import torch
import numpy as np
import torch.nn.functional as F


def detect(
    model,
    bev_pillars,
    top_k: int = 50,
    n_classes: int = 3,
    down_ratio: int = 4,
    peak_thresh: float = 0.2,
    class_idx: int = None,
):
    outputs = model(bev_pillars.unsqueeze(0))
    outputs["hm_cen"] = _sigmoid(outputs["hm_cen"])
    outputs["cen_offset"] = _sigmoid(outputs["cen_offset"])

    # detections size (batch_size, K, 10)
    detections = decode(
        outputs["hm_cen"],
        outputs["cen_offset"],
        outputs["direction"],
        outputs["z_coor"],
        outputs["dim"],
        K=top_k,
    )
    detections = detections.cpu().numpy().astype(np.float32)

    if class_idx is not None:
        detections = post_process(
            detections, n_classes, down_ratio, peak_thresh, class_idx=class_idx
        )
    else:
        detections = post_process(detections, n_classes, down_ratio, peak_thresh)

    return detections[0]


def decode(hm_cen, cen_offset, direction, z_coor, dim, K=40):
    batch_size, num_classes, height, width = hm_cen.size()

    hm_cen = _nms(hm_cen)
    scores, inds, clses, ys, xs = _topk(hm_cen, K=K)
    if cen_offset is not None:
        cen_offset = _transpose_and_gather_feat(cen_offset, inds)
        cen_offset = cen_offset.view(batch_size, K, 2)
        xs = xs.view(batch_size, K, 1) + cen_offset[:, :, 0:1]
        ys = ys.view(batch_size, K, 1) + cen_offset[:, :, 1:2]
    else:
        xs = xs.view(batch_size, K, 1) + 0.5
        ys = ys.view(batch_size, K, 1) + 0.5

    direction = _transpose_and_gather_feat(direction, inds)
    direction = direction.view(batch_size, K, 2)
    z_coor = _transpose_and_gather_feat(z_coor, inds)
    z_coor = z_coor.view(batch_size, K, 1)
    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch_size, K, 3)
    clses = clses.view(batch_size, K, 1).float()
    scores = scores.view(batch_size, K, 1)

    # (scores x 1, ys x 1, xs x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
    # (scores-0:1, ys-1:2, xs-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
    # detections: [batch_size, K, 10]
    detections = torch.cat([scores, xs, ys, z_coor, dim, direction, clses], dim=2)

    return detections


def post_process(
    detections,
    num_classes=3,
    down_ratio=4,
    peak_thresh=0.2,
    class_idx: int = None,
    discretization_coefficient: float = 50 / 608,
):
    """
    :param detections: [batch_size, K, 10]
    # (scores x 1, xs x 1, ys x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
    # (scores-0:1, xs-1:2, ys-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
    :return:
    """
    # TODO: Need to consider rescale to the original scale: x, y

    ret = []
    for i in range(detections.shape[0]):
        top_preds = {}
        classes = detections[i, :, -1]
        for j in range(num_classes):
            inds = classes == j
            # x, y, z, h, w, l, yaw
            top_preds[j] = np.concatenate(
                [
                    detections[i, inds, 0:1],
                    detections[i, inds, 1:2] * down_ratio,
                    detections[i, inds, 2:3] * down_ratio,
                    detections[i, inds, 3:4],
                    detections[i, inds, 4:5],
                    detections[i, inds, 5:6] / discretization_coefficient,
                    detections[i, inds, 6:7] / discretization_coefficient,
                    get_yaw(detections[i, inds, 7:9]).astype(np.float32),
                ],
                axis=1,
            )

            # Filter by peak_thresh
            if len(top_preds[j]) > 0:
                keep_inds = top_preds[j][:, 0] > peak_thresh
                top_preds[j] = top_preds[j][keep_inds]

            # Workaround for vehicle-only detection
            if class_idx and j != class_idx:
                top_preds[j] = []

        ret.append(top_preds)

    return ret


def convert_det_to_real_values(
    detections: np.ndarray,
    discretization_coefficient: float = 50 / 608,
    x_min: float = 0.0,
    y_min: float = -25.0,
    z_min: float = -2.73,
    num_classes: int = 3,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    z_offset: float = 0.0,
    backwards: bool = False,
    rot_90: bool = False,
) -> np.ndarray:
    kitti_dets = []
    for cls_id in range(num_classes):
        if len(detections[cls_id]) > 0:
            for det in detections[cls_id]:
                # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                _score, _x, _y, _z, _h, _w, _l, _yaw = det
                _yaw = -_yaw
                x = _y * discretization_coefficient + x_min
                y = _x * discretization_coefficient + y_min
                z = _z + z_min
                w = _w * discretization_coefficient
                l = _l * discretization_coefficient
                x += x_offset
                y += y_offset
                z += z_offset

                if backwards:
                    kitti_dets.append(
                        [
                            _score,
                            cls_id,
                            x * -1,
                            y * -1,
                            z,
                            _h,
                            w,
                            l,
                            _yaw + np.deg2rad(180),
                        ]
                    )
                elif rot_90:
                    kitti_dets.append(
                        [_score, cls_id, -y, x, z, _h, w, l, _yaw - np.deg2rad(90)]
                    )
                else:
                    kitti_dets.append([_score, cls_id, x, y, z, _h, w, l, _yaw])

    return np.array(kitti_dets)


def get_yaw(direction):
    return np.arctan2(direction[:, 0:1], direction[:, 1:2])


def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()

    return heat * keep


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (torch.floor_divide(topk_inds, width)).float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (torch.floor_divide(topk_ind, K)).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
