import pcl
import torch
import numpy as np


def rasterize_bev_pillars(
    point_cloud: torch.Tensor,
    discretization_coefficient: float = 50 / 608,
    bev_height: int = 608,
    bev_width: int = 608,
    z_max: float = 1.27,
    z_min: float = -2.73,
    n_lasers: int = 128,
    device: str = "cuda",
) -> torch.Tensor:
    """Optimized PyTorch version of kitti_bev_utils.makeBEVMap"""
    height = bev_height + 1
    width = bev_width + 1

    _point_cloud = torch.clone(point_cloud)  # Local copy required?
    _point_cloud = _point_cloud.to(device, dtype=torch.float32)

    # Discretize x and y coordinates
    _point_cloud[:, 0] = torch.floor(_point_cloud[:, 0] / discretization_coefficient)
    _point_cloud[:, 1] = torch.floor(_point_cloud[:, 1] / discretization_coefficient)

    # Get unique indices to rasterize and unique counts to compute the point density
    _, unique_indices, _, unique_counts = unique_torch(_point_cloud[:, 0:2], dim=0)
    _point_cloud_top = _point_cloud[unique_indices]

    # Intensity, height and density maps
    intensity_map = torch.zeros((height, width), dtype=torch.float32, device=device)
    height_map = torch.zeros((height, width), dtype=torch.float32, device=device)
    density_map = torch.zeros((height, width), dtype=torch.float32, device=device)

    x_indices = _point_cloud_top[:, 0].int()
    y_indices = _point_cloud_top[:, 1].int()

    intensity_map[x_indices, y_indices] = _point_cloud_top[:, 3]

    max_height = np.float32(np.abs(z_max - z_min))
    height_map[x_indices, y_indices] = _point_cloud_top[:, 2] / max_height

    normalized_counts = torch.log(unique_counts + 1) / np.log(n_lasers)
    normalized_counts = torch.clamp(normalized_counts, min=0.0, max=1.0)
    density_map[x_indices, y_indices] = normalized_counts

    ihd_map = torch.zeros(
        (3, bev_height, bev_width), dtype=torch.float32, device=device
    )
    ihd_map[0, :, :] = intensity_map[:bev_height, :bev_width]
    ihd_map[1, :, :] = height_map[:bev_height, :bev_width]
    ihd_map[2, :, :] = density_map[:bev_height, :bev_width]

    return ihd_map


def unique_torch(x: torch.Tensor, dim: int = 0) -> tuple:
    # Src: https://github.com/pytorch/pytorch/issues/36748
    unique, inverse, counts = torch.unique(
        x, dim=dim, sorted=True, return_inverse=True, return_counts=True
    )
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]

    return unique, index, inverse, counts


def filter_point_cloud(
    point_cloud: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    mask = np.where(
        (point_cloud[:, 0] >= x_min)
        & (point_cloud[:, 0] <= x_max)
        & (point_cloud[:, 1] >= y_min)
        & (point_cloud[:, 1] <= y_max)
        & (point_cloud[:, 2] >= z_min)
        & (point_cloud[:, 2] <= z_max)
    )
    point_cloud = point_cloud[mask]

    return point_cloud


def preprocess_point_cloud(pc: pcl.PointCloud) -> np.ndarray:
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