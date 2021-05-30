import glob
import torch
from torch.utils.data import Dataset, DataLoader


def shapenet_voxels(path_to_data, batch_size=16, size=32):
    """ShapeNet voxel dataloader.

    Args:
        path_to_data (string): Path to ShapeNet voxel files.
        batch_size (int):
        size (int): Size of voxel cube side.
    """
    dataset = VoxelDataset(path_to_data, size)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def shapenet_point_clouds(path_to_data, batch_size=16):
    """ShapeNet point cloud dataloader.

    Args:
        path_to_data (string): Path to ShapeNet point cloud files.
        batch_size (int):
    """
    dataset = PointCloudDataset(path_to_data)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


class VoxelDataset(Dataset):
    """Three dimensional voxel datasets.

    Args:
        data_dir (torch.utils.Dataset): Path to directory where voxel data is
            stored.
        size (int): Size of voxel cube side.
        threshold (float): If interpolating, threshold to use to determine voxel
            occupancy. Works best with low values.
    """
    def __init__(self, data_dir, size=32, threshold=0.05):
        self.data_dir = data_dir
        self.voxel_paths = glob.glob(data_dir + "/*.pt")
        self.voxel_paths.sort()  # Ensure consistent ordering of voxels
        self.size = size
        self.threshold = threshold

    def __getitem__(self, index):
        # Shape (depth, height, width)
        voxels = torch.load(self.voxel_paths[index])
        # Unsqueeze to get shape (1, depth, height, width)
        voxels = voxels.unsqueeze(0)
        # Optionally resize
        if self.size != 32:
            # Need to add batch dimension for interpolate function
            voxels = torch.nn.functional.interpolate(voxels.unsqueeze(0).float(), self.size, 
                                                     mode='trilinear')[0]
            # Convert back to byte datatype
            voxels = voxels > self.threshold
        return voxels, 0  # Return unused label to match image datasets

    def __len__(self):
        return len(self.voxel_paths)


class PointCloudDataset(Dataset):
    """Three dimensional point cloud datasets. Each datapoint is a tensor of shape
    (num_points, 4), where the first 3 columns correspond to the x, y, z coordinates 
    of the point and the fourth column is a label (0 or 1) corresponding to whether
    the point is inside or outside the object.

    Args:
        data_dir (torch.utils.Dataset): Path to directory where point cloud data 
            is stored.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.point_cloud_paths = glob.glob(data_dir + "/*.pt")
        self.point_cloud_paths.sort()  # Ensure consistent ordering of point clouds

    def __getitem__(self, index):
        # Shape (num_points, 4)
        point_cloud = torch.load(self.point_cloud_paths[index])
        # Change coordinates [-.5, .5] -> [-1., 1.]
        point_cloud[:, :3] = 2. * point_cloud[:, :3] 
        return point_cloud, 0  # Return unused label to match image datasets

    def __len__(self):
        return len(self.point_cloud_paths)