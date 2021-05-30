import torch


class GridDataConverter():
    """Module used to convert grid data to coordinates and features.

    Args:
        data_shape (tuple of ints): Tuple of the form (feature_dim,
            coordinate_dim_1, coordinate_dim_2, ...). For example, for an
            image this would be (feature_dim, height, width). For a voxel grid this
            would be (1, depth, height, width).
        normalize (bool): If True normalizes coordinates to lie in [-1, 1].
        normalize_features (bool): If True normalizes features (i.e. RGB values)
            to lie in [-1, 1].
    """
    def __init__(self, device, data_shape, normalize=True,
                 normalize_features=False):
        self.device = device
        self.data_shape = data_shape
        self.normalize = normalize
        self.normalize_features = normalize_features
        self.coordinate_dim = len(data_shape[1:])
        # Since first dimension of data_shape corresponds to feature dimension,
        # only consider size of coordinate dimensions to determine coordinates
        # Tensor.nonzero() returns a tensor of shape (num_points, num_coordinates)
        # with the coordinates of the data. For example for an image of size
        # (3, 32, 32), this would return a (32 * 32, 2) dimensional tensor with
        # entries (0, 0), (0, 1), (0, 2), ..., (0, 31), (1, 0), (1, 1), ...
        self.coordinates = torch.ones(data_shape[1:]).nonzero(as_tuple=False).float().to(device)
        # Optionally normalize coordinates to lie in [-1, 1]
        if self.normalize:
            self.coordinates = normalize_coordinates(self.coordinates, data_shape[1])

    def to_coordinates_and_features(self, data):
        """Given a datapoint (e.g. an image), convert to coordinates and
        features at each coordinate.

        Args:
            data (torch.Tensor): Shape self.data_shape.
        """
        # This will be a tensor of shape (num_points, feature_dim)
        features = data.view(data.shape[0], -1).T
        if self.normalize_features:
            # Image features are in [0, 1], convert to [-1, 1]
            features = 2. * features - 1.
        return self.coordinates, features

    def batch_to_coordinates_and_features(self, data_batch):
        """Given a batch of datapoints (e.g. images), converts to coordinates
        and features at each coordinate.

        Args:
            data_batch (torch.Tensor): Shape (batch_size,) + self.data_shape.
        """
        batch_size, feature_dim = data_batch.shape[0], data_batch.shape[1]
        # This will be a tensor of shape (batch_size, feature_dim, num_points)
        features_batch = data_batch.view(batch_size, feature_dim, -1)
        # This will be a tensor of shape (batch_size, num_points, feature_dim)
        features_batch = features_batch.transpose(2, 1)
        if self.normalize_features:
            # Image features are in [0, 1], convert to [-1, 1]
            features_batch = 2. * features_batch - 1.
        # This will have shape (batch_size, num_points, coordinate_dim)
        coordinates_batch = self.coordinates.unsqueeze(0).repeat(batch_size, 1, 1)
        return coordinates_batch, features_batch

    def to_data(self, coordinates, features, resolution=None):
        """Converts tensor of features to grid data representation.

        Args:
            coordinates (torch.Tensor): Unused argument.
            features (torch.Tensor): Shape (num_points, feature_dim).
            resolution (tuple of ints): Resolution at which feature vector was
                sampled. If None returns default resolution. As an example,
                for images, we could set resolution = (64, 64).
        """
        if self.normalize_features:
            # [-1, 1] -> [0, 1]
            features = .5 * (features + 1.)
        if resolution is None:
            return features.T.view(self.data_shape)
        else:
            return self._superresolution_to_data(features, resolution)

    def batch_to_data(self, coordinates, features):
        """Converts tensor of batch of features to grid data representation.

        Args:
            coordinates (torch.Tensor): Unused argument.
            features (torch.Tensor): Shape (batch_size, num_points, feature_dim).
        """
        if self.normalize_features:
            # [-1, 1] -> [0, 1]
            features = .5 * (features + 1.)
        batch_size, _, feature_dim = features.shape
        # (batch_size, num_points, feature_dim) -> (batch_size, *coordinate_dims, feature_dim)
        features = features.view(batch_size, *self.data_shape[1:], feature_dim)
        # (batch_size, *coordinate_dims, feature_dim) -> (batch_size, feature_dim, *coordinate_dims)
        permutation = (0, -1) + tuple(range(1, self.num_coordinate_dims + 1))
        return features.permute(*permutation)

    def unnormalized_coordinates(self, coordinates):
        """
        """
        unnormalized_coordinates = coordinates / 2 + 0.5
        return unnormalized_coordinates * (self.data_shape[1] - 1)

    def superresolve_coordinates(self, resolution):
        """Returns coordinates at a given resolution.

        Args:
            resolution (tuple of ints): Resolution at which to return
                coordinates.
        """
        superresolution_coordinates = torch.ones(resolution).nonzero(as_tuple=False).float().to(self.device)
        max_coordinate = resolution[0]  # Always normalize by first dimension
        if self.normalize:
            superresolution_coordinates = normalize_coordinates(superresolution_coordinates,
                                                                max_coordinate)
        return superresolution_coordinates

    def _superresolution_to_data(self, features, resolution):
        """Converts tensor of features to traditional data representation.

        Args:
            features (torch.Tensor): Shape (num_points, feature_dim).
            resolution (tuple of ints): Resolution at which feature vector was
                sampled. If None returns default resolution.
        """
        data_shape = (self.data_shape[0],) + resolution
        return features.T.view(data_shape)


class PointCloudDataConverter():
    """Module used to convert point cloud to coordinates and features.

    Args:
        data_shape (tuple of ints): Tuple of the form (feature_dim,
            coordinate_dim_1, coordinate_dim_2, ...). While point
            clouds do not have a data_shape this will be used when sampling
            points on grid to generate samples.
        normalize (bool): If True normalizes coordinates to lie in [-1, 1].
        normalize_features (bool): If True normalizes features (e.g. RGB or occupancy
            values to lie in [-1, 1].
    
    Notes:
        We assume point cloud is given as a tensor of shape (num_points, 4),
        where the first 3 columns correspond to (x, y, z) locations and the
        last column corresponds to a binary occupancy value.
    """
    def __init__(self, device, data_shape, normalize=True,
                 normalize_features=False):
        self.device = device
        self.data_shape = data_shape
        self.normalize = normalize
        self.normalize_features = normalize_features
        self.coordinate_dim = len(data_shape[1:])
        self.coordinates = torch.ones(data_shape[1:]).nonzero(as_tuple=False).float().to(device)
        # Optionally normalize so coordinates lie in [-1, 1]
        if self.normalize:
            self.coordinates = normalize_coordinates(self.coordinates, data_shape[1])

    def to_coordinates_and_features(self, data):
        """Given a datapoint convert to coordinates and features at each 
        coordinate.

        Args:
            data (torch.Tensor): Shape (num_points, 4), where first 3 columns
                corresponds to spatial location in [-0.5, 0.5].
        """
        coordinates = data[:, :3]  # Shape (num_points, 3)
        features = data[:, 3:]  # Shape (num_points, 1)
        if self.normalize_features:
            # Features are in [0, 1], convert to [-1, 1]
            features = 2. * features - 1.
        return coordinates, features

    def batch_to_coordinates_and_features(self, data_batch):
        """Given a batch of point clouds converts to coordinates
        and features at each coordinate.

        Args:
            data_batch (torch.Tensor): Shape (batch_size, num_points, 4).
        """
        # Shape (batch_size, num_points, 3)
        coordinates_batch = data_batch[:, :, :3]
        # Shape (batch_size, num_points, 1)
        features_batch = data_batch[:, :, 3:]
        if self.normalize_features:
            # Image features are in [0, 1], convert to [-1, 1]
            features_batch = 2. * features_batch - 1.
        return coordinates_batch, features_batch

    def to_data(self, coordinates, features, resolution=None):
        """Converts tensor of features to point cloud representation.

        Args:
            coordinates (torch.Tensor): Shape (num_points, 3).
            features (torch.Tensor): Shape (num_points, 1).
            resolution (tuple of ints): Unused argument.
        """
        if self.normalize_features:
            # [-1, 1] -> [0, 1]
            features = .5 * (features + 1.)
        return torch.cat([coordinates, features], dim=-1)

    def batch_to_data(self, coordinates, features):
        """Converts tensor of batch features to point cloud representation.

        Args:
            coordinates (torch.Tensor): Shape (batch_size, num_points, 3)
            features (torch.Tensor): Shape (batch_size, num_points, 1).
        """
        return self.to_data(coordinates, features)

    def unnormalized_coordinates(self, coordinates):
        """
        """
        unnormalized_coordinates = coordinates / 2 + 0.5
        return unnormalized_coordinates * (self.data_shape[1] - 1)

    def superresolve_coordinates(self, resolution):
        """Returns coordinates at a given resolution.

        Args:
            resolution (tuple of ints): Resolution at which to return
                coordinates.
        """
        superresolution_coordinates = torch.ones(resolution).nonzero(as_tuple=False).float().to(self.device)
        max_coordinate = resolution[0]  # Always normalize by first dimension
        if self.normalize:
            superresolution_coordinates = normalize_coordinates(superresolution_coordinates,
                                                                max_coordinate)
        return superresolution_coordinates


def normalize_coordinates(coordinates, max_coordinate):
    """Normalizes coordinates to [-1, 1] range.

    Args:
        coordinates (torch.Tensor):
        max_coordinate (float): Maximum coordinate in original grid.
    """
    # Get points in range [-0.5, 0.5]
    normalized_coordinates = coordinates / (max_coordinate - 1) - 0.5
    # Convert to range [-1, 1]
    normalized_coordinates *= 2
    return normalized_coordinates
