import numpy as np
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


class ERA5Converter():
    """Module used to convert ERA5 data to spherical coordinates and features.

    Args:
        data_shape (tuple of ints): Tuple of the form (num_lats, num_lons).
        normalize (bool): This argument is only kept for compatibility.
            Coordinates will always lie in [-1, 1] since we use spherical
            coordinates with r=1.
        normalize_features (bool): If True normalizes features (e.g. temperature
            values) to lie in [-1, 1]. This assumes features from the dataloader
            lie in [0, 1].

    Notes:
        We assume the spherical data is given as a tensor of shape
        (3, num_lats, num_longs), where the first dimension contains latitude
        values, the second dimension longitude values and the third dimension
        temperature values.

        The coordinates are given by:
            x = cos(latitude) cos(longitude)
            y = cos(latitude) sin(longitude)
            z = sin(latitude).
    """
    def __init__(self, device, data_shape, normalize=True,
                 normalize_features=False):
        self.device = device
        self.data_shape = data_shape
        self.normalize = normalize
        self.normalize_features = normalize_features
        # Initialize coordinates
        self.latitude = np.linspace(90., -90., data_shape[0])
        self.longitude = np.linspace(0., 360. - (360. / data_shape[1]),
                                     data_shape[1])
        # Create a grid of latitude and longitude values (num_lats, num_lons)
        longitude_grid, latitude_grid = np.meshgrid(self.longitude,
                                                    self.latitude)
        # Shape (3, num_lats, num_lons) (add bogus temperature dimension to be
        # compatible with coordinates and features transformation function)
        data_tensor = np.stack([latitude_grid,
                                longitude_grid,
                                np.zeros_like(longitude_grid)])
        data_tensor = torch.Tensor(data_tensor).to(device)
        # Shape (num_lats, num_lons, 3)
        self.coordinates, _ = era5_to_coordinates_and_features(data_tensor)
        # (num_lats, num_lons, 3) -> (num_lats * num_lons, 3)
        self.coordinates = self.coordinates.view(-1, 3)
        # Store to use when converting to from coordinates and features to data
        self.latitude_grid = torch.Tensor(latitude_grid).to(device)
        self.longitude_grid = torch.Tensor(longitude_grid).to(device)

    def to_coordinates_and_features(self, data):
        """Given a datapoint convert to coordinates and features at each
        coordinate.

        Args:
            data (torch.Tensor): Shape (3, num_lats, num_lons) where latitudes
                and longitudes are in degrees and temperatures are in [0, 1].
        """
        # Shapes (num_lats, num_lons, 3), (num_lats, num_lons, 1)
        coordinates, features = era5_to_coordinates_and_features(data)
        if self.normalize_features:
            # Features are in [0, 1], convert to [-1, 1]
            features = 2. * features - 1.
        # Flatten features and coordinates
        # (num_lats, num_lons, 1) -> (num_lats * num_lons, 1)
        features = features.view(-1, 1)
        # (num_lats, num_lons, 3) -> (num_lats * num_lons, 3)
        coordinates = coordinates.view(-1, 3)
        return coordinates, features

    def batch_to_coordinates_and_features(self, data_batch):
        """Given a batch of datapoints, convert to coordinates and features at
        each coordinate.

        Args:
            data_batch (torch.Tensor): Shape (batch_size, 3, num_lats, num_lons)
                where latitudes and longitudes are in degrees and temperatures
                are in [0, 1].
        """
        batch_size = data_batch.shape[0]
        # Shapes (batch_size, num_lats, num_lons, 3), (batch_size, num_lats, num_lons, 1)
        coordinates_batch, features_batch = era5_to_coordinates_and_features(data_batch)
        if self.normalize_features:
            # Image features are in [0, 1], convert to [-1, 1]
            features_batch = 2. * features_batch - 1.
        # Flatten features and coordinates
        # (batch_size, num_lats, num_lons, 1) -> (batch_size, num_lats * num_lons, 1)
        features_batch = features_batch.view(batch_size, -1, 1)
        # (batch_size, num_lats, num_lons, 3) -> (batch_size, num_lats * num_lons, 3)
        coordinates_batch = coordinates_batch.view(batch_size, -1, 3)
        return coordinates_batch, features_batch

    def to_data(self, coordinates, features, resolution=None):
        """Converts tensors of features and coordinates to ERA5 data.

        Args:
            coordinates (torch.Tensor): Unused argument.
            features (torch.Tensor): Shape (num_lats * num_lons, 1).
            resolution (tuple of ints): Unused argument.

        Notes:
            Since we don't use subsampling or superresolution for ERA5
            data, this function ignores passed coordinates tensor and
            assumes we use self.coordinates.
        """
        if self.normalize_features:
            # [-1, 1] -> [0, 1]
            features = .5 * (features + 1.)
        # Reshape features (num_lats * num_lons, 1) -> (1, num_lats, num_lons)
        features = features.view(1, *self.data_shape)
        # Shape (3, num_lats, num_lons)
        return torch.cat([self.latitude_grid.unsqueeze(0),
                          self.longitude_grid.unsqueeze(0),
                          features], dim=0)

    def batch_to_data(self, coordinates, features):
        """Converts tensor of batch features to point cloud representation.

        Args:
            coordinates (torch.Tensor): Unused argument.
            features (torch.Tensor): Shape (batch_size, num_lats, num_lons, 1).
        """
        batch_size = features.shape[0]
        if self.normalize_features:
            # [-1, 1] -> [0, 1]
            features = .5 * (features + 1.)
        # Reshape features (batch_size, num_lats * num_lons, 1) -> (batch_size, 1, num_lats, num_lons)
        features = features.view(batch_size, 1, *self.data_shape)
        # Shape (batch_size, 1, num_lats, num_lons)
        batch_lat_grid = self.latitude_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        batch_lon_grid = self.longitude_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        # Shape (batch_size, 3, num_lats, num_lons)
        return torch.cat([batch_lat_grid, batch_lon_grid, features], dim=1)

    def unnormalized_coordinates(self, coordinates):
        """
        """
        unnormalized_coordinates = coordinates / 2 + 0.5
        return unnormalized_coordinates * (self.data_shape[1] - 1)

    def superresolve_coordinates(self, resolution):
        """Not implemented for spherical data."""
        raise NotImplementedError


def era5_to_coordinates_and_features(data, use_spherical=True):
    """
    Converts ERA5 data lying on the globe to spherical coordinates and features.
    The coordinates are given by:
        x = cos(latitude) cos(longitude)
        y = cos(latitude) sin(longitude)
        z = sin(latitude).
    The features are temperatures.

    Args:
        data (torch.Tensor): Tensor of shape ({batch,} 3, num_lats, num_lons)
            as returned by the ERA5 dataloader (batch dimension optional).
            The first dimension contains latitudes, the second longitudes
            and the third temperatures.
        use_spherical (bool): If True, uses spherical coordinates, otherwise
            uses normalized latitude and longitude directly.

    Returns:
        Tuple of coordinates and features where coordinates has shape
        ({batch,} num_lats, num_lons, 2 or 3) and features has shape
        ({batch,} num_lats, num_lons, 1).
    """
    assert data.ndim in (3, 4)

    if data.ndim == 3:
        latitude, longitude, temperature = data
    elif data.ndim == 4:
        latitude, longitude, temperature = data[:, 0], data[:, 1], data[:, 2]

    # Create coordinate tensor
    if use_spherical:
        coordinates = torch.zeros(latitude.shape + (3,)).to(latitude.device)
        long_rad = deg_to_rad(longitude)
        lat_rad = deg_to_rad(latitude)
        coordinates[..., 0] = torch.cos(lat_rad) * torch.cos(long_rad)
        coordinates[..., 1] = torch.cos(lat_rad) * torch.sin(long_rad)
        coordinates[..., 2] = torch.sin(lat_rad)
    else:
        coordinates = torch.zeros(latitude.shape + (2,)).to(latitude.device)
        # Longitude [0, 360] -> [-1, 1]
        coordinates[..., 0] = longitude / 180. - 1.
        # Latitude [-90, 90] -> [-.5, .5]
        coordinates[..., 1] = latitude / 180.
    # Feature tensor is given by temperatures (unsqueeze to ensure we have
    # feature dimension)
    features = temperature.unsqueeze(-1)

    return coordinates, features


def deg_to_rad(degrees):
    return np.pi * degrees / 180.


def rad_to_deg(radians):
    return 180. * radians / np.pi


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
