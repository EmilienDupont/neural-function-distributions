# Based on https://github.com/DylanWusee/pointconv_pytorch
import torch
import torch.nn as nn


def index_points(points, idx):
    """Returns subsamples of points tensor index by idx tensor.

    Args:
        points (torch.Tensor): Shape (batch_size, num_points, feature_dim)
        idx (torch.Tensor): Index tensor of shape (batch_size, num_output_points) OR
            (batch_size, num_output_points, num_neighbors).

    Returns:
        out_points (torch.Tensor): Indexed points, with shape (batch_size, num_output_points, feature_dim)
            OR (batch_size, num_output_points, num_neighbors, feature_dim).
    """
    batch_size = points.shape[0]
    # Create tensor of batch indices with shape (batch_size,)
    batch_indices = torch.arange(batch_size, dtype=torch.long).to(points.device)
    if len(idx.shape) == 2:
        # Repeat batch indices to match shape of idx tensor
        batch_indices = batch_indices.view((batch_size, 1)).repeat((1,) + idx.shape[1:])
    elif len(idx.shape) == 3:
        # Repeat batch indices to match shape of idx tensor
        batch_indices = batch_indices.view((batch_size, 1, 1)).repeat((1,) + idx.shape[1:])
    # Pytorch tensor indexing magic
    out_points = points[batch_indices, idx, :]
    return out_points


def farthest_point_sample(coordinates, num_output_points, norm_order=2., deterministic=False):
    """Subsamples coordinates by randomly selecting a first coordinate and then iteratively
    choosing the num_output_points - 1 remaining coordinates as the furthest from previously
    chosen coordinates.

    Args:
        coordinates (torch.Tensor): Shape (batch_size, num_points, coordinate_dim).
        num_output_points (int): Number of points to select among coordinates.
        norm_order (float): Order of the norm to use to measure distance. Defaults to L2 norm.
        deterministic (bool): If True, uses first point in batch as initial point, otherwise
            randomly samples initial point.

    Returns:
        subsampled_idx: Subsampled coordinate indices of shape (batch_size, num_output_points).
    """
    device = coordinates.device
    batch_size, num_points, coordinate_dim = coordinates.shape
    # Initialize indices of points to be selected
    subsampled_idx = torch.zeros(batch_size, num_output_points, dtype=torch.long).to(device)
    # Initialize distances to very large number
    distances = torch.ones(batch_size, num_points).to(device) * 1e10
    # Select initial "farthest" point
    if deterministic:
        # Select first point in batch as initial point
        farthest = torch.zeros(batch_size, dtype=torch.long).to(device)
    else:
        # Randomly select initial point for each batch batch element
        farthest = torch.randint(0, num_points, (batch_size,), dtype=torch.long).to(device)
    batch_indices = torch.arange(batch_size, dtype=torch.long).to(device)
    for i in range(num_output_points):
        # Set selected index to be farthest from previously selected indices
        subsampled_idx[:, i] = farthest
        # Select current farthest centroid
        centroid = coordinates[batch_indices, farthest, :].view(batch_size, 1, coordinate_dim)
        # Calculate distance from current centroid to all other points
        current_dist = torch.linalg.norm(coordinates - centroid, dim=-1, ord=norm_order)
        # Update distances with newly calculated distances
        mask = current_dist < distances
        distances[mask] = current_dist[mask]
        # Select farthest point with updated distances (note torch.max returns both values
        # and indices, so we select indices)
        farthest = torch.max(distances, dim=-1)[1]
    return subsampled_idx


def knn_point(num_neighbors, coordinates, subsampled_coordinates, norm_order=2.0):
    """Finds num_neighbors nearest neighbors of subsampled_coordinates among points
    in coordinates.

    Args:
        num_neighbors (int): Number of neighbors to sample in local region.
        coordinates (torch.Tensor): All coordinates, shape (batch_size, num_points, coordinate_dim).
        subsampled_coordinates: Subsampled points used as centers for which to find k-nearest neighbors.
            Has shape (batch_size, num_output_points, coordinate_dim)
        norm_order (float): Order of the norm to use to measure distance. Defaults to L2 norm.

    Returns:
        neighbors_idx: Indices of neighbors for each point. Shape (batch_size, num_output_points, num_neighbors)
    """
    # torch.cdist computes the (batched) distance between each pair of the two collections
    # of points. For input of shape (B, N, D) and (B, M, D) returns a tensor of shape
    # (B, N, M).
    distances = torch.cdist(subsampled_coordinates, coordinates, p=norm_order)
    # Returns indices of num_neighbors nearest points
    _, neighbors_idx = torch.topk(distances, num_neighbors, dim=-1, largest=False, sorted=False)
    return neighbors_idx


def sample_and_group(coordinates, features, num_output_points, num_neighbors, density=None, norm_order=2.0,
                     deterministic=False, same_coordinates=False):
    """
    Args:
        coordinates (torch.Tensor): Coordinates of input points.
            Shape (batch_size, num_points, coordinate_dim).
        features (torch.Tensor): Features of input points.
            Shape (batch_size, num_points, in_channels).
        num_output_points (int): Number of representative points to choose among input points.
        num_neighbors (int): Number of points to sample in neighborhood of each representative point.
            This is roughly equivalent to kernel size in regular convolution (i.e. using more neighbors
            corresponds to using a larger kernel).
        norm_order (float): Order of the norm to use to measure distance. Defaults to L2 norm.
        deterministic (bool):
        same_coordinates (bool): If the coordinates are the same across the batch, this should be set to
            True. If True, will only perform nearest neighbor search on a single element of batch. This
            can reduce compute time if there is a large number of points.

    Returns:
        out_coordinates (torch.Tensor): Shape (batch_size, num_output_points, coordinate_dim)
        group_features (torch.Tensor): Shape (batch_size, num_output_points, num_neighbors, in_channels)
        group_coordinates_centered (torch.Tensor): Shape (batch_size, num_output_points, num_neighbors, coordinate_dim)
    """
    if same_coordinates:
        # Select only first element of batch of coordinates to find nearest neighbors
        coordinates = coordinates[0:1]  # Shape (1, num_points, coordinate_dim)

    if num_output_points == coordinates.shape[1]:
        # If number of output points is the same as number of input points, no
        # need to subsample coordinates
        out_coordinates = coordinates
    else:
        # Sample num_output_points coordinates among the input coordinates to
        # obtain an index tensor of shape (batch_size, num_output_points)
        subsampled_idx = farthest_point_sample(coordinates, num_output_points, norm_order=norm_order,
                                               deterministic=deterministic)
        # Use subsampled index to choose sampled coordinates which will have shape
        # (batch_size, num_output_points, coordinate_dim)
        out_coordinates = index_points(coordinates, subsampled_idx)
    # For each output point, obtain its num_neighbors nearest neighbors to return
    # an index tensor of shape (batch_size, num_output_points, num_neighbors)
    neighbors_idx = knn_point(num_neighbors, coordinates, out_coordinates, norm_order=norm_order)
    # Use neighbors_idx to extract coordinates of nearest neighbors for each
    # output point to obtain tensor of shape
    # (batch_size, num_output_points, num_neighbors, coordinate_dim)
    group_coordinates = index_points(coordinates, neighbors_idx)
    # Center grouped_coordinates around their respective centroid
    batch_size, _, coordinate_dim = coordinates.shape
    group_coordinates_centered = group_coordinates - out_coordinates.view(batch_size, num_output_points, 1, coordinate_dim)

    if same_coordinates:
        batch_size = features.shape[0]
        # Repeat coordinates and neighbor tensors to restore total batch
        out_coordinates = out_coordinates.repeat(batch_size, 1, 1)
        neighbors_idx = neighbors_idx.repeat(batch_size, 1, 1)
        group_coordinates_centered = group_coordinates_centered.repeat(batch_size, 1, 1, 1)

    # Use neighbors_idx to extract features of nearest neighbors for each output
    # point to obtain tensor of shape
    # (batch_size, num_output_points, num_neighbors, feature_dim)
    group_features = index_points(features, neighbors_idx)

    if density is None:
        return out_coordinates, group_features, group_coordinates_centered
    else:
        group_density = index_points(density, neighbors_idx)
        return out_coordinates, group_features, group_coordinates_centered, group_density


class WeightNet(nn.Module):
    """MLP mapping coordinates to weights of convolution filters.

    Args:
        coordinate_dim (int):
        layer_sizes (list of ints): Sizes of layers in the MLP. Last layer should
            correspond to C_mid in the paper.
        add_batchnorm (bool): If True adds batchnorm.

    Notes:
        The MLPs are implemented as 1x1 convolutions, since we want to apply
        the same MLP individually to each point.
    """
    def __init__(self, coordinate_dim, layer_sizes=(16,), non_linearity=nn.LeakyReLU(0.2),
                 add_batchnorm=False):
        super(WeightNet, self).__init__()
        self.coordinate_dim = coordinate_dim
        self.layer_sizes = layer_sizes
        self.non_linearity = non_linearity
        self.add_batchnorm = add_batchnorm
        self._init_neural_net()

    def _init_neural_net(self):
        forward_layers = []
        prev_num_channels = self.coordinate_dim
        for num_channels in self.layer_sizes:
            # Implement pointwise MLP as 1x1 convolution
            forward_layers.append(nn.Conv2d(prev_num_channels, num_channels, 1))
            if self.add_batchnorm:
                forward_layers.append(nn.BatchNorm2d(num_channels))
            forward_layers.append(self.non_linearity)
            prev_num_channels = num_channels
        self.forward_layers = nn.Sequential(*forward_layers)

    def forward(self, centered_coordinates):
        """Maps centered coordinates to convolution filter values.

        Args:
            centered_coordinates (torch.Tensor): Shape (batch_size, coordinate_dim, num_neighbors, num_points).
        """
        return self.forward_layers(centered_coordinates)


class PointConv(nn.Module):
    """Single layer of sampling + grouping + PointConv.

    Args:
        coordinate_dim (int): Coordinate dimension of inputs.
        num_output_points (int): Number of representative points to choose as centers of the convolution.
            Each point corresponds to a convolution, so there will be num_output_points output by this
            layer. This is roughly analogous to the number of "pixels" in the outputs of a regular
            convolution.
        num_neighbors (int): Number of points to sample in neighborhood of each representative point.
            This is roughly equivalent to kernel size in regular convolution (i.e. using more neighbors
            corresponds to using a larger kernel).
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mid_channels (tuple of ints): Number of channels in hidden layers of MLP parameterizing the
            weights. Last entry of this tuple corresponds to C_mid from the paper.
        group_all (bool): If True, groups all points into a single point. If this is True, both
            num_output_points and num_neighbors will be ignored. Indeed, this corresponds to using
            num_output_points=1 and num_neighbors=num_input_points.
        norm_order (float): Order of the norm to use to measure distance. Defaults to L2 norm.
        add_batchnorm (bool): If True, adds batchnorm to WeightNet.
        deterministic (bool): If True, uses a deterministic algorithm to select query points, otherwise
            uses a random algorithm.
        same_coordinates (bool): If True, uses single batch trick to accelerate nearest neighbor
            computation. WARNING: if this is set to True and batch of coordinates passed to model
            does not contain the same coordinates, model will give garbage results.
    """
    def __init__(self, coordinate_dim, num_output_points, num_neighbors, in_channels, out_channels,
                 mid_channels=(16,), group_all=False, norm_order=2.0, add_batchnorm=False, 
                 deterministic=False, same_coordinates=False):
        super(PointConv, self).__init__()
        self.num_output_points = num_output_points
        self.num_neighbors = num_neighbors
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.group_all = group_all
        self.norm_order = norm_order
        self.add_batchnorm = add_batchnorm
        self.deterministic = deterministic
        self.same_coordinates = same_coordinates

        # Neural net parameterizing the convolution kernels. It takes as input coordinates, e.g.
        # (x, y) and returns the kernel value at that point. Note that since we are using the
        # efficient PointConv trick, this neural net actually outputs the middle (hidden) layer
        # of this MLP
        self.weight_net = WeightNet(coordinate_dim, mid_channels, add_batchnorm=add_batchnorm)
        # Final linear layer that maps the intermediate representation to the final output
        # features using the efficient PointConv trick
        self.linear = nn.Linear(mid_channels[-1] * in_channels, out_channels)

    def forward(self, coordinates, features):
        """
        Args:
            coordinates (torch.Tensor): Shape (batch_size, num_points, coordinate_dim).
            features (torch.Tensor): Shape (batch_size, num_points, in_channels).

        Returns:
            out_coordinates (torch.Tensor): Shape (batch_size, num_output_points, coordinate_dim)
            out_features (torch.Tensor): Shape (batch_size, num_output_points, out_channels)
        """
        batch_size, num_points, _ = coordinates.shape

        # Subsample points
        if self.group_all:
            # If using group_all, group all points into a single point
            out_coordinates, out_features, group_coordinates_centered = sample_and_group(coordinates, features,
                                                                                         num_output_points=1,
                                                                                         num_neighbors=coordinates.shape[1],
                                                                                         norm_order=self.norm_order,
                                                                                         deterministic=self.deterministic)
        else:
            # out_coordinates shape (batch_size, num_output_points, coordinate_dim)
            # out_features shape (batch_size, num_output_points, num_neighbors, in_channels)
            # For each output point, find its nearest neighbors and center them around output point
            # group_coordinates_centered shape (batch_size, num_output_points, num_neighbors, coordinate_dim)
            out_coordinates, out_features, group_coordinates_centered = sample_and_group(coordinates, features,
                                                                                         self.num_output_points, self.num_neighbors,
                                                                                         norm_order=self.norm_order,
                                                                                         deterministic=self.deterministic,
                                                                                         same_coordinates=self.same_coordinates)

        # We change this shape so we can apply MLP on each point, which for a 1x1 conv means we apply the
        # same function to each pixel. Therefore, move all point dimensions to last two dimensions (which
        # correspond to height and width as required).
        # Note all of this could be abstracted into the forward of the weightnet
        # Shape (batch_size, num_output_points, num_neighbors, coordinate_dim) -> (batch_size, coordinate_dim, num_neighbors, num_output_points)
        group_coordinates_centered = group_coordinates_centered.permute(0, 3, 2, 1)
        # weights has shape (batch_size, mid_channels, num_neighbors, num_output_points)
        weights = self.weight_net(group_coordinates_centered)

        # For matrix multiplication (which will act on last two dimensions) we need to change
        # out_features from shape (batch_size, num_output_points, num_neighbors, in_channels) to
        # (batch_size, num_output_points, in_channels, num_neighbors) and weights from shape
        # (batch_size, mid_channels, num_neighbors, num_output_points) to shape
        # (batch_size, num_output_points, num_neighbors, mid_channels). The matrix multiplication
        # will then yield (in_channels, num_neighbors) * (num_neighbors, mid_channels) = (in_channels, mid_channels)
        # The resulting out_features will have shape (batch_size, num_output_points, in_channels, mid_channels)
        # which we flatten to (batch_size, num_output_points, in_channels * mid_channels)
        out_features = torch.matmul(input=out_features.permute(0, 1, 3, 2),
                                    other=weights.permute(0, 3, 2, 1)).view(batch_size, self.num_output_points, -1)

        # Shape of out_features is (batch_size, num_output_points, mid_channels * in_channels)
        # We apply a linear layer (which only acts on the last dimension and treats everything else as "batch_size") to obtain
        # out_features of shape (batch_size, num_output_points, out_channels)
        out_features = self.linear(out_features)

        return out_coordinates, out_features


class AvgPool(nn.Module):
    """Average pooling layer for point clouds.

    Args:
        num_output_points (int): Number of query points to choose as centers of
            pooling operations.
        num_neighbors (int): Number of points to sample in neighborhood of each
            query point.
        norm_order (float): Order of the norm to use to measure distance.
            Defaults to L2 norm.
        deterministic (bool): If True, uses a deterministic algorithm to select
            query points, otherwise uses a random algorithm.
        same_coordinates (bool): If True, uses single batch trick to accelerate
            nearest neighbor computation. WARNING: if this is set to True and
            batch of coordinates passed to model does not contain the same
            coordinates, model will give garbage results.
    """
    def __init__(self, num_output_points, num_neighbors, norm_order=2.0,
                 deterministic=False, same_coordinates=False):
        super(AvgPool, self).__init__()
        self.num_output_points = num_output_points
        self.num_neighbors = num_neighbors
        self.norm_order = norm_order
        self.deterministic = deterministic
        self.same_coordinates = same_coordinates

    def forward(self, coordinates, features):
        """
        """
        # out_coordinates shape (batch_size, num_output_points, coordinate_dim)
        # out_features shape (batch_size, num_output_points, num_neighbors, in_channels)
        # For each output point, find its nearest neighbors and center them around output point
        # group_coordinates_centered shape (batch_size, num_output_points, num_neighbors, coordinate_dim)
        out_coordinates, out_features, group_coordinates_centered = sample_and_group(coordinates, features,
                                                                                     self.num_output_points, self.num_neighbors,
                                                                                     norm_order=self.norm_order,
                                                                                     deterministic=self.deterministic,
                                                                                     same_coordinates=self.same_coordinates)
        # Take mean of features over neighbors for each query point, shape
        # (batch_size, num_output_points, num_neighbors, feature_dim) ->
        # (batch_size, num_output_points, feature_dim)
        out_features = out_features.mean(dim=2)
        return out_coordinates, out_features


class FeatureNonLinearity(nn.Module):
    """Small wrapper to apply non linearities to features.
    """
    def __init__(self, non_linearity):
        super(FeatureNonLinearity, self).__init__()
        self.non_linearity = non_linearity

    def forward(self, coordinates, features):
        return coordinates, self.non_linearity(features)


class FeatureBatchNorm(nn.Module):
    """
    """
    def __init__(self, batchnorm):
        super(FeatureBatchNorm, self).__init__()
        self.batchnorm = batchnorm

    def forward(self, coordinates, features):
        # Batchnorm expects channel dimension to be the second dimension
        # So permute (batch_size, num_points, in_channels) ->
        # (batch_size, in_channels, num_points) apply batchnorm and then
        # permute back
        bn_features = self.batchnorm(features.permute(0, 2, 1)).permute(0, 2, 1)
        return coordinates, bn_features
