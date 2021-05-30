import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointconv import PointConv, FeatureNonLinearity, FeatureBatchNorm, AvgPool


class PointConvDiscriminator(nn.Module):
    """
    Args:
        layer_configs (list of dicts): List of dictionaries, each specifying a pointconv
            layer. Must contain keys "out_channels", "num_output_points", "num_neighbors"
            and "mid_channels" if PointConv layer. Should *not* contain key "in_channels" 
            as this is predetermined. If AvgPool layer, should not contain key 
            "out_channels".
        linear_layer_sizes (list of ints): Specify size of hidden layers in linear layers
            applied after pointconv. Note the last element of this list must be 1 (since
            discriminator outputs a single scalar value).
    """
    def __init__(self, coordinate_dim, feature_dim, layer_configs, linear_layer_sizes=(),
                 non_linearity=nn.LeakyReLU(0.2), add_sigmoid=True, norm_order=2.0,
                 add_batchnorm=False, add_weightnet_batchnorm=False, deterministic=False,
                 same_coordinates=False):
        super(PointConvDiscriminator, self).__init__()
        self.coordinate_dim = coordinate_dim
        self.feature_dim = feature_dim
        self.linear_layer_sizes = linear_layer_sizes
        self.layer_configs = layer_configs
        self.add_sigmoid = add_sigmoid
        self.norm_order = norm_order
        self.add_batchnorm = add_batchnorm
        self.add_weightnet_batchnorm = add_weightnet_batchnorm
        self.deterministic = deterministic
        self.same_coordinates = same_coordinates

        # Ensure layers are in a module list so they are registered as learnable parameters
        self.layers = nn.ModuleList()
        in_channels = feature_dim  # Initial number of input channels is feature dimension
        for i, layer_config in enumerate(layer_configs):
            # If key "out_channels" is contained in dictionary, must be PointConv
            # layer, otherwise it is AvgPool layer
            if "out_channels" in layer_config:
                self.layers.append(PointConv(coordinate_dim=coordinate_dim, in_channels=in_channels, norm_order=norm_order,
                                             add_batchnorm=add_weightnet_batchnorm, deterministic=deterministic,
                                             same_coordinates=same_coordinates, **layer_config))
                # in_channels of next layer is out_channels of current layer
                in_channels = layer_config["out_channels"]

                # Don't add batchnorm or non linearity to final pointconv layer
                # if there are no subsequent linear layers
                if i == len(layer_configs) - 1 and len(linear_layer_sizes) == 0:
                    pass
                else:
                    if self.add_batchnorm:
                        self.layers.append(FeatureBatchNorm(nn.BatchNorm1d(layer_config["out_channels"])))
                    self.layers.append(FeatureNonLinearity(non_linearity))
            else:
                self.layers.append(AvgPool(norm_order=norm_order, deterministic=deterministic, 
                                           same_coordinates=same_coordinates, **layer_config))

        # Add linear layers
        if len(self.linear_layer_sizes):
            # Output size of pointconv layer has size (batch_size, num_output_points, out_channels)
            # As we flatten output, input size of first fully connected layer will be num_output_points * out_channels
            prev_num_units = layer_configs[-1]["num_output_points"] * in_channels
            linear_layers = []
            for i, num_units in enumerate(self.linear_layer_sizes):
                linear_layers.append(nn.Linear(prev_num_units, num_units))
                # If not last layer, apply non linearity to features
                if i != len(self.linear_layer_sizes) - 1:
                    linear_layers.append(non_linearity)
                prev_num_units = num_units
            self.linear_layers = nn.Sequential(*linear_layers)
        else:
            self.linear_layers = nn.Identity()

        # Output dim is size of last linear layer if there are linear layers, otherwise it is
        # out_channels * num_output_points of last convolution (i.e. dimension of flattened output
        # of convolution). Note that in_channels is set to size of last convolution
        # channel in loop when constructing model
        if len(self.linear_layer_sizes):
            self.output_dim = self.linear_layer_sizes[-1]
        else:
            self.output_dim = in_channels * self.layer_configs[-1]["num_output_points"]

    def forward(self, coordinates, features):
        """
        Args:
            coordinates (torch.Tensor): Shape (batch_size, num_points, coordinate_dim).
            features (torch.Tensor): Shape (batch_size, num_points, in_channels).
        """
        batch_size, _, _ = coordinates.shape
        # Apply PointConv layers
        for i, layer in enumerate(self.layers):
            coordinates, features = layer(coordinates, features)
        # Flatten output to apply linear layers
        features = features.view(batch_size, -1)
        features = self.linear_layers(features)
        if self.add_sigmoid:
            return torch.sigmoid(features)
        else:
            return features
