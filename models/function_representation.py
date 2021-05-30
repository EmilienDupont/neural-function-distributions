import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FunctionRepresentation(nn.Module):
    """Function to represent a single datapoint. For example this could be a
    function that takes pixel coordinates as input and returns RGB values, i.e.
    f(x, y) = (r, g, b).

    Args:
        coordinate_dim (int): Dimension of input (coordinates).
        feature_dim (int): Dimension of output (features).
        layer_sizes (tuple of ints): Specifies size of each hidden layer.
        encoding (torch.nn.Module): Encoding layer, usually one of
            Identity or FourierFeatures.
        final_non_linearity (torch.nn.Module): Final non linearity to use.
            Usually nn.Sigmoid() or nn.Tanh().
    """
    def __init__(self, coordinate_dim, feature_dim, layer_sizes, encoding,
                 non_linearity=nn.ReLU(), final_non_linearity=nn.Sigmoid()):
        super(FunctionRepresentation, self).__init__()
        self.coordinate_dim = coordinate_dim
        self.feature_dim = feature_dim
        self.layer_sizes = layer_sizes
        self.encoding = encoding
        self.non_linearity = non_linearity
        self.final_non_linearity = final_non_linearity

        self._init_neural_net()

    def _init_neural_net(self):
        """
        """
        # First layer transforms coordinates into a positional encoding
        # Check output dimension of positional encoding
        if isinstance(self.encoding, nn.Identity):
            prev_num_units = self.coordinate_dim  # No encoding, so same output dimension
        else:
            prev_num_units = self.encoding.feature_dim
        # Build MLP layers
        forward_layers = []
        for num_units in self.layer_sizes:
            forward_layers.append(nn.Linear(prev_num_units, num_units))
            forward_layers.append(self.non_linearity)
            prev_num_units = num_units
        forward_layers.append(nn.Linear(prev_num_units, self.feature_dim))
        forward_layers.append(self.final_non_linearity)
        self.forward_layers = nn.Sequential(*forward_layers)

    def forward(self, coordinates):
        """Forward pass. Given a set of coordinates, returns feature at every
        coordinate.

        Args:
            coordinates (torch.Tensor): Shape (batch_size, coordinate_dim)
        """
        encoded = self.encoding(coordinates)
        return self.forward_layers(encoded)

    def get_weight_shapes(self):
        """Returns lists of shapes of weights and biases in the network."""
        weight_shapes = []
        bias_shapes = []
        for param in self.forward_layers.parameters():
            if len(param.shape) == 1:
                bias_shapes.append(param.shape)
            if len(param.shape) == 2:
                weight_shapes.append(param.shape)
        return weight_shapes, bias_shapes

    def get_weights_and_biases(self):
        """Returns list of weights and biases in the network."""
        weights = []
        biases = []
        for param in self.forward_layers.parameters():
            if len(param.shape) == 1:
                biases.append(param)
            if len(param.shape) == 2:
                weights.append(param)
        return weights, biases

    def set_weights_and_biases(self, weights, biases):
        """Sets weights and biases in the network.

        Args:
            weights (list of torch.Tensor):
            biases (list of torch.Tensor):

        Notes:
            The inputs to this function should have the same form as the outputs
            of self.get_weights_and_biases.
        """
        weight_idx = 0
        bias_idx = 0
        with torch.no_grad():
            for param in self.forward_layers.parameters():
                if len(param.shape) == 1:
                    param.copy_(biases[bias_idx])
                    bias_idx += 1
                if len(param.shape) == 2:
                    param.copy_(weights[weight_idx])
                    weight_idx += 1

    def duplicate(self):
        """Returns a FunctionRepresentation instance with random weights."""
        # Extract device
        device = next(self.parameters()).device
        # Create new function representation and put it on same device
        return FunctionRepresentation(self.coordinate_dim, self.feature_dim,
                                      self.layer_sizes, self.encoding,
                                      self.non_linearity,
                                      self.final_non_linearity).to(device)

    def sample_grid(self, data_converter, resolution=None):
        """Returns function values evaluated on grid.

        Args:
            data_converter (data.conversion.DataConverter):
            resolution (tuple of ints): Resolution of grid on which to evaluate
                features. If None uses default resolution.
        """
        # Predict features at every coordinate in a grid
        if resolution is None:
            coordinates = data_converter.coordinates
        else:
            coordinates = data_converter.superresolve_coordinates(resolution)
        features = self(coordinates)
        # Convert features into appropriate data format (e.g. images)
        return data_converter.to_data(coordinates, features, resolution)

    def stateless_forward(self, coordinates, weights, biases):
        """Computes forward pass of function representation given a set of
        weights and biases without using the state of the PyTorch module.

        Args:
            coordinates (torch.Tensor): Tensor of shape (num_points, coordinate_dim).
            weights (list of torch.Tensor): List of tensors containing weights
                of linear layers of neural network.
            biases (list of torch.Tensor): List of tensors containing biases of
                linear layers of neural network.

        Notes:
            This is useful for computing forward pass for a specific function
            representation (i.e. for a given set of weights and biases). However,
            it might be easiest to just change the weights of the network directly
            and then perform forward pass.
            Doing the current way is definitely more error prone because we have
            to mimic the forward pass, instead of just directly using it.

        Return:
            Returns a tensor of shape (num_points, feature_dim)
        """
        # Positional encoding is first layer of function representation
        # model, so apply this transformation to coordinates
        hidden = self.encoding(coordinates)
        # Apply linear layers and non linearities
        for i in range(len(weights)):
            hidden = F.linear(hidden, weights[i], biases[i])
            if i == len(weights) - 1:
                hidden = self.final_non_linearity(hidden)
            else:
                hidden = self.non_linearity(hidden)
        return hidden

    def batch_stateless_forward(self, coordinates, weights, biases):
        """Stateless forward pass for multiple function representations.

        Args:
            coordinates (torch.Tensor): Batch of coordinates of shape
                (batch_size, num_points, coordinate_dim).
            weights (dict of list of torch.Tensor): Batch of list of tensors
                containing weights of linear layers for each neural network.
            biases (dict of list of torch.Tensor): Batch of list of tensors
                containing biases of linear layers for each neural network.

        Return:
            Returns a tensor of shape (batch_size, num_points, feature_dim).
        """
        features = []
        for i in range(coordinates.shape[0]):
            features.append(
                self.stateless_forward(coordinates[i], weights[i], biases[i]).unsqueeze(0)
            )
        return torch.cat(features, dim=0)

    def _get_config(self):
        return {"coordinate_dim": self.coordinate_dim,
                "feature_dim": self.feature_dim,
                "layer_sizes": self.layer_sizes,
                "encoding": self.encoding,
                "non_linearity": self.non_linearity,
                "final_non_linearity": self.final_non_linearity}


class FourierFeatures(nn.Module):
    """Random Fourier features.

    Args:
        frequency_matrix (torch.Tensor): Matrix of frequencies to use
            for Fourier features. Shape (num_frequencies, num_coordinates).
            This is referred to as B in the paper.
        learnable_features (bool): If True, fourier features are learnable,
            otherwise they are fixed.
    """
    def __init__(self, frequency_matrix, learnable_features=False):
        super(FourierFeatures, self).__init__()
        if learnable_features:
            self.frequency_matrix = nn.Parameter(frequency_matrix)
        else:
            # Register buffer adds a key to the state dict of the model. This will
            # track the attribute without registering it as a learnable parameter.
            # We require this so frequency matrix will also be moved to GPU when
            # we call .to(device) on the model
            self.register_buffer('frequency_matrix', frequency_matrix)
        self.learnable_features = learnable_features
        self.num_frequencies = frequency_matrix.shape[0]
        self.coordinate_dim = frequency_matrix.shape[1]
        # Factor of 2 since we consider both a sine and cosine encoding
        self.feature_dim = 2 * self.num_frequencies

    def forward(self, coordinates):
        """Creates Fourier features from coordinates.

        Args:
            coordinates (torch.Tensor): Shape (num_points, coordinate_dim)
        """
        # The coordinates variable contains a batch of vectors of dimension
        # coordinate_dim. We want to perform a matrix multiply of each of these
        # vectors with the frequency matrix. I.e. given coordinates of
        # shape (num_points, coordinate_dim) we perform a matrix multiply by
        # the transposed frequency matrix of shape (coordinate_dim, num_frequencies)
        # to obtain an output of shape (num_points, num_frequencies).
        prefeatures = torch.matmul(coordinates, self.frequency_matrix.T)
        # Calculate cosine and sine features
        cos_features = torch.cos(2 * math.pi * prefeatures)
        sin_features = torch.sin(2 * math.pi * prefeatures)
        # Concatenate sine and cosine features
        return torch.cat((cos_features, sin_features), dim=1)        
