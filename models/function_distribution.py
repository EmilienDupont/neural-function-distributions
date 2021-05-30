import torch
import torch.nn as nn
from models.function_representation import FunctionRepresentation
from torch.distributions import Normal


class FunctionDistribution(nn.Module):
    """Distribution of functions.

    Args:
        hypernetwork (HyperNetwork):
    """
    def __init__(self, hypernetwork):
        super(FunctionDistribution, self).__init__()
        self.hypernetwork = hypernetwork
        self.latent_dim = self.hypernetwork.latent_dim
        # Extract hypernetwork device so we can move latent distribution to
        # correct device
        hypernet_device = next(self.hypernetwork.parameters()).device
        self.latent_distribution = Normal(torch.zeros(self.latent_dim).to(hypernet_device),
                                          torch.ones(self.latent_dim).to(hypernet_device))

    def forward(self, latents):
        """Returns weights and biases for each latent vector in latents tensor.

        Args:
            latents (torch.Tensor): Shape (batch_size, latent_dim)
        """
        return self.hypernetwork(latents)

    def latent_to_function(self, latents):
        """Returns a FunctionRepresentation instance for each latent vector in
        latents tensor.

        Args:
            latents (torch.Tensor): Shape (batch_size, latent_dim)
        """
        all_weights, all_biases = self(latents)
        function_representations = []
        for i in range(latents.shape[0]):
            # Create an "empty" function representation
            function_rep = self.hypernetwork.function_representation.duplicate()
            # Set weights and biases as predicted by hypernetwork
            function_rep.set_weights_and_biases(all_weights[i], all_biases[i])
            function_representations.append(function_rep)
        return function_representations

    def sample_function(self, num_samples=1):
        """Samples functions from function distribution.

        Args:
            num_samples (int): Number of functions to sample.

        Returns:
            List of function representations.
        """
        latent_samples = self.latent_distribution.sample((num_samples,))
        return self.latent_to_function(latent_samples)

    def sample_data(self, data_converter, num_samples=1, resolution=None):
        """Samples functions from function distributions and evaluates functions
        on grid to return image.

        Args:
            data_converter (conversion.DataConverter):
            num_samples (int):
            resolution (tuple of ints):
        """
        samples = []
        function_representations = self.sample_function(num_samples)
        for function_rep in function_representations:
            samples.append(function_rep.sample_grid(data_converter, resolution))
        return samples

    def latent_to_data(self, latents, data_converter, resolution=None):
        """Converts each latent vector to a FunctionRepresentation instance and
        uses this instance to sample data on a grid for each function. Returns
        a list of data.

        Args:
            latents (torch.Tensor): Shape (batch_size, latent_dim)
            data_converter (conversion.DataConverter):
            resolution (tuple of ints):
        """
        samples = []
        function_representations = self.latent_to_function(latents)
        for function_representation in function_representations:
            samples.append(function_representation.sample_grid(data_converter,
                                                               resolution))
        return samples

    def sample_prior(self, num_samples):
        """Returns a batch of samples from prior.

        Args:
            num_samples (int): Number of samples to draw.
        """
        return self.latent_distribution.sample((num_samples,))

    def _stateless_sample(self, coordinates):
        """Samples a batch of functions from function distribution, then
        performs a stateless evaluation of the functions at the given
        coordinates to return the predicted features.

        Args:
            coordinates (torch.Tensor): Batch of coordinates of shape
                (batch_size, num_points, coordinate_dim).
        """
        latents = self.sample_prior(coordinates.shape[0])
        return self._stateless_forward(coordinates, latents)

    def _stateless_forward(self, coordinates, latents):
        """Computes a set of functions based on the latent variables and
        performs a stateless evaluation of the functions at the given
        coordinates to return the predicted features.

        Args:
            coordinates (torch.Tensor): Batch of coordinates of shape
                (batch_size, num_points, coordinate_dim).
            latents (torch.Tensor): Shape (batch_size, latent_dim)
        """
        # Compute weights and biases of functions
        weights, biases = self(latents)
        # Perform a stateless evaluation of functions at given coordinates
        return self.hypernetwork.function_representation.batch_stateless_forward(coordinates,
                                                                                 weights,
                                                                                 biases)

    def _get_config(self):
        """Returns config for function distribution network."""
        return {"hypernetwork": self.hypernetwork._get_config(),
                "function_representation": self.hypernetwork.function_representation._get_config()}

    def save_model(self, path):
        """Saves model to given path.

        Args:
            path (string): File extension should be ".pt".
        """
        torch.save({'config': self._get_config(), 'state_dict': self.state_dict()}, path)


class HyperNetwork(nn.Module):
    """Hypernetwork that outputs the weights of a function representation.

    Args:
        function_representation (models.function_representation.FunctionRepresentation):
        latent_dim (int): Dimension of latent vectors.
        layer_sizes (tuple of ints): Specifies size of each hidden layer.
        non_linearity (torch.nn.Module):
    """
    def __init__(self, function_representation, latent_dim, layer_sizes,
                 non_linearity):
        super(HyperNetwork, self).__init__()
        self.function_representation = function_representation
        self.latent_dim = latent_dim
        self.layer_sizes = layer_sizes
        self.non_linearity = non_linearity
        self._infer_output_shapes()
        self._init_neural_net()

    def _infer_output_shapes(self):
        """Uses function representation to infer correct output shapes for
        hypernetwork (i.e. so dimension matches size of weights in function
        representation) network."""
        self.weight_shapes, self.bias_shapes = self.function_representation.get_weight_shapes()
        num_layers = len(self.weight_shapes)

        # Calculate output dimension
        self.output_dim = 0
        for i in range(num_layers):
            # Add total number of weights in weight matrix
            self.output_dim += self.weight_shapes[i][0] * self.weight_shapes[i][1]
            # Add total number of weights in bias vector
            self.output_dim += self.bias_shapes[i][0]

        # Calculate partition of output of network, so that output network can
        # be reshaped into weights of the function representation network
        # Partition first part of output into weight matrices
        start_index = 0
        self.weight_partition = []
        for i in range(num_layers):
            weight_size = self.weight_shapes[i][0] * self.weight_shapes[i][1]
            self.weight_partition.append((start_index, start_index + weight_size))
            start_index += weight_size

        # Partition second part of output into bias matrices
        self.bias_partition = []
        for i in range(num_layers):
            bias_size = self.bias_shapes[i][0]
            self.bias_partition.append((start_index, start_index + bias_size))
            start_index += bias_size

        self.num_layers_function_representation = num_layers

    def _init_neural_net(self):
        """Initializes weights of hypernetwork."""
        forward_layers = []
        prev_num_units = self.latent_dim
        for num_units in self.layer_sizes:
            forward_layers.append(nn.Linear(prev_num_units, num_units))
            forward_layers.append(self.non_linearity)
            prev_num_units = num_units
        forward_layers.append(nn.Linear(prev_num_units, self.output_dim))
        self.forward_layers = nn.Sequential(*forward_layers)

    def output_to_weights(self, output):
        """Converts output of function distribution network into list of weights
        and biases for function representation networks.

        Args:
            output (torch.Tensor): Output of neural network as a tensor of shape
                (batch_size, self.output_dim).

        Notes:
            Each element in batch will correspond to a separate function
            representation network, therefore there will be batch_size sets of
            weights and biases.
        """
        all_weights = {}
        all_biases = {}
        # Compute weights and biases separately for each element in batch
        for i in range(output.shape[0]):
            weights = []
            biases = []
            # Add weight matrices
            for j, (start_index, end_index) in enumerate(self.weight_partition):
                weight = output[i, start_index:end_index]
                weights.append(weight.view(*self.weight_shapes[j]))
            # Add bias vectors
            for j, (start_index, end_index) in enumerate(self.bias_partition):
                bias = output[i, start_index:end_index]
                biases.append(bias.view(*self.bias_shapes[j]))
            # Add weights and biases for this function representation to batch
            all_weights[i] = weights
            all_biases[i] = biases
        return all_weights, all_biases

    def forward(self, latents):
        """Compute weights of function representations from latent vectors.

        Args:
            latents (torch.Tensor): Shape (batch_size, latent_dim).
        """
        output = self.forward_layers(latents)
        return self.output_to_weights(output)

    def _get_config(self):
        """ """
        return {"latent_dim": self.latent_dim, "layer_sizes": self.layer_sizes,
                "non_linearity": self.non_linearity}


def load_function_distribution(device, path):
    """
    """
    all_dicts = torch.load(path, map_location=lambda storage, loc: storage)
    config, state_dict = all_dicts["config"], all_dicts["state_dict"]
    # Initialize function representation
    config_rep = config["function_representation"]
    encoding = config_rep["encoding"].to(device)
    if hasattr(encoding, 'frequency_matrix'):
        encoding.frequency_matrix = encoding.frequency_matrix.to(device)
    function_representation = FunctionRepresentation(config_rep["input_dim"],
                                                     config_rep["output_dim"],
                                                     config_rep["layer_sizes"],
                                                     encoding,
                                                     config_rep["non_linearity"],
                                                     config_rep["final_non_linearity"]).to(device)
    # Initialize hypernetwork
    config_hyp = config["hypernetwork"]
    hypernetwork = HyperNetwork(function_representation, config_hyp["latent_dim"],
                                config_hyp["layer_sizes"], config_hyp["non_linearity"]).to(device)
    # Initialize function distribution
    function_distribution = FunctionDistribution(hypernetwork).to(device)
    # Load weights of function distribution
    function_distribution.load_state_dict(state_dict)
    return function_distribution
