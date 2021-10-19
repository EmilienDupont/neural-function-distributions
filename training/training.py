import json
import torch
from viz.plots import plot_voxels_batch, plot_point_cloud_batch
from torchvision.utils import save_image


class Trainer():
    """Trains a function distribution.

    Args:
        device (torch.device):
        function_distribution (models.function_distribution.FunctionDistribution):
        discriminator (models.discrimnator.PointConvDiscriminator):
        data_converter (data.conversion.{GridDataConverter, PointCloudDataConverter}):
        lr (float): Learning rate for hypernetwork.
        lr_disc (float): Learning rate for discriminator.
        betas (tuple of ints): Beta parameters to use in Adam optimizer. Usually
            this is either (0.5, 0.999) or (0., 0.9).
        r1_weight (float): Weight to use for R1 regularization.
        max_num_points (int or None): If not None, randomly samples max_num_points
            points from the coordinates and features before passing through
            generator and discriminator.
        is_voxel (bool): If True, considers data as voxels.
        is_point_cloud (bool): If True, considers data as point clouds.
        is_era5 (bool): If True, considers data as ERA5 surface temperatures.
        print_freq (int): Frequency with which to print loss.
        save_dir (string): Path to a directory where experiment logs and images
            will be saved.
        model_save_freq (int): Frequency (in epochs) at which to save model.
    """
    def __init__(self, device, function_distribution, discriminator,
                 data_converter, lr=2e-4, lr_disc=2e-4, betas=(0.5, 0.999),
                 r1_weight=0, max_num_points=None, is_voxel=False,
                 is_point_cloud=False, is_era5=False, print_freq=1, save_dir='',
                 model_save_freq=0):
        self.device = device
        self.function_distribution = function_distribution
        self.discriminator = discriminator
        self.data_converter = data_converter
        self.r1_weight = r1_weight
        self.max_num_points = max_num_points
        self.is_voxel = is_voxel
        self.is_point_cloud = is_point_cloud
        self.is_era5 = is_era5

        self.bce = torch.nn.BCELoss()

        # We only want to learn weights of hypernetwork (i.e. forward_layers)
        # not of function representation itself
        self.optimizer = torch.optim.Adam(
            self.function_distribution.hypernetwork.forward_layers.parameters(),
            lr=lr, betas=betas
        )
        # Optimizer for discriminator
        self.optimizer_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_disc, betas=betas
        )

        self.print_freq = print_freq
        self.save_dir = save_dir
        # Ensure number of samples saved is not too large when data itself
        # is large
        if data_converter.data_shape[1] < 65:
            self.num_samples_to_save = 32
        elif data_converter.data_shape[1] >= 65:
            self.num_samples_to_save = 16
        if self.is_voxel or self.is_point_cloud:
            self.num_samples_to_save = 16
        self.model_save_freq = model_save_freq
        self._init_logs()

    def _init_logs(self):
        """Initializes logs to track model performance during training."""
        self.logged_items = ("generator", "discriminator", "disc_real", "disc_fake")
        if self.r1_weight:
            self.logged_items = self.logged_items + ("grad_squared",)
        self.logs = {logged_item : [] for logged_item in self.logged_items}
        self.epoch_logs = {logged_item : [] for logged_item in self.logged_items}
        # Track the gradients of the discriminator during training
        self.gradient_logs = {name: [] for name, _ in self.discriminator.named_parameters()}
        # Track the gradients of the generator (hypernetwork) during training
        self.hyper_gradient_logs = {name: [] for name, _ in self.function_distribution.hypernetwork.forward_layers.named_parameters()}

    def _log_epoch_losses(self, iterations_per_epoch):
        """
        """
        for logged_item in self.logged_items:
            self.epoch_logs[logged_item].append(mean(self.logs[logged_item][-iterations_per_epoch:]))

    def _save_logs(self):
        """
        """
        # Save regular logs
        with open(self.save_dir + '/logs.json', 'w') as f:
            json.dump(self.logs, f)
        # Save epoch logs
        with open(self.save_dir + '/epoch_logs.json', 'w') as f:
            json.dump(self.epoch_logs, f)
        # Save gradient logs
        with open(self.save_dir + '/gradient_logs.json', 'w') as f:
            json.dump(self.gradient_logs, f)
        # Save gradient logs for hypernetwork
        with open(self.save_dir + '/hyper_gradient_logs.json', 'w') as f:
            json.dump(self.hyper_gradient_logs, f)

    def _save_data_samples(self, latents, filename):
        """
        """
        with torch.no_grad():
            samples = self.function_distribution.latent_to_data(latents,
                                                                self.data_converter)
            # Convert list of samples to a batch of tensors
            samples = torch.cat([sample.unsqueeze(0) for sample in samples], dim=0)
        # Save samples as grid
        if self.is_voxel:
            # Voxels lie in [0, 1], so use 0.5 as a threshold
            plot_voxels_batch(samples.detach().cpu() > .5, save_fig=self.save_dir + "/" + filename,
                              ncols=self.num_samples_to_save // 4)
        elif self.is_point_cloud:
            plot_point_cloud_batch(samples.detach().cpu(), save_fig=self.save_dir + "/" + filename,
                              ncols=self.num_samples_to_save // 4)
        elif self.is_era5:
            # ERA5 data has shape (batch_size, 3, num_lats, num_lons) where 3rd
            # dimension of 2nd axis corresponds to temperature, so extract this
            # (will correspond to grayscale image)
            save_image(samples[:, 2:3].detach().cpu(), self.save_dir + "/" + filename,
                       nrow=self.num_samples_to_save // 4)
        else:
            save_image(samples.detach().cpu(), self.save_dir + "/" + filename,
                       nrow=self.num_samples_to_save // 4)

    def train(self, dataloader, epochs):
        """
        """
        if self.save_dir:
            # Randomly sample latents from the prior to monitor how
            # generation improves during training
            self.random_latents = self.function_distribution.sample_prior(self.num_samples_to_save).to(self.device)
            # Generate samples before any training begins
            self._save_data_samples(self.random_latents, "samples_{}.png".format(0))
            # Save real samples (first few samples of dataset)
            real_samples = torch.cat([dataloader.dataset[i][0].unsqueeze(0)
                                     for i in range(self.num_samples_to_save)], dim=0)
            if self.is_voxel:
                plot_voxels_batch(real_samples.float() > .5, save_fig=self.save_dir + "/ground_truth.png",
                              ncols=self.num_samples_to_save // 4)
            elif self.is_point_cloud:
                plot_point_cloud_batch(real_samples.float(), save_fig=self.save_dir + "/ground_truth.png",
                              ncols=self.num_samples_to_save // 4)
            elif self.is_era5:
                save_image(real_samples[:, 2:3], self.save_dir + "/ground_truth.png",
                           nrow=self.num_samples_to_save // 4)
            else:
                save_image(real_samples, self.save_dir + "/ground_truth.png",
                           nrow=self.num_samples_to_save // 4)

        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            for i, batch in enumerate(dataloader):
                self.train_batch(batch)
                if i % self.print_freq == 0:
                    if self.r1_weight:
                        print("Iteration {}/{}: generator {:.3f}, discriminator {:.3f}, disc_real {:.3f}, disc_fake {:.3f}, 100 x grad_squared {:.3f}".format(i + 1, len(dataloader), self.logs["generator"][-1], self.logs["discriminator"][-1], self.logs["disc_real"][-1], self.logs["disc_fake"][-1], 100 * self.logs["grad_squared"][-1]))
                    else:
                        print("Iteration {}/{}: generator {:.3f}, discriminator {:.3f}, disc_real {:.3f}, disc_fake {:.3f}".format(i + 1, len(dataloader), self.logs["generator"][-1], self.logs["discriminator"][-1], self.logs["disc_real"][-1], self.logs["disc_fake"][-1]))
            self._log_epoch_losses(len(dataloader))
            if self.r1_weight:
                print("\nEpoch {}: generator {:.3f}, discriminator {:.3f}, disc_real {:.3f}, disc_fake {:.3f}, 100 x grad_squared {:.3f}".format(epoch + 1, self.epoch_logs["generator"][-1], self.epoch_logs["discriminator"][-1], self.epoch_logs["disc_real"][-1], self.epoch_logs["disc_fake"][-1], 100 * self.epoch_logs["grad_squared"][-1]))
            else:
                print("\nEpoch {}: generator {:.3f}, discriminator {:.3f}, disc_real {:.3f}, disc_fake {:.3f}".format(epoch + 1, self.epoch_logs["generator"][-1], self.epoch_logs["discriminator"][-1], self.epoch_logs["disc_real"][-1], self.epoch_logs["disc_fake"][-1]))
            # Optionally save logs, samples and model
            if self.save_dir:
                self._save_logs()
                self._save_data_samples(self.random_latents, "img_random_{}.png".format(epoch + 1))
                self.function_distribution.save_model(self.save_dir + "/model.pt")
                # If model_save_freq is non-zero save intermediate versions of
                # the model
                if epoch != 0 and self.model_save_freq != 0 and epoch % self.model_save_freq == 0:
                    self.function_distribution.save_model(self.save_dir + "/model_{}.pt".format(epoch))

    def train_batch(self, batch):
        """
        """
        # Extract data
        data, _ = batch
        data = data.to(self.device)

        # Extract coordinates and features from data
        coordinates, features = self.data_converter.batch_to_coordinates_and_features(data)

        # Optionally randomly subsample coordinates and features
        if self.max_num_points:
            set_size = coordinates.shape[1]
            subset_indices = random_indices(self.max_num_points, set_size)
            # Select a subsample of coordinates and features
            coordinates = coordinates[:, subset_indices]
            features = features[:, subset_indices]

        # Sample features from function distribution to train discriminator
        generated_features = self.function_distribution._stateless_sample(coordinates)
        self._train_discriminator(coordinates, features, generated_features)

        # Sample different features to train generator
        generated_features_gen = self.function_distribution._stateless_sample(coordinates)
        self._train_generator(coordinates, generated_features_gen)

    def _train_discriminator(self, coordinates, real_features, generated_features):
        """
        Args:
            coordinates (torch.Tensor): Tensor of shape (batch_size, num_points, coordinate_dim).
            real_features (torch.Tensor): Features of real data. Tensor of shape
                (batch_size, num_points, feature_dim).
            generated_features (torch.Tensor): Features of data generated from
                function distribution. Tensor of shape (batch_size, num_points, feature_dim).
        """
        self.optimizer_disc.zero_grad()

        # Generate batches of target probabilities
        batch_size = coordinates.shape[0]
        y_real = torch.ones((batch_size, 1)).to(self.device)  # Prob of being real for data is 1
        y_fake = torch.zeros((batch_size, 1)).to(self.device)  # Prob of being real for generated is 0

        # Calculate loss on real data
        # Probabilty of data being real according to discriminator
        if self.r1_weight:
            real_features.requires_grad_()  # Ensure we can take gradients for R1 regularization
        prob_real = self.discriminator(coordinates, real_features)

        # Calculate loss on fake data (note that we detach generated_features
        # as discriminator does not need gradients from generator)
        prob_fake = self.discriminator(coordinates, generated_features.detach())

        # Calculate losses on real and fake data
        disc_real_loss = self.bce(prob_real, y_real)
        disc_fake_loss = self.bce(prob_fake, y_fake)

        # Loss is sum of fake and real loss
        disc_loss = disc_real_loss + disc_fake_loss
        if self.r1_weight:
            # Take gradient of discriminator with respect to input on real data
            # and calculate its square norm
            grad_squared = norm_gradient_squared(prob_real, real_features).mean()
            disc_loss += 0.5 * self.r1_weight * grad_squared
        disc_loss.backward()
        self.optimizer_disc.step()

        # Log loss
        self.logs["discriminator"].append(disc_loss.item())
        self.logs["disc_real"].append(disc_real_loss.item())
        self.logs["disc_fake"].append(disc_fake_loss.item())
        if self.r1_weight:
            self.logs["grad_squared"].append(grad_squared.item())

        # Log gradient magnitudes of discriminator
        for name, param in self.discriminator.named_parameters():
            self.gradient_logs[name].append(param.grad.abs().mean().item())

    def _train_generator(self, coordinates, generated_features):
        """
        Args:
            coordinates (torch.Tensor): Tensor of shape (batch_size, num_points, coordinate_dim).
            generated_features (torch.Tensor): Features of data generated from
                function distribution. Tensor of shape (batch_size, num_points, feature_dim).
        """
        self.optimizer.zero_grad()

        # Generate batches of target probabilities
        batch_size = coordinates.shape[0]
        y_real = torch.ones((batch_size, 1)).to(self.device)

        # Probability that discriminator thinks samples are real
        prob_generated = self.discriminator(coordinates, generated_features)

        # Want to fool discriminator, i.e. change generator so that D outputs 1 on its samples
        generator_loss = self.bce(prob_generated, y_real)
        generator_loss.backward()
        self.optimizer.step()

        # Log loss
        self.logs["generator"].append(generator_loss.item())

        # Log gradient magnitudes of generator
        for name, param in self.function_distribution.hypernetwork.forward_layers.named_parameters():
            self.hyper_gradient_logs[name].append(param.grad.abs().mean().item())


def norm_gradient_squared(outputs, inputs, sum_over_points=True):
    """Computes square of the norm of the gradient of outputs with respect to
    inputs.

    Args:
        outputs (torch.Tensor): Shape (batch_size, 1). Usually the output of
            discriminator on real data.
        inputs (torch.Tensor): Shape (batch_size, num_points, coordinate_dim + feature_dim)
            or shape (batch_size, num_points, feature_dim) depending on whether gradient
            is over coordinates and features or only features.
        sum_over_points (bool): If True, sums over num_points dimension, otherwise takes mean.

    Notes:
        This is inspired by the function in this repo
        https://github.com/LMescheder/GAN_stability/blob/master/gan_training/train.py
    """
    batch_size, num_points, _ = inputs.shape
    # Compute gradient of outputs with respect to inputs
    grad_outputs = torch.autograd.grad(
        outputs=outputs.sum(), inputs=inputs,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    # Square gradients
    grad_outputs_squared = grad_outputs.pow(2)
    # Return norm of squared gradients for each example in batch. We sum over
    # features, to return a tensor of shape (batch_size, num_points).
    regularization = grad_outputs_squared.sum(dim=2)
    # We can now either take mean or sum over num_points dimension
    if sum_over_points:
        return regularization.sum(dim=1)
    else:
        return regularization.mean(dim=1)


def random_indices(num_indices, max_idx):
    """Generates a set of num_indices random indices (without replacement)
    between 0 and max_idx - 1.

    Args:
        num_indices (int): Number of indices to include.
        max_idx (int): Maximum index.
    """
    # It is wasteful to compute the entire permutation, but it looks like
    # PyTorch does not have other functions to do this
    permutation = torch.randperm(max_idx)
    # Select first num_indices indices (this will be random since permutation is
    # random)
    return permutation[:num_indices]


def mean(array):
    """Returns mean of a list.

    Args:
        array (list of ints or floats):
    """
    return sum(array) / len(array)
