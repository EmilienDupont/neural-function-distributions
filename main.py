import json
import os
import sys
import time
import torch
from training.training import Trainer
from data.conversion import GridDataConverter, PointCloudDataConverter
from data.dataloaders import mnist, celebahq
from data.dataloaders3d import shapenet_voxels, shapenet_point_clouds
from models.discriminator import PointConvDiscriminator
from models.function_distribution import HyperNetwork, FunctionDistribution
from models.function_representation import FunctionRepresentation, FourierFeatures


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get config file from command line arguments
if len(sys.argv) != 2:
    raise(RuntimeError("Wrong arguments, use python main.py <config_path>"))
config_path = sys.argv[1]

# Open config file
with open(config_path) as f:
    config = json.load(f)

if config["path_to_data"] == "":
    raise(RuntimeError("Path to data not specified. Modify path_to_data attribute in config to point to data."))

# Create a folder to store experiment results
timestamp = time.strftime("%Y-%m-%d_%H-%M")
directory = "{}_{}".format(timestamp, config["id"])
if not os.path.exists(directory):
    os.makedirs(directory)

# Save config file in experiment directory
with open(directory + '/config.json', 'w') as f:
    json.dump(config, f)

# Setup dataloader
is_voxel = False
is_point_cloud = False
if config["dataset"] == 'mnist':
    dataloader = mnist(path_to_data=config["path_to_data"], 
                       batch_size=config["training"]["batch_size"], 
                       size=config["resolution"], 
                       train=True)
    input_dim = 2
    output_dim = 1
    data_shape = (1, config["resolution"], config["resolution"])
elif config["dataset"] == 'celebahq':
    dataloader = celebahq(path_to_data=config["path_to_data"],
                          batch_size=config["training"]["batch_size"], 
                          size=config["resolution"])
    input_dim = 2
    output_dim = 3
    data_shape = (3, config["resolution"], config["resolution"])
elif config["dataset"] == 'shapenet_voxels':
    dataloader = shapenet_voxels(path_to_data=config["path_to_data"],
                                 batch_size=config["training"]["batch_size"], 
                                 size=config["resolution"])
    input_dim = 3
    output_dim = 1
    data_shape = (1, config["resolution"], config["resolution"], config["resolution"])
    is_voxel = True
elif config["dataset"] == 'shapenet_point_clouds':
    dataloader = shapenet_point_clouds(path_to_data=config["path_to_data"],
                                       batch_size=config["training"]["batch_size"])
    input_dim = 3
    output_dim = 1
    data_shape = (1, config["resolution"], config["resolution"], config["resolution"])
    is_point_cloud = True


# Setup data converter
if is_point_cloud:
    data_converter = PointCloudDataConverter(device, data_shape, normalize_features=True)
else:
    data_converter = GridDataConverter(device, data_shape, normalize_features=True)


# Setup encoding for function distribution
num_frequencies = config["generator"]["encoding"]["num_frequencies"]
std_dev = config["generator"]["encoding"]["std_dev"]
if num_frequencies:
    frequency_matrix = torch.normal(mean=torch.zeros(num_frequencies, input_dim),
                                    std=std_dev).to(device)
    encoding = FourierFeatures(frequency_matrix)
else:
    encoding = torch.nn.Identity()

# Setup generator models
final_non_linearity = torch.nn.Tanh()
non_linearity = torch.nn.LeakyReLU(0.1)
function_representation = FunctionRepresentation(input_dim, output_dim,
                                                config["generator"]["layer_sizes"],
                                                encoding, non_linearity,
                                                final_non_linearity).to(device)
hypernetwork = HyperNetwork(function_representation, config["generator"]["latent_dim"],
                            config["generator"]["hypernet_layer_sizes"], non_linearity).to(device)
function_distribution = FunctionDistribution(hypernetwork).to(device)

# Setup discriminator
discriminator = PointConvDiscriminator(input_dim, output_dim, config["discriminator"]["layer_configs"],
                                        linear_layer_sizes=config["discriminator"]["linear_layer_sizes"],
                                        norm_order=config["discriminator"]["norm_order"],
                                        add_sigmoid=True,
                                        add_batchnorm=config["discriminator"]["add_batchnorm"],
                                        add_weightnet_batchnorm=config["discriminator"]["add_weightnet_batchnorm"],
                                        deterministic=config["discriminator"]["deterministic"],
                                        same_coordinates=config["discriminator"]["same_coordinates"]).to(device)


print("\nFunction distribution")
print(hypernetwork)
print("Number of parameters: {}".format(count_parameters(hypernetwork)))

print("\nDiscriminator")
print(discriminator)
print("Number of parameters: {}".format(count_parameters(discriminator)))

# Setup trainer
trainer = Trainer(device, function_distribution, discriminator, data_converter,
                  lr=config["training"]["lr"], lr_disc=config["training"]["lr_disc"],
                  r1_weight=config["training"]["r1_weight"],
                  max_num_points=config["training"]["max_num_points"],
                  print_freq=config["training"]["print_freq"], save_dir=directory,
                  model_save_freq=config["training"]["model_save_freq"],
                  is_voxel=is_voxel, is_point_cloud=is_point_cloud)
trainer.train(dataloader, config["training"]["epochs"])