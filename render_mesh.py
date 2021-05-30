import argparse
import torch
from data.conversion import GridDataConverter, PointCloudDataConverter
from data.dataloaders3d import VoxelDataset
from models.function_representation import FourierFeatures
from models.function_distribution import load_function_distribution
from torchvision.utils import save_image
from viz.render import voxels_to_torch3d_mesh, voxels_to_cubified_mesh, render_mesh


# It is strongly recommended to use GPU for rendering
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read command line args
parser = argparse.ArgumentParser()
parser.add_argument("-sp", "--save_path", help="Path to save images")
parser.add_argument("-mr", "--model_resolution", help="Resolution at which model was trained", type=int)
parser.add_argument("-mp", "--model_path", help="Path to model")
parser.add_argument("-ns", "--num_samples", help="Number of samples to generate", type=int, default=1)
parser.add_argument("-bs", "--batch_size", help="Batch size to use for rendering", type=int, default=1)
parser.add_argument("-rs", "--resolution", help="Resolution at which to sample 3d model", type=int, default=32)
parser.add_argument("-th", "--threshold", help="Threshold at which to consider voxel occupied", type=float, default=0.5)
parser.add_argument("-sm", "--smooth", help="Whether to smooth mesh", action="store_true")
parser.add_argument("-cu", "--cubify", help="Whether to cubify mesh", action="store_true")
parser.add_argument("-gp", "--gpu", help="Whether to use gpu", action="store_true")
parser.add_argument("-pc", "--point_cloud", help="Whether model is a point cloud model", action="store_true")
parser.add_argument("-mrp", "--multi_resolution_plot", help="Whether to make multi resolution plot", action="store_true")
parser.add_argument("-nr", "--nrow", help="Number of images per row", type=int, default=4)


args = parser.parse_args()

if args.gpu:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

num_batches = int((args.num_samples - 0.5) // args.batch_size) + 1
all_images = []

# Load a model
func_dist = load_function_distribution(device, args.model_path)
# Set up data converter
if args.point_cloud:
    data_converter = PointCloudDataConverter(device, (1,) + (args.model_resolution,) * 3, normalize_features=True)
else:
    data_converter = GridDataConverter(device, (1,) + (args.model_resolution,) * 3, normalize_features=True)

if args.multi_resolution_plot:
    # First resolution to plot will be given resolution
    resolution = args.resolution
    # Initialize latent that will be used for every resolution
    latent = func_dist.sample_prior(1)

for i in range(num_batches):
    # Clear CUDA cache to avoid OOM error
    torch.cuda.empty_cache()

    print("Batch {}/{}".format(i + 1, num_batches))
    with torch.no_grad():
        # Sample from function distribution
        if args.multi_resolution_plot:
            samples = []
            for i in range(args.batch_size):
                samples += func_dist.latent_to_data(latent, data_converter, resolution=(resolution,) * 3)
                resolution = 2 * resolution  # Double resolution for next sample
            # Threshold to create voxels
            voxels = [(sample > args.threshold).detach().float().to(device) for sample in samples]
        else:
            samples = func_dist.sample_data(data_converter, num_samples=args.batch_size, 
                                            resolution=(args.resolution,) * 3)
            # If point clouds, convert to voxels for marching cubes algorithm
            if args.point_cloud:
                samples_ = []
                for i in range(args.batch_size):
                    # Point cloud data is stored as a tensor of shape (num_points, 4) where
                    # 4th column represent the occupancy value. Extract occupancy features
                    # of shape (num_points, 1)
                    features = samples[i][:, 3:]
                    sample = features.T.view((1,) + (args.resolution,) * 3)
                    samples_.append(sample.permute(0, 1, 3, 2))
                samples = samples_
            samples = torch.cat([sample.unsqueeze(0) for sample in samples], dim=0).detach()
            # Threshold to create voxels
            voxels = (samples > args.threshold).float().to(device)

    print("Completed voxel generation.")
    with torch.no_grad():
        # Convert voxels to mesh and render
        if args.cubify:
            mesh = voxels_to_cubified_mesh(voxels)
        else:
            mesh = voxels_to_torch3d_mesh(voxels, args.smooth)
        print("Completed voxel to mesh conversion.")
        images = render_mesh(device, mesh, flat=args.cubify)
        print("Completed rendering.")
        all_images.append(images)

# Save images
all_images = torch.cat(all_images, dim=0)
save_image(all_images, base_dir + args.save_path, nrow=args.nrow, pad_value=1.)