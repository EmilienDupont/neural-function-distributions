import matplotlib
matplotlib.use('Agg')  # This is hacky (useful for running on VMs)
import matplotlib.pyplot as plt


def plot_voxels_batch(voxels, ncols=4, save_fig=''):
    """Plots batches of voxels.
    
    Args:
        voxels (torch.Tensor): Shape (batch_size, 1, depth, height, width).
        ncols (int): Number of columns in grid of images.
    """
    batch_size, _, voxel_size, _, _ = voxels.shape
    nrows = int((batch_size - 0.5) / ncols) + 1
    fig = plt.figure()
    
    # Permutation to get better angle of chair data
    voxels = voxels.permute(0, 1, 2, 4, 3)
    
    for i in range(batch_size):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')

        # Non zero voxels define coordinates of visible points
        coords = voxels[i, 0].nonzero(as_tuple=False)
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=1)            
    
        # Set limits to size of voxel grid
        ax.set_xlim(0, voxel_size - 1)
        ax.set_ylim(0, voxel_size - 1)
        ax.set_zlim(0, voxel_size - 1)

    plt.tight_layout()
    
    # Optionally save figure
    if len(save_fig):
        plt.savefig(save_fig, format='png', dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        plt.show()


def plot_point_cloud_batch(point_clouds, ncols=4, threshold=0.5, save_fig=''):
    """Plots batches of point clouds
    
    Args:
        point_clouds (torch.Tensor): Shape (batch_size, num_points, 4).
        ncols (int): Number of columns in grid of images.
        threshold (float): Value above which to consider point cloud occupied.
    """
    batch_size = point_clouds.shape[0]
    nrows = int((batch_size - 0.5) / ncols) + 1
    fig = plt.figure()
    
    for i in range(batch_size):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')

        # Extract coordinates with feature values above threshold (corresponding
        # to occupied points)
        coordinates = point_clouds[i, :, :3]
        features = point_clouds[i, :, -1]
        coordinates = coordinates[features > threshold]
        ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], s=1)            
    
        # Set limits of plot
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

    plt.tight_layout()
    
    # Optionally save figure
    if len(save_fig):
        plt.savefig(save_fig, format='png', dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        plt.show()