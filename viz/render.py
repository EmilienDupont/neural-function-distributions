import mcubes
import torch
from pytorch3d.ops import cubify
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardFlatShader,
    TexturesUV
)


def voxels_to_mesh(voxels, smooth, normalize=True, scale=1.1):
    """Converts a batch of binary voxels to a mesh.

    Args:
        voxels (torch.Tensor): Shape (batch_size, channels, depth, height, width).
            Values must be between 0 and 1, i.e. threshold should be applied before
            passing tensor through this function.
        smooth (bool): If True, smoothes mesh.
        normalize (bool): Normalizes mesh to lie in [-.5, .5]^3 cube.
    
    Returns:
        all_vertices (list of torch.Tensor): List of tensors containing vertices for
            each voxel grid in batch.
        all_faces (list of torch.Tensor): List of tensors containing faces for
            each voxel grid in batch.
    """
    if isinstance(voxels, list):
        is_list = True
        device = voxels[0].device
    else:
        is_list = False
        device = voxels.device
    all_vertices = []
    all_faces = []
    # Threshold above which to consider voxel occupied
    threshold = 0.0 if smooth else 0.5    
    for i in range(len(voxels)):
        # Convert single voxel grid to numpy in order to run marching cubes
        if is_list:
            voxels_np = voxels[i][0].cpu().numpy()
        else:
            voxels_np = voxels[i, 0].cpu().numpy()
        if smooth:
            voxels_np = mcubes.smooth(voxels_np)
        # Apply marching cubes algorithm to obtain vertices and faces of mesh
        vertices, faces = mcubes.marching_cubes(voxels_np, threshold)
        if normalize:
            # [0, voxel_size - 1] -> [0, 1]
            vertices /= voxels_np.shape[-1] - 1
            # [0, 1] -> [-0.5, 0.5]
            vertices -= .5
        # Scale vertices
        vertices *= scale
        all_vertices.append(torch.from_numpy(vertices).float().to(device))
        all_faces.append(torch.from_numpy(faces.astype(int)).to(device))
    return all_vertices, all_faces


def get_uniform_texture(all_vertices, color=0.85):
    """Returns a texture object corresponding to all vertices having the same color.
    """
    vertices_rgb = [color * torch.ones_like(vertices) for vertices in all_vertices]
    return TexturesVertex(verts_features=vertices_rgb)


def voxels_to_torch3d_mesh(voxels, smooth, normalize=True, scale=1.1,
                           color=0.85):
    """Converts a batch of binary voxels to a pytorch3d mesh.

    Args:
        voxels (torch.Tensor): Shape (batch_size, channels, depth, height, width).
            Values must be either 0 and 1, i.e. threshold should be applied before
            passing tensor through this function.
        smooth (bool): If True, smoothes mesh.
        color (float): Value between 0 and 1 controlling the color of mesh (0 is black,
            1 is white).
    """
    all_vertices, all_faces = voxels_to_mesh(voxels, smooth, normalize, scale)
    # Create gray texture in order to later render mesh
    textures = get_uniform_texture(all_vertices, color)
    return Meshes(all_vertices, all_faces, textures=textures)


def voxels_to_cubified_mesh(voxels, normalize=True, scale=1.1, color=0.85):
    """Converts a batch of binary voxels to a mesh where every occupied voxel
    is replaced with a cube.
    
    Args:

    """
    # Cubify voxels (remove channel dimension and permute to fit camera angle)
    voxels_ = voxels[:, 0]
    voxels_ = voxels_.permute(0, 3, 2, 1)  # Swap x and z
    mesh = cubify(voxels_, 0.5, align="center")
    # Extract vertices and faces in order to scale and normalize
    all_vertices = mesh.verts_list()
    all_faces = mesh.faces_list()
    # Normalize and scale
    if normalize:
        # [-1, 1] -> [-0.5, 0.5] * scale
        all_vertices = [scale * vertices / 2 for vertices in all_vertices]
    # Create gray texture in order to later render mesh
    textures = get_uniform_texture(all_vertices, color)
    return Meshes(all_vertices, all_faces, textures=textures)


def render_mesh(device, mesh, flat=False):
    """Renders a mesh using a pytorch3d renderer. Note that cameras, lighting and other
    rendering details have been optimized for rendering the shapenet dataset.

    Args:
        device ():
        mesh (pytorch3d.structures.Meshes): Mesh (or meshes) to render.
        flat (bool): If True, renders with flat shading otherwise uses a Phong shader.
    """
    # Set up camera (position optimized for shapenet dataset)
    R, T = look_at_view_transform(eye=((1.2, 0.8, -0.6),), at=((0.0, -0.1, 0.0),))
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Set up rasterizer
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1
    )

    # Set up point light
    lights = PointLights(
        device=device, 
        location=((2.0, 1.0, 1.0),)
    )

    # Create shader
    if flat:
        shader = HardFlatShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    else:
        shader = SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    # Create renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=shader
    )

    # Render meshes
    images = renderer(mesh)  # (batch_size, height, width, channels)
    images = images.permute(0, 3, 1, 2)  # (batch_size, channels, height, width)
    # Remove alpha channel
    return images[:, :3]
