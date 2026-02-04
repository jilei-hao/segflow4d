import torch
import torch.nn.functional as F
from common.types.image_wrapper import ImageWrapper

def warp_mesh_vertices(vertices, warp_grid, fixed_images: ImageWrapper, moving_images: ImageWrapper):
    """
    Warp mesh vertices using the displacement field from get_warped_coordinates.
    
    Args:
        vertices: Tensor of shape [N_vertices, 3] in physical coordinates
        warp_grid: Output from get_warped_coordinates, shape [1, H, W, D, 3]
        fixed_images: ImageWrapper to get physical space info
        moving_images: ImageWrapper to get moving space info
    
    Returns:
        warped_vertices: Tensor of shape [N_vertices, 3] in moving image physical space
    """
    # 1. Convert physical coordinates to normalized [-1, 1] grid coordinates
    origin = torch.tensor(fixed_images.get_origin(), device=vertices.device, dtype=torch.float32)      # [3]
    spacing = torch.tensor(fixed_images.get_spacing(), device=vertices.device, dtype=torch.float32)    # [3]
    shape = torch.tensor(warp_grid.shape[1:4], device=vertices.device, dtype=torch.float32)  # [H, W, D]
    
    # Physical to voxel
    voxel_coords = (vertices - origin) / spacing
    # Voxel to normalized [-1, 1]
    normalized_coords = 2.0 * voxel_coords / (shape - 1) - 1.0
    
    # 2. Sample the warp field at vertex locations
    # grid_sample expects [N, C, D, H, W] and sample points [N, 1, 1, N_points, 3]
    warp_field = warp_grid.permute(0, 4, 3, 1, 2)  # [1, 3, D, H, W]
    sample_points = normalized_coords.view(1, 1, 1, -1, 3)  # [1, 1, 1, N_vertices, 3]
    
    # Sample coordinates at each vertex
    sampled_coords = F.grid_sample(
        warp_field, sample_points, 
        mode='bilinear', align_corners=True, padding_mode='border'
    )  # [1, 3, 1, 1, N_vertices]
    
    # 3. The sampled values are the NEW normalized coordinates in moving image space
    new_normalized = sampled_coords.squeeze().T  # [N_vertices, 3]
    
    # 4. Convert back to physical space (moving image space)
    moving_origin_tuple = moving_images.get_origin()      # [3]
    moving_spacing_tuple = moving_images.get_spacing()    # [3]
    
    moving_origin = torch.tensor(moving_origin_tuple, device=vertices.device, dtype=torch.float32)
    moving_spacing = torch.tensor(moving_spacing_tuple, device=vertices.device, dtype=torch.float32)
    
    # Normalized to voxel
    new_voxel = (new_normalized + 1.0) * (shape - 1) / 2.0
    # Voxel to physical (in moving image space)
    warped_vertices = new_voxel * moving_spacing + moving_origin
    
    return warped_vertices


def main():
    pass

if __name__ == "__main__":
    main()