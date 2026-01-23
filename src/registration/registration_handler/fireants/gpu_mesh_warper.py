import torch
import torch.nn.functional as F

def warp_mesh_vertices(vertices, warp_grid, fixed_images, moving_images):
    """
    Warp mesh vertices using the displacement field from get_warped_coordinates.
    
    Args:
        vertices: Tensor of shape [N_vertices, 3] in physical coordinates
        warp_grid: Output from get_warped_coordinates, shape [1, H, W, D, 3]
        fixed_images: BatchedImages to get physical space info
        moving_images: BatchedImages to get moving space info
    
    Returns:
        warped_vertices: Tensor of shape [N_vertices, 3] in moving image physical space
    """
    # 1. Convert physical coordinates to normalized [-1, 1] grid coordinates
    origin = fixed_images.origins[0]      # [3]
    spacing = fixed_images.spacings[0]    # [3]
    shape = torch.tensor(warp_grid.shape[1:4], device=vertices.device, dtype=torch.float32)  # [D, H, W]
    
    # Physical to voxel
    voxel_coords = (vertices - origin) / spacing
    # Voxel to normalized [-1, 1]
    normalized_coords = 2.0 * voxel_coords / (shape - 1) - 1.0
    
    # 2. Sample the warp field at vertex locations
    # grid_sample expects [N, C, D, H, W] and sample points [N, 1, 1, N_points, 3]
    warp_field = warp_grid.permute(0, 4, 1, 2, 3)  # [1, 3, H, W, D]
    sample_points = normalized_coords.view(1, 1, 1, -1, 3)  # [1, 1, 1, N_vertices, 3]
    
    # Sample coordinates at each vertex
    sampled_coords = F.grid_sample(
        warp_field, sample_points, 
        mode='bilinear', align_corners=True, padding_mode='border'
    )  # [1, 3, 1, 1, N_vertices]
    
    # 3. The sampled values are the NEW normalized coordinates in moving image space
    new_normalized = sampled_coords.squeeze().T  # [N_vertices, 3]
    
    # 4. Convert back to physical space (moving image space)
    moving_origin = moving_images.origins[0]      # [3]
    moving_spacing = moving_images.spacings[0]    # [3]
    
    # Normalized to voxel
    new_voxel = (new_normalized + 1.0) * (shape - 1) / 2.0
    # Voxel to physical (in moving image space)
    warped_vertices = new_voxel * moving_spacing + moving_origin
    
    return warped_vertices