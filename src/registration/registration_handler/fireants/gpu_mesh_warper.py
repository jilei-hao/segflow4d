import torch
import torch.nn.functional as F
from common.types.image_wrapper import ImageWrapper
import logging

logger = logging.getLogger(__name__)

def warp_mesh_vertices(vertices, warp_grid, fixed_images: ImageWrapper, moving_images: ImageWrapper):
    """
    Warp mesh vertices using the warp field from get_warped_coordinates.

    The warp_grid maps each voxel in the fixed image's normalized [-1,1] space to
    the corresponding coordinate in the moving image's normalized [-1,1] space.
    Physical-to-voxel and voxel-to-physical conversions must account for the
    image direction matrix (e.g. flipped Z axis).

    Args:
        vertices: Tensor of shape [N_vertices, 3] in physical coordinates (x, y, z)
        warp_grid: Output from get_warped_coordinates, shape [1, D, H, W, 3]
                   Last dim is (x, y, z) normalized coordinates in moving space.
        fixed_images: ImageWrapper for fixed/reference image space
        moving_images: ImageWrapper for moving image space

    Returns:
        warped_vertices: Tensor of shape [N_vertices, 3] in moving image physical space
    """
    dev = vertices.device

    # --- Fixed image metadata ---
    fixed_origin = torch.tensor(fixed_images.get_origin(), device=dev, dtype=torch.float32)       # (ox, oy, oz)
    fixed_spacing = torch.tensor(fixed_images.get_spacing(), device=dev, dtype=torch.float32)     # (sx, sy, sz)
    fixed_direction = torch.tensor(fixed_images.get_direction(), device=dev, dtype=torch.float32).reshape(3, 3)
    fixed_size = torch.tensor(fixed_images.get_dimensions(), device=dev, dtype=torch.float32)     # (W, H, D)

    # 1. Physical to voxel (must account for direction matrix)
    #    ITK convention: physical = origin + direction @ diag(spacing) @ index
    #    Inverse:        index = diag(1/spacing) @ direction^T @ (physical - origin)
    #    For row vectors: index = (physical - origin) @ direction / spacing
    centered = vertices - fixed_origin                          # [N, 3]
    voxel_coords = (centered @ fixed_direction) / fixed_spacing # [N, 3] in (i, j, k) order

    # 2. Voxel to normalized [-1, 1]
    #    SimpleITK GetSize() returns (W, H, D) matching (i, j, k) = (x, y, z) index order
    normalized_coords = 2.0 * voxel_coords / (fixed_size - 1) - 1.0  # [N, 3] as (x, y, z) for grid_sample

    logger.debug(f"Voxel coords range:  X[{voxel_coords[:,0].min():.1f}, {voxel_coords[:,0].max():.1f}], "
                 f"Y[{voxel_coords[:,1].min():.1f}, {voxel_coords[:,1].max():.1f}], "
                 f"Z[{voxel_coords[:,2].min():.1f}, {voxel_coords[:,2].max():.1f}]")
    logger.debug(f"Normalized range:    X[{normalized_coords[:,0].min():.4f}, {normalized_coords[:,0].max():.4f}], "
                 f"Y[{normalized_coords[:,1].min():.4f}, {normalized_coords[:,1].max():.4f}], "
                 f"Z[{normalized_coords[:,2].min():.4f}, {normalized_coords[:,2].max():.4f}]")

    # 3. Sample the warp field at vertex locations
    #    warp_grid is [1, D, H, W, 3] → need [1, 3, D, H, W] for grid_sample input
    warp_field = warp_grid.permute(0, 4, 1, 2, 3)              # [1, 3, D, H, W]
    #    grid_sample grid: [N, D_out, H_out, W_out, 3] with last dim (x, y, z)
    sample_points = normalized_coords.view(1, 1, 1, -1, 3)     # [1, 1, 1, N_vertices, 3]

    sampled_coords = F.grid_sample(
        warp_field, sample_points,
        mode='bilinear', align_corners=True, padding_mode='border'
    )  # [1, 3, 1, 1, N_vertices]

    # 4. Sampled values are normalized coordinates in moving image space
    new_normalized = sampled_coords.squeeze().T                 # [N_vertices, 3]

    logger.debug(f"Sampled norm range:  X[{new_normalized[:,0].min():.4f}, {new_normalized[:,0].max():.4f}], "
                 f"Y[{new_normalized[:,1].min():.4f}, {new_normalized[:,1].max():.4f}], "
                 f"Z[{new_normalized[:,2].min():.4f}, {new_normalized[:,2].max():.4f}]")

    # --- Moving image metadata ---
    moving_origin = torch.tensor(moving_images.get_origin(), device=dev, dtype=torch.float32)
    moving_spacing = torch.tensor(moving_images.get_spacing(), device=dev, dtype=torch.float32)
    moving_direction = torch.tensor(moving_images.get_direction(), device=dev, dtype=torch.float32).reshape(3, 3)
    moving_size = torch.tensor(moving_images.get_dimensions(), device=dev, dtype=torch.float32)   # (W, H, D)

    # 5. Normalized to voxel in moving space
    new_voxel = (new_normalized + 1.0) * (moving_size - 1) / 2.0   # [N, 3]

    # 6. Voxel to physical in moving space (account for direction)
    #    physical = origin + direction @ diag(spacing) @ index
    #    For row vectors: physical = origin + (index * spacing) @ direction^T
    warped_vertices = moving_origin + (new_voxel * moving_spacing) @ moving_direction.T

    logger.debug(f"Warped range:        X[{warped_vertices[:,0].min():.2f}, {warped_vertices[:,0].max():.2f}], "
                 f"Y[{warped_vertices[:,1].min():.2f}, {warped_vertices[:,1].max():.2f}], "
                 f"Z[{warped_vertices[:,2].min():.2f}, {warped_vertices[:,2].max():.2f}]")

    return warped_vertices


def main():
    pass

if __name__ == "__main__":
    main()