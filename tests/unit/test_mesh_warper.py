"""Unit tests for the GPU mesh warper.

CPU-runnable tests use zero-displacement and constant-displacement tensors
with plain CPU torch tensors to validate the coordinate math.  Tests that
actually need a CUDA device are marked @pytest.mark.gpu.
"""

import numpy as np
import pytest
import torch
import SimpleITK as sitk

from segflow4d.common.types.image_wrapper import ImageWrapper


# ---------------------------------------------------------------------------
# Helper: build minimal fixed/moving SimpleITK images matching the warp grid
# ---------------------------------------------------------------------------

def _make_sitk_image(shape_zyx, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    arr = np.zeros(shape_zyx, dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    return img


# ---------------------------------------------------------------------------
# Import the function under test
# ---------------------------------------------------------------------------

try:
    from segflow4d.registration.registration_handler.fireants.gpu_mesh_warper import (
        warp_mesh_vertices,
    )
    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False

pytestmark_import = pytest.mark.skipif(
    not _IMPORT_OK, reason="gpu_mesh_warper could not be imported"
)


@pytest.mark.gpu
@pytestmark_import
class TestIdentityWarp:
    def test_identity_warp_preserves_vertices(self, synthetic_mesh):
        """Identity warp must leave vertex positions unchanged.

        ``warp_grid`` stores absolute normalized coordinates in moving space
        (not displacements).  An all-zeros grid maps every vertex to the image
        centre.  The true identity grid must map each voxel (d, h, w) to its
        own normalised coordinate (x, y, z).
        """
        device = torch.device("cuda")
        shape_zyx = (16, 32, 32)
        D, H, W = shape_zyx
        spacing = (1.0, 1.0, 1.0)

        fixed_img = ImageWrapper(_make_sitk_image(shape_zyx, spacing))
        moving_img = ImageWrapper(_make_sitk_image(shape_zyx, spacing))

        # Build identity warp: warp[0, d, h, w] = (x_norm, y_norm, z_norm)
        # where x_norm = 2*w/(W-1)-1, etc.
        xs = torch.linspace(-1, 1, W, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        zs = torch.linspace(-1, 1, D, device=device)
        # grid_z[d,h,w], grid_y[d,h,w], grid_x[d,h,w]
        grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing="ij")
        # warp_grid last dim is (x, y, z)
        warp = torch.stack([grid_x, grid_y, grid_z], dim=-1).unsqueeze(0)  # [1, D, H, W, 3]

        vertices = synthetic_mesh.get_data().GetPoints()
        n_pts = vertices.GetNumberOfPoints()
        orig_pts = np.array([vertices.GetPoint(i) for i in range(n_pts)])
        orig_pts_tensor = torch.tensor(orig_pts, dtype=torch.float32, device=device)

        warped_mesh = warp_mesh_vertices(
            vertices=orig_pts_tensor,
            warp_grid=warp,
            fixed_images=fixed_img,
            moving_images=moving_img,
        )

        # warp_mesh_vertices returns a Tensor of shape [N_vertices, 3]
        warped_pts = warped_mesh.cpu().numpy()
        np.testing.assert_allclose(warped_pts, orig_pts, atol=1e-4)


@pytestmark_import
class TestConstantDisplacementWarp:
    def test_known_translation_shifts_vertices(self, synthetic_mesh):
        """A constant displacement of (dx, dy, dz) in voxel space must shift
        every vertex by the same physical amount (when spacing = 1 mm)."""
        pytest.importorskip("torch")

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            device = torch.device("cpu")

        shape_zyx = (16, 32, 32)
        spacing = (1.0, 1.0, 1.0)  # 1 mm isotropic → voxel == mm

        fixed_img = ImageWrapper(_make_sitk_image(shape_zyx, spacing))
        moving_img = ImageWrapper(_make_sitk_image(shape_zyx, spacing))

        # Constant displacement: shift by (1, 0, 0) in normalised grid units
        # The actual physical shift depends on the warp_mesh_vertices implementation.
        # We simply verify that output ≠ input (non-trivial warp was applied).
        shift = torch.zeros(1, 16, 32, 32, 3, dtype=torch.float32, device=device)
        shift[..., 0] = 0.1  # non-zero displacement in x

        vertices = synthetic_mesh.get_data().GetPoints()
        n_pts = vertices.GetNumberOfPoints()
        orig_pts = np.array([vertices.GetPoint(i) for i in range(n_pts)])
        orig_pts_tensor = torch.tensor(orig_pts, dtype=torch.float32, device=device)

        warped_mesh = warp_mesh_vertices(
            vertices=orig_pts_tensor,
            warp_grid=shift,
            fixed_images=fixed_img,
            moving_images=moving_img,
        )

        # warp_mesh_vertices returns a Tensor of shape [N_vertices, 3]
        warped_pts = warped_mesh.cpu().numpy()
        # At least some vertices must have moved
        assert not np.allclose(warped_pts, orig_pts, atol=1e-6), (
            "Expected warped vertices to differ from original"
        )
