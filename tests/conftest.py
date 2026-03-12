"""
Shared pytest fixtures for the SegFlow4D test suite.

All fixtures here are CPU-only and have no GPU dependency.
"""

import numpy as np
import pytest
import SimpleITK as sitk
import torch


def pytest_runtest_setup(item):
    """Auto-skip tests marked 'gpu' when CUDA is not available."""
    if any(item.iter_markers(name="gpu")):
        if not torch.cuda.is_available():
            pytest.skip("requires a CUDA-capable GPU")


@pytest.fixture(autouse=True)
def reset_registration_manager():
    """Reset the RegistrationManager singleton before and after every test.

    The RegistrationManager is a global singleton; without this reset a test
    that initialises it with FireANTS would corrupt all subsequent tests that
    expect a different backend (e.g. Greedy).
    """
    from segflow4d.registration.registration_manager.factory import RegistrationManager
    RegistrationManager.reset()
    yield
    RegistrationManager.reset()

from segflow4d.common.types.image_wrapper import ImageWrapper
from segflow4d.common.types.mesh_wrapper import MeshWrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sphere_array(shape, center, radius, inside_val=1, dtype=np.float32):
    """Return a numpy array with a filled sphere."""
    arr = np.zeros(shape, dtype=dtype)
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    dist = np.sqrt(
        (zz - center[0]) ** 2 +
        (yy - center[1]) ** 2 +
        (xx - center[2]) ** 2
    )
    arr[dist <= radius] = inside_val
    return arr


def _make_sitk_image(array, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """Wrap a ZYX numpy array into a SimpleITK image with given metadata."""
    img = sitk.GetImageFromArray(array)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    return img


# ---------------------------------------------------------------------------
# 3-D image fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_image_3d():
    """32×32×16 float32 image with a sphere of intensity 1.0 in the centre."""
    shape = (16, 32, 32)  # ZYX
    center = (8, 16, 16)
    arr = _make_sphere_array(shape, center, radius=6, inside_val=1.0)
    # Add background gradient so registration has texture
    bg = np.random.default_rng(42).uniform(0.0, 0.1, shape).astype(np.float32)
    arr = arr + bg
    img = _make_sitk_image(arr, spacing=(1.0, 1.0, 1.0))
    return ImageWrapper(img)


@pytest.fixture
def synthetic_seg_3d():
    """32×32×16 label image with label 1 (outer sphere) and label 2 (inner sphere)."""
    shape = (16, 32, 32)  # ZYX
    center = (8, 16, 16)
    arr = np.zeros(shape, dtype=np.int16)
    # outer sphere → label 1
    arr += _make_sphere_array(shape, center, radius=6, inside_val=1, dtype=np.int16)
    # inner sphere → label 2 (overwrites inner part of label 1)
    arr += _make_sphere_array(shape, center, radius=3, inside_val=1, dtype=np.int16)
    arr = np.clip(arr, 0, 2).astype(np.int16)
    img = _make_sitk_image(arr, spacing=(1.0, 1.0, 1.0))
    return ImageWrapper(img)


# ---------------------------------------------------------------------------
# 4-D image fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_image_4d():
    """
    5-timepoint 4D image. Each TP is a 32×32×16 float32 volume with the same
    sphere slightly shifted in X to simulate cardiac motion.
    """
    shape = (16, 32, 32)  # ZYX
    spacing_4d = (1.0, 1.0, 1.0, 1.0)  # x, y, z, t

    volumes = []
    for tp in range(5):
        center = (8, 16, 16 + tp)  # 1-voxel shift per TP
        arr = _make_sphere_array(shape, center, radius=6, inside_val=1.0)
        bg = np.random.default_rng(tp).uniform(0.0, 0.1, shape).astype(np.float32)
        volumes.append(arr + bg)

    # Stack to (5, 16, 32, 32), then construct 4D SITK image
    stack = np.stack(volumes, axis=0)  # shape (T, Z, Y, X)
    img_4d = sitk.GetImageFromArray(stack)
    # SITK 4D: size = (X, Y, Z, T), spacing = (sx, sy, sz, st)
    img_4d.SetSpacing(spacing_4d)
    img_4d.SetOrigin((0.0, 0.0, 0.0, 0.0))
    return ImageWrapper(img_4d)


# ---------------------------------------------------------------------------
# Mesh fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_mesh():
    """A VTK sphere poly-data centered at (16, 16, 8) in image space."""
    from vtkmodules.vtkFiltersSources import vtkSphereSource
    src = vtkSphereSource()
    src.SetCenter(16.0, 16.0, 8.0)
    src.SetRadius(5.0)
    src.SetPhiResolution(12)
    src.SetThetaResolution(12)
    src.Update()
    return MeshWrapper(src.GetOutput())


# ---------------------------------------------------------------------------
# Warp / identity displacement field
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_warp_identity():
    """
    Identity dense displacement field shaped [1, D, H, W, 3] (all zeros).
    Matches a 32×32×16 image (W=32, H=32, D=16).
    """
    import torch
    return torch.zeros(1, 16, 32, 32, 3, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary directory for file output tests."""
    out = tmp_path / "output"
    out.mkdir()
    return out
