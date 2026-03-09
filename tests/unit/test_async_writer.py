"""Unit tests for the AsyncWriter file-writing utility.

Each test creates a fresh AsyncWriter instance (not the global singleton)
to avoid cross-test state contamination.
"""

import os
import pytest
import SimpleITK as sitk
import numpy as np

from segflow4d.utility.file_writer.async_writer import AsyncWriter
from segflow4d.common.types.image_wrapper import ImageWrapper
from segflow4d.common.types.mesh_wrapper import MeshWrapper


def _make_simple_image():
    arr = np.zeros((8, 8, 4), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    return ImageWrapper(img)


def _make_simple_mesh():
    from vtkmodules.vtkFiltersSources import vtkSphereSource
    src = vtkSphereSource()
    src.SetRadius(1.0)
    src.Update()
    return MeshWrapper(src.GetOutput())


class TestAsyncWriterImage:
    def test_submit_image_creates_nifti_file(self, tmp_output_dir):
        writer = AsyncWriter()
        out_path = str(tmp_output_dir / "test_image.nii.gz")
        writer.submit_image(_make_simple_image(), out_path)
        writer.shutdown(wait=True)
        assert os.path.isfile(out_path), f"Expected file at {out_path}"

    def test_multiple_image_writes_all_complete(self, tmp_output_dir):
        writer = AsyncWriter()
        n = 8
        paths = [str(tmp_output_dir / f"img_{i:02d}.nii.gz") for i in range(n)]
        for p in paths:
            writer.submit_image(_make_simple_image(), p)
        writer.shutdown(wait=True)
        for p in paths:
            assert os.path.isfile(p), f"Missing file: {p}"


class TestAsyncWriterMesh:
    def test_submit_mesh_creates_vtp_file(self, tmp_output_dir):
        writer = AsyncWriter()
        out_path = str(tmp_output_dir / "test_mesh.vtp")
        writer.submit_mesh(_make_simple_mesh(), out_path)
        writer.shutdown(wait=True)
        assert os.path.isfile(out_path), f"Expected file at {out_path}"

    def test_submit_vtk_mesh_creates_vtk_file(self, tmp_output_dir):
        writer = AsyncWriter()
        out_path = str(tmp_output_dir / "test_mesh.vtk")
        writer.submit_mesh(_make_simple_mesh(), out_path)
        writer.shutdown(wait=True)
        assert os.path.isfile(out_path), f"Expected file at {out_path}"


class TestAsyncWriterShutdown:
    def test_shutdown_waits_for_all_tasks(self, tmp_output_dir):
        """shutdown(wait=True) must not return until all queued tasks are done."""
        writer = AsyncWriter()
        n = 12
        paths = [str(tmp_output_dir / f"shutdown_{i:02d}.nii.gz") for i in range(n)]
        for p in paths:
            writer.submit_image(_make_simple_image(), p)
        writer.shutdown(wait=True)  # returns only after all writes finish
        missing = [p for p in paths if not os.path.isfile(p)]
        assert len(missing) == 0, f"Missing files after shutdown: {missing}"
