"""Warped meshes must follow the anatomy, for every backend and both high-res strategies
(sf4d-200 KNOWN_ISSUES B7, B11, B12).

The earlier additional-mesh e2e tests only checked that the output files exist and
have the right vertex count. That let a greedy bug through. The greedy handler
warped meshes with the forward deformable field alone, which dropped the affine and
pushed the small residual the wrong way, so meshes barely moved while the
segmentation tracked the motion correctly. The ANTs handler had the same bug,
and it also could not run in the pipeline at all: it did not accept the
additional_meshes_to_reslice keyword that the strategies pass.

Here a textured ellipsoid moves and stretches between timepoints, and the image has a
non-zero origin, with an identity or a flipped direction. The reference TP's surface,
in LPS physical coordinates, goes in as an additional mesh through the in-memory API
that avrp-handler uses. At each target TP the warped mesh must:
  - be much closer to the true (analytic) surface than the unwarped mesh is;
  - have its centroid much closer to the true centre than the unwarped mesh does;
  - lie on the propagated segmentation's surface (mesh and segmentation agree).
"""

import os

import numpy as np
import pytest
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from segflow4d.common.types.propagation_input import PropagationInputFactory
from segflow4d.propagation.propagation_pipeline import PropagationPipeline

SHAPE_ZYX = (64, 64, 64)
ORIGIN = np.array([-30.0, 12.0, 5.0])
N_TP = 3
STEP_CENTRE = np.array([1.5, -1.0, 0.75])   # mm per TP, physical space
STEP_RADIUS_Y = 0.8                          # mm per TP
R0 = np.array([7.0, 7.0, 7.0])

# Ratio of warped to unwarped error that still counts as "follows". Measured after the
# fix: <= 0.33 (greedy) and <= 0.2 (FireANTs). Before the fix: about 1.0.
FOLLOW_RATIO = 0.5
# Mesh-to-segmentation boundary distance, sampled at the nearest voxel of a 1 mm
# distance map. About 0.6 mm here even when the two agree, so 1 mm is a 1-voxel bound.
MESH_SEG_MAX_MM = 1.0

BACKENDS = [
    # Seeded: unseeded greedy is not reproducible (KNOWN_ISSUES H1). On this case
    # its SASD result sometimes landed just over the follow ratio.
    pytest.param("greedy", {"metric": "SSD", "affine_dof": 6,
                            "smooth_sigma_pre_mm": 2.0, "smooth_sigma_post_mm": 0.5,
                            "affine_iterations": [100, 50],
                            "deformable_iterations": [100, 50, 25], "seed": 7},
                 marks=pytest.mark.greedy, id="greedy"),
    # FireANTs multi-scale fails on volumes this small (FFT downsample to 0 points), so
    # single scale, as the other synthetic FireANTs e2e tests do.
    pytest.param("fireants", {"scales": [1], "affine_iterations": [200],
                              "deformable_iterations": [200], "loss_type": "mse"},
                 marks=pytest.mark.gpu, id="fireants"),
    # A 2-level affine schedule, given as lists the way YAML delivers it. antspyx
    # ignored list schedules until the handler passed them on as tuples. Seeded:
    # unseeded ANTs affine sampling now and then yields a bad SASD step affine,
    # which SASD compounds (KNOWN_ISSUES B14). That is a robustness question; this
    # test is about the transform chain. The seed reaches ANTs only since the B15
    # fix. Multithreaded reductions still leave runs slightly non-identical.
    pytest.param("ants", {"aff_iterations": [100, 50], "aff_shrink_factors": [2, 1],
                          "aff_smoothing_sigmas": [1, 0], "reg_iterations": [40, 20],
                          "random_seed": 7},
                 marks=pytest.mark.ants, id="ants"),
]
# Star registers each target to the reference directly. SASD initialises each
# deformable registration with a chained affine, which takes a different chain
# through every backend's reslice and mesh code.
COMBOS = ["sequential_star", "sequential_sasd"]
DIRECTIONS = [
    pytest.param(np.eye(3), id="identity-direction"),
    pytest.param(np.diag([1.0, -1.0, 1.0]), id="flipped-y-direction"),
]


def _centre0(direction):
    return ORIGIN + direction @ (np.array(SHAPE_ZYX[::-1], dtype=float) / 2 - np.array([4.0, 2.0, 2.0]))


def _truth(t, direction):
    return _centre0(direction) + STEP_CENTRE * t, R0 + np.array([0.0, STEP_RADIUS_Y * t, 0.0])


def _ellipsoid(center, radii, direction):
    zz, yy, xx = np.mgrid[0:SHAPE_ZYX[0], 0:SHAPE_ZYX[1], 0:SHAPE_ZYX[2]].astype(np.float64)
    phys = ORIGIN + np.stack([xx, yy, zz], axis=-1) @ direction.T
    rho = np.sqrt((((phys - center) / radii) ** 2).sum(axis=-1))
    texture = np.where(rho <= 1.0, 0.6 + 0.4 * np.cos(4 * np.pi * rho), 0.0).astype(np.float32)
    return texture, rho <= 1.0


def _image(arr, direction):
    img = sitk.GetImageFromArray(arr)
    img.SetOrigin(tuple(ORIGIN))
    img.SetDirection(tuple(direction.flatten()))
    return img


def _surface(center, radii):
    src = vtk.vtkSphereSource()
    src.SetRadius(1.0)
    src.SetThetaResolution(32)
    src.SetPhiResolution(32)
    src.Update()
    poly = vtk.vtkPolyData()
    poly.DeepCopy(src.GetOutput())
    pts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64) * radii + center
    poly.GetPoints().SetData(numpy_to_vtk(pts, deep=True))
    return poly


def _errors(pts, center, radii):
    """(mean |normalised radius - 1| x R0 mm, centroid distance mm) against the truth."""
    rho = np.sqrt((((pts - center) / radii) ** 2).sum(axis=1))
    return np.abs(rho - 1).mean() * R0[0], np.linalg.norm(pts.mean(axis=0) - center)


def _read_points(path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    return vtk_to_numpy(reader.GetOutput().GetPoints().GetData()).astype(np.float64)


def _mesh_to_seg_mm(pts, seg_frame):
    sdf = sitk.SignedMaurerDistanceMap(seg_frame > 0, insideIsPositive=False,
                                       squaredDistance=False, useImageSpacing=True)
    arr = sitk.GetArrayFromImage(sdf)
    size = np.array(sdf.GetSize()) - 1
    dists = []
    for p in pts:
        i = np.clip(np.round(sdf.TransformPhysicalPointToContinuousIndex(tuple(p))).astype(int), 0, size)
        dists.append(abs(arr[i[2], i[1], i[0]]))
    return float(np.mean(dists))


@pytest.mark.slow
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("combo", COMBOS)
@pytest.mark.parametrize("backend,backend_options", BACKENDS)
def test_additional_mesh_follows_the_anatomy(tmp_path, backend, backend_options, combo, direction):
    if backend == "greedy":
        pytest.importorskip("picsl_greedy", reason="picsl-greedy not installed")
    if backend == "ants":
        pytest.importorskip("ants", reason="antspyx not installed")

    frames = [_image(_ellipsoid(*_truth(t, direction), direction)[0], direction) for t in range(N_TP)]
    seg_ref = _image(_ellipsoid(*_truth(0, direction), direction)[1].astype(np.int16), direction)
    mesh_ref = _surface(*_truth(0, direction))
    ref_pts = vtk_to_numpy(mesh_ref.GetPoints().GetData()).astype(np.float64).copy()

    out_dir = str(tmp_path / "out")
    factory = PropagationInputFactory()
    factory.set_image_4d(sitk.JoinSeries(frames))
    factory.add_tp_input_group(tp_ref=1, tp_target=list(range(2, N_TP + 1)), seg_ref=seg_ref,
                               additional_meshes_ref={"mesh1": mesh_ref})
    factory.set_options(lowres_factor=0.5, registration_backend=backend, dilation_radius=4,
                        write_result_to_disk=True, output_directory=out_dir, debug=False,
                        debug_output_directory="", minimum_required_vram_gb=0,
                        propagation_strategy_combo=combo, roi_crop_padding_voxels=0,
                        **backend_options)
    PropagationPipeline(factory.build()).run()

    seg4d = sitk.ReadImage(os.path.join(out_dir, "seg-4d.nii.gz"))
    for tp in range(2, N_TP + 1):
        center, radii = _truth(tp - 1, direction)
        warped = _read_points(os.path.join(out_dir, "additional-mesh", f"mesh1_tp-{tp:03d}.vtp"))
        surf, ctr = _errors(warped, center, radii)
        surf_still, ctr_still = _errors(ref_pts, center, radii)
        assert surf < FOLLOW_RATIO * surf_still, (
            f"TP {tp}: mesh surface error {surf:.3f} mm vs {surf_still:.3f} mm unwarped; "
            f"the mesh is not following the motion"
        )
        assert ctr < FOLLOW_RATIO * ctr_still, (
            f"TP {tp}: mesh centroid error {ctr:.3f} mm vs {ctr_still:.3f} mm unwarped"
        )

        frame = sitk.Extract(seg4d, list(seg4d.GetSize()[:3]) + [0], [0, 0, 0, tp - 1])
        frame.CopyInformation(frames[tp - 1])
        gap = _mesh_to_seg_mm(warped, frame)
        assert gap < MESH_SEG_MAX_MM, (
            f"TP {tp}: warped mesh is {gap:.3f} mm from the propagated segmentation's surface"
        )
