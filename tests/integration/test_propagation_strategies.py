"""Integration tests for propagation strategies.

The registration backend is replaced by a mock so no GPU or real
registration is invoked.  We verify orchestration logic only:
- The correct timepoints are submitted to the backend
- Results are collected and written back into TPData correctly
"""

import concurrent.futures
import pytest
from unittest.mock import MagicMock, patch

from segflow4d.common.types.image_wrapper import ImageWrapper
from segflow4d.common.types.mesh_wrapper import MeshWrapper
from segflow4d.common.types.tp_data import TPData
from segflow4d.common.types.propagation_options import PropagationOptions
from segflow4d.propagation.propagation_strategy.sequential_propagation_strategy import (
    SequentialPropagationStrategy,
)
from segflow4d.propagation.propagation_strategy.star_propagation_strategy import (
    StarPropagationStrategy,
)
from segflow4d.propagation.propagation_strategy.sasd_propagation_strategy import (
    SASDPropagationStrategy,
)
from segflow4d.common.types.registration_methods import REGISTRATION_METHODS

import SimpleITK as sitk
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image_wrapper():
    arr = np.zeros((8, 8, 4), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return ImageWrapper(img)


def _make_mesh_wrapper():
    from vtkmodules.vtkFiltersSources import vtkSphereSource
    src = vtkSphereSource()
    src.SetRadius(1.0)
    src.Update()
    return MeshWrapper(src.GetOutput())


def _make_identity_result():
    """Return a TPData that looks like a successful identity registration result."""
    result = TPData()
    result.resliced_image = _make_image_wrapper()
    result.resliced_segmentation_mesh = _make_mesh_wrapper()
    result.warp_image = _make_image_wrapper()
    return result


def _make_options():
    return PropagationOptions(
        lowres_scale_factor=0.5,
        dilation_radius=2,
        registration_backend="FIREANTS",
        registration_backend_options={},
        write_result_to_disk=False,
        output_directory="",
        debug=False,
        debug_output_directory="",
        minimum_required_vram_gb=0,
    )


def _make_tp_data_dict(tp_list, ref_tp):
    """Build a minimal dict[int, TPData] suitable for strategy input."""
    data = {}
    for tp in tp_list:
        td = TPData()
        td.image = _make_image_wrapper()
        td.image_low_res = _make_image_wrapper()
        td.mask = _make_image_wrapper()
        if tp == ref_tp:
            td.segmentation = _make_image_wrapper()
            td.segmentation_mesh = _make_mesh_wrapper()
            td.resliced_image = _make_image_wrapper()
        data[tp] = td
    return data


# ---------------------------------------------------------------------------
# Mock future helper
# ---------------------------------------------------------------------------

def _make_future(value):
    f = concurrent.futures.Future()
    f.set_result(value)
    return f


# ---------------------------------------------------------------------------
# Sequential strategy tests
# ---------------------------------------------------------------------------

class TestSequentialPropagationStrategy:
    def _make_mock_manager(self, result_factory=None):
        manager = MagicMock()
        if result_factory is None:
            manager.submit.return_value = _make_future(_make_identity_result())
        else:
            manager.submit.side_effect = lambda *a, **kw: _make_future(result_factory())
        return manager

    def test_processes_each_tp_pair_once(self):
        """submit() must be called exactly N-1 times for N timepoints."""
        tp_list = [0, 1, 2, 3]
        ref_tp = 0
        tp_data = _make_tp_data_dict(tp_list, ref_tp)
        mock_manager = self._make_mock_manager()

        with patch(
            "segflow4d.propagation.propagation_strategy.sequential_propagation_strategy.RegistrationManager"
        ) as mock_rm_class:
            mock_rm_class.get_instance.return_value = mock_manager
            strategy = SequentialPropagationStrategy()
            strategy.propagate(tp_data, _make_options())

        assert mock_manager.submit.call_count == len(tp_list) - 1

    def test_processes_in_order(self):
        """Source TPs in submit() calls must match sequential order."""
        tp_list = [0, 1, 2, 3]
        ref_tp = 0
        tp_data = _make_tp_data_dict(tp_list, ref_tp)
        mock_manager = self._make_mock_manager()

        with patch(
            "segflow4d.propagation.propagation_strategy.sequential_propagation_strategy.RegistrationManager"
        ) as mock_rm_class:
            mock_rm_class.get_instance.return_value = mock_manager
            strategy = SequentialPropagationStrategy()
            strategy.propagate(tp_data, _make_options())

        calls = mock_manager.submit.call_args_list
        for i, call in enumerate(calls):
            kw = call.kwargs
            # fixed image must be tp_list[i+1], moving must be tp_list[i]
            assert kw["img_fixed"] is tp_data[tp_list[i + 1]].image
            assert kw["img_moving"] is tp_data[tp_list[i]].image

    def test_resliced_image_propagated_to_next_tp(self):
        """Each TP's resliced_image must be set after propagation completes."""
        tp_list = [0, 1, 2]
        ref_tp = 0
        tp_data = _make_tp_data_dict(tp_list, ref_tp)
        mock_manager = self._make_mock_manager()

        with patch(
            "segflow4d.propagation.propagation_strategy.sequential_propagation_strategy.RegistrationManager"
        ) as mock_rm_class:
            mock_rm_class.get_instance.return_value = mock_manager
            strategy = SequentialPropagationStrategy()
            result_data = strategy.propagate(tp_data, _make_options())

        for tp in tp_list[1:]:
            assert result_data[tp].resliced_image is not None, (
                f"TP {tp} should have resliced_image after propagation"
            )


# ---------------------------------------------------------------------------
# Star strategy tests
# ---------------------------------------------------------------------------

class TestStarPropagationStrategy:
    def _make_mock_manager(self):
        manager = MagicMock()
        manager.submit.side_effect = lambda *a, **kw: _make_future(_make_identity_result())
        return manager

    def test_all_tps_submitted(self):
        """Star strategy must submit one job per non-ref TP."""
        tp_list = [0, 1, 2, 3, 4]
        ref_tp = 0
        tp_data = _make_tp_data_dict(tp_list, ref_tp)
        mock_manager = self._make_mock_manager()

        with patch(
            "segflow4d.propagation.propagation_strategy.star_propagation_strategy.RegistrationManager"
        ) as mock_rm_class:
            mock_rm_class.get_instance.return_value = mock_manager
            strategy = StarPropagationStrategy()
            strategy.propagate(tp_data, _make_options())

        assert mock_manager.submit.call_count == len(tp_list) - 1

    def test_all_target_tps_have_resliced_segmentation(self):
        """After star propagation every target TP must have a segmentation mesh."""
        tp_list = [0, 1, 2, 3]
        ref_tp = 0
        tp_data = _make_tp_data_dict(tp_list, ref_tp)
        mock_manager = self._make_mock_manager()

        with patch(
            "segflow4d.propagation.propagation_strategy.star_propagation_strategy.RegistrationManager"
        ) as mock_rm_class:
            mock_rm_class.get_instance.return_value = mock_manager
            strategy = StarPropagationStrategy()
            result = strategy.propagate(tp_data, _make_options())

        for tp in tp_list[1:]:
            assert result[tp].resliced_segmentation_mesh is not None, (
                f"TP {tp} missing resliced_segmentation_mesh"
            )


# ---------------------------------------------------------------------------
# SASD strategy tests
# ---------------------------------------------------------------------------

class TestSASDPropagationStrategy:
    """SASD: a sequential affine chain, composed per target, then a parallel
    deformable star initialised with each target's composed affine.

    With ``sequential_sasd`` SASD is now the high-res path, so it carries the
    segmentation *and* the caller's additional meshes. sf4d-200 KNOWN_ISSUES B2:
    it used to drop the additional meshes silently.
    """

    # Three non-commuting step affines, so a wrong composition order is caught.
    ROT_Z_90 = np.array([[0., -1., 0., 0.], [1., 0., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    SHIFT_X = np.array([[1., 0., 0., 1.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    SCALE_X = np.diag([2., 1., 1., 1.])

    def _run(self, tp_list, step_affines):
        ref_tp = tp_list[0]
        tp_data = _make_tp_data_dict(tp_list, ref_tp)
        ref_meshes = {"root": _make_mesh_wrapper()}
        tp_data[ref_tp].additional_meshes = ref_meshes
        returned_meshes = {"root": _make_mesh_wrapper()}
        fixed_image_to_tp = {id(tp_data[tp].image): tp for tp in tp_list}

        affines = iter(step_affines)  # phase 1 is sequential, so call order = chain order
        deformable_calls: dict[int, dict] = {}

        def submit(method, **kwargs):
            if method == REGISTRATION_METHODS.RUN_AFFINE_ONLY:
                return _make_future(TPData(affine_matrix=next(affines)))
            deformable_calls[fixed_image_to_tp[id(kwargs["img_fixed"])]] = kwargs
            result = _make_identity_result()
            result.resliced_meshes = returned_meshes
            return _make_future(result)

        manager = MagicMock()
        manager.submit.side_effect = submit
        with patch(
            "segflow4d.propagation.propagation_strategy.sasd_propagation_strategy.RegistrationManager"
        ) as mock_rm_class:
            mock_rm_class.get_instance.return_value = manager
            result = SASDPropagationStrategy().propagate(tp_data, _make_options())

        return result, deformable_calls, ref_meshes, returned_meshes, tp_data

    def test_additional_meshes_are_resliced_for_every_target(self):
        tp_list = [0, 1, 2, 3]
        result, calls, ref_meshes, returned_meshes, _ = self._run(
            tp_list, [self.ROT_Z_90, self.SHIFT_X, self.SCALE_X]
        )

        assert sorted(calls) == tp_list[1:]
        for tp in tp_list[1:]:
            assert calls[tp]["additional_meshes_to_reslice"] is ref_meshes, (
                f"TP {tp}: the reference's additional meshes were not handed to the backend"
            )
            assert result[tp].resliced_meshes is returned_meshes, (
                f"TP {tp}: resliced_meshes not written back"
            )

    def test_deformable_init_is_the_chained_affine_rightmost_first(self):
        """Target k is initialised with A_k @ ... @ A_1, as the class docstring states.

        This pins the documented order. Whether that order matches each
        backend's affine convention is a separate, open question (B7 / H5).
        """
        tp_list = [0, 1, 2, 3]
        a1, a2, a3 = self.ROT_Z_90, self.SHIFT_X, self.SCALE_X
        _, calls, _, _, _ = self._run(tp_list, [a1, a2, a3])

        expected = {1: a1, 2: a2 @ a1, 3: a3 @ a2 @ a1}
        for tp, matrix in expected.items():
            np.testing.assert_allclose(calls[tp]["init_affine_matrix"], matrix)
        assert not np.allclose(a1 @ a2, a2 @ a1), "test affines must not commute"

    def test_deformable_stage_registers_every_target_against_the_reference(self):
        """Star, not chain: each target's deformable moving image is the reference's,
        not its neighbour's, so a deformable error cannot accumulate along the chain."""
        tp_list = [5, 6, 7, 8]
        _, calls, _, _, tp_data = self._run(tp_list, [self.ROT_Z_90, self.SHIFT_X, self.SCALE_X])

        ref = tp_data[5]
        for tp in tp_list[1:]:
            assert calls[tp]["img_moving"] is ref.image
            assert calls[tp]["img_to_reslice"] is ref.resliced_image
            assert calls[tp]["mesh_to_reslice"] is ref.segmentation_mesh
