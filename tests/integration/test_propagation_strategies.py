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
