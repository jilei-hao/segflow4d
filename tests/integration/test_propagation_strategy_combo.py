"""Integration tests: propagation_strategy_combo cross-examination.

Verifies that PropagationPipeline._run_unidirectional_propagation selects
the correct low-res and high-res propagation strategies for every
(propagation_strategy_combo, registration_backend) combination.

Combos:
  sequential_star  →  LR = Sequential,  HR = Star
  sasd_star        →  LR = SASD,         HR = Star

Backends: fireants (GPU-backed), greedy (CPU-backed)

No real registration is performed; both the RegistrationManager singleton
and the strategy implementations are replaced with lightweight mocks.
"""

import concurrent.futures
import pytest
import numpy as np
import SimpleITK as sitk
from unittest.mock import MagicMock, patch

from segflow4d.common.types.image_wrapper import ImageWrapper
from segflow4d.common.types.tp_data import TPData
from segflow4d.common.types.propagation_options import PropagationOptions
from segflow4d.common.types.propagation_strategy_name import PropagationStrategyName
from segflow4d.propagation.propagation_pipeline import PropagationPipeline
from segflow4d.propagation.propagation_strategy.propagation_strategy_factory import PropagationStrategyFactory
from segflow4d.propagation.tp_partition_input import TPPartitionInput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image_wrapper():
    arr = np.zeros((8, 16, 16), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    return ImageWrapper(img)


def _make_tp_data(tp_list: list[int], ref_tp: int) -> dict[int, TPData]:
    """Build a minimal tp_data dict for _run_unidirectional_propagation."""
    data = {}
    for tp in tp_list:
        td = TPData(
            image=_make_image_wrapper(),
            image_low_res=_make_image_wrapper(),
        )
        if tp == ref_tp:
            td.mask_low_res = _make_image_wrapper()
        data[tp] = td
    return data


def _make_options(backend: str, combo: str) -> PropagationOptions:
    return PropagationOptions(
        lowres_scale_factor=0.5,
        dilation_radius=2,
        registration_backend=backend,
        registration_backend_options={},
        propagation_strategy_combo=combo,
    )


def _make_mock_propagation_input(options: PropagationOptions) -> MagicMock:
    mock_input = MagicMock()
    mock_input.options = options
    mock_input.image_4d = MagicMock()
    mock_input.tp_input_groups = []
    return mock_input


def _make_mock_ref_input() -> MagicMock:
    ref = MagicMock(spec=TPPartitionInput)
    ref.seg_ref = _make_image_wrapper()
    ref.seg_mesh_ref = None
    return ref


def _make_mock_strategy():
    """
    Return a mock strategy whose propagate() sets resliced_image on every TP.

    The LR stage requires resliced_image to be non-None for all timepoints
    (propagation_pipeline raises RuntimeError otherwise).
    """
    mock_strategy = MagicMock()

    def fake_propagate(tp_input_data: dict, options):
        for tp in tp_input_data:
            tp_input_data[tp].resliced_image = _make_image_wrapper()
        return tp_input_data

    mock_strategy.propagate.side_effect = fake_propagate
    return mock_strategy


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

BACKENDS = ["fireants", "greedy"]

COMBO_EXPECTED_LR = [
    ("sequential_star", PropagationStrategyName.SEQUENTIAL),
    ("sasd_star",       PropagationStrategyName.SASD),
]


# ---------------------------------------------------------------------------
# Tests: correct strategy selected for each (combo, backend) pair
# ---------------------------------------------------------------------------

class TestStrategyComboSelection:
    """_run_unidirectional_propagation must select strategies matching the combo."""

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("combo,expected_lr", COMBO_EXPECTED_LR)
    def test_lr_strategy_matches_combo(self, backend, combo, expected_lr):
        """Low-res strategy name must match the combo; high-res must be STAR."""
        options = _make_options(backend, combo)
        mock_input = _make_mock_propagation_input(options)
        mock_ref_input = _make_mock_ref_input()
        tp_list = [0, 1, 2]
        tp_data = _make_tp_data(tp_list, ref_tp=0)

        requested_names: list[str] = []

        def fake_create_strategy(name: str):
            requested_names.append(name)
            return _make_mock_strategy()

        with patch(
            "segflow4d.propagation.propagation_pipeline.RegistrationManager"
        ) as MockRM, patch.object(
            PropagationStrategyFactory,
            "create_propagation_strategy",
            side_effect=fake_create_strategy,
        ), patch(
            "segflow4d.propagation.propagation_pipeline.create_high_res_mask",
            return_value=_make_image_wrapper(),
        ):
            MockRM.return_value = MagicMock()
            pipeline = PropagationPipeline(mock_input)
            pipeline._run_unidirectional_propagation(
                tp_data=tp_data,
                ref_input=mock_ref_input,
                tp_list=tp_list,
                options=options,
            )

        assert len(requested_names) == 2, (
            f"Expected 2 strategy factory calls, got {len(requested_names)}: {requested_names}"
        )
        assert requested_names[0] == expected_lr, (
            f"combo={combo!r}, backend={backend!r}: "
            f"expected LR strategy {expected_lr!r}, got {requested_names[0]!r}"
        )
        assert requested_names[1] == PropagationStrategyName.STAR, (
            f"combo={combo!r}, backend={backend!r}: "
            f"expected HR strategy {PropagationStrategyName.STAR!r}, got {requested_names[1]!r}"
        )

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("combo,_", COMBO_EXPECTED_LR)
    def test_both_strategies_propagate_called_once(self, backend, combo, _):
        """Both the LR and HR strategy.propagate() methods are called exactly once."""
        options = _make_options(backend, combo)
        mock_input = _make_mock_propagation_input(options)
        mock_ref_input = _make_mock_ref_input()
        tp_list = [0, 1, 2]
        tp_data = _make_tp_data(tp_list, ref_tp=0)

        created_strategies: list[MagicMock] = []

        def fake_create_strategy(name: str):
            s = _make_mock_strategy()
            created_strategies.append(s)
            return s

        with patch(
            "segflow4d.propagation.propagation_pipeline.RegistrationManager"
        ) as MockRM, patch.object(
            PropagationStrategyFactory,
            "create_propagation_strategy",
            side_effect=fake_create_strategy,
        ), patch(
            "segflow4d.propagation.propagation_pipeline.create_high_res_mask",
            return_value=_make_image_wrapper(),
        ):
            MockRM.return_value = MagicMock()
            pipeline = PropagationPipeline(mock_input)
            pipeline._run_unidirectional_propagation(
                tp_data=tp_data,
                ref_input=mock_ref_input,
                tp_list=tp_list,
                options=options,
            )

        assert len(created_strategies) == 2
        for i, s in enumerate(created_strategies):
            s.propagate.assert_called_once(), f"Strategy {i} propagate() not called once"

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("combo,expected_lr", COMBO_EXPECTED_LR)
    def test_result_contains_all_timepoints(self, backend, combo, expected_lr):
        """The returned dict must contain entries for every TP in tp_list."""
        options = _make_options(backend, combo)
        mock_input = _make_mock_propagation_input(options)
        mock_ref_input = _make_mock_ref_input()
        tp_list = [0, 1, 2]
        tp_data = _make_tp_data(tp_list, ref_tp=0)

        with patch(
            "segflow4d.propagation.propagation_pipeline.RegistrationManager"
        ) as MockRM, patch.object(
            PropagationStrategyFactory,
            "create_propagation_strategy",
            side_effect=lambda name: _make_mock_strategy(),
        ), patch(
            "segflow4d.propagation.propagation_pipeline.create_high_res_mask",
            return_value=_make_image_wrapper(),
        ):
            MockRM.return_value = MagicMock()
            pipeline = PropagationPipeline(mock_input)
            result = pipeline._run_unidirectional_propagation(
                tp_data=tp_data,
                ref_input=mock_ref_input,
                tp_list=tp_list,
                options=options,
            )

        assert set(result.keys()) == set(tp_list), (
            f"Result keys {set(result.keys())} do not match tp_list {set(tp_list)}"
        )


# ---------------------------------------------------------------------------
# Test: unknown combo raises ValueError for both backends
# ---------------------------------------------------------------------------

class TestUnknownComboRaisesError:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_invalid_combo_raises_value_error(self, backend):
        """An unrecognised combo string must raise ValueError."""
        options = _make_options(backend, "not_a_valid_combo")
        mock_input = _make_mock_propagation_input(options)
        mock_ref_input = _make_mock_ref_input()
        tp_list = [0, 1, 2]
        tp_data = _make_tp_data(tp_list, ref_tp=0)

        with patch(
            "segflow4d.propagation.propagation_pipeline.RegistrationManager"
        ) as MockRM:
            MockRM.return_value = MagicMock()
            pipeline = PropagationPipeline(mock_input)

        with pytest.raises(ValueError, match="Unknown propagation_strategy_combo"):
            pipeline._run_unidirectional_propagation(
                tp_data=tp_data,
                ref_input=mock_ref_input,
                tp_list=tp_list,
                options=options,
            )


# ---------------------------------------------------------------------------
# Test: PropagationOptions combo defaults and round-trips
# ---------------------------------------------------------------------------

class TestPropagationOptionsCombo:
    def test_default_combo_is_sequential_star(self):
        """PropagationOptions.propagation_strategy_combo defaults to 'sequential_star'."""
        opts = PropagationOptions(
            lowres_scale_factor=0.5,
            dilation_radius=2,
            registration_backend="greedy",
            registration_backend_options={},
        )
        assert opts.propagation_strategy_combo == "sequential_star"

    @pytest.mark.parametrize("combo", ["sequential_star", "sasd_star"])
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_combo_stored_correctly(self, combo, backend):
        """Combo value set in PropagationOptions is readable unchanged."""
        opts = _make_options(backend, combo)
        assert opts.propagation_strategy_combo == combo

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_backend_stored_correctly(self, backend):
        """registration_backend value is stored correctly in PropagationOptions."""
        opts = _make_options(backend, "sequential_star")
        assert opts.registration_backend == backend
