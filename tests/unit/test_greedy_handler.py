"""Unit tests for the Greedy registration handler and its options.

Option / resolve tests (TestGreedyRegistrationOptions,
TestGreedyHandlerResolveOptions) do not require picsl-greedy to be
installed — they only exercise pure Python dataclass and resolution
logic.

Handler integration tests (TestGreedyHandlerRegistration) do require
picsl-greedy and are therefore marked @pytest.mark.greedy.  They are
automatically skipped when the package is not installed.

Usage::

    # Run all greedy tests (skips integration tests if picsl-greedy absent)
    pytest tests/unit/test_greedy_handler.py -v

    # Run only the integration tests (requires picsl-greedy)
    pytest tests/unit/test_greedy_handler.py -m greedy -v
"""

import numpy as np
import pytest
import SimpleITK as sitk

from segflow4d.common.types.image_wrapper import ImageWrapper
from segflow4d.common.types.propagation_options import PropagationOptions
from segflow4d.registration.registration_handler.greedy.greedy_registration_options import (
    GreedyRegistrationOptions,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_sitk_sphere(shape=(16, 16, 16), radius=4, dtype=np.float32):
    """Return a SimpleITK image with a filled sphere of intensity 1.0."""
    arr = np.zeros(shape, dtype=dtype)
    c = [s // 2 for s in shape]
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    dist = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
    arr[dist <= radius] = 1.0
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    return img


def _make_seg_sphere(shape=(16, 16, 16), radius=4):
    """Return a SimpleITK int16 label image with label 1 inside a sphere."""
    arr = np.zeros(shape, dtype=np.int16)
    c = [s // 2 for s in shape]
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    dist = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
    arr[dist <= radius] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    return img


def _minimal_greedy_opts(**overrides):
    """GreedyRegistrationOptions with very few iterations for speed."""
    defaults = dict(affine_iterations=[5], deformable_iterations=[5])
    defaults.update(overrides)
    return GreedyRegistrationOptions(**defaults)


def _minimal_prop_opts(greedy_opts=None):
    """PropagationOptions wrapping a minimal GreedyRegistrationOptions."""
    return PropagationOptions(
        lowres_resample_factor=2.0,
        dilation_radius=2,
        registration_backend="greedy",
        registration_backend_options=greedy_opts or _minimal_greedy_opts(),
    )


# ---------------------------------------------------------------------------
# GreedyRegistrationOptions — no external deps needed
# ---------------------------------------------------------------------------

class TestGreedyRegistrationOptions:
    def test_default_construction(self):
        opts = GreedyRegistrationOptions()
        assert opts.metric == "NCC"
        assert opts.affine_dof == 12
        assert opts.verbosity == 0
        assert opts.threads is None

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric must be"):
            GreedyRegistrationOptions(metric="INVALID")

    def test_invalid_affine_dof_raises(self):
        with pytest.raises(ValueError, match="affine_dof must be"):
            GreedyRegistrationOptions(affine_dof=9)

    def test_valid_affine_dofs(self):
        for dof in (6, 7, 12):
            opts = GreedyRegistrationOptions(affine_dof=dof)
            assert opts.affine_dof == dof

    def test_invalid_verbosity_raises(self):
        with pytest.raises(ValueError, match="verbosity must be"):
            GreedyRegistrationOptions(verbosity=5)

    def test_negative_smooth_sigma_pre_raises(self):
        with pytest.raises(ValueError, match="smooth_sigma_pre_mm"):
            GreedyRegistrationOptions(smooth_sigma_pre_mm=-1.0)

    def test_negative_smooth_sigma_post_raises(self):
        with pytest.raises(ValueError, match="smooth_sigma_post_mm"):
            GreedyRegistrationOptions(smooth_sigma_post_mm=-0.1)

    def test_zero_threads_raises(self):
        with pytest.raises(ValueError, match="threads must be"):
            GreedyRegistrationOptions(threads=0)

    def test_positive_threads_accepted(self):
        opts = GreedyRegistrationOptions(threads=4)
        assert opts.threads == 4

    def test_empty_affine_iterations_raises(self):
        with pytest.raises(ValueError, match="affine_iterations"):
            GreedyRegistrationOptions(affine_iterations=[])

    def test_empty_deformable_iterations_raises(self):
        with pytest.raises(ValueError, match="deformable_iterations"):
            GreedyRegistrationOptions(deformable_iterations=[])

    def test_affine_schedule_single_level(self):
        opts = GreedyRegistrationOptions(affine_iterations=[100])
        assert opts.affine_schedule() == "100"

    def test_affine_schedule_multi_level(self):
        opts = GreedyRegistrationOptions(affine_iterations=[100, 50, 25])
        assert opts.affine_schedule() == "100x50x25"

    def test_deformable_schedule_format(self):
        opts = GreedyRegistrationOptions(deformable_iterations=[200, 100])
        assert opts.deformable_schedule() == "200x100"

    def test_metric_flag_ncc_default_radius(self):
        opts = GreedyRegistrationOptions(metric="NCC", metric_radius=[2, 2, 2])
        assert opts.metric_flag() == "NCC 2x2x2"

    def test_metric_flag_ncc_custom_radius(self):
        opts = GreedyRegistrationOptions(metric="NCC", metric_radius=[3, 3, 3])
        assert opts.metric_flag() == "NCC 3x3x3"

    def test_metric_flag_ssd(self):
        opts = GreedyRegistrationOptions(metric="SSD")
        assert opts.metric_flag() == "SSD"

    def test_metric_flag_nmi(self):
        opts = GreedyRegistrationOptions(metric="NMI")
        assert opts.metric_flag() == "NMI"


# ---------------------------------------------------------------------------
# GreedyRegistrationHandler._resolve_options — no external deps needed
# ---------------------------------------------------------------------------

class TestGreedyHandlerResolveOptions:
    """Tests for the _resolve_options static method (no picsl-greedy needed)."""

    @pytest.fixture(autouse=True)
    def handler_cls(self):
        from segflow4d.registration.registration_handler.greedy.greedy_registration_handler import (
            GreedyRegistrationHandler,
        )
        self.resolve = GreedyRegistrationHandler._resolve_options

    def test_resolve_from_propagation_options_with_greedy_opts(self):
        greedy_opts = GreedyRegistrationOptions(metric="SSD", affine_dof=6)
        prop_opts = PropagationOptions(
            lowres_resample_factor=2.0,
            dilation_radius=2,
            registration_backend="greedy",
            registration_backend_options=greedy_opts,
        )
        resolved = self.resolve(prop_opts)
        assert isinstance(resolved, GreedyRegistrationOptions)
        assert resolved.metric == "SSD"
        assert resolved.affine_dof == 6

    def test_resolve_from_dict_with_backend_options(self):
        options_dict = {
            "registration_backend_options": {
                "metric": "NMI",
                "affine_dof": 7,
                "deformable_iterations": [100, 50],
            }
        }
        resolved = self.resolve(options_dict)
        assert isinstance(resolved, GreedyRegistrationOptions)
        assert resolved.metric == "NMI"
        assert resolved.affine_dof == 7
        assert resolved.deformable_iterations == [100, 50]

    def test_resolve_from_empty_dict_uses_defaults(self):
        options_dict = {"registration_backend_options": {}}
        resolved = self.resolve(options_dict)
        assert isinstance(resolved, GreedyRegistrationOptions)
        assert resolved.metric == "NCC"  # default value

    def test_resolve_invalid_type_raises(self):
        prop_opts = PropagationOptions(
            lowres_resample_factor=2.0,
            dilation_radius=2,
            registration_backend="greedy",
            registration_backend_options=object(),  # invalid type
        )
        with pytest.raises((ValueError, TypeError)):
            self.resolve(prop_opts)


# ---------------------------------------------------------------------------
# GreedyRegistrationHandler — requires picsl-greedy (@pytest.mark.greedy)
# ---------------------------------------------------------------------------

@pytest.mark.greedy
class TestGreedyHandlerRegistration:
    """Integration tests for the full greedy registration pipeline.

    Requires picsl-greedy to be installed::

        pip install segflow4d[greedy]   # or: pip install picsl-greedy

    Automatically skipped when the package is missing.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_greedy(self):
        pytest.importorskip(
            "picsl_greedy",
            reason="picsl-greedy not installed — skipping greedy handler tests",
        )

    @pytest.fixture
    def handler(self):
        from segflow4d.registration.registration_handler.greedy.greedy_registration_handler import (
            GreedyRegistrationHandler,
        )
        return GreedyRegistrationHandler()

    def test_run_registration_returns_tp_data(self, handler):
        from segflow4d.common.types.tp_data import TPData

        img_fixed = ImageWrapper(_make_sitk_sphere())
        img_moving = ImageWrapper(_make_sitk_sphere())
        seg = ImageWrapper(_make_seg_sphere())

        result = handler.run_registration_and_reslice(
            img_fixed=img_fixed,
            img_moving=img_moving,
            img_to_reslice=seg,
            mesh_to_reslice=None,
            options=_minimal_prop_opts(),
        )

        assert isinstance(result, TPData)
        assert result.resliced_image is not None
        assert result.warp_image is not None
        assert result.affine_matrix is not None

    def test_resliced_image_shape_matches_fixed(self, handler):
        shape = (16, 16, 16)
        img_fixed = ImageWrapper(_make_sitk_sphere(shape=shape))
        img_moving = ImageWrapper(_make_sitk_sphere(shape=shape))
        seg = ImageWrapper(_make_seg_sphere(shape=shape))

        result = handler.run_registration_and_reslice(
            img_fixed=img_fixed,
            img_moving=img_moving,
            img_to_reslice=seg,
            mesh_to_reslice=None,
            options=_minimal_prop_opts(),
        )

        assert result.resliced_image.get_data().GetSize() == img_fixed.get_data().GetSize()

    def test_affine_matrix_is_numpy_array(self, handler):
        img_fixed = ImageWrapper(_make_sitk_sphere())
        img_moving = ImageWrapper(_make_sitk_sphere())
        seg = ImageWrapper(_make_seg_sphere())

        result = handler.run_registration_and_reslice(
            img_fixed=img_fixed,
            img_moving=img_moving,
            img_to_reslice=seg,
            mesh_to_reslice=None,
            options=_minimal_prop_opts(),
        )

        assert isinstance(result.affine_matrix, np.ndarray)

    def test_warp_field_is_image_wrapper(self, handler):
        img_fixed = ImageWrapper(_make_sitk_sphere())
        img_moving = ImageWrapper(_make_sitk_sphere())
        seg = ImageWrapper(_make_seg_sphere())

        result = handler.run_registration_and_reslice(
            img_fixed=img_fixed,
            img_moving=img_moving,
            img_to_reslice=seg,
            mesh_to_reslice=None,
            options=_minimal_prop_opts(),
        )

        assert isinstance(result.warp_image, ImageWrapper)

    def test_mask_application_does_not_crash(self, handler):
        """Passing masks should not raise an error."""
        shape = (16, 16, 16)
        img_fixed = ImageWrapper(_make_sitk_sphere(shape=shape))
        img_moving = ImageWrapper(_make_sitk_sphere(shape=shape))
        seg = ImageWrapper(_make_seg_sphere(shape=shape))
        mask = ImageWrapper(_make_sitk_sphere(shape=shape, dtype=np.uint8))

        result = handler.run_registration_and_reslice(
            img_fixed=img_fixed,
            img_moving=img_moving,
            img_to_reslice=seg,
            mesh_to_reslice=None,
            options=_minimal_prop_opts(),
            mask_fixed=mask,
            mask_moving=mask,
        )

        assert result.resliced_image is not None

    def test_get_device_type_is_cpu(self, handler):
        assert handler.get_device_type() == "cpu"

    def test_identity_registration_preserves_labels(self, handler):
        """Registering identical images should produce a seg close to the input."""
        shape = (16, 16, 16)
        seg_sitk = _make_seg_sphere(shape=shape)
        img = ImageWrapper(_make_sitk_sphere(shape=shape))
        seg = ImageWrapper(seg_sitk)

        result = handler.run_registration_and_reslice(
            img_fixed=img,
            img_moving=img,
            img_to_reslice=seg,
            mesh_to_reslice=None,
            options=_minimal_prop_opts(),
        )

        pred = sitk.GetArrayFromImage(result.resliced_image.get_data()).astype(np.int32)
        gt = sitk.GetArrayFromImage(seg_sitk).astype(np.int32)

        intersection = np.sum((pred == 1) & (gt == 1))
        union = np.sum((pred == 1) | (gt == 1))
        dice = 2 * intersection / (np.sum(pred == 1) + np.sum(gt == 1) + 1e-6)
        assert dice >= 0.75, f"Identity registration Dice {dice:.3f} < 0.75"
