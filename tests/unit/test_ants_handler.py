"""Unit tests for the classic ANTs (antspyx) registration handler and its options.

Option / resolve tests (TestAntsRegistrationOptions,
TestAntsHandlerResolveOptions) do not require antspyx to be installed —
they only exercise pure Python dataclass and resolution logic.

Handler integration tests (TestAntsHandlerRegistration) do require
antspyx and are therefore marked ``@pytest.mark.ants``.  They are
automatically skipped when the package is not installed.

Usage::

    # Run all ants tests (skips integration tests if antspyx absent)
    pytest tests/unit/test_ants_handler.py -v

    # Run only the integration tests (requires antspyx)
    pytest tests/unit/test_ants_handler.py -m ants -v
"""

import numpy as np
import pytest
import SimpleITK as sitk

from segflow4d.common.types.image_wrapper import ImageWrapper
from segflow4d.common.types.propagation_options import PropagationOptions
from segflow4d.registration.registration_handler.ants.ants_registration_options import (
    AntsRegistrationOptions,
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


def _minimal_ants_opts(**overrides):
    """AntsRegistrationOptions with very small schedules for fast tests."""
    defaults = dict(
        transform_type='SyN',
        aff_iterations=(20, 10),
        reg_iterations=(10, 5),
        aff_shrink_factors=(2, 1),
        aff_smoothing_sigmas=(1, 0),
    )
    defaults.update(overrides)
    return AntsRegistrationOptions(**defaults)


def _minimal_prop_opts(ants_opts=None):
    return PropagationOptions(
        lowres_scale_factor=0.5,
        dilation_radius=2,
        registration_backend="ants",
        registration_backend_options=ants_opts or _minimal_ants_opts(),
    )


# ---------------------------------------------------------------------------
# AntsRegistrationOptions — no external deps needed
# ---------------------------------------------------------------------------

class TestAntsRegistrationOptions:
    def test_default_construction(self):
        opts = AntsRegistrationOptions()
        assert opts.transform_type == 'SyN'
        assert opts.metric == 'mattes'
        assert opts.label_interpolation == 'genericLabel'
        assert opts.threads is None
        assert opts.verbose is False
        # Defaults should coerce list-defaults into tuples
        assert isinstance(opts.aff_iterations, tuple)
        assert isinstance(opts.reg_iterations, tuple)

    def test_invalid_transform_type_raises(self):
        with pytest.raises(ValueError, match="transform_type must be"):
            AntsRegistrationOptions(transform_type="NonsensePreset")

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric must be"):
            AntsRegistrationOptions(metric="INVALID")

    def test_invalid_label_interpolation_raises(self):
        with pytest.raises(ValueError, match="label_interpolation must be"):
            AntsRegistrationOptions(label_interpolation="linear")

    def test_non_positive_grad_step_raises(self):
        with pytest.raises(ValueError, match="grad_step"):
            AntsRegistrationOptions(grad_step=0.0)
        with pytest.raises(ValueError, match="grad_step"):
            AntsRegistrationOptions(grad_step=-0.1)

    def test_negative_flow_sigma_raises(self):
        with pytest.raises(ValueError, match="flow_sigma"):
            AntsRegistrationOptions(flow_sigma=-1.0)

    def test_negative_total_sigma_raises(self):
        with pytest.raises(ValueError, match="total_sigma"):
            AntsRegistrationOptions(total_sigma=-0.5)

    def test_non_positive_syn_sampling_raises(self):
        with pytest.raises(ValueError, match="syn_sampling"):
            AntsRegistrationOptions(syn_sampling=0)

    def test_empty_aff_iterations_raises(self):
        with pytest.raises(ValueError, match="aff_iterations"):
            AntsRegistrationOptions(aff_iterations=())

    def test_empty_reg_iterations_raises(self):
        with pytest.raises(ValueError, match="reg_iterations"):
            AntsRegistrationOptions(reg_iterations=())

    def test_negative_iteration_entry_raises(self):
        with pytest.raises(ValueError, match="aff_iterations"):
            AntsRegistrationOptions(aff_iterations=(100, -1))
        with pytest.raises(ValueError, match="reg_iterations"):
            AntsRegistrationOptions(reg_iterations=(40, -5, 0))

    def test_zero_iteration_entry_accepted(self):
        # ANTs treats a 0 entry as "skip this level" — valid input.
        opts = AntsRegistrationOptions(reg_iterations=(40, 20, 0))
        assert opts.reg_iterations == (40, 20, 0)

    def test_zero_threads_raises(self):
        with pytest.raises(ValueError, match="threads must be"):
            AntsRegistrationOptions(threads=0)

    def test_positive_threads_accepted(self):
        opts = AntsRegistrationOptions(threads=4)
        assert opts.threads == 4

    def test_list_iterations_coerced_to_tuple(self):
        # When loaded from a YAML dict, iteration schedules arrive as lists.
        opts = AntsRegistrationOptions(
            aff_iterations=[100, 50],
            reg_iterations=[40, 20, 0],
        )
        assert opts.aff_iterations == (100, 50)
        assert opts.reg_iterations == (40, 20, 0)

    def test_valid_transform_type_presets(self):
        for tt in ('Rigid', 'Affine', 'SyN', 'SyNRA', 'SyNOnly',
                   'antsRegistrationSyN[s]'):
            opts = AntsRegistrationOptions(transform_type=tt)
            assert opts.transform_type == tt


# ---------------------------------------------------------------------------
# AntsRegistrationHandler._resolve_options — no external deps needed
# ---------------------------------------------------------------------------

class TestAntsHandlerResolveOptions:
    """Tests for the _resolve_options static method (no antspyx needed)."""

    @pytest.fixture(autouse=True)
    def handler_cls(self):
        from segflow4d.registration.registration_handler.ants.ants_registration_handler import (
            AntsRegistrationHandler,
        )
        self.resolve = AntsRegistrationHandler._resolve_options

    def test_resolve_from_propagation_options_with_ants_opts(self):
        ants_opts = AntsRegistrationOptions(transform_type="SyNRA", metric="CC")
        prop_opts = PropagationOptions(
            lowres_scale_factor=0.5,
            dilation_radius=2,
            registration_backend="ants",
            registration_backend_options=ants_opts,
        )
        resolved = self.resolve(prop_opts)
        assert isinstance(resolved, AntsRegistrationOptions)
        assert resolved.transform_type == "SyNRA"
        assert resolved.metric == "CC"

    def test_resolve_from_dict_with_backend_options(self):
        options_dict = {
            "registration_backend_options": {
                "transform_type": "Affine",
                "metric": "MI",
                "reg_iterations": [40, 20],
                "threads": 2,
            }
        }
        resolved = self.resolve(options_dict)
        assert isinstance(resolved, AntsRegistrationOptions)
        assert resolved.transform_type == "Affine"
        assert resolved.metric == "MI"
        assert resolved.reg_iterations == (40, 20)
        assert resolved.threads == 2

    def test_resolve_from_empty_dict_uses_defaults(self):
        resolved = self.resolve({"registration_backend_options": {}})
        assert isinstance(resolved, AntsRegistrationOptions)
        assert resolved.transform_type == "SyN"
        assert resolved.metric == "mattes"

    def test_resolve_invalid_type_raises(self):
        prop_opts = PropagationOptions(
            lowres_scale_factor=0.5,
            dilation_radius=2,
            registration_backend="ants",
            registration_backend_options=object(),
        )
        with pytest.raises((ValueError, TypeError)):
            self.resolve(prop_opts)


# ---------------------------------------------------------------------------
# Factory wiring — does not require antspyx (only the handler module
# import path is resolved; the lazy ants import happens later).
# ---------------------------------------------------------------------------

class TestAntsHandlerFactory:
    def test_factory_returns_ants_handler(self):
        from segflow4d.registration.registration_handler.registration_handler_factory import (
            RegistrationHandlerFactory,
        )
        from segflow4d.registration.registration_handler.ants.ants_registration_handler import (
            AntsRegistrationHandler,
        )

        handler = RegistrationHandlerFactory.create_registration_handler("ants")
        assert isinstance(handler, AntsRegistrationHandler)
        assert handler.get_device_type() == "cpu"

    def test_ants_is_cpu_only_backend(self):
        from segflow4d.registration.registration_manager.factory import _CPU_ONLY_BACKENDS

        assert "ants" in _CPU_ONLY_BACKENDS


# ---------------------------------------------------------------------------
# AntsRegistrationHandler — requires antspyx (@pytest.mark.ants)
# ---------------------------------------------------------------------------

@pytest.mark.ants
class TestAntsHandlerRegistration:
    """Integration tests for the full ANTs registration pipeline.

    Requires antspyx to be installed::

        pip install segflow4d[ants]   # or: pip install antspyx

    Automatically skipped when the package is missing.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_ants(self):
        pytest.importorskip(
            "ants",
            reason="antspyx not installed — skipping ants handler tests",
        )

    @pytest.fixture
    def handler(self):
        from segflow4d.registration.registration_handler.ants.ants_registration_handler import (
            AntsRegistrationHandler,
        )
        return AntsRegistrationHandler()

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
        warp_sitk = result.warp_image.get_data()
        # ANTs writes vector displacement fields with 3 components per voxel.
        assert warp_sitk.GetNumberOfComponentsPerPixel() == 3

    def test_resliced_segmentation_preserves_pixel_type(self, handler):
        img_fixed = ImageWrapper(_make_sitk_sphere())
        img_moving = ImageWrapper(_make_sitk_sphere())
        seg_sitk = _make_seg_sphere()
        seg = ImageWrapper(seg_sitk)

        result = handler.run_registration_and_reslice(
            img_fixed=img_fixed,
            img_moving=img_moving,
            img_to_reslice=seg,
            mesh_to_reslice=None,
            options=_minimal_prop_opts(),
        )

        assert result.resliced_image.get_data().GetPixelID() == seg_sitk.GetPixelID()

    def test_mask_application_does_not_crash(self, handler):
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

    def test_affine_only_returns_matrix(self, handler):
        img_fixed = ImageWrapper(_make_sitk_sphere())
        img_moving = ImageWrapper(_make_sitk_sphere())

        result = handler.run_affine_only(
            img_fixed=img_fixed,
            img_moving=img_moving,
            options=_minimal_prop_opts(),
        )

        assert result.affine_matrix is not None
        assert isinstance(result.affine_matrix, np.ndarray)
        assert result.affine_matrix.shape == (4, 4)

    def test_affine_with_syn_returns_matrix(self, handler):
        """SyNRA includes an affine stage; affine_matrix should be populated."""
        img_fixed = ImageWrapper(_make_sitk_sphere())
        img_moving = ImageWrapper(_make_sitk_sphere())
        seg = ImageWrapper(_make_seg_sphere())

        result = handler.run_registration_and_reslice(
            img_fixed=img_fixed,
            img_moving=img_moving,
            img_to_reslice=seg,
            mesh_to_reslice=None,
            options=_minimal_prop_opts(_minimal_ants_opts(transform_type='SyNRA')),
        )

        assert result.affine_matrix is not None
        assert result.affine_matrix.shape == (4, 4)

    def test_identity_registration_preserves_labels(self, handler):
        """Registering identical images should produce a seg close to the input."""
        shape = (16, 16, 16)
        seg_sitk = _make_seg_sphere(shape=shape)
        img = ImageWrapper(_make_sitk_sphere(shape=shape))
        seg = ImageWrapper(seg_sitk)

        identity_opts = _minimal_ants_opts(
            aff_iterations=(100, 50),
            reg_iterations=(40, 20, 0),
            aff_shrink_factors=(2, 1),
            aff_smoothing_sigmas=(1, 0),
            random_seed=1,
        )
        result = handler.run_registration_and_reslice(
            img_fixed=img,
            img_moving=img,
            img_to_reslice=seg,
            mesh_to_reslice=None,
            options=_minimal_prop_opts(identity_opts),
        )

        pred = sitk.GetArrayFromImage(result.resliced_image.get_data()).astype(np.int32)
        gt = sitk.GetArrayFromImage(seg_sitk).astype(np.int32)

        dice = 2 * np.sum((pred == 1) & (gt == 1)) / (
            np.sum(pred == 1) + np.sum(gt == 1) + 1e-6
        )
        assert dice >= 0.75, f"Identity registration Dice {dice:.3f} < 0.75"

    def test_sitk_ants_roundtrip_preserves_metadata(self, handler):
        """The internal SITK<->ANTs converters must preserve geometry."""
        from segflow4d.registration.registration_handler.ants.ants_registration_handler import (
            AntsRegistrationHandler,
        )
        import ants  # already importorskip'd above

        # Build a non-trivial image: non-unit spacing, non-zero origin,
        # default identity direction (most ANTs versions only accept
        # diagonal-friendly directions in from_numpy).
        arr = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        sitk_img = sitk.GetImageFromArray(arr)
        sitk_img.SetSpacing((1.5, 2.0, 0.75))
        sitk_img.SetOrigin((-3.0, 4.0, 7.0))

        ants_img = AntsRegistrationHandler._sitk_to_ants(sitk_img, ants)
        back = AntsRegistrationHandler._ants_to_sitk(ants_img)

        assert back.GetSpacing() == pytest.approx(sitk_img.GetSpacing())
        assert back.GetOrigin() == pytest.approx(sitk_img.GetOrigin())
        assert back.GetSize() == sitk_img.GetSize()
        np.testing.assert_array_equal(
            sitk.GetArrayFromImage(back),
            sitk.GetArrayFromImage(sitk_img),
        )
